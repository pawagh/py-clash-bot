"""Reward shaping for the Clash Royale RL environment.

Layers (all active, ordered by leverage):

    1.  Crown differential (CROWN_REWARD * delta) — tower destruction.
    2.  Tower-HP shaping (TOWER_HP_REWARD / TOWER_HP_PENALTY) — reads
        HP bars directly, gives dense per-step signal between crowns.
    3.  Elixir economy:
          * overflow penalty at max elixir (ELIXIR_OVERFLOW_PENALTY),
          * card-play validity filter: if an attempted play did not
            actually spend elixir, it's punished (INVALID_PLAY_PENALTY)
            instead of rewarded.
    4.  Positional priors:
          * defensive-lane bonus for placing a card in the same lane as
            a visible enemy unit (LANE_DEFENSE_BONUS).
        (Opponent-side placement is already impossible by construction:
         action_map's ARENA clamps y to the agent's own half.)
    5.  Time-structured shaping:
          * urgency — small per-step penalty that scales up as the
            battle progresses (URGENCY_COEF * fraction),
          * overtime-aware step-cost scaling (STEP_COST * 0.5 once the
            battle passes OVERTIME_START_SECONDS).
    6.  Terminal bonus / penalty (WIN_BONUS, LOSS_PENALTY) on the first
        frame after the battle ends.

All CV readers share a single frame passed into calculate(), so the
per-step wall-clock cost of reward shaping is a handful of masked
rectangle crops rather than six screenshot round-trips.

Calibration: RewardCalculator.dump_debug() writes an annotated PNG
showing every detector's sample points and current reading. Use it
against a live battle to tune the *_COORDS / HSV constants.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

from rl.bridge import (
    ARENA_MIDLINE_Y,
    TroopDetection,
    classify_detection,
    get_screen,
    is_battle_over,
    is_in_battle,
    summarize_detections,
)


@dataclass
class StepInfo:
    """Lightweight side-channel the env passes into RewardCalculator.

    Fields:
        card_played: True if the env attempted a card-play this step.
        no_op: True if the action decoded to the no-op slot.
        play_x: pixel x of the placement (only meaningful when
            card_played=True). Used for lane defense detection.
        play_y: pixel y of the placement. Used for lane-opened bonus
            (was the play in the enemy half of an opened lane?) and
            for radius-based counter-push attribution.
    """

    card_played: bool = False
    no_op: bool = False
    play_x: int = -1
    play_y: int = -1


@dataclass
class _PendingPlay:
    """A card-play whose downstream effect we resolve a few steps later.

    Used by the counter-push / negative-trade / wasted-spell shaping:
    we record the enemy detection count within a radius around the
    placement at play-time, and on resolution compare against the
    current count to attribute kills (or lack thereof) to the play.
    """

    step: int
    x: int
    y: int
    enemies_in_radius: int


class RewardCalculator:
    """Tracks per-episode state and emits a scalar reward per step."""

    # ------------------------------------------------------------------
    # Reward coefficients. Tune after the first 50k-step training run.
    # ------------------------------------------------------------------
    STEP_COST: ClassVar[float] = -0.005
    NOOP_COST: ClassVar[float] = -0.002
    CARD_PLAYED_BONUS: ClassVar[float] = 0.01
    WIN_BONUS: ClassVar[float] = 5.0
    LOSS_PENALTY: ClassVar[float] = -5.0
    CROWN_REWARD: ClassVar[float] = 2.0

    # Elixir economy
    ELIXIR_OVERFLOW_PENALTY: ClassVar[float] = -0.02
    INVALID_PLAY_PENALTY: ClassVar[float] = -0.03

    # Tower HP shaping (per unit HP fraction lost)
    TOWER_HP_REWARD: ClassVar[float] = 0.5
    TOWER_HP_PENALTY: ClassVar[float] = -0.5

    # Positional priors
    LANE_DEFENSE_BONUS: ClassVar[float] = 0.03

    # Unit-mass differential (board-state shaping). Per pixel-mass delta
    # step-to-step. Capped both ways to prevent a single frame of mask
    # flicker from dominating a reward, and to cap the marginal value of
    # "unit-count farming" strategies (spamming cheap skeletons to
    # inflate friendly mass). Elixir regeneration already bounds the
    # steady-state friendly mass; the clip is a second line of defense.
    UNIT_DIFF_REWARD: ClassVar[float] = 0.003
    UNIT_DIFF_CLIP: ClassVar[float] = 0.15

    # Detection-based unit count differential. When Roboflow is online
    # we get exact integer counts of friendly/enemy troops on the board
    # instead of pixel masses, so the per-unit weight is much higher
    # than UNIT_DIFF_REWARD. Capped per-step to prevent a flicker of
    # detection (a unit briefly missed for one frame and recovered the
    # next) from blowing up the reward.
    TROOP_COUNT_REWARD: ClassVar[float] = 0.05
    TROOP_COUNT_CLIP: ClassVar[float] = 0.20
    # Bonus when the agent plays into a lane that already has a visible
    # enemy troop on the agent's side. Higher than LANE_DEFENSE_BONUS
    # because detections are far more reliable than red-HP-bar HSV.
    TROOP_LANE_DEFENSE_BONUS: ClassVar[float] = 0.05

    # Counter-push / negative-trade / wasted-spell shaping.
    # On each play we record enemy detections within PLAY_RADIUS_PX of
    # the placement; PLAY_RESOLVE_WINDOW steps later we re-count and:
    #   * killed enemies in radius      -> +COUNTER_PUSH_BONUS_PER_KILL
    #   * play near enemies, none died  -> NEGATIVE_TRADE_PENALTY
    #   * play in empty area (spell?)   -> WASTED_SPELL_PENALTY
    # This handles arrows/fireball misuse without needing to know which
    # slot is a spell — playing any card in an empty area earns a small
    # penalty proportional to the elixir wasted.
    PLAY_RESOLVE_WINDOW: ClassVar[int] = 3  # steps; ~1.8s at FRAME_SKIP_SECONDS=0.6
    PLAY_RADIUS_PX: ClassVar[int] = 100
    COUNTER_PUSH_BONUS_PER_KILL: ClassVar[float] = 0.10
    NEGATIVE_TRADE_PENALTY: ClassVar[float] = -0.05
    WASTED_SPELL_PENALTY: ClassVar[float] = -0.04

    # Lane-opened bonus: once an opp princess tower is destroyed, the
    # agent can place troops on the enemy half of that lane. This is
    # a high-leverage action, so reward it explicitly — relying solely
    # on the eventual tower-HP / crown reward leaves too thin a
    # gradient for the policy to discover the behavior.
    LANE_OPENED_BONUS: ClassVar[float] = 0.20
    # Tower-HP threshold below which we consider a tower "destroyed"
    # for the lane-opened check. The HP reader can read 0.0-0.05 due
    # to debris animations on the destroyed-tower frame.
    LANE_OPENED_HP_THRESHOLD: ClassVar[float] = 0.05

    # Hasty-play penalty. Discourages spending elixir at low-elixir,
    # low-pressure moments — i.e. the agent should learn to bank
    # elixir for either an incoming defensive engagement or an empty-
    # lane initiation push, instead of dumping cards the moment they
    # become affordable. Only fires when there is neither an enemy on
    # the agent's half nor a friendly unit already pushing.
    HASTY_PLAY_PENALTY: ClassVar[float] = -0.04
    HASTY_PLAY_ELIXIR_THRESHOLD: ClassVar[int] = 5

    # Time-structured shaping
    URGENCY_COEF: ClassVar[float] = -0.002
    OVERTIME_STEP_COST_SCALE: ClassVar[float] = 0.5
    OVERTIME_START_SECONDS: ClassVar[float] = 180.0
    MAX_BATTLE_SECONDS: ClassVar[float] = 300.0

    # ------------------------------------------------------------------
    # Crown-reader tuning (calibrated for 419x633).
    # ------------------------------------------------------------------
    SCOREBOARD_BAND: ClassVar[tuple[int, int]] = (22, 70)
    SELF_X_RANGE: ClassVar[tuple[int, int]] = (130, 208)
    OPP_X_RANGE: ClassVar[tuple[int, int]] = (212, 290)
    GOLD_HSV_LOWER: ClassVar[tuple[int, int, int]] = (15, 120, 150)
    GOLD_HSV_UPPER: ClassVar[tuple[int, int, int]] = (40, 255, 255)
    MIN_CROWN_AREA: ClassVar[int] = 40
    MAX_CROWNS_PER_SIDE: ClassVar[int] = 3

    # ------------------------------------------------------------------
    # Elixir-reader tuning. Mirrors pyclashbot.bot.fight.ELIXIR_COORDS
    # (row, col) = (y, x). ELIXIR_COLOR is the magenta-ish fill.
    # ------------------------------------------------------------------
    ELIXIR_COORDS: ClassVar[tuple[tuple[int, int], ...]] = (
        (613, 149),
        (613, 165),
        (613, 188),
        (613, 212),
        (613, 240),
        (613, 262),
        (613, 287),
        (613, 314),
        (613, 339),
        (613, 364),
    )
    ELIXIR_COLOR_BGR: ClassVar[tuple[int, int, int]] = (240, 137, 244)
    ELIXIR_COLOR_TOL: ClassVar[int] = 50
    MAX_ELIXIR: ClassVar[int] = 10

    # ------------------------------------------------------------------
    # Tower HP bar sample points. Each entry is (x, y) center of a
    # small horizontal strip covering the bar. Order:
    #   [self_left_princess, self_king, self_right_princess,
    #    opp_left_princess,  opp_king,  opp_right_princess]
    # Values calibrated for a 419x633 frame.
    # ------------------------------------------------------------------
    TOWER_HP_COORDS: ClassVar[tuple[tuple[int, int], ...]] = (
        (104, 484),  # self left princess
        (210, 543),  # self king
        (313, 484),  # self right princess
        (104, 112),  # opp left princess
        (210, 56),   # opp king
        (313, 112),  # opp right princess
    )
    # Horizontal half-length of the HP-bar strip we sample. The bar is
    # ~30px wide; 15 on each side of the center.
    HP_BAR_HALF_WIDTH: ClassVar[int] = 15
    HP_BAR_HALF_HEIGHT: ClassVar[int] = 2
    # HSV bounds for "healthy" (green) and "damaged" (red/orange) fill.
    HP_ACTIVE_HSV_LOWER: ClassVar[tuple[int, int, int]] = (0, 100, 100)
    HP_ACTIVE_HSV_UPPER: ClassVar[tuple[int, int, int]] = (90, 255, 255)
    HP_EMPTY_HSV_LOWER: ClassVar[tuple[int, int, int]] = (0, 0, 0)
    HP_EMPTY_HSV_UPPER: ClassVar[tuple[int, int, int]] = (179, 60, 80)
    HP_MIN_SAMPLED_PIXELS: ClassVar[int] = 10

    # ------------------------------------------------------------------
    # Enemy-presence detection for lane defense. Enemy HP bars over
    # their units are bright red; detecting them in the agent's half
    # of the arena tells us which lane has an incoming threat.
    # ------------------------------------------------------------------
    ENEMY_PRESENCE_BAND: ClassVar[tuple[int, int]] = (316, 520)
    LANE_SPLIT_X: ClassVar[int] = 210
    # Red wraps around the hue circle; we mask both ends.
    ENEMY_RED_HSV_LOWER_A: ClassVar[tuple[int, int, int]] = (0, 150, 150)
    ENEMY_RED_HSV_UPPER_A: ClassVar[tuple[int, int, int]] = (10, 255, 255)
    ENEMY_RED_HSV_LOWER_B: ClassVar[tuple[int, int, int]] = (170, 150, 150)
    ENEMY_RED_HSV_UPPER_B: ClassVar[tuple[int, int, int]] = (179, 255, 255)
    ENEMY_MIN_PIXELS: ClassVar[int] = 20

    # ------------------------------------------------------------------
    # Unit-mass detection for board-state differential. Scans the full
    # playable arena (above the hand tray, below the scoreboard) for
    # blue (friendly) vs red (enemy) HP-bar pixels. Each unit on screen
    # draws a small HP bar above itself; counting these pixels gives a
    # proxy for "which side is winning the board right now".
    # ------------------------------------------------------------------
    ARENA_BAND: ClassVar[tuple[int, int]] = (75, 520)
    # Friendly HP bars are blue (often cyan-blue).
    FRIENDLY_BLUE_HSV_LOWER: ClassVar[tuple[int, int, int]] = (95, 120, 140)
    FRIENDLY_BLUE_HSV_UPPER: ClassVar[tuple[int, int, int]] = (130, 255, 255)

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self.prev_my_crowns = 0
        self.prev_opp_crowns = 0
        self.prev_elixir = 0
        self.prev_tower_hps_my: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.prev_tower_hps_opp: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.prev_unit_diff: int = 0
        self.prev_troop_count_diff: int = 0
        self.prev_friendly_count: int = 0
        self._was_in_battle = False
        self._battle_start_time: float | None = None
        self._step_idx: int = 0
        self._pending_plays: list[_PendingPlay] = []

    def reset(self) -> None:
        self.prev_my_crowns = 0
        self.prev_opp_crowns = 0
        self.prev_elixir = 0
        self.prev_tower_hps_my = (1.0, 1.0, 1.0)
        self.prev_tower_hps_opp = (1.0, 1.0, 1.0)
        self.prev_unit_diff = 0
        self.prev_troop_count_diff = 0
        self.prev_friendly_count = 0
        self._was_in_battle = False
        self._battle_start_time = None
        self._step_idx = 0
        self._pending_plays = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def calculate(
        self,
        info: StepInfo,
        frame: np.ndarray | None = None,
        detections: list[TroopDetection] | None = None,
    ) -> tuple[float, bool]:
        """Return (reward, terminated) for the current step.

        If `frame` is None, one is pulled from the emulator; pass one
        in from the env step loop to avoid a redundant screenshot.

        If `detections` is provided (Roboflow troop detection enabled),
        the unit-mass and lane-defense components are driven by those
        detections instead of HSV color masks. Detection-based signals
        are far more reliable than HSV (no false positives from UI
        chrome, no missed units due to non-red HP bars), so when
        available they get higher coefficients. When `detections=None`,
        we fall back to the HSV components silently.
        """
        if frame is None:
            try:
                frame = get_screen()
            except Exception:
                frame = None

        currently_in_battle = is_in_battle()
        if currently_in_battle and self._battle_start_time is None:
            self._battle_start_time = time.monotonic()

        self._step_idx += 1
        overtime = self._is_overtime()
        step_scale = self.OVERTIME_STEP_COST_SCALE if overtime else 1.0
        reward = self.STEP_COST * step_scale

        if info.no_op:
            reward += self.NOOP_COST

        # Snapshot current elixir for hasty-play check (before _score_elixir
        # mutates self.prev_elixir).
        current_elixir = self._read_elixir(frame) if frame is not None else self.prev_elixir

        reward += self._score_elixir(frame, info)
        reward += self._score_crowns(frame)
        reward += self._score_tower_hp(frame)

        if detections is not None:
            # Detection-based shaping is strictly better than HSV when
            # available — replace, don't stack.
            reward += self._score_troop_count_diff(detections)
            reward += self._score_troop_lane_defense(detections, info)
            # Counter-push / negative-trade / wasted-spell shaping
            # needs detections to attribute kills to placements.
            reward += self._resolve_pending_plays(detections)
            self._register_play_if_any(info, detections)
            reward += self._score_hasty_play(detections, info, current_elixir)
        else:
            reward += self._score_unit_differential(frame)
            reward += self._score_lane_defense(frame, info)

        # Lane-opened bonus uses tower-HP state only (no detections needed),
        # so it runs in both detection and no-detection paths.
        reward += self._score_lane_opened(info)

        reward += self._score_urgency()

        terminated = False
        battle_ended = (
            self._was_in_battle and not currently_in_battle and is_battle_over()
        )
        if battle_ended:
            terminated = True
            reward += self._terminal_reward(
                self.prev_my_crowns, self.prev_opp_crowns
            )
        self._was_in_battle = self._was_in_battle or currently_in_battle
        return reward, terminated

    # ------------------------------------------------------------------
    # Reward components
    # ------------------------------------------------------------------
    def _score_elixir(self, frame: np.ndarray | None, info: StepInfo) -> float:
        """Elixir economy: overflow penalty + card-play validity filter."""
        reward = 0.0
        curr_elixir = self._read_elixir(frame) if frame is not None else self.prev_elixir

        if info.card_played:
            # A real play consumes 2-9 elixir. If elixir did NOT drop,
            # the play was illegal (wrong card slot, insufficient
            # elixir, or mis-click). Punish instead of reward.
            if curr_elixir < self.prev_elixir:
                reward += self.CARD_PLAYED_BONUS
            else:
                reward += self.INVALID_PLAY_PENALTY

        if curr_elixir >= self.MAX_ELIXIR:
            reward += self.ELIXIR_OVERFLOW_PENALTY

        self.prev_elixir = curr_elixir
        return reward

    def _score_crowns(self, frame: np.ndarray | None) -> float:
        my_crowns, opp_crowns = self._read_crowns(frame)
        reward = 0.0
        if my_crowns > self.prev_my_crowns:
            reward += self.CROWN_REWARD * (my_crowns - self.prev_my_crowns)
        if opp_crowns > self.prev_opp_crowns:
            reward -= self.CROWN_REWARD * (opp_crowns - self.prev_opp_crowns)
        self.prev_my_crowns = my_crowns
        self.prev_opp_crowns = opp_crowns
        return reward

    def _score_tower_hp(self, frame: np.ndarray | None) -> float:
        """Dense shaping on tower HP deltas. Only negative deltas count
        (HP never regenerates in Clash Royale), which filters out
        reader noise.
        """
        my_hps, opp_hps = self._read_tower_hps(frame)
        reward = 0.0
        for i in range(3):
            opp_drop = self.prev_tower_hps_opp[i] - opp_hps[i]
            if opp_drop > 0:
                reward += self.TOWER_HP_REWARD * opp_drop
            my_drop = self.prev_tower_hps_my[i] - my_hps[i]
            if my_drop > 0:
                reward += self.TOWER_HP_PENALTY * my_drop
        self.prev_tower_hps_my = my_hps
        self.prev_tower_hps_opp = opp_hps
        return reward

    def _score_lane_defense(
        self, frame: np.ndarray | None, info: StepInfo
    ) -> float:
        """Bonus for playing into the lane where a visible enemy is."""
        if not info.card_played or info.play_x < 0 or frame is None:
            return 0.0
        left_present, right_present = self._read_enemy_presence(frame)
        played_left = info.play_x < self.LANE_SPLIT_X
        if played_left and left_present:
            return self.LANE_DEFENSE_BONUS
        if not played_left and right_present:
            return self.LANE_DEFENSE_BONUS
        return 0.0

    def _score_troop_count_diff(
        self, detections: list[TroopDetection]
    ) -> float:
        """Detection-based unit count differential (board-state shaping).

        Equivalent in spirit to `_score_unit_differential` but uses
        Roboflow integer counts instead of HSV pixel masses. Telescopes
        over the episode (a unit added then later killed yields net
        zero), so it cannot be exploited by spamming cheap troops.
        Per-step delta is hard-clipped to TROOP_COUNT_CLIP to bound the
        impact of one-frame detection flicker.
        """
        friendly, enemy = summarize_detections(detections)
        curr_diff = friendly - enemy
        delta = curr_diff - self.prev_troop_count_diff
        self.prev_troop_count_diff = curr_diff
        shaped = self.TROOP_COUNT_REWARD * delta
        return max(-self.TROOP_COUNT_CLIP, min(self.TROOP_COUNT_CLIP, shaped))

    def _score_troop_lane_defense(
        self,
        detections: list[TroopDetection],
        info: StepInfo,
    ) -> float:
        """Bonus for placing a card in the lane of an enemy on our half.

        An enemy detection counts as a threat when its center is below
        ARENA_MIDLINE_Y (i.e. already past the bridge into the agent's
        half). The lane is determined by x vs LANE_SPLIT_X. Unlike the
        HSV version this works regardless of HP-bar visibility, so
        spawn-protected and freshly-deployed enemies still register.
        """
        if not info.card_played or info.play_x < 0:
            return 0.0

        threat_left = False
        threat_right = False
        for det in detections:
            if classify_detection(det) != "enemy":
                continue
            if det.y < ARENA_MIDLINE_Y:
                continue
            if det.x < self.LANE_SPLIT_X:
                threat_left = True
            else:
                threat_right = True

        played_left = info.play_x < self.LANE_SPLIT_X
        if played_left and threat_left:
            return self.TROOP_LANE_DEFENSE_BONUS
        if not played_left and threat_right:
            return self.TROOP_LANE_DEFENSE_BONUS
        return 0.0

    def _enemies_in_radius(
        self,
        detections: list[TroopDetection],
        x: int,
        y: int,
        radius: int,
    ) -> int:
        """Count enemy detections within `radius` of (x, y)."""
        if x < 0 or y < 0:
            return 0
        n = 0
        r2 = radius * radius
        for det in detections:
            if classify_detection(det) != "enemy":
                continue
            dx = det.x - x
            dy = det.y - y
            if dx * dx + dy * dy <= r2:
                n += 1
        return n

    def _register_play_if_any(
        self,
        info: StepInfo,
        detections: list[TroopDetection],
    ) -> None:
        """Record a fresh play for delayed counter-push attribution."""
        if not info.card_played or info.play_x < 0 or info.play_y < 0:
            return
        enemies = self._enemies_in_radius(
            detections, info.play_x, info.play_y, self.PLAY_RADIUS_PX
        )
        self._pending_plays.append(
            _PendingPlay(
                step=self._step_idx,
                x=info.play_x,
                y=info.play_y,
                enemies_in_radius=enemies,
            )
        )

    def _resolve_pending_plays(
        self,
        detections: list[TroopDetection],
    ) -> float:
        """Resolve plays whose effect window has elapsed.

        For each play that's now PLAY_RESOLVE_WINDOW steps old:
          * killed enemies in radius   -> +COUNTER_PUSH_BONUS_PER_KILL * killed
          * enemies were nearby, none died -> NEGATIVE_TRADE_PENALTY
          * no enemies were ever nearby   -> WASTED_SPELL_PENALTY
        """
        reward = 0.0
        survivors: list[_PendingPlay] = []
        for p in self._pending_plays:
            age = self._step_idx - p.step
            if age < self.PLAY_RESOLVE_WINDOW:
                survivors.append(p)
                continue

            curr = self._enemies_in_radius(
                detections, p.x, p.y, self.PLAY_RADIUS_PX
            )
            killed = p.enemies_in_radius - curr
            if killed > 0:
                reward += self.COUNTER_PUSH_BONUS_PER_KILL * killed
            elif p.enemies_in_radius > 0:
                # Engaged enemies but failed to kill any — bad trade.
                reward += self.NEGATIVE_TRADE_PENALTY
            else:
                # Played in an empty area. If it was a spell (arrows /
                # fireball), this caught nothing. If it was a troop,
                # it's just a deployment — but spending elixir on a
                # placement that touches nothing is wasteful enough
                # for a small penalty.
                reward += self.WASTED_SPELL_PENALTY
        self._pending_plays = survivors
        return reward

    def _score_lane_opened(self, info: StepInfo) -> float:
        """Bonus for placing on the enemy half of a lane whose princess is dead.

        Uses self.prev_tower_hps_opp which was just updated by
        _score_tower_hp this step. Index layout:
          [0] left princess, [1] king, [2] right princess.
        """
        if not info.card_played or info.play_x < 0 or info.play_y < 0:
            return 0.0
        # Only enemy half placements qualify.
        if info.play_y >= ARENA_MIDLINE_Y:
            return 0.0
        threshold = self.LANE_OPENED_HP_THRESHOLD
        played_left = info.play_x < self.LANE_SPLIT_X
        if played_left and self.prev_tower_hps_opp[0] <= threshold:
            return self.LANE_OPENED_BONUS
        if not played_left and self.prev_tower_hps_opp[2] <= threshold:
            return self.LANE_OPENED_BONUS
        return 0.0

    def _score_hasty_play(
        self,
        detections: list[TroopDetection],
        info: StepInfo,
        elixir: int,
    ) -> float:
        """Penalize low-elixir plays with no defensive or offensive context.

        Fires only when *all* of:
          * a card was played,
          * elixir at play time was below HASTY_PLAY_ELIXIR_THRESHOLD,
          * no enemy detections on the agent's half (no defense needed),
          * no friendly detections on the enemy half (no push to support).

        Encourages banking elixir for either an incoming engagement or a
        committed push, instead of the current "spend the moment it's
        affordable" behavior.
        """
        if not info.card_played:
            return 0.0
        if elixir >= self.HASTY_PLAY_ELIXIR_THRESHOLD:
            return 0.0

        any_enemy_threat = False
        any_friendly_pushing = False
        for det in detections:
            side = classify_detection(det)
            if side == "enemy" and det.y >= ARENA_MIDLINE_Y:
                any_enemy_threat = True
            elif side == "friendly" and det.y < ARENA_MIDLINE_Y:
                any_friendly_pushing = True
            if any_enemy_threat and any_friendly_pushing:
                break
        if any_enemy_threat or any_friendly_pushing:
            return 0.0
        return self.HASTY_PLAY_PENALTY

    def _score_unit_differential(self, frame: np.ndarray | None) -> float:
        """Dense shaping on friendly-vs-enemy unit mass on the board.

        Computes per-step delta of (friendly_pixels - enemy_pixels), so
        the total reward telescopes and "spam cheap units" strategies
        net zero over the life of the farmed units (they die, the
        friendly mass deflates, cancelling the early gains). Further
        guarded by a hard per-step clip.
        """
        if frame is None:
            return 0.0
        friendly, enemy = self._read_unit_mass(frame)
        curr_diff = friendly - enemy
        delta = curr_diff - self.prev_unit_diff
        self.prev_unit_diff = curr_diff
        shaped = self.UNIT_DIFF_REWARD * delta
        return max(-self.UNIT_DIFF_CLIP, min(self.UNIT_DIFF_CLIP, shaped))

    def _score_urgency(self) -> float:
        """Tiny per-step penalty that scales up as the battle drags on."""
        frac = self._battle_fraction()
        return self.URGENCY_COEF * frac

    def _terminal_reward(self, my_crowns: int, opp_crowns: int) -> float:
        if my_crowns > opp_crowns:
            return self.WIN_BONUS
        if my_crowns < opp_crowns:
            return self.LOSS_PENALTY
        return 0.0  # draw

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------
    def _elapsed(self) -> float:
        if self._battle_start_time is None:
            return 0.0
        return time.monotonic() - self._battle_start_time

    def _battle_fraction(self) -> float:
        return min(1.0, self._elapsed() / self.MAX_BATTLE_SECONDS)

    def _is_overtime(self) -> bool:
        return self._elapsed() >= self.OVERTIME_START_SECONDS

    # ------------------------------------------------------------------
    # CV readers
    # ------------------------------------------------------------------
    def _read_crowns(
        self, frame: np.ndarray | None
    ) -> tuple[int, int]:
        """Count destroyed-tower crown indicators, monotonic + capped."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return self.prev_my_crowns, self.prev_opp_crowns

        my_count = self._count_crowns(frame, self.SELF_X_RANGE)
        opp_count = self._count_crowns(frame, self.OPP_X_RANGE)

        my_count = max(my_count, self.prev_my_crowns)
        opp_count = max(opp_count, self.prev_opp_crowns)
        return (
            min(my_count, self.MAX_CROWNS_PER_SIDE),
            min(opp_count, self.MAX_CROWNS_PER_SIDE),
        )

    @classmethod
    def _count_crowns(cls, frame: np.ndarray, x_range: tuple[int, int]) -> int:
        y0, y1 = cls.SCOREBOARD_BAND
        x0, x1 = x_range
        h, w = frame.shape[:2]
        y0, y1 = max(0, y0), min(h, y1)
        x0, x1 = max(0, x0), min(w, x1)
        if y1 <= y0 or x1 <= x0:
            return 0

        band = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array(cls.GOLD_HSV_LOWER, dtype=np.uint8),
            np.array(cls.GOLD_HSV_UPPER, dtype=np.uint8),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        big = [c for c in contours if cv2.contourArea(c) >= cls.MIN_CROWN_AREA]
        return min(len(big), cls.MAX_CROWNS_PER_SIDE)

    def _read_elixir(self, frame: np.ndarray | None) -> int:
        """Count filled segments of the elixir bar (0..10).

        The bar fills left-to-right; we scan from position 1 up and
        return the highest filled slot. Gracefully handles pre-battle
        frames by returning 0.
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return self.prev_elixir

        h, w = frame.shape[:2]
        target = np.array(self.ELIXIR_COLOR_BGR, dtype=np.int16)
        tol = self.ELIXIR_COLOR_TOL
        count = 0
        for y, x in self.ELIXIR_COORDS:
            if y >= h or x >= w:
                break
            px = frame[y, x].astype(np.int16)
            if np.all(np.abs(px - target) <= tol):
                count += 1
            else:
                break
        return count

    def _read_tower_hps(
        self, frame: np.ndarray | None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Sample the six HP bars and return (my_hps, opp_hps)."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return self.prev_tower_hps_my, self.prev_tower_hps_opp

        def sample(i: int, prev: float) -> float:
            cx, cy = self.TOWER_HP_COORDS[i]
            return self._hp_from_bar(frame, cx, cy, prev)

        my = (
            sample(0, self.prev_tower_hps_my[0]),
            sample(1, self.prev_tower_hps_my[1]),
            sample(2, self.prev_tower_hps_my[2]),
        )
        opp = (
            sample(3, self.prev_tower_hps_opp[0]),
            sample(4, self.prev_tower_hps_opp[1]),
            sample(5, self.prev_tower_hps_opp[2]),
        )
        return my, opp

    @classmethod
    def _hp_from_bar(
        cls,
        frame: np.ndarray,
        cx: int,
        cy: int,
        prev: float,
    ) -> float:
        """Return HP fraction in [0, 1] from the bar centered at (cx, cy).

        The HP bar is a filled strip that shrinks rightward as HP drops.
        We count how many pixels in the strip match "active fill"
        (saturated green/yellow/red) vs "empty bar" (dark). HP is
        active / (active + empty); if neither category has enough
        pixels (e.g. tower undamaged → no bar visible), we return the
        previous HP unchanged.
        """
        h, w = frame.shape[:2]
        x0 = max(0, cx - cls.HP_BAR_HALF_WIDTH)
        x1 = min(w, cx + cls.HP_BAR_HALF_WIDTH + 1)
        y0 = max(0, cy - cls.HP_BAR_HALF_HEIGHT)
        y1 = min(h, cy + cls.HP_BAR_HALF_HEIGHT + 1)
        if x1 <= x0 or y1 <= y0:
            return prev

        strip = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

        active = cv2.inRange(
            hsv,
            np.array(cls.HP_ACTIVE_HSV_LOWER, dtype=np.uint8),
            np.array(cls.HP_ACTIVE_HSV_UPPER, dtype=np.uint8),
        )
        empty = cv2.inRange(
            hsv,
            np.array(cls.HP_EMPTY_HSV_LOWER, dtype=np.uint8),
            np.array(cls.HP_EMPTY_HSV_UPPER, dtype=np.uint8),
        )

        n_active = int(np.count_nonzero(active))
        n_empty = int(np.count_nonzero(empty))
        total = n_active + n_empty
        if total < cls.HP_MIN_SAMPLED_PIXELS:
            # Tower undamaged or animation frame with no bar visible.
            return prev

        hp = n_active / total
        # Never allow HP to regenerate within an episode.
        return min(hp, prev)

    def _read_unit_mass(
        self, frame: np.ndarray
    ) -> tuple[int, int]:
        """Count friendly (blue) and enemy (red) HP-bar pixels on the board.

        Returns (friendly_pixels, enemy_pixels) summed over the full
        playable arena. Each unit on screen draws a ~20-40 px HP bar
        above itself as soon as it takes or deals damage; a healthy
        but undamaged unit may not have one (fine for RL — we care
        about contested state, not pristine spawns).
        """
        h, w = frame.shape[:2]
        y0, y1 = self.ARENA_BAND
        y0, y1 = max(0, y0), min(h, y1)
        if y1 <= y0:
            return 0, 0

        band = frame[y0:y1, :]
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)

        enemy_mask = cv2.bitwise_or(
            cv2.inRange(
                hsv,
                np.array(self.ENEMY_RED_HSV_LOWER_A, dtype=np.uint8),
                np.array(self.ENEMY_RED_HSV_UPPER_A, dtype=np.uint8),
            ),
            cv2.inRange(
                hsv,
                np.array(self.ENEMY_RED_HSV_LOWER_B, dtype=np.uint8),
                np.array(self.ENEMY_RED_HSV_UPPER_B, dtype=np.uint8),
            ),
        )
        friendly_mask = cv2.inRange(
            hsv,
            np.array(self.FRIENDLY_BLUE_HSV_LOWER, dtype=np.uint8),
            np.array(self.FRIENDLY_BLUE_HSV_UPPER, dtype=np.uint8),
        )

        # Zero out the scoreboard region and the hand-tray/elixir UI
        # to avoid counting blue/red UI chrome as units.
        self._mask_out_ui(friendly_mask)
        self._mask_out_ui(enemy_mask)

        return (
            int(np.count_nonzero(friendly_mask)),
            int(np.count_nonzero(enemy_mask)),
        )

    @staticmethod
    def _mask_out_ui(mask: np.ndarray) -> None:
        """Zero UI chrome areas in an arena-band mask in place.

        The arena band starts at ARENA_BAND[0] (above the top
        scoreboard's crown-count row), so the top ~5 rows may overlap
        the bottom of the scoreboard tail. Similarly the bottom of the
        band passes just above the hand tray which has blue/red
        elements in some visual themes. Defensive zeroing keeps the
        mass differential honest.
        """
        # Top scoreboard tail (first 5 rows of the band).
        if mask.shape[0] > 5:
            mask[:5, :] = 0
        # Bottom hand-tray overlap (last 5 rows of the band).
        if mask.shape[0] > 5:
            mask[-5:, :] = 0

    def _read_enemy_presence(
        self, frame: np.ndarray
    ) -> tuple[bool, bool]:
        """Is there an enemy unit in the left / right lane of our half?

        Enemy HP bars are red; detecting them in the agent's half of
        the arena signals an incoming threat. Returns (left, right)
        booleans.
        """
        h, w = frame.shape[:2]
        y0, y1 = self.ENEMY_PRESENCE_BAND
        y0, y1 = max(0, y0), min(h, y1)
        if y1 <= y0:
            return False, False

        band = frame[y0:y1, :]
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        mask_a = cv2.inRange(
            hsv,
            np.array(self.ENEMY_RED_HSV_LOWER_A, dtype=np.uint8),
            np.array(self.ENEMY_RED_HSV_UPPER_A, dtype=np.uint8),
        )
        mask_b = cv2.inRange(
            hsv,
            np.array(self.ENEMY_RED_HSV_LOWER_B, dtype=np.uint8),
            np.array(self.ENEMY_RED_HSV_UPPER_B, dtype=np.uint8),
        )
        mask = cv2.bitwise_or(mask_a, mask_b)

        split = min(max(0, self.LANE_SPLIT_X), w)
        left_px = int(np.count_nonzero(mask[:, :split]))
        right_px = int(np.count_nonzero(mask[:, split:]))
        return (
            left_px >= self.ENEMY_MIN_PIXELS,
            right_px >= self.ENEMY_MIN_PIXELS,
        )

    # ------------------------------------------------------------------
    # Debug / calibration
    # ------------------------------------------------------------------
    def dump_debug(self, out_path: str | Path = "reward_debug.png") -> Path:
        """Annotated screenshot showing every detector's reading.

        Run while a battle is in progress to verify calibration. The
        output PNG shows:
            * scoreboard crown detection boxes + per-side counts,
            * each elixir sample pixel + current elixir count,
            * tower HP sample strips + HP fraction,
            * enemy-presence red mask overlay + per-lane booleans.
        """
        frame = get_screen()
        annotated = frame.copy()

        # Crown boxes
        for x_range, color, label in (
            (self.SELF_X_RANGE, (80, 200, 80), "self"),
            (self.OPP_X_RANGE, (80, 80, 220), "opp"),
        ):
            y0, y1 = self.SCOREBOARD_BAND
            x0, x1 = x_range
            cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 1)
            count = self._count_crowns(frame, x_range)
            cv2.putText(
                annotated, f"{label}:{count}", (x0, y1 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

        # Elixir pixels
        elixir = self._read_elixir(frame)
        for i, (y, x) in enumerate(self.ELIXIR_COORDS):
            c = (0, 255, 0) if i < elixir else (0, 0, 255)
            cv2.circle(annotated, (x, y), 2, c, -1)
        cv2.putText(
            annotated, f"elixir:{elixir}/10", (370, 620),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
        )

        # Tower HP bars
        my_hps, opp_hps = self._read_tower_hps(frame)
        labels = ["sL", "sK", "sR", "oL", "oK", "oR"]
        for i, (cx, cy) in enumerate(self.TOWER_HP_COORDS):
            x0 = cx - self.HP_BAR_HALF_WIDTH
            x1 = cx + self.HP_BAR_HALF_WIDTH
            y0 = cy - self.HP_BAR_HALF_HEIGHT
            y1 = cy + self.HP_BAR_HALF_HEIGHT
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 255), 1)
            hp = (my_hps + opp_hps)[i]
            cv2.putText(
                annotated, f"{labels[i]}:{hp:.2f}", (cx - 14, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1,
            )

        # Unit-mass differential
        friendly_mass, enemy_mass = self._read_unit_mass(frame)
        y0, y1 = self.ARENA_BAND
        cv2.rectangle(
            annotated, (0, y0), (annotated.shape[1] - 1, y1),
            (200, 200, 0), 1,
        )
        cv2.putText(
            annotated,
            f"F:{friendly_mass} E:{enemy_mass} d:{friendly_mass - enemy_mass:+d}",
            (10, y0 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1,
        )

        # Enemy presence
        left_p, right_p = self._read_enemy_presence(frame)
        y0, y1 = self.ENEMY_PRESENCE_BAND
        cv2.rectangle(
            annotated, (0, y0), (self.LANE_SPLIT_X, y1),
            (0, 200, 255) if left_p else (120, 120, 120), 1,
        )
        cv2.rectangle(
            annotated, (self.LANE_SPLIT_X, y0), (annotated.shape[1] - 1, y1),
            (0, 200, 255) if right_p else (120, 120, 120), 1,
        )
        cv2.putText(
            annotated, f"L:{int(left_p)} R:{int(right_p)}", (10, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1,
        )

        path = Path(out_path)
        cv2.imwrite(str(path), annotated)
        return path
