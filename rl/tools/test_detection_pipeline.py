"""Exercise the FULL detection pipeline the RL policy will see.

Where `rl.tools.test_roboflow` only tests `detect_troops()` (raw model
output), this tool shows what the policy actually consumes after our
post-processing — the (det_class, det_pos, det_side, det_conf) tensors
that go into the observation. That's what matters for whether the agent
can learn from the new clash-royale-bhjq1/2 model.

What it probes:

  * Raw Roboflow detections (class names + bboxes + confidence).
  * Tower synthesis: 6 fixed slots that should appear in every obs
    regardless of what the model emits.
  * HP-bar color team classification: each troop detection should be
    classified friendly or enemy from the colored bar above its bbox,
    with y-position fallback only when no bar is visible.
  * troop_class_to_id() vocab matches: how many model-emitted class
    names successfully resolve to a card ID vs. fall through to UNKNOWN.

Outputs:

  * Console summary of all four signals.
  * detection_pipeline.png: frame with bbox overlays color-coded by
    classified team (red=enemy, blue=friendly, yellow=tower, gray=unknown).
  * detection_pipeline.json: full per-slot dump.

Usage:

    uv run python -m rl.tools.test_detection_pipeline
    uv run python -m rl.tools.test_detection_pipeline --battle-mode auto
    uv run python -m rl.tools.test_detection_pipeline --frames 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pyclashbot.utils.platform import is_macos
from rl.bridge import (
    DEFAULT_TROOP_MODEL_ID,
    NUM_DETECTION_SLOTS,
    SCREEN_H,
    SCREEN_W,
    SIDE_EMPTY,
    SIDE_ENEMY,
    SIDE_FRIENDLY,
    SIDE_TOWER,
    TOWER_POSITIONS,
    classify_detection_with_frame,
    detect_troops,
    encode_detections,
    get_card_id_map,
    get_emulator,
    get_screen,
    is_in_battle,
    start_battle,
    troop_class_to_id,
)


REQUIRED_ENV_VARS = ["ROBOFLOW_API_KEY"]

# Color-code the annotated frame so a glance tells you whether the
# pipeline is classifying teams correctly.
SIDE_COLORS_BGR = {
    SIDE_FRIENDLY: (255, 200, 0),    # cyan-blue
    SIDE_ENEMY: (0, 0, 255),         # red
    SIDE_TOWER: (0, 255, 255),       # yellow
    SIDE_EMPTY: (160, 160, 160),     # gray (shouldn't appear with conf>0)
}
SIDE_NAMES = {
    SIDE_FRIENDLY: "FRIEND",
    SIDE_ENEMY: "ENEMY",
    SIDE_TOWER: "TOWER",
    SIDE_EMPTY: "empty",
}


def _check_env() -> None:
    missing = [k for k in REQUIRED_ENV_VARS if not os.environ.get(k)]
    if missing:
        print("ERROR: missing env vars:", ", ".join(missing))
        raise SystemExit(1)


def _boot_emulator() -> None:
    if is_macos():
        get_emulator(emulator_type="bluestacks", render_settings={"graphics_renderer": "vlcn"})
    else:
        get_emulator(emulator_type="memu", render_mode="directx", debug_mode=True)


def _wait_for_user_in_battle(troop_settle_secs: float) -> np.ndarray:
    print("\nManual mode: navigate into a battle, then press Enter.")
    input("Press Enter when ready... ")
    if troop_settle_secs > 0:
        print(f"  Letting troops settle for {troop_settle_secs:.0f}s...")
        time.sleep(troop_settle_secs)
    frame = get_screen()
    if not is_in_battle():
        print("  [!] is_in_battle() returned False — captured frame may not be a battle.")
    return frame


def _auto_navigate(troop_settle_secs: float) -> np.ndarray:
    print("\nAuto-navigating to Trophy Road via start_battle()...")
    if not start_battle():
        print("  [!] start_battle() failed; falling back to manual mode.")
        return _wait_for_user_in_battle(troop_settle_secs)
    print(f"  Battle started; sleeping {troop_settle_secs:.0f}s for troops...")
    time.sleep(troop_settle_secs)
    return get_screen()


def _summarize_raw(detections: list) -> tuple[int, int]:
    """Print raw model output. Return (n_detections, n_unknown_class)."""
    print(f"\n=== Raw model output: {len(detections)} detections ===")
    if not detections:
        print("  (none)")
        return 0, 0

    by_class: dict[str, list[float]] = {}
    for d in detections:
        by_class.setdefault(d.cls, []).append(d.confidence)

    n_unknown = 0
    for cls in sorted(by_class):
        confs = by_class[cls]
        avg = sum(confs) / len(confs)
        hi = max(confs)
        cls_id = troop_class_to_id(cls)
        if cls_id == 0:
            n_unknown += 1
            tag = "  [UNKNOWN -> id 0]"
        else:
            tag = f"  -> id {cls_id}"
        print(f"  {cls:28s} n={len(confs):2d}  avg={avg:.2f}  max={hi:.2f}{tag}")
    return len(detections), n_unknown


def _summarize_pipeline(
    cls_arr: np.ndarray,
    pos: np.ndarray,
    side: np.ndarray,
    conf: np.ndarray,
) -> None:
    print(f"\n=== Pipeline output (post encode_detections, K={NUM_DETECTION_SLOTS}) ===")
    id_to_name = {v: k for k, v in get_card_id_map().items()}

    n_friend = int((side == SIDE_FRIENDLY).sum())
    n_enemy = int((side == SIDE_ENEMY).sum())
    n_tower = int((side == SIDE_TOWER).sum())
    n_empty = int((side == SIDE_EMPTY).sum())
    print(f"  friendly: {n_friend:2d}   enemy: {n_enemy:2d}   "
          f"tower: {n_tower:2d}   empty: {n_empty:2d}")
    print(f"  expected tower count: {len(TOWER_POSITIONS)} (synthesized)")
    if n_tower < len(TOWER_POSITIONS):
        print("  [!] fewer tower slots than expected — synthesis may be broken.")

    print("\n  per-slot dump (only non-empty):")
    print(f"  {'i':>2}  {'class':<14}  {'side':<7}  {'x':>5}  {'y':>5}  {'conf':>5}")
    for i in range(NUM_DETECTION_SLOTS):
        if side[i] == SIDE_EMPTY and conf[i] == 0:
            continue
        cls_name = id_to_name.get(int(cls_arr[i]), "?")[:14]
        side_name = SIDE_NAMES[int(side[i])]
        x_px = int(pos[i, 0] * SCREEN_W)
        y_px = int(pos[i, 1] * SCREEN_H)
        print(
            f"  {i:>2}  {cls_name:<14}  {side_name:<7}  "
            f"{x_px:>5}  {y_px:>5}  {float(conf[i]):>5.2f}"
        )


def _summarize_team_classification(
    detections: list,
    frame: np.ndarray,
) -> None:
    """Probe what HP-bar sampling decided vs. what y-fallback would say."""
    print("\n=== Team classification breakdown ===")
    if not detections:
        print("  (no troop detections to classify)")
        return
    print(f"  {'class':<22}  {'pos':<10}  {'team':<10}  {'method'}")
    arena_mid = SCREEN_H / 2  # rough — actual midline is 300, see ARENA_MIDLINE_Y
    for d in detections:
        team = classify_detection_with_frame(d, frame)
        # What would y-fallback alone have said?
        y_team = "enemy" if d.y < 300 else "friendly"
        method = "HP-bar" if team != y_team else "y-fallback (or matched HP-bar)"
        print(f"  {d.cls:<22}  ({int(d.x):3d},{int(d.y):3d})  "
              f"{team:<10}  {method}")


def _annotate(
    frame: np.ndarray,
    detections: list,
    pos: np.ndarray,
    side: np.ndarray,
    conf: np.ndarray,
    out_path: Path,
) -> None:
    annotated = frame.copy()
    # Draw raw bboxes (per-troop, since towers are synthetic, no bbox).
    for d in detections:
        team = classify_detection_with_frame(d, frame)
        if team == "friendly":
            color = SIDE_COLORS_BGR[SIDE_FRIENDLY]
        elif team == "enemy":
            color = SIDE_COLORS_BGR[SIDE_ENEMY]
        elif team == "tower":
            color = SIDE_COLORS_BGR[SIDE_TOWER]
        else:
            color = SIDE_COLORS_BGR[SIDE_EMPTY]
        x0, y0, x1, y1 = d.bbox_xyxy
        cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 1)
        cv2.putText(
            annotated, f"{d.cls} {d.confidence:.2f} {team}",
            (x0, max(y0 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA,
        )

    # Draw synthesized tower markers (filled circles so they're distinct
    # from model bboxes).
    for tx, ty, _team_side in TOWER_POSITIONS:
        cv2.circle(annotated, (tx, ty), 8, SIDE_COLORS_BGR[SIDE_TOWER], 2)
        cv2.putText(
            annotated, "T", (tx - 4, ty + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, SIDE_COLORS_BGR[SIDE_TOWER], 1, cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), annotated)


def _dump_per_slot(
    cls_arr: np.ndarray,
    pos: np.ndarray,
    side: np.ndarray,
    conf: np.ndarray,
    out_path: Path,
) -> None:
    id_to_name = {v: k for k, v in get_card_id_map().items()}
    payload = []
    for i in range(NUM_DETECTION_SLOTS):
        payload.append({
            "slot": i,
            "class_id": int(cls_arr[i]),
            "class_name": id_to_name.get(int(cls_arr[i]), "?"),
            "x_norm": float(pos[i, 0]),
            "y_norm": float(pos[i, 1]),
            "x_px": int(pos[i, 0] * SCREEN_W),
            "y_px": int(pos[i, 1] * SCREEN_H),
            "side": SIDE_NAMES[int(side[i])],
            "side_id": int(side[i]),
            "conf": float(conf[i]),
        })
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)


def run_one_frame(frame: np.ndarray, out_dir: Path, suffix: str = "") -> int:
    print(f"\nFrame shape: {frame.shape}")

    t0 = time.monotonic()
    try:
        detections = detect_troops(frame)
    except RuntimeError as e:
        print(f"detect_troops failed: {e}")
        return 1
    detect_ms = (time.monotonic() - t0) * 1000

    t0 = time.monotonic()
    cls_arr, pos, side, conf = encode_detections(detections, frame=frame)
    encode_ms = (time.monotonic() - t0) * 1000

    print(f"\nLatency: detect={detect_ms:.0f}ms  encode={encode_ms:.0f}ms")

    n_total, n_unknown = _summarize_raw(detections)
    if n_total > 0:
        pct_known = 100.0 * (n_total - n_unknown) / n_total if n_total else 0
        print(f"  -> {n_total - n_unknown}/{n_total} class names mapped to card IDs ({pct_known:.0f}%)")

    _summarize_pipeline(cls_arr, pos, side, conf)
    _summarize_team_classification(detections, frame)

    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"detection_pipeline{suffix}.png"
    json_path = out_dir / f"detection_pipeline{suffix}.json"
    _annotate(frame, detections, pos, side, conf, img_path)
    _dump_per_slot(cls_arr, pos, side, conf, json_path)
    print(f"\nAnnotated: {img_path}")
    print(f"Per-slot:  {json_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--battle-mode", choices=["manual", "auto"], default="manual")
    p.add_argument("--troop-settle-secs", type=float, default=15.0)
    p.add_argument("--frames", type=int, default=1,
                   help="How many frames to capture/analyze in sequence (default: 1).")
    p.add_argument("--frame-interval", type=float, default=2.0,
                   help="Seconds between successive frame captures (default: 2.0).")
    p.add_argument("--out-dir", type=Path, default=Path("."))
    args = p.parse_args()

    _check_env()

    print("Booting emulator...")
    _boot_emulator()
    print(f"\nROBOFLOW config:")
    print(f"  api_url:  {os.environ.get('ROBOFLOW_API_URL', 'http://localhost:9001 (default)')}")
    print(f"  model_id: {os.environ.get('ROBOFLOW_TROOP_MODEL_ID', f'{DEFAULT_TROOP_MODEL_ID} (default)')}")

    if args.battle_mode == "auto":
        frame = _auto_navigate(args.troop_settle_secs)
    else:
        frame = _wait_for_user_in_battle(args.troop_settle_secs)

    rc = run_one_frame(frame, args.out_dir, suffix="" if args.frames == 1 else "_0")
    if rc != 0:
        return rc

    for i in range(1, args.frames):
        time.sleep(args.frame_interval)
        print(f"\n========== Frame {i + 1}/{args.frames} ==========")
        rc = run_one_frame(get_screen(), args.out_dir, suffix=f"_{i}")
        if rc != 0:
            return rc

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
