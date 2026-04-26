"""Sanity-check the Roboflow troop detection bridge on a live BlueStacks frame.

Prerequisites:

    1. Run a local Roboflow inference server. Either:
         * GUI: download the .dmg from
           https://github.com/roboflow/inference/releases/latest
           drag to /Applications, launch it (menu-bar icon).
         * CLI: `uv sync --group roboflow` then `uv run inference server start`.
       Either way it listens on http://localhost:9001.
    2. Get a private API key at https://app.roboflow.com/settings/api
    3. Export env vars before running this script:
           export ROBOFLOW_API_KEY='<your key>'
           # Optional. Defaults to 'clash-royale-of3d3/1' if unset.
           export ROBOFLOW_TROOP_MODEL_ID='<project-slug>/<version>'

IMPORTANT — model_id format:
    The SDK requires exactly two parts: '<project-slug>/<version>'.
    NOT '<workspace>/<project>/<version>' (that returns 400 Invalid Model ID).
    Universe project slugs have unique random suffixes, so two parts is enough.

To find the right model_id: open a public Clash Royale detection project on
Universe, click 'Deploy' (top right), and copy the string from 'Copy Model ID'.
Default points at https://universe.roboflow.com/clashroyale/clash-royale-of3d3
which is a public model with 72 classes, accessible to any valid API key.

Usage:

    # Default: prompt before capture so you can manually start a battle.
    uv run python -m rl.tools.test_roboflow

    # Auto-navigate into a Trophy Road match using rl.bridge.start_battle():
    uv run python -m rl.tools.test_roboflow --battle-mode auto

    # Latency benchmark (run 20 back-to-back detections; you must already be
    # inside a battle when iteration 1 fires):
    uv run python -m rl.tools.test_roboflow --battle-mode auto --benchmark 20

Output:

    * Console summary: detections per class, confidence stats, latency.
    * ``roboflow_test.png``: frame with bounding boxes overlaid.
    * ``roboflow_raw.json``: full upstream response so you can inspect the
      exact shape of the model output.

Decision criteria from this test:

    * p95 latency << FRAME_SKIP_SECONDS (0.6s = 600ms)? → Roboflow is viable.
    * p95 latency ≈ 600ms? → increase FRAME_SKIP_SECONDS or add a GPU.
    * p95 latency > 1s? → drop Roboflow for HSV-based unit detection.
    * Detections sensible (troops tagged, towers not falsely tagged)? → classes ok.
    * Empty detections even with troops on field? → check roboflow_raw.json:
      either the model's not loading (first call downloads ~200MB of weights)
      or the screen resolution differs from what the model expects.
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
    detect_troops,
    get_emulator,
    get_screen,
    is_in_battle,
    start_battle,
)


REQUIRED_ENV_VARS = ["ROBOFLOW_API_KEY"]
BUDGET_MS = 600  # must leave headroom for action dispatch + env bookkeeping


def _check_env() -> None:
    missing = [k for k in REQUIRED_ENV_VARS if not os.environ.get(k)]
    if missing:
        print("ERROR: missing required environment variables:")
        for k in missing:
            print(f"  {k}")
        print("\nSee the module docstring for setup instructions.")
        raise SystemExit(1)


def _preflight(frame: np.ndarray) -> int:
    """Issue one warmup detection before the main workload.

    Fails fast if the server isn't running, the model id is wrong, or
    auth is broken — so we don't spin through 20 iterations before
    finding out. Also forces the inference server to download model
    weights (~200MB on first call), which would otherwise skew the
    benchmark's first iteration.
    """
    print("\nPreflight: one warmup detection (may take 30-60s on first run while")
    print("the inference server downloads ~200MB of model weights)...")
    start = time.monotonic()
    try:
        detections = detect_troops(frame)
    except RuntimeError as e:
        print(f"\nPreflight FAILED:\n\n{e}\n")
        return 1
    except Exception as e:  # noqa: BLE001  -- surface unexpected errors verbatim
        print(f"\nPreflight FAILED with unexpected error:\n  {type(e).__name__}: {e}\n")
        return 1
    elapsed_ms = (time.monotonic() - start) * 1000
    print(f"  OK ({elapsed_ms:.0f}ms, {len(detections)} detections)")
    return 0


def _boot_emulator() -> np.ndarray:
    if is_macos():
        get_emulator(emulator_type="bluestacks", render_settings={"graphics_renderer": "vlcn"})
    else:
        get_emulator(emulator_type="memu", render_mode="directx", debug_mode=True)

    frame = get_screen()
    if frame is None or frame.size == 0:
        raise RuntimeError("Emulator returned empty frame; is BlueStacks running?")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(f"Expected BGR frame, got shape={frame.shape}")
    return frame


def _wait_for_user_in_battle(troop_settle_secs: float) -> np.ndarray:
    """Pause until the user manually navigates into a battle, then capture.

    Useful when the auto-navigator (start_battle) has trouble — for
    example because Trophy Road is gated behind a tutorial, or you want
    to test on a 2v2 / Ladder match the auto-navigator doesn't pick.
    """
    print(
        "\nManual mode: navigate into a Clash Royale battle inside the "
        "BlueStacks 'pyclashbot-136' window."
    )
    print(f"Wait until BOTH sides have troops on the field, then add ~{troop_settle_secs:.0f}s")
    print("of buffer so the model sees a non-trivial frame.")
    input("Press Enter when ready to capture... ")
    if troop_settle_secs > 0:
        print(f"  Letting troops settle for {troop_settle_secs:.0f}s...")
        time.sleep(troop_settle_secs)
    frame = get_screen()
    if not is_in_battle():
        print("  [!] is_in_battle() returned False — the captured frame may not be a battle.")
    return frame


def _auto_navigate_into_battle(troop_settle_secs: float) -> np.ndarray:
    """Use rl.bridge.start_battle() to enter a Trophy Road match.

    After the battle starts, sleep so opponent troops can spawn before
    we grab a frame. If start_battle() fails (e.g. the account is
    mid-tutorial), fall back to manual mode.
    """
    print("\nAuto mode: navigating to a Trophy Road battle via rl.bridge.start_battle()...")
    if not start_battle():
        print("  [!] start_battle() failed. Falling back to manual mode.")
        return _wait_for_user_in_battle(troop_settle_secs)
    print(f"  Battle started. Waiting {troop_settle_secs:.0f}s for troops to spawn...")
    time.sleep(troop_settle_secs)
    return get_screen()


def _summarize(detections: list) -> None:
    if not detections:
        print("  (no detections)")
        return
    by_class: dict[str, list[float]] = {}
    for d in detections:
        by_class.setdefault(d.cls, []).append(d.confidence)
    for cls in sorted(by_class):
        confs = by_class[cls]
        avg = sum(confs) / len(confs)
        hi = max(confs)
        print(f"  {cls:28s} n={len(confs):2d}  avg={avg:.2f}  max={hi:.2f}")


def _annotate(frame: np.ndarray, detections: list, out_path: Path) -> None:
    annotated = frame.copy()
    for d in detections:
        x0, y0, x1, y1 = d.bbox_xyxy
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 1)
        label = f"{d.cls} {d.confidence:.2f}"
        cv2.putText(
            annotated, label, (x0, max(y0 - 3, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), annotated)


def _dump_raw(detections: list, out_path: Path) -> None:
    payload = [
        {
            "cls": d.cls,
            "confidence": d.confidence,
            "x": d.x,
            "y": d.y,
            "width": d.width,
            "height": d.height,
            "raw": d.raw,
        }
        for d in detections
    ]
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)


def run_single(frame: np.ndarray, out_dir: Path) -> int:
    print("\nRunning single detection...")
    start = time.monotonic()
    try:
        detections = detect_troops(frame)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1
    elapsed_ms = (time.monotonic() - start) * 1000

    print(f"Detected {len(detections)} objects in {elapsed_ms:.0f}ms (budget: {BUDGET_MS}ms)")
    _summarize(detections)

    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / "roboflow_test.png"
    raw_path = out_dir / "roboflow_raw.json"
    _annotate(frame, detections, img_path)
    _dump_raw(detections, raw_path)
    print(f"\nAnnotated frame: {img_path}")
    print(f"Raw predictions: {raw_path}")

    if not detections:
        print(
            "\n[!] No detections returned. Possible causes:\n"
            "    1. The emulator isn't showing an active battle frame with visible troops.\n"
            "       Inspect roboflow_test.png and confirm what the model actually saw.\n"
            "    2. All detections fell below the min_confidence threshold (default 0.25).\n"
            "       Re-run with a lower threshold by editing detect_troops(min_confidence=0.1).\n"
            "    3. The model trained on a different screen resolution or skin and doesn't\n"
            "       generalise to BlueStacks 419x633 frames. Try a different Universe model."
        )
    return 0


def run_benchmark(iterations: int) -> int:
    print(f"\nRunning latency benchmark: {iterations} iterations...")
    print("Tip: keep units on the field during the run for realistic latency.")

    times: list[float] = []
    det_counts: list[int] = []
    for i in range(iterations):
        frame = get_screen()
        start = time.monotonic()
        try:
            detections = detect_troops(frame)
        except RuntimeError as e:
            print(f"  iter {i + 1}: ERROR {e}")
            return 1
        elapsed_ms = (time.monotonic() - start) * 1000
        times.append(elapsed_ms)
        det_counts.append(len(detections))
        print(f"  iter {i + 1:2d}: {elapsed_ms:6.0f}ms   n={len(detections)}")

    times_sorted = sorted(times)
    p50 = times_sorted[len(times_sorted) // 2]
    p95 = times_sorted[min(int(len(times_sorted) * 0.95), len(times_sorted) - 1)]
    avg = sum(times) / len(times)
    print("\n=== Latency summary ===")
    print(f"  min:  {min(times):6.0f}ms")
    print(f"  p50:  {p50:6.0f}ms")
    print(f"  mean: {avg:6.0f}ms")
    print(f"  p95:  {p95:6.0f}ms")
    print(f"  max:  {max(times):6.0f}ms")
    print(f"  detections/iter: min={min(det_counts)}  mean={sum(det_counts) / len(det_counts):.1f}  max={max(det_counts)}")

    print(f"\nStep budget: {BUDGET_MS}ms (FRAME_SKIP_SECONDS=0.6s)")
    if p95 > BUDGET_MS:
        print(
            f"  [!] p95 is OVER BUDGET by {p95 - BUDGET_MS:.0f}ms.\n"
            "      Options: (a) accept a bigger FRAME_SKIP_SECONDS, (b) enable GPU on the\n"
            "      inference server, (c) fall back to HSV-based unit detection."
        )
        return 2
    print(f"  OK — p95 has {BUDGET_MS - p95:.0f}ms headroom.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=int, default=0,
                        help="Run N back-to-back detections to measure latency (0=single run)")
    parser.add_argument("--out-dir", type=Path, default=Path("."),
                        help="Where to write roboflow_test.png and roboflow_raw.json")
    parser.add_argument("--battle-mode", choices=["manual", "auto", "skip"], default="manual",
                        help="manual: prompt before capture so you can navigate yourself "
                             "(default). auto: invoke rl.bridge.start_battle() automatically. "
                             "skip: capture immediately after boot (only useful for debugging).")
    parser.add_argument("--troop-settle-secs", type=float, default=15.0,
                        help="Seconds to wait between confirming you're in a battle and "
                             "capturing, so opponent troops have time to spawn (default: 15).")
    args = parser.parse_args()

    _check_env()

    print("Booting emulator (first run may take 30-60s)...")
    boot_frame = _boot_emulator()
    print(f"Boot frame: shape={boot_frame.shape}")

    print("\nProbing Roboflow configuration:")
    print(f"  API URL:  {os.environ.get('ROBOFLOW_API_URL', 'http://localhost:9001')}")
    print(f"  model id: {os.environ.get('ROBOFLOW_TROOP_MODEL_ID', f'{DEFAULT_TROOP_MODEL_ID} (default)')}")

    # Run preflight on the post-boot frame (probably the main menu) so we
    # confirm auth + connectivity BEFORE waiting for a battle. That way a
    # broken setup fails fast rather than after the user navigates.
    rc = _preflight(boot_frame)
    if rc != 0:
        return rc

    if args.battle_mode == "manual":
        frame = _wait_for_user_in_battle(args.troop_settle_secs)
    elif args.battle_mode == "auto":
        frame = _auto_navigate_into_battle(args.troop_settle_secs)
    else:
        frame = boot_frame
    print(f"Capture frame: shape={frame.shape}")

    if args.benchmark > 0:
        return run_benchmark(args.benchmark)
    return run_single(frame, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
