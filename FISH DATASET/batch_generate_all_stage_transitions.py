import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def _process_one(root: Path, export_py: str, board_py: str, out_dir: str, fish: str, frame: int, skip_existing: bool) -> tuple[str, str]:
    stage_dir = root / out_dir / fish / f"frame_{frame:06d}"
    top_board = stage_dir / "TOP_STAGE_TRANSITION_BOARD.jpg"
    front_board = stage_dir / "FRONT_STAGE_TRANSITION_BOARD.jpg"

    if bool(skip_existing) and top_board.exists() and front_board.exists():
        return ("skipped", "")

    try:
        subprocess.run(
            [
                "python",
                export_py,
                "--dataset-root",
                str(root),
                "--fish-id",
                fish,
                "--frame-index",
                str(frame),
                "--out-dir",
                out_dir,
            ],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "python",
                board_py,
                "--stage-dir",
                str(stage_dir),
            ],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
        return ("ok", "")
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "")[-500:]
        return ("fail", msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=".")
    ap.add_argument(
        "--csv",
        default="fish_frames_200_filtered_200_per_fish.csv",
        help="CSV with FishID and FrameIndex rows to export.",
    )
    ap.add_argument("--out-dir", default="final stages all frames")
    ap.add_argument("--max-rows", type=int, default=0, help="0 means all rows.")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers for faster generation (recommended 4-6).",
    )
    args = ap.parse_args()

    root = Path(args.dataset_root).resolve()
    csv_path = root / args.csv
    if not csv_path.exists():
        raise RuntimeError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "FishID" not in df.columns or "FrameIndex" not in df.columns:
        raise RuntimeError("CSV must contain FishID and FrameIndex")

    rows = df[["FishID", "FrameIndex"]].copy()
    rows["FishID"] = rows["FishID"].astype(str)
    rows["FrameIndex"] = pd.to_numeric(rows["FrameIndex"], errors="coerce").astype("Int64")
    rows = rows.dropna(subset=["FrameIndex"])
    rows["FrameIndex"] = rows["FrameIndex"].astype(int)

    if int(args.max_rows) > 0:
        rows = rows.head(int(args.max_rows)).copy()

    tasks = [(str(r.FishID), int(r.FrameIndex)) for r in rows.itertuples(index=False)]
    total = len(tasks)
    ok = 0
    skipped = 0
    fail = 0
    failures = []

    export_py = str(root / "export_pipeline_stage_images.py")
    board_py = str(root / "make_stage_transition_board.py")

    workers = max(1, int(args.workers))
    workers = min(workers, max(1, (os.cpu_count() or 4)))
    print(f"starting total={total} workers={workers}")

    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fish, frame in tasks:
            fu = ex.submit(_process_one, root, export_py, board_py, args.out_dir, fish, frame, bool(args.skip_existing))
            futures[fu] = (fish, frame)

        done = 0
        for fu in as_completed(futures):
            done += 1
            fish, frame = futures[fu]
            status, err = fu.result()
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                fail += 1
                failures.append((fish, frame, err))

            if done % 50 == 0 or done == total:
                print(f"progress {done}/{total} ok={ok} skipped={skipped} fail={fail}")

    report = root / args.out_dir / "_batch_generation_report.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"source_csv={csv_path.name}\n")
        f.write(f"total_rows={total}\n")
        f.write(f"ok={ok}\n")
        f.write(f"skipped={skipped}\n")
        f.write(f"fail={fail}\n")
        if failures:
            f.write("failures:\n")
            for fish, frame, err in failures[:200]:
                f.write(f"{fish},frame={frame},{err}\n")

    print("done")
    print(f"out_dir={root / args.out_dir}")
    print(f"report={report}")
    print(f"ok={ok} skipped={skipped} fail={fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

