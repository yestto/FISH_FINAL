from pathlib import Path
import cv2
import pandas as pd
from build_fish_dataset import build_background, static_exclude_mask, motion_mask, combine_large_components


def keep_largest_component(mask: "cv2.typing.MatLike") -> "cv2.typing.MatLike":
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype("uint8"), connectivity=8)
    if num <= 1:
        return (mask > 0).astype("uint8") * 0
    best = 1
    best_area = int(stats[1, cv2.CC_STAT_AREA])
    for i in range(2, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a > best_area:
            best = i
            best_area = a
    out = ((labels == best).astype("uint8")) * 255
    return out


def find_video(root: Path, fish_id: str, view_folder: str) -> Path | None:
    folder = root / fish_id / view_folder
    if not folder.exists():
        return None
    videos = sorted(folder.glob("*.mp4"))
    return videos[0] if videos else None


def main() -> int:
    root = Path(__file__).resolve().parent
    csv_path = root / "fish_frames_200.csv"
    out_dir = root / "proof_frames_fish_frames_200"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "FishID" not in df.columns or "FrameIndex" not in df.columns:
        raise RuntimeError("CSV must contain FishID and FrameIndex columns")

    caps: dict[tuple[str, str], cv2.VideoCapture] = {}
    bg_cache: dict[str, dict[str, object]] = {}
    rows_out: list[dict[str, object]] = []
    rows_full: list[dict[str, object]] = []

    for row_idx, row in df.iterrows():
        fish_id = str(row["FishID"])
        frame_idx = int(row["FrameIndex"])

        fish_out = out_dir / fish_id
        fish_out.mkdir(parents=True, exist_ok=True)

        top_key = (fish_id, "top")
        front_key = (fish_id, "front")

        if top_key not in caps:
            top_video = find_video(root, fish_id, "top view")
            if top_video is None:
                raise RuntimeError(f"Top video not found for {fish_id}")
            caps[top_key] = cv2.VideoCapture(str(top_video))
            if fish_id not in bg_cache:
                top_bg = build_background(str(top_video))
                top_ex = static_exclude_mask(top_bg, "top")
                bg_cache[fish_id] = {"top_bg": top_bg, "top_ex": top_ex}

        if front_key not in caps:
            front_video = find_video(root, fish_id, "front view")
            if front_video is None:
                raise RuntimeError(f"Front video not found for {fish_id}")
            caps[front_key] = cv2.VideoCapture(str(front_video))
            if fish_id not in bg_cache:
                bg_cache[fish_id] = {}
            front_bg = build_background(str(front_video))
            front_ex = static_exclude_mask(front_bg, "front")
            bg_cache[fish_id]["front_bg"] = front_bg
            bg_cache[fish_id]["front_ex"] = front_ex

        top_cap = caps[top_key]
        front_cap = caps[front_key]

        top_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        front_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ok_t, top_frame = top_cap.read()
        ok_f, front_frame = front_cap.read()
        if not ok_t or top_frame is None or not ok_f or front_frame is None:
            continue

        base = f"{fish_id}_frame{frame_idx:06d}"
        top_path = fish_out / f"{base}_top.jpg"
        front_path = fish_out / f"{base}_front.jpg"
        top_mask_path = fish_out / f"{base}_top_mask.png"
        front_mask_path = fish_out / f"{base}_front_mask.png"

        cv2.imwrite(str(top_path), top_frame)
        cv2.imwrite(str(front_path), front_frame)
        if fish_id in bg_cache and all(k in bg_cache[fish_id] for k in ["top_bg", "top_ex", "front_bg", "front_ex"]):
            top_mask = motion_mask(
                top_frame,
                bg_cache[fish_id]["top_bg"],
                bg_cache[fish_id]["top_ex"],
                diff_thresh=12,
            )
            front_mask = motion_mask(
                front_frame,
                bg_cache[fish_id]["front_bg"],
                bg_cache[fish_id]["front_ex"],
                diff_thresh=10,
            )
            top_mask = combine_large_components(top_mask, min_area_px=300)
            front_mask = combine_large_components(front_mask, min_area_px=250)
            top_mask = keep_largest_component(top_mask)
            front_mask = keep_largest_component(front_mask)
            cv2.imwrite(str(top_mask_path), top_mask)
            cv2.imwrite(str(front_mask_path), front_mask)

        rec = {
            "FishID": fish_id,
            "FrameIndex": frame_idx,
            "TopFramePath": str(top_path),
            "FrontFramePath": str(front_path),
            "TopMaskPath": str(top_mask_path),
            "FrontMaskPath": str(front_mask_path),
        }
        rows_out.append(rec)
        rows_full.append({"RowIndex": int(row_idx), **rec})

    for cap in caps.values():
        cap.release()

    manifest = pd.DataFrame(rows_out).drop_duplicates(subset=["FishID", "FrameIndex"]).sort_values(
        by=["FishID", "FrameIndex"]
    )
    manifest_path = out_dir / "manifest_saved_frames.csv"
    manifest.to_csv(manifest_path, index=False)

    annotated_dir = out_dir / "annotated_all"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = ["Weight (g)", "Length (cm)", "Width (cm)", "Height (cm)", "Area (cm²)", "Perimeter (cm)"]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    lookup = (
        df.drop_duplicates(subset=["FishID", "FrameIndex"], keep="first")
        .set_index(["FishID", "FrameIndex"], drop=False)
    )
    saved_annotated = 0
    for _, r in manifest.iterrows():
        fish_id = str(r["FishID"])
        frame_idx = int(r["FrameIndex"])
        if (fish_id, frame_idx) not in lookup.index:
            continue
        m = lookup.loc[(fish_id, frame_idx)]
        top = cv2.imread(str(r["TopFramePath"]))
        front = cv2.imread(str(r["FrontFramePath"]))
        if top is None or front is None:
            continue
        h = max(top.shape[0], front.shape[0])
        if top.shape[0] < h:
            top = cv2.copyMakeBorder(top, 0, h - top.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if front.shape[0] < h:
            front = cv2.copyMakeBorder(front, 0, h - front.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panel = cv2.hconcat([top, front])
        t1 = f"FishID={fish_id} FrameIndex={frame_idx} Weight={float(m['Weight (g)']):.2f}g"
        t2 = (
            f"L={float(m['Length (cm)']):.2f} W={float(m['Width (cm)']):.2f} "
            f"H={float(m['Height (cm)']):.2f} A={float(m['Area (cm²)']):.2f} P={float(m['Perimeter (cm)']):.2f}"
        )
        cv2.putText(panel, t1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, t1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, t2, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, t2, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        out_path = annotated_dir / f"{fish_id}_frame{frame_idx:06d}_annotated.jpg"
        cv2.imwrite(str(out_path), panel)
        saved_annotated += 1

    rows_full_df = pd.DataFrame(rows_full)
    rows_full_manifest_path = out_dir / "manifest_rows_all.csv"
    rows_full_df.to_csv(rows_full_manifest_path, index=False)

    annotated_rows_dir = out_dir / "annotated_rows_all"
    annotated_rows_dir.mkdir(parents=True, exist_ok=True)
    saved_row_panels = 0
    for _, r in rows_full_df.iterrows():
        fish_id = str(r["FishID"])
        frame_idx = int(r["FrameIndex"])
        row_idx = int(r["RowIndex"])
        if (fish_id, frame_idx) not in lookup.index:
            continue
        m = lookup.loc[(fish_id, frame_idx)]
        top = cv2.imread(str(r["TopFramePath"]))
        front = cv2.imread(str(r["FrontFramePath"]))
        if top is None or front is None:
            continue
        h = max(top.shape[0], front.shape[0])
        if top.shape[0] < h:
            top = cv2.copyMakeBorder(top, 0, h - top.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if front.shape[0] < h:
            front = cv2.copyMakeBorder(front, 0, h - front.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panel = cv2.hconcat([top, front])
        t1 = f"Row={row_idx} FishID={fish_id} FrameIndex={frame_idx} Weight={float(m['Weight (g)']):.2f}g"
        t2 = (
            f"L={float(m['Length (cm)']):.2f} W={float(m['Width (cm)']):.2f} "
            f"H={float(m['Height (cm)']):.2f} A={float(m['Area (cm²)']):.2f} P={float(m['Perimeter (cm)']):.2f}"
        )
        cv2.putText(panel, t1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, t1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, t2, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, t2, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        out_path = annotated_rows_dir / f"row{row_idx:06d}_{fish_id}_frame{frame_idx:06d}_annotated.jpg"
        cv2.imwrite(str(out_path), panel)
        saved_row_panels += 1

    print(f"Input rows: {len(df)}")
    print(f"Saved unique fish+frame rows: {len(manifest)}")
    print(f"Output directory: {out_dir}")
    print(f"Manifest CSV: {manifest_path}")
    print(f"Annotated panels saved (unique pairs): {saved_annotated}")
    print(f"Annotated directory (unique pairs): {annotated_dir}")
    print(f"Rows manifest CSV: {rows_full_manifest_path}")
    print(f"Annotated panels saved (all rows): {saved_row_panels}")
    print(f"Annotated directory (all rows): {annotated_rows_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
