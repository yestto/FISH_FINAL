import argparse
from pathlib import Path

import cv2
import numpy as np


def _read(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _tile(stages: list[tuple[str, Path]], title: str, out_path: Path, cell_w: int = 420, cell_h: int = 260, cols: int = 4) -> None:
    rows = (len(stages) + cols - 1) // cols
    pad = 18
    header_h = 70
    label_h = 38
    board_w = cols * cell_w + (cols + 1) * pad
    board_h = header_h + rows * (cell_h + label_h) + (rows + 1) * pad
    board = np.full((board_h, board_w, 3), 20, dtype=np.uint8)

    cv2.putText(board, title, (pad, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2, cv2.LINE_AA)

    for i, (label, path) in enumerate(stages):
        r = i // cols
        c = i % cols
        x0 = pad + c * (cell_w + pad)
        y0 = header_h + pad + r * (cell_h + label_h)
        img = _fit(_read(path), cell_w, cell_h)
        board[y0 : y0 + cell_h, x0 : x0 + cell_w] = img
        cv2.rectangle(board, (x0, y0), (x0 + cell_w, y0 + cell_h), (80, 80, 80), 1)
        cv2.putText(board, label, (x0 + 6, y0 + cell_h + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), board)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", required=True, help="Folder created by export_pipeline_stage_images.py, e.g. .../frame_000289")
    args = ap.parse_args()
    d = Path(args.stage_dir).resolve()

    top = [
        ("01 Raw", d / "01_top_raw.jpg"),
        ("02 Background", d / "02_top_background.jpg"),
        ("03 AbsDiff", d / "03_top_absdiff.jpg"),
        ("04 Diff Gray", d / "04_top_diff_gray.jpg"),
        ("05 Threshold", d / "05_top_threshold.jpg"),
        ("06 Exclude Mask", d / "06_top_exclude_mask.jpg"),
        ("07 After Exclude", d / "07_top_after_exclude.jpg"),
        ("08 Open", d / "08_top_open.jpg"),
        ("09 Close", d / "09_top_close.jpg"),
        ("10 Erode", d / "10_top_erode.jpg"),
        ("11 Dilate", d / "11_top_dilate.jpg"),
        ("12 Components", d / "12_top_components.jpg"),
        ("13 Geometry Overlay", d / "13_top_geometry_overlay.jpg"),
        ("40 Final Mask", d / "40_top_final_mask.png"),
    ]

    front = [
        ("21 Raw", d / "21_front_raw.jpg"),
        ("22 Background", d / "22_front_background.jpg"),
        ("23 AbsDiff", d / "23_front_absdiff.jpg"),
        ("24 Diff Gray", d / "24_front_diff_gray.jpg"),
        ("25 Threshold", d / "25_front_threshold.jpg"),
        ("26 Exclude Mask", d / "26_front_exclude_mask.jpg"),
        ("27 After Exclude", d / "27_front_after_exclude.jpg"),
        ("28 Open", d / "28_front_open.jpg"),
        ("29 Close", d / "29_front_close.jpg"),
        ("30 Erode", d / "30_front_erode.jpg"),
        ("31 Dilate", d / "31_front_dilate.jpg"),
        ("32 Components", d / "32_front_components.jpg"),
        ("33 Height Overlay", d / "33_front_height_overlay.jpg"),
        ("41 Final Mask", d / "41_front_final_mask.png"),
    ]

    _tile(top, "Top View Stage-by-Stage Transition", d / "TOP_STAGE_TRANSITION_BOARD.jpg")
    _tile(front, "Front View Stage-by-Stage Transition", d / "FRONT_STAGE_TRANSITION_BOARD.jpg")
    print("Saved:")
    print(d / "TOP_STAGE_TRANSITION_BOARD.jpg")
    print(d / "FRONT_STAGE_TRANSITION_BOARD.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

