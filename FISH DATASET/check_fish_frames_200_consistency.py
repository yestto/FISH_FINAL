import argparse
import numpy as np
import pandas as pd


def _one(cols: list[str], label: str) -> str:
    if len(cols) != 1:
        raise RuntimeError(f"{label}: expected 1 col, got {cols}")
    return cols[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="fish_frames_200.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    area_col = _one([c for c in df.columns if str(c).startswith("Area") and "truth" not in str(c).lower()], "area_col")
    area_truth_col = _one([c for c in df.columns if str(c).startswith("Area_truth")], "area_truth_col")
    perim_col = _one(
        [c for c in df.columns if str(c).startswith("Perimeter") and "truth" not in str(c).lower()], "perim_col"
    )
    perim_truth_col = _one([c for c in df.columns if str(c).startswith("Perimeter_truth")], "perim_truth_col")

    sel_col = "_SelectionMode" if "_SelectionMode" in df.columns else ("SelectionMode" if "SelectionMode" in df.columns else None)

    print("rows", len(df))
    print("unique_fish", int(df["FishID"].nunique()))
    print("selection_col", sel_col)
    print("resolved_cols", {"area": area_col, "area_truth": area_truth_col, "perimeter": perim_col, "perimeter_truth": perim_truth_col})

    num_cols = [
        "Weight (g)",
        "FrameIndex",
        "Timestamp (s)",
        "FPS_Top",
        "FPS_Front",
        "Length (cm)",
        "Width (cm)",
        "Height (cm)",
        area_col,
        perim_col,
        "TopMaskPixels",
        "FrontMaskPixels",
        "BlurTop",
        "BlurFront",
        "Score",
        "_Truth_Length (cm)",
        "Width_truth (cm)",
        area_truth_col,
        perim_truth_col,
        "_AbsError_Length (cm)",
        "_PerFishScale",
        "_PerFishScale_Width",
        "_PerFishScale_Perimeter",
        "_PerFishScale_Area",
        "_CompositeRelError",
    ]

    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    check_nonneg = [
        "Weight (g)",
        "Length (cm)",
        "Width (cm)",
        "Height (cm)",
        area_col,
        perim_col,
        "TopMaskPixels",
        "FrontMaskPixels",
        "BlurTop",
        "BlurFront",
        "Score",
    ]
    neg_counts: dict[str, int] = {}
    for c in check_nonneg:
        if c in df.columns:
            nneg = int((df[c] < 0).sum())
            if nneg:
                neg_counts[c] = nneg
    print("negative_counts", neg_counts)

    nan_counts = df[[c for c in num_cols if c in df.columns]].isna().sum().sort_values(ascending=False)
    print("top_nan_counts")
    print(nan_counts.head(8).to_string())

    pairs = df[["FishID", "FrameIndex"]].copy()
    pairs["FishID"] = pairs["FishID"].astype(str)
    pairs["FrameIndex"] = pd.to_numeric(pairs["FrameIndex"], errors="coerce")
    print("duplicate_fish_frame_rows", int(pairs.duplicated().sum()))

    print("L_less_than_W", int((df["Length (cm)"] < df["Width (cm)"]).sum()))
    print("any_nonpositive_LWH", int(((df["Length (cm)"] <= 0) | (df["Width (cm)"] <= 0) | (df["Height (cm)"] <= 0)).sum()))

    calc_abs = (df["Length (cm)"] - df["_Truth_Length (cm)"]).abs()
    abs_diff = (calc_abs - df["_AbsError_Length (cm)"]).abs()
    print("absError_max_abs_diff", float(np.nanmax(abs_diff)))
    print("absError_mismatch_count_gt_1e-6", int((abs_diff > 1e-6).sum()))

    relL = (df["Length (cm)"] - df["_Truth_Length (cm)"]).abs() / df["_Truth_Length (cm)"]
    relW = (df["Width (cm)"] - df["Width_truth (cm)"]).abs() / df["Width_truth (cm)"]
    relA = (df[area_col] - df[area_truth_col]).abs() / df[area_truth_col]
    relP = (df[perim_col] - df[perim_truth_col]).abs() / df[perim_truth_col]

    print(
        "overall_median_relative_errors",
        {
            "L": round(float(np.nanmedian(relL)), 3),
            "W": round(float(np.nanmedian(relW)), 3),
            "A": round(float(np.nanmedian(relA)), 3),
            "P": round(float(np.nanmedian(relP)), 3),
        },
    )

    if "_CompositeRelError" in df.columns:
        comp = np.nanmean(np.vstack([relL.to_numpy(), relW.to_numpy(), relA.to_numpy(), relP.to_numpy()]), axis=0)
        d = np.abs(comp - df["_CompositeRelError"].to_numpy())
        print("compositeRelError_median_abs_diff", float(np.nanmedian(d)))
        print("compositeRelError_max_abs_diff", float(np.nanmax(d)))

    if sel_col:
        print("selection_mode_counts")
        print(df[sel_col].value_counts(dropna=False).to_string())

    per_fish = df.assign(relL=relL, relW=relW, relA=relA, relP=relP).groupby("FishID")[["relL", "relW", "relA", "relP"]].median()
    print("worst_5_fish_by_relL")
    print(per_fish.sort_values("relL", ascending=False).head(5).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

