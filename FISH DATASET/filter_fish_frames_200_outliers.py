import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _one(cols: list[str], label: str) -> str:
    if len(cols) != 1:
        raise RuntimeError(f"{label}: expected 1 col, got {cols}")
    return cols[0]


def _resolve_cols(df: pd.DataFrame) -> dict[str, str]:
    area = _one([c for c in df.columns if str(c).startswith("Area") and "truth" not in str(c).lower()], "area_col")
    area_truth = _one([c for c in df.columns if str(c).startswith("Area_truth")], "area_truth_col")
    perim = _one(
        [c for c in df.columns if str(c).startswith("Perimeter") and "truth" not in str(c).lower()], "perim_col"
    )
    perim_truth = _one([c for c in df.columns if str(c).startswith("Perimeter_truth")], "perim_truth_col")
    sel = "_SelectionMode" if "_SelectionMode" in df.columns else ("SelectionMode" if "SelectionMode" in df.columns else "")
    return {
        "area": area,
        "area_truth": area_truth,
        "perim": perim,
        "perim_truth": perim_truth,
        "sel": sel,
    }


def _to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="fish_frames_200.csv")
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--keep-per-fish", type=int, default=200)
    ap.add_argument(
        "--per-fish-robust-z",
        type=float,
        default=0.0,
        help="Optional within-fish robust z-score filter using median+MAD on L/W/H/A/P; set 0 to disable (recommended 3.5).",
    )
    ap.add_argument(
        "--height-max-ratio",
        type=float,
        default=0.0,
        help="Optional plausibility filter: drop rows where Height > (height_max_ratio * Length); set 0 to disable (recommended 1.2).",
    )
    ap.add_argument("--out-prefix", default="fish_frames_200_filtered")
    args = ap.parse_args()

    src = Path(args.csv)
    df_raw = pd.read_csv(src)
    cols = _resolve_cols(df_raw)

    df = _to_num(
        df_raw,
        [
            "FrameIndex",
            "Weight (g)",
            "Length (cm)",
            "Width (cm)",
            "Height (cm)",
            cols["area"],
            cols["perim"],
            "_Truth_Length (cm)",
            "Width_truth (cm)",
            cols["area_truth"],
            cols["perim_truth"],
            "Score",
        ],
    )

    relL = (df["Length (cm)"] - df["_Truth_Length (cm)"]).abs() / df["_Truth_Length (cm)"]
    relW = (df["Width (cm)"] - df["Width_truth (cm)"]).abs() / df["Width_truth (cm)"]
    relA = (df[cols["area"]] - df[cols["area_truth"]]).abs() / df[cols["area_truth"]]
    relP = (df[cols["perim"]] - df[cols["perim_truth"]]).abs() / df[cols["perim_truth"]]
    comp = np.nanmean(np.vstack([relL.to_numpy(), relW.to_numpy(), relA.to_numpy(), relP.to_numpy()]), axis=0)

    df = df.assign(_relL=relL, _relW=relW, _relA=relA, _relP=relP, _CompositeRelErrorRecalc=comp)

    q = float(args.quantile)
    if not (0.0 < q < 1.0):
        raise RuntimeError("--quantile must be in (0,1)")
    thr = float(np.nanpercentile(comp, q * 100.0))

    kept = df[df["_CompositeRelErrorRecalc"] <= thr].copy()
    outliers = df[df["_CompositeRelErrorRecalc"] > thr].copy()

    zthr = float(args.per_fish_robust_z)
    outliers_within_fish = df.iloc[0:0].copy()
    if zthr > 0:
        metric_cols = ["Length (cm)", "Width (cm)", "Height (cm)", cols["area"], cols["perim"]]

        def _robust_z(x: pd.Series) -> pd.Series:
            med = float(np.nanmedian(x.to_numpy()))
            mad = float(np.nanmedian(np.abs(x.to_numpy() - med)))
            if np.isfinite(mad) and mad > 0:
                return 0.6745 * (x - med) / mad
            q75 = float(np.nanpercentile(x.to_numpy(), 75))
            q25 = float(np.nanpercentile(x.to_numpy(), 25))
            iqr = q75 - q25
            if np.isfinite(iqr) and iqr > 0:
                return 0.7413 * (x - med) / iqr
            sd = float(np.nanstd(x.to_numpy()))
            if np.isfinite(sd) and sd > 0:
                return (x - float(np.nanmean(x.to_numpy()))) / sd
            return pd.Series(np.zeros(len(x), dtype=np.float64), index=x.index)

        flags: list[pd.Series] = []
        for c in metric_cols:
            z = kept.groupby("FishID")[c].transform(_robust_z).abs()
            flags.append(z > zthr)
        bad = flags[0]
        for f in flags[1:]:
            bad = bad | f

        outliers_within_fish = kept[bad].copy()
        kept = kept[~bad].copy()

    out_prefix = Path(args.out_prefix)
    outliers_path = out_prefix.with_name(out_prefix.name + "_outliers.csv")
    outliers_within_fish_path = out_prefix.with_name(out_prefix.name + "_outliers_within_fish.csv")
    kept_all_path = out_prefix.with_name(out_prefix.name + "_allrows.csv")
    kept_unique_path = out_prefix.with_name(out_prefix.name + "_unique_frames.csv")
    kept_fixed_path = out_prefix.with_name(out_prefix.name + "_200_per_fish.csv")
    report_path = out_prefix.with_name(out_prefix.name + "_report.txt")

    kept.to_csv(kept_all_path, index=False)
    outliers.to_csv(outliers_path, index=False)
    if zthr > 0:
        outliers_within_fish.to_csv(outliers_within_fish_path, index=False)

    kept_unique = kept.drop_duplicates(subset=["FishID", "FrameIndex"], keep="first").copy()
    kept_unique.to_csv(kept_unique_path, index=False)

    hr = float(args.height_max_ratio)
    outliers_height_path = out_prefix.with_name(out_prefix.name + "_outliers_height.csv")
    outliers_height = df.iloc[0:0].copy()
    if hr > 0:
        bad_h = kept["Height (cm)"] > (hr * kept["Length (cm)"])
        outliers_height = kept[bad_h].copy()
        kept = kept[~bad_h].copy()
        outliers_height.to_csv(outliers_height_path, index=False)

    k = int(args.keep_per_fish)
    if k <= 0:
        raise RuntimeError("--keep-per-fish must be positive")

    sort_cols = ["_CompositeRelErrorRecalc"]
    if "Score" in kept.columns:
        sort_cols.append("Score")
    kept_sorted = kept.sort_values(sort_cols, ascending=[True] + [False] * (len(sort_cols) - 1))

    fixed_parts: list[pd.DataFrame] = []
    for fish_id, g in kept_sorted.groupby("FishID", sort=False):
        g = g.copy()
        if len(g) >= k:
            fixed_parts.append(g.head(k))
        else:
            fixed_parts.append(g)
            need = k - len(g)
            if len(g) > 0:
                fixed_parts.append(g.head(1).loc[g.head(1).index.repeat(need)])

    kept_fixed = pd.concat(fixed_parts, ignore_index=True)
    kept_fixed.to_csv(kept_fixed_path, index=False)

    def med_rel(d: pd.DataFrame) -> dict[str, float]:
        return {
            "relL_med": float(np.nanmedian(d["_relL"].to_numpy())),
            "relW_med": float(np.nanmedian(d["_relW"].to_numpy())),
            "relA_med": float(np.nanmedian(d["_relA"].to_numpy())),
            "relP_med": float(np.nanmedian(d["_relP"].to_numpy())),
            "comp_med": float(np.nanmedian(d["_CompositeRelErrorRecalc"].to_numpy())),
            "comp_p95": float(np.nanpercentile(d["_CompositeRelErrorRecalc"].to_numpy(), 95)),
            "comp_max": float(np.nanmax(d["_CompositeRelErrorRecalc"].to_numpy())),
        }

    overall_before = med_rel(df)
    overall_after = med_rel(kept)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Source: {src.name}\n")
        f.write(f"Rows: {len(df)} | Unique fish: {df['FishID'].nunique()}\n")
        f.write(f"Composite rel-error quantile: {q} | threshold: {thr:.6f}\n")
        f.write(f"Kept rows: {len(kept)} | Outliers removed: {len(outliers)}\n")
        if zthr > 0:
            f.write(f"Within-fish robust-z filter: {zthr} | Removed: {len(outliers_within_fish)}\n")
        if hr > 0:
            f.write(f"Height plausibility filter: Height <= {hr} * Length | Removed: {len(outliers_height)}\n")
        f.write("\nOverall median relative errors BEFORE:\n")
        for k2, v2 in overall_before.items():
            f.write(f"  {k2}: {v2:.6f}\n")
        f.write("\nOverall median relative errors AFTER:\n")
        for k2, v2 in overall_after.items():
            f.write(f"  {k2}: {v2:.6f}\n")
        f.write("\nSelectionMode counts (kept):\n")
        if cols["sel"] and cols["sel"] in kept.columns:
            for key, val in kept[cols["sel"]].value_counts(dropna=False).to_dict().items():
                f.write(f"  {key}: {val}\n")

    print("threshold", thr)
    print("kept_rows", len(kept), "outliers_removed", len(outliers))
    print("outputs")
    print(" ", kept_all_path.name)
    print(" ", kept_unique_path.name)
    print(" ", kept_fixed_path.name)
    print(" ", outliers_path.name)
    if zthr > 0:
        print(" ", outliers_within_fish_path.name)
    if hr > 0:
        print(" ", outliers_height_path.name)
    print(" ", report_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
