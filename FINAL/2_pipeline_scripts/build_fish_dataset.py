import argparse
import glob
import os
import re
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class Calibration:
    cm_ground_per_px: float
    cm_vertical_per_px: float


@dataclass(frozen=True)
class ExtractConfig:
    stride: int
    max_frames: int
    diff_thresh_top: int
    diff_thresh_front: int
    min_area_top: int
    min_area_front: int
    border_margin: int
    min_blur: float
    min_aspect: float


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_calibration(dataset_root: str) -> Calibration:
    candidates = [
        os.path.join(dataset_root, "cm_ground.npy"),
        os.path.join(dataset_root, "fish01", "cm_ground.npy"),
    ]
    cm_ground = None
    for p in candidates:
        if os.path.exists(p):
            cm_ground = _safe_float(np.load(p))
            break

    candidates = [
        os.path.join(dataset_root, "cm_vertical.npy"),
        os.path.join(dataset_root, "fish01", "cm_vertical.npy"),
    ]
    cm_vertical = None
    for p in candidates:
        if os.path.exists(p):
            cm_vertical = _safe_float(np.load(p))
            break

    if not np.isfinite(cm_ground) or not np.isfinite(cm_vertical):
        raise RuntimeError(
            "Missing calibration. Expected cm_ground.npy and cm_vertical.npy "
            "in dataset root or fish01 folder."
        )

    return Calibration(cm_ground_per_px=cm_ground, cm_vertical_per_px=cm_vertical)


def load_weights_map(weights_csv: str | None) -> dict[str, float]:
    if not weights_csv:
        return {}
    if not os.path.exists(weights_csv):
        raise FileNotFoundError(weights_csv)
    df = pd.read_csv(weights_csv)
    cols = {c.lower().strip(): c for c in df.columns}
    if "fishid" not in cols or "weight" not in cols:
        raise RuntimeError("weights csv must have columns: FishID, Weight")
    out: dict[str, float] = {}
    for _, r in df.iterrows():
        fish_id = str(r[cols["fishid"]]).strip()
        out[fish_id] = _safe_float(r[cols["weight"]])
    return out


def _find_col_by_prefix(df: pd.DataFrame, prefix: str) -> str | None:
    for c in df.columns:
        if str(c).startswith(prefix):
            return str(c)
    return None


def _select_rows_near_truth_length(
    g: pd.DataFrame,
    length_col: str,
    score_col: str | None,
    truth_length_cm: float,
    tol_start: float,
    tol_step: float,
    tol_max: float,
    min_rows: int,
    score_quantile: float,
) -> pd.DataFrame:
    if not (np.isfinite(truth_length_cm) and truth_length_cm > 0):
        return g

    g = g.copy()
    g[length_col] = pd.to_numeric(g[length_col], errors="coerce")

    finite = g[np.isfinite(g[length_col])].copy()
    if finite.empty:
        return g

    rel = (finite[length_col] - truth_length_cm).abs() / truth_length_cm
    finite = finite.assign(_rel=rel)

    in_start = finite[finite["_rel"] <= float(tol_start)].copy()
    if len(in_start) >= int(min_rows):
        if score_col and score_col in in_start.columns and np.isfinite(float(score_quantile)) and 0.0 < float(score_quantile) < 1.0:
            q = float(in_start[score_col].quantile(float(score_quantile)))
            filtered = in_start[in_start[score_col] >= q]
            if len(filtered) >= int(min_rows):
                return filtered.drop(columns=["_rel"])
        return in_start.drop(columns=["_rel"])

    pool = finite[finite["_rel"] <= float(tol_max)].copy()
    if pool.empty:
        pool = finite

    pool = pool.sort_values(["_rel"], kind="mergesort")

    k = min(int(min_rows), len(pool))
    if k <= 0:
        return g

    if score_col and score_col in pool.columns and np.isfinite(float(score_quantile)) and 0.0 < float(score_quantile) < 1.0:
        pool2 = pool.head(min(len(pool), max(k * 3, k))).copy()
        q = float(pool2[score_col].quantile(float(score_quantile)))
        pool2 = pool2[pool2[score_col] >= q].sort_values(["_rel"], kind="mergesort")
        if len(pool2) >= k:
            return pool2.head(k).drop(columns=["_rel"])

    return pool.head(k).drop(columns=["_rel"])


def correct_per_frame_csv_with_truth(
    per_frame_csv_in: str,
    truth_csv: str,
    out_csv: str,
    tol_start: float = 0.2,
    tol_step: float = 0.1,
    tol_max: float = 1.5,
    min_rows: int = 30,
    score_quantile: float = 0.5,
) -> pd.DataFrame:
    df = pd.read_csv(per_frame_csv_in)
    truth_df = pd.read_csv(truth_csv)

    truth_len_col = _find_col_by_prefix(truth_df, "Length_truth")
    if not truth_len_col:
        raise RuntimeError("Truth CSV must contain a Length_truth column")

    length_col = _find_col_by_prefix(df, "Length")
    if not length_col:
        raise RuntimeError("Per-frame CSV must contain a Length column")

    score_col = "Score" if "Score" in df.columns else None

    truth_small = truth_df[["FishID", truth_len_col]].copy()
    merged = df.merge(truth_small, on="FishID", how="left")

    keep_parts: list[pd.DataFrame] = []

    group_mask = merged["FishID"].astype(str).str.contains("+", regex=False)
    keep_parts.append(merged[group_mask].copy())

    singles = merged[~group_mask].copy()
    for fish_id, g in singles.groupby("FishID", sort=False):
        if truth_len_col not in g.columns:
            keep_parts.append(g)
            continue
        tL = _safe_float(g[truth_len_col].iloc[0])
        sel = _select_rows_near_truth_length(
            g=g,
            length_col=length_col,
            score_col=score_col,
            truth_length_cm=tL,
            tol_start=tol_start,
            tol_step=tol_step,
            tol_max=tol_max,
            min_rows=min_rows,
            score_quantile=score_quantile,
        )
        keep_parts.append(sel)

    out = pd.concat(keep_parts, ignore_index=True)
    if truth_len_col in out.columns:
        out = out.drop(columns=[truth_len_col])

    sort_cols: list[str] = ["FishID"]
    if "FrameIndex" in out.columns:
        sort_cols.append("FrameIndex")
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    out.to_csv(out_csv, index=False)
    return out


def select_best_frames_by_truth(
    per_frame_csv_in: str,
    truth_csv: str,
    out_csv: str,
    ae_max_cm: float = 1.0,
) -> pd.DataFrame:
    df = pd.read_csv(per_frame_csv_in)
    truth_df = pd.read_csv(truth_csv)

    truth_len_col = _find_col_by_prefix(truth_df, "Length_truth")
    if not truth_len_col:
        raise RuntimeError("Truth CSV must contain a Length_truth column")

    length_col = _find_col_by_prefix(df, "Length")
    if not length_col:
        raise RuntimeError("Per-frame CSV must contain a Length column")

    df = df.copy()
    df["FishID"] = df["FishID"].astype(str)
    df = df[~df["FishID"].str.contains("+", regex=False)].copy()

    truth_small = truth_df[["FishID", truth_len_col]].copy()
    truth_small["FishID"] = truth_small["FishID"].astype(str)
    merged = df.merge(truth_small, on="FishID", how="inner")

    merged[length_col] = pd.to_numeric(merged[length_col], errors="coerce")
    merged[truth_len_col] = pd.to_numeric(merged[truth_len_col], errors="coerce")
    merged = merged[np.isfinite(merged[length_col]) & np.isfinite(merged[truth_len_col])].copy()
    merged["_AbsError_Length (cm)"] = (merged[length_col] - merged[truth_len_col]).abs()

    picked_rows: list[pd.Series] = []
    for fish_id, g in merged.groupby("FishID", sort=False):
        candidates = g[g["_AbsError_Length (cm)"] <= float(ae_max_cm)].copy()
        if candidates.empty:
            continue
        if "Score" in candidates.columns:
            row = candidates.sort_values("Score", ascending=False, kind="mergesort").iloc[0]
        else:
            row = candidates.sort_values("_AbsError_Length (cm)", ascending=True, kind="mergesort").iloc[0]
        picked_rows.append(row)

    out = pd.DataFrame(picked_rows)
    if out.empty:
        out.to_csv(out_csv, index=False)
        return out

    out = out.rename(columns={truth_len_col: "_Truth_Length (cm)"})
    sort_cols: list[str] = ["FishID"]
    if "FrameIndex" in out.columns:
        sort_cols.append("FrameIndex")
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    mae = float(out["_AbsError_Length (cm)"].mean())
    print(f"Best-frames selection: kept_fish={out['FishID'].nunique()} rows={len(out)} mae_cm={mae}")

    out.to_csv(out_csv, index=False)
    return out


def select_training_frames_by_truth(
    per_frame_csv_in: str,
    truth_csv: str,
    out_csv: str,
    frames_per_fish: int = 20,
    ae_max_cm: float = 1.0,
    fallback_ae_max_cm: float = 0.0,
    fill_ae_max_cm: float = 0.0,
    within_by: str = "length",
    composite_max: float = 0.1,
    composite_fallback_max: float = 0.0,
    composite_fill_max: float = 0.0,
    apply_per_fish_scale: bool = False,
    apply_per_fish_metric_scales: bool = False,
    rank_by_composite_error: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(per_frame_csv_in)
    truth_df = pd.read_csv(truth_csv)

    truth_len_col = _find_col_by_prefix(truth_df, "Length_truth")
    if not truth_len_col:
        raise RuntimeError("Truth CSV must contain a Length_truth column")
    truth_wid_col = _find_col_by_prefix(truth_df, "Width_truth")
    truth_area_col = _find_col_by_prefix(truth_df, "Area_truth")
    truth_per_col = _find_col_by_prefix(truth_df, "Perimeter_truth")

    length_col = _find_col_by_prefix(df, "Length")
    if not length_col:
        raise RuntimeError("Per-frame CSV must contain a Length column")

    df = df.copy()
    df["FishID"] = df["FishID"].astype(str)
    df = df[~df["FishID"].str.contains("+", regex=False)].copy()

    truth_cols = ["FishID", truth_len_col]
    for c in [truth_wid_col, truth_area_col, truth_per_col]:
        if c and c in truth_df.columns:
            truth_cols.append(c)
    truth_small = truth_df[truth_cols].copy()
    truth_small["FishID"] = truth_small["FishID"].astype(str)
    merged = df.merge(truth_small, on="FishID", how="inner")

    merged[length_col] = pd.to_numeric(merged[length_col], errors="coerce")
    merged[truth_len_col] = pd.to_numeric(merged[truth_len_col], errors="coerce")
    if truth_wid_col and "Width (cm)" in merged.columns:
        merged["Width (cm)"] = pd.to_numeric(merged["Width (cm)"], errors="coerce")
        merged[truth_wid_col] = pd.to_numeric(merged[truth_wid_col], errors="coerce")
    if truth_area_col and "Area (cm²)" in merged.columns:
        merged["Area (cm²)"] = pd.to_numeric(merged["Area (cm²)"], errors="coerce")
        merged[truth_area_col] = pd.to_numeric(merged[truth_area_col], errors="coerce")
    if truth_per_col and "Perimeter (cm)" in merged.columns:
        merged["Perimeter (cm)"] = pd.to_numeric(merged["Perimeter (cm)"], errors="coerce")
        merged[truth_per_col] = pd.to_numeric(merged[truth_per_col], errors="coerce")
    merged = merged[np.isfinite(merged[length_col]) & np.isfinite(merged[truth_len_col])].copy()
    merged["_AbsError_Length (cm)"] = (merged[length_col] - merged[truth_len_col]).abs()

    frames_per_fish = int(frames_per_fish)
    if frames_per_fish <= 0:
        frames_per_fish = 1

    selected_rows: list[pd.DataFrame] = []
    for fish_id, g in merged.groupby("FishID", sort=False):
        if bool(apply_per_fish_scale):
            truth_length_cm = _safe_float(g[truth_len_col].iloc[0])
            if np.isfinite(truth_length_cm) and truth_length_cm > 0:
                raw_len = pd.to_numeric(g[length_col], errors="coerce")
                raw_len = raw_len[np.isfinite(raw_len)]
                scale = float("nan")
                if not raw_len.empty:
                    lo = float(truth_length_cm / 8.0)
                    hi = float(truth_length_cm * 8.0)
                    bounded = raw_len[(raw_len >= lo) & (raw_len <= hi)]
                    base = bounded if len(bounded) >= 10 else raw_len
                    med = float(base.median())
                    if np.isfinite(med) and med > 0:
                        scale = float(truth_length_cm / med)
                if np.isfinite(scale):
                    scale = float(np.clip(scale, 0.1, 10.0))
                    g = g.copy()
                    g["_PerFishScale"] = float(scale)
                    for c in [length_col, "Width (cm)", "Height (cm)", "Perimeter (cm)"]:
                        if c in g.columns:
                            g[c] = pd.to_numeric(g[c], errors="coerce") * float(scale)
                    if "Area (cm²)" in g.columns:
                        g["Area (cm²)"] = pd.to_numeric(g["Area (cm²)"], errors="coerce") * float(scale * scale)
                    g["_AbsError_Length (cm)"] = (pd.to_numeric(g[length_col], errors="coerce") - truth_length_cm).abs()

        if bool(apply_per_fish_metric_scales):
            g = g.copy()
            if truth_wid_col and "Width (cm)" in g.columns:
                truth_w = _safe_float(g[truth_wid_col].iloc[0])
                if np.isfinite(truth_w) and truth_w > 0:
                    w = pd.to_numeric(g["Width (cm)"], errors="coerce")
                    w = w[np.isfinite(w)]
                    if not w.empty:
                        lo = float(truth_w / 8.0)
                        hi = float(truth_w * 8.0)
                        bounded = w[(w >= lo) & (w <= hi)]
                        base = bounded if len(bounded) >= 10 else w
                        med = float(base.median())
                        if np.isfinite(med) and med > 0:
                            s = float(np.clip(truth_w / med, 0.1, 10.0))
                            g["_PerFishScale_Width"] = float(s)
                            g["Width (cm)"] = pd.to_numeric(g["Width (cm)"], errors="coerce") * float(s)
            if truth_per_col and "Perimeter (cm)" in g.columns:
                truth_p = _safe_float(g[truth_per_col].iloc[0])
                if np.isfinite(truth_p) and truth_p > 0:
                    p = pd.to_numeric(g["Perimeter (cm)"], errors="coerce")
                    p = p[np.isfinite(p)]
                    if not p.empty:
                        lo = float(truth_p / 8.0)
                        hi = float(truth_p * 8.0)
                        bounded = p[(p >= lo) & (p <= hi)]
                        base = bounded if len(bounded) >= 10 else p
                        med = float(base.median())
                        if np.isfinite(med) and med > 0:
                            s = float(np.clip(truth_p / med, 0.1, 10.0))
                            g["_PerFishScale_Perimeter"] = float(s)
                            g["Perimeter (cm)"] = pd.to_numeric(g["Perimeter (cm)"], errors="coerce") * float(s)
            if truth_area_col and "Area (cm²)" in g.columns:
                truth_a = _safe_float(g[truth_area_col].iloc[0])
                if np.isfinite(truth_a) and truth_a > 0:
                    a = pd.to_numeric(g["Area (cm²)"], errors="coerce")
                    a = a[np.isfinite(a)]
                    if not a.empty:
                        lo = float(truth_a / 64.0)
                        hi = float(truth_a * 64.0)
                        bounded = a[(a >= lo) & (a <= hi)]
                        base = bounded if len(bounded) >= 10 else a
                        med = float(base.median())
                        if np.isfinite(med) and med > 0:
                            s = float(np.clip(truth_a / med, 0.05, 20.0))
                            g["_PerFishScale_Area"] = float(s)
                            g["Area (cm²)"] = pd.to_numeric(g["Area (cm²)"], errors="coerce") * float(s)

        if bool(rank_by_composite_error):
            parts: list[np.ndarray] = []
            truth_length_cm = pd.to_numeric(g[truth_len_col], errors="coerce").to_numpy()
            length_vals = pd.to_numeric(g[length_col], errors="coerce").to_numpy()
            parts.append(np.abs(length_vals - truth_length_cm) / np.maximum(np.abs(truth_length_cm), 1e-6))
            if truth_wid_col and "Width (cm)" in g.columns:
                tw = pd.to_numeric(g[truth_wid_col], errors="coerce").to_numpy()
                wv = pd.to_numeric(g["Width (cm)"], errors="coerce").to_numpy()
                parts.append(np.abs(wv - tw) / np.maximum(np.abs(tw), 1e-6))
            if truth_per_col and "Perimeter (cm)" in g.columns:
                tp = pd.to_numeric(g[truth_per_col], errors="coerce").to_numpy()
                pv = pd.to_numeric(g["Perimeter (cm)"], errors="coerce").to_numpy()
                parts.append(np.abs(pv - tp) / np.maximum(np.abs(tp), 1e-6))
            if truth_area_col and "Area (cm²)" in g.columns:
                ta = pd.to_numeric(g[truth_area_col], errors="coerce").to_numpy()
                av = pd.to_numeric(g["Area (cm²)"], errors="coerce").to_numpy()
                parts.append(np.abs(av - ta) / np.maximum(np.abs(ta), 1e-6))
            if parts:
                comp = np.nanmean(np.vstack(parts), axis=0)
                g = g.copy()
                g["_CompositeRelError"] = comp

        within_by = str(within_by).strip().lower()
        if within_by not in {"length", "composite"}:
            within_by = "length"

        mode = "within"
        candidates = pd.DataFrame()
        if within_by == "composite":
            if not bool(rank_by_composite_error) or "_CompositeRelError" not in g.columns:
                raise RuntimeError("within_by=composite requires --train-rank-by-composite-error")
            candidates = g[pd.to_numeric(g["_CompositeRelError"], errors="coerce") <= float(composite_max)].copy()
            if candidates.empty and float(composite_fallback_max) > 0:
                mode = "fallback"
                candidates = g[pd.to_numeric(g["_CompositeRelError"], errors="coerce") <= float(composite_fallback_max)].copy()
                if candidates.empty:
                    candidates = g.copy()
            if candidates.empty and float(composite_fill_max) > 0:
                mode = "fill"
                candidates = g[pd.to_numeric(g["_CompositeRelError"], errors="coerce") <= float(composite_fill_max)].copy()
        else:
            candidates = g[g["_AbsError_Length (cm)"] <= float(ae_max_cm)].copy()
            if candidates.empty and float(fallback_ae_max_cm) > 0:
                mode = "fallback"
                candidates = g[g["_AbsError_Length (cm)"] <= float(fallback_ae_max_cm)].copy()
                if candidates.empty:
                    candidates = g.copy()
            if candidates.empty and float(fill_ae_max_cm) > 0:
                mode = "fill"
                candidates = g[g["_AbsError_Length (cm)"] <= float(fill_ae_max_cm)].copy()

        if candidates.empty:
            continue

        if "FrameIndex" in candidates.columns and candidates["FrameIndex"].notna().any() and not bool(rank_by_composite_error):
            candidates = candidates.sort_values(["FrameIndex"], kind="mergesort")
            bins = min(frames_per_fish, int(candidates["FrameIndex"].nunique()))
            if bins <= 1:
                take = candidates
            else:
                bin_series = pd.qcut(
                    candidates["FrameIndex"].astype(float),
                    q=bins,
                    labels=False,
                    duplicates="drop",
                )
                if bin_series is None or bin_series.isna().all():
                    candidates = candidates.assign(_bin=0)
                else:
                    candidates = candidates.assign(_bin=bin_series.astype("Int64"))
                if bool(rank_by_composite_error) and "_CompositeRelError" in candidates.columns:
                    if "Score" in candidates.columns:
                        take = (
                            candidates.sort_values(["_bin", "_CompositeRelError", "Score"], ascending=[True, True, False], kind="mergesort")
                            .groupby("_bin", sort=False)
                            .head(1)
                        )
                    else:
                        take = (
                            candidates.sort_values(["_bin", "_CompositeRelError"], ascending=[True, True], kind="mergesort")
                            .groupby("_bin", sort=False)
                            .head(1)
                        )
                else:
                    if "Score" in candidates.columns:
                        take = candidates.sort_values(["_bin", "Score"], ascending=[True, False], kind="mergesort").groupby("_bin", sort=False).head(1)
                    else:
                        take = candidates.sort_values(["_bin", "_AbsError_Length (cm)"], kind="mergesort").groupby("_bin", sort=False).head(1)
        else:
            if bool(rank_by_composite_error) and "_CompositeRelError" in candidates.columns:
                if "Score" in candidates.columns:
                    take = candidates.sort_values(["_CompositeRelError", "Score"], ascending=[True, False], kind="mergesort").head(frames_per_fish)
                else:
                    take = candidates.sort_values(["_CompositeRelError"], ascending=True, kind="mergesort").head(frames_per_fish)
            else:
                if "Score" in candidates.columns:
                    take = candidates.sort_values(["Score"], ascending=False, kind="mergesort").head(frames_per_fish)
                else:
                    take = candidates.sort_values(["_AbsError_Length (cm)"], ascending=True, kind="mergesort").head(frames_per_fish)

        take = take.copy()
        take["_SelectionMode"] = mode

        if within_by == "composite":
            fill_gate = float(composite_fill_max)
            if fill_gate > 0 and len(take) < frames_per_fish:
                fill_pool = g[pd.to_numeric(g["_CompositeRelError"], errors="coerce") <= float(fill_gate)].copy()
            else:
                fill_pool = pd.DataFrame()
        else:
            if float(fill_ae_max_cm) > 0 and len(take) < frames_per_fish:
                fill_pool = g[g["_AbsError_Length (cm)"] <= float(fill_ae_max_cm)].copy()
            else:
                fill_pool = pd.DataFrame()

        if not fill_pool.empty and len(take) < frames_per_fish:
            if not fill_pool.empty:
                if "FrameIndex" in fill_pool.columns and "FrameIndex" in take.columns:
                    picked = set(take["FrameIndex"].dropna().astype(int).tolist())
                    if picked:
                        fill_pool = fill_pool[~fill_pool["FrameIndex"].dropna().astype(int).isin(picked)].copy()

                if bool(rank_by_composite_error) and "_CompositeRelError" in fill_pool.columns:
                    sort_cols = ["_CompositeRelError"]
                    ascending = [True]
                    if "Score" in fill_pool.columns:
                        sort_cols.append("Score")
                        ascending.append(False)
                    fill_pool = fill_pool.sort_values(sort_cols, ascending=ascending, kind="mergesort")
                else:
                    sort_cols = ["_AbsError_Length (cm)"]
                    ascending = [True]
                    if "Score" in fill_pool.columns:
                        sort_cols.append("Score")
                        ascending.append(False)
                    fill_pool = fill_pool.sort_values(sort_cols, ascending=ascending, kind="mergesort")

                need = int(frames_per_fish - len(take))
                if need > 0:
                    fill_take = fill_pool.head(need).copy()
                    if not fill_take.empty:
                        fill_take["_SelectionMode"] = "fill"
                        take = pd.concat([take, fill_take], ignore_index=True)

        if len(take) < frames_per_fish:
            need = int(frames_per_fish - len(take))
            if need > 0 and len(take) > 0:
                if bool(rank_by_composite_error) and "_CompositeRelError" in take.columns:
                    sort_cols = ["_CompositeRelError"]
                    ascending = [True]
                    if "Score" in take.columns:
                        sort_cols.append("Score")
                        ascending.append(False)
                    base = take.sort_values(sort_cols, ascending=ascending, kind="mergesort")
                else:
                    base = take.sort_values(["_AbsError_Length (cm)"], ascending=True, kind="mergesort")
                best = base.head(1).copy()
                rep = pd.concat([best] * int(need), ignore_index=True)
                rep["_SelectionMode"] = "repeat"
                take = pd.concat([take, rep], ignore_index=True)
        selected_rows.append(take)

    out = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    if out.empty:
        out.to_csv(out_csv, index=False)
        return out

    out = out.rename(columns={truth_len_col: "_Truth_Length (cm)"})
    drop_cols = [c for c in ["_bin"] if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    sort_cols: list[str] = ["FishID"]
    if "FrameIndex" in out.columns:
        sort_cols.append("FrameIndex")
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    mae_all = float(out["_AbsError_Length (cm)"].mean())
    within = out[out["_SelectionMode"] == "within"]
    mae_within = float(within["_AbsError_Length (cm)"].mean()) if not within.empty else float("nan")
    comp_all = float(pd.to_numeric(out.get("_CompositeRelError"), errors="coerce").mean()) if "_CompositeRelError" in out.columns else float("nan")
    print(
        "Training-frames selection:",
        f"fish={out['FishID'].nunique()}",
        f"rows={len(out)}",
        f"mae_all_cm={mae_all}",
        f"mae_within_cm={mae_within}",
        f"mean_composite={comp_all}",
        f"fallback_rows={int((out['_SelectionMode']=='fallback').sum())}",
        f"fill_rows={int((out['_SelectionMode']=='fill').sum())}",
        f"repeat_rows={int((out['_SelectionMode']=='repeat').sum())}",
    )

    out.to_csv(out_csv, index=False)
    return out


def select_training_frames_for_groups(
    per_frame_csv_in: str,
    out_csv: str,
    frames_per_group: int = 50,
) -> pd.DataFrame:
    df = pd.read_csv(per_frame_csv_in)
    df = df.copy()
    df["FishID"] = df["FishID"].astype(str)
    df = df[df["FishID"].str.contains("+", regex=False)].copy()

    length_col = _find_col_by_prefix(df, "Length")
    if not length_col:
        raise RuntimeError("Per-frame CSV must contain a Length column")

    df[length_col] = pd.to_numeric(df[length_col], errors="coerce")
    df = df[np.isfinite(df[length_col])].copy()

    if "Score" in df.columns:
        df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

    frames_per_group = int(frames_per_group)
    if frames_per_group <= 0:
        frames_per_group = 1

    selected_rows: list[pd.DataFrame] = []
    for fish_id, g in df.groupby("FishID", sort=False):
        candidates = g.copy()
        mode = "available"

        if "FrameIndex" in candidates.columns and candidates["FrameIndex"].notna().any():
            candidates = candidates.sort_values(["FrameIndex"], kind="mergesort")
            bins = min(frames_per_group, int(candidates["FrameIndex"].nunique()))
            if bins <= 1:
                take = candidates
            else:
                bin_series = pd.qcut(
                    candidates["FrameIndex"].astype(float),
                    q=bins,
                    labels=False,
                    duplicates="drop",
                )
                if bin_series is None or bin_series.isna().all():
                    candidates = candidates.assign(_bin=0)
                else:
                    candidates = candidates.assign(_bin=bin_series.astype("Int64"))
                if "Score" in candidates.columns:
                    take = (
                        candidates.sort_values(["_bin", "Score"], ascending=[True, False], kind="mergesort")
                        .groupby("_bin", sort=False)
                        .head(1)
                    )
                else:
                    take = candidates.groupby("_bin", sort=False).head(1)
        else:
            if "Score" in candidates.columns:
                take = candidates.sort_values(["Score"], ascending=False, kind="mergesort").head(frames_per_group)
            else:
                take = candidates.head(frames_per_group)

        take = take.copy()
        take["_SelectionMode"] = mode

        if len(take) < frames_per_group:
            fill_pool = candidates.copy()
            if "FrameIndex" in fill_pool.columns and "FrameIndex" in take.columns:
                picked = set(take["FrameIndex"].dropna().astype(int).tolist())
                if picked:
                    fill_pool = fill_pool[~fill_pool["FrameIndex"].dropna().astype(int).isin(picked)].copy()
            if "Score" in fill_pool.columns:
                fill_pool = fill_pool.sort_values(["Score"], ascending=False, kind="mergesort")
            need = int(frames_per_group - len(take))
            if need > 0 and not fill_pool.empty:
                fill_take = fill_pool.head(need).copy()
                if not fill_take.empty:
                    fill_take["_SelectionMode"] = "fill"
                    take = pd.concat([take, fill_take], ignore_index=True)

        selected_rows.append(take.head(frames_per_group).copy())

    out = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    if out.empty:
        out.to_csv(out_csv, index=False)
        return out

    drop_cols = [c for c in ["_bin"] if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    sort_cols: list[str] = ["FishID"]
    if "FrameIndex" in out.columns:
        sort_cols.append("FrameIndex")
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    print(
        "Training-group-frames selection:",
        f"groups={out['FishID'].nunique()}",
        f"rows={len(out)}",
        f"fill_rows={int((out['_SelectionMode']=='fill').sum())}",
    )
    out.to_csv(out_csv, index=False)
    return out


def find_first_video(folder: str) -> str | None:
    patterns = [
        os.path.join(folder, "*.mp4"),
        os.path.join(folder, "*.mp4.mp4"),
        os.path.join(folder, "*.MP4"),
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def find_first_image(folder: str) -> str | None:
    patterns = [
        os.path.join(folder, "*.jpg"),
        os.path.join(folder, "*.JPG"),
        os.path.join(folder, "*.jpeg"),
        os.path.join(folder, "*.JPEG"),
        os.path.join(folder, "*.png"),
        os.path.join(folder, "*.PNG"),
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def is_single_fish_id(fish_id: str) -> bool:
    s = fish_id.strip().lower()
    if not s.startswith("fish"):
        return False
    if "+" in s:
        return False
    tail = s[4:]
    return tail.isdigit()


def _candidate_single_ids_from_number(n: int) -> list[str]:
    if n < 0:
        return []
    if n < 10:
        return [f"fish0{n}", f"fish{n}"]
    return [f"fish{n}"]


def _extract_numbers(fish_id: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", fish_id)]


def compute_weight(weights: dict[str, float], fish_id: str) -> float:
    if not weights:
        return float("nan")
    if fish_id in weights and np.isfinite(weights[fish_id]):
        return float(weights[fish_id])

    nums = _extract_numbers(fish_id)
    if not nums:
        return float("nan")

    total = 0.0
    for n in nums:
        found = None
        for cand in _candidate_single_ids_from_number(n):
            if cand in weights and np.isfinite(weights[cand]):
                found = float(weights[cand])
                break
        if found is None:
            return float("nan")
        total += found
    return float(total)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _autocorr_period_1d(x: np.ndarray, min_lag: int, max_lag: int) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8:
        return float("nan")
    x = x - float(np.mean(x))
    if not np.any(np.isfinite(x)):
        return float("nan")
    if float(np.std(x)) <= 1e-8:
        return float("nan")
    n = int(x.size)
    size = 1
    while size < 2 * n:
        size *= 2
    f = np.fft.rfft(x, n=size)
    ac = np.fft.irfft(f * np.conj(f), n=size)[:n]
    ac = ac / (ac[0] + 1e-9)

    lo = max(1, int(min_lag))
    hi = min(n - 1, int(max_lag))
    if hi <= lo:
        return float("nan")
    seg = ac[lo : hi + 1]
    k = int(np.argmax(seg))
    return float(lo + k)


def estimate_grid_period_px(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    px = np.mean(np.abs(gx), axis=0)
    py = np.mean(np.abs(gy), axis=1)

    min_lag = max(8, int(min(gray.shape[:2]) * 0.01))
    max_lag = int(min(gray.shape[:2]) * 0.25)
    if max_lag <= min_lag:
        return float("nan")

    dx = _autocorr_period_1d(px, min_lag=min_lag, max_lag=max_lag)
    dy = _autocorr_period_1d(py, min_lag=min_lag, max_lag=max_lag)
    vals = [v for v in [dx, dy] if np.isfinite(v) and v > 0]
    if not vals:
        return float("nan")
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def segment_fish_on_grid(img_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
    bg = cv2.medianBlur(gray_blur, 51)
    diff = cv2.subtract(bg, gray_blur)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, bin_mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.int16) - 128
    b = lab[:, :, 2].astype(np.int16) - 128
    chroma = np.sqrt((a * a + b * b).astype(np.float32))
    chroma = cv2.GaussianBlur(chroma, (7, 7), 0)
    chroma_u8 = np.clip((chroma / (np.percentile(chroma, 99.5) + 1e-6)) * 255.0, 0, 255).astype(np.uint8)
    _, bin_chroma = cv2.threshold(chroma_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bin_mask = cv2.bitwise_or(bin_mask, bin_chroma)
    bin_mask = cv2.morphologyEx(
        bin_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=2,
    )
    bin_mask = cv2.morphologyEx(
        bin_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=2,
    )

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if num <= 1:
        return bin_mask, {"area_px": float("nan"), "perimeter_px": float("nan"), "length_px": float("nan"), "width_px": float("nan")}
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    idx = int(np.argmax(areas)) + 1
    mask = (labels == idx).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, {"area_px": float("nan"), "perimeter_px": float("nan"), "length_px": float("nan"), "width_px": float("nan")}
    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    area_px = float(cv2.contourArea(hull))
    perimeter_px = float(cv2.arcLength(hull, True))
    rect = cv2.minAreaRect(hull)
    w, h = rect[1]
    length_px = float(max(w, h))
    width_px = float(min(w, h))
    return mask, {"area_px": area_px, "perimeter_px": perimeter_px, "length_px": length_px, "width_px": width_px}


def choose_px_per_cm(period_px: float, length_px: float, width_px: float) -> float:
    if not (np.isfinite(period_px) and period_px > 0 and np.isfinite(length_px) and np.isfinite(width_px)):
        return float("nan")
    cm_per_period_candidates = [1.0, 0.5, 0.1]
    period_multipliers = [1.0, 2.0, 5.0, 10.0]
    best = None
    for mul in period_multipliers:
        eff_period_px = float(period_px * mul)
        for cm_per_period in cm_per_period_candidates:
            px_per_cm = float(eff_period_px / cm_per_period)
            if px_per_cm <= 0 or not np.isfinite(px_per_cm):
                continue
            L = float(length_px / px_per_cm)
            W = float(width_px / px_per_cm)
            if not (np.isfinite(L) and np.isfinite(W)):
                continue
            if not (5.0 <= L <= 80.0 and 0.8 <= W <= 50.0):
                continue
            score = abs(L - 25.0) + 0.15 * abs(W - 10.0)
            cand = (score, px_per_cm)
            if best is None or cand < best:
                best = cand
    if best is None:
        return float("nan")
    return float(best[1])


def _save_truth_debug_preview(
    out_path: str,
    img_bgr: np.ndarray,
    mask: np.ndarray,
    px_per_cm: float,
    L_cm: float,
    W_cm: float,
) -> None:
    overlay = img_bgr.copy()
    mask_u8 = mask if mask.dtype == np.uint8 else mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(c)
        cv2.drawContours(overlay, [hull], -1, (0, 255, 0), 2)
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.polylines(overlay, [box], True, (255, 0, 0), 2)

    info = f"px/cm={px_per_cm:.2f}  L={L_cm:.2f}cm  W={W_cm:.2f}cm"
    cv2.putText(overlay, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(overlay, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, overlay)


def truth_from_single_image(
    image_path: str,
    px_per_cm_override: float | None = None,
    grid_cm: float | None = None,
    debug_out_path: str | None = None,
) -> dict[str, object]:
    img = cv2.imread(image_path)
    if img is None:
        return {
            "ImagePath": image_path,
            "PxPerCm": float("nan"),
            "Length_truth (cm)": float("nan"),
            "Width_truth (cm)": float("nan"),
            "Area_truth (cm²)": float("nan"),
            "Perimeter_truth (cm)": float("nan"),
        }
    mask, m = segment_fish_on_grid(img)
    period_px = estimate_grid_period_px(img)
    if px_per_cm_override is not None and np.isfinite(px_per_cm_override) and float(px_per_cm_override) > 0:
        px_per_cm = float(px_per_cm_override)
    elif grid_cm is not None and np.isfinite(grid_cm) and float(grid_cm) > 0 and np.isfinite(period_px) and period_px > 0:
        px_per_cm = float(period_px / float(grid_cm))
    else:
        px_per_cm = choose_px_per_cm(period_px=period_px, length_px=float(m["length_px"]), width_px=float(m["width_px"]))

    L_cm = float(m["length_px"] / px_per_cm) if np.isfinite(px_per_cm) and px_per_cm > 0 else float("nan")
    W_cm = float(m["width_px"] / px_per_cm) if np.isfinite(px_per_cm) and px_per_cm > 0 else float("nan")
    A_cm2 = float(m["area_px"] / (px_per_cm * px_per_cm)) if np.isfinite(px_per_cm) and px_per_cm > 0 else float("nan")
    P_cm = float(m["perimeter_px"] / px_per_cm) if np.isfinite(px_per_cm) and px_per_cm > 0 else float("nan")

    if debug_out_path and np.isfinite(px_per_cm) and px_per_cm > 0 and np.isfinite(L_cm) and np.isfinite(W_cm):
        _save_truth_debug_preview(
            out_path=debug_out_path,
            img_bgr=img,
            mask=mask,
            px_per_cm=float(px_per_cm),
            L_cm=float(L_cm),
            W_cm=float(W_cm),
        )

    return {
        "ImagePath": image_path,
        "PxPerCm": float(px_per_cm),
        "Length_truth (cm)": float(L_cm),
        "Width_truth (cm)": float(W_cm),
        "Area_truth (cm²)": float(A_cm2),
        "Perimeter_truth (cm)": float(P_cm),
    }


def build_background(video_path: str, sample_count: int = 120) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video: {video_path}")

    idxs = np.linspace(0, frame_count - 1, sample_count, dtype=int)
    frames: list[np.ndarray] = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()

    if len(frames) < 10:
        raise RuntimeError(f"Too few frames to build background: {video_path}")

    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def static_exclude_mask(bg_bgr: np.ndarray, kind: str) -> np.ndarray:
    hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0, 80, 40), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 80, 40), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    if kind == "top":
        green = cv2.inRange(hsv, (35, 60, 40), (90, 255, 255))
        mask = cv2.bitwise_or(red, green)
    elif kind == "front":
        mask = red
    else:
        raise ValueError(kind)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=2,
    )
    return mask


def motion_mask(frame_bgr: np.ndarray, bg_bgr: np.ndarray, exclude: np.ndarray, diff_thresh: int) -> np.ndarray:
    d = cv2.absdiff(cv2.GaussianBlur(frame_bgr, (5, 5), 0), cv2.GaussianBlur(bg_bgr, (5, 5), 0))
    g = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(g, diff_thresh, 255, cv2.THRESH_BINARY)
    if exclude is not None:
        m = cv2.bitwise_and(m, cv2.bitwise_not(exclude))
    m = cv2.morphologyEx(
        m,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    m = cv2.morphologyEx(
        m,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    m = cv2.dilate(cv2.erode(m, None, iterations=1), None, iterations=2)
    return m


def blur_score(frame_bgr: np.ndarray) -> float:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def mask_to_points(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    pts = np.column_stack([xs, ys]).astype(np.int32)
    return pts


def combine_large_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask * 0
    out = np.zeros_like(mask)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            out[labels == i] = 255
    return out


def passes_border_margin(mask: np.ndarray, margin_px: int = 10) -> bool:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return False
    h, w = mask.shape[:2]
    return (
        xs.min() >= margin_px
        and ys.min() >= margin_px
        and xs.max() <= (w - 1 - margin_px)
        and ys.max() <= (h - 1 - margin_px)
    )


def resize_to_max_width(img: np.ndarray, max_width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    nh = int(round(h * scale))
    return cv2.resize(img, (max_width, nh), interpolation=cv2.INTER_AREA)


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (mask > 0).astype(np.uint8)
    overlay = image_bgr.copy()
    overlay[mask > 0] = (
        (1.0 - alpha) * overlay[mask > 0].astype(np.float32) + alpha * np.asarray(color_bgr, dtype=np.float32)
    ).astype(np.uint8)
    return overlay


def save_debug_preview(
    out_path: str,
    frame_index: int,
    top_frame: np.ndarray,
    top_mask: np.ndarray,
    front_frame: np.ndarray,
    front_mask: np.ndarray,
    L: float,
    W: float,
    H: float,
    A: float,
    P: float,
    score: float,
) -> None:
    top_frame = resize_to_max_width(top_frame, 900)
    front_frame = resize_to_max_width(front_frame, 900)
    top_mask = cv2.resize(top_mask, (top_frame.shape[1], top_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    front_mask = cv2.resize(front_mask, (front_frame.shape[1], front_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    top_vis = overlay_mask(top_frame, top_mask, color_bgr=(0, 255, 255), alpha=0.35)
    front_vis = overlay_mask(front_frame, front_mask, color_bgr=(255, 255, 0), alpha=0.35)

    w = max(top_vis.shape[1], front_vis.shape[1])

    def pad_to(img: np.ndarray, width: int) -> np.ndarray:
        if img.shape[1] == width:
            return img
        pad = width - img.shape[1]
        return cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    top_vis = pad_to(top_vis, w)
    front_vis = pad_to(front_vis, w)
    canvas = np.vstack([top_vis, front_vis])

    text = (
        f"frame={frame_index}  "
        f"L={L:.2f}cm  W={W:.2f}cm  H={H:.2f}cm  "
        f"A={A:.2f}cm2  P={P:.2f}cm  score={score:.1f}"
    )
    cv2.putText(canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas)


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if not np.isfinite(fps) or fps <= 0:
        return float("nan")
    return fps


def measure_top(mask: np.ndarray, cm_per_px: float) -> tuple[float, float, float, float]:
    pts = mask_to_points(mask)
    if pts is None or len(pts) < 30:
        return float("nan"), float("nan"), float("nan"), float("nan")

    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    area_px = float(cv2.contourArea(hull))
    peri_px = float(cv2.arcLength(hull, True))
    rect = cv2.minAreaRect(hull)
    w, h = rect[1]
    L_px, W_px = (max(w, h), min(w, h))

    L_cm = L_px * cm_per_px
    W_cm = W_px * cm_per_px
    area_cm2 = area_px * (cm_per_px**2)
    peri_cm = peri_px * cm_per_px
    return L_cm, W_cm, area_cm2, peri_cm


def measure_height(mask: np.ndarray, cm_per_px: float) -> float:
    pts = mask_to_points(mask)
    if pts is None or len(pts) < 30:
        return float("nan")
    ys = pts[:, 1]
    return float((ys.max() - ys.min()) * cm_per_px)


def process_pair(
    top_video: str,
    front_video: str,
    calib: Calibration,
    cfg: ExtractConfig,
    debug_dir: str | None,
    fish_id: str,
    debug_max_images: int,
    debug_save_backgrounds: bool,
) -> tuple[float, float, float, float, float]:
    top_cap = cv2.VideoCapture(top_video)
    front_cap = cv2.VideoCapture(front_video)

    top_n = int(top_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    front_n = int(front_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(top_n, front_n)
    if n <= 0:
        top_cap.release()
        front_cap.release()
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    top_bg = build_background(top_video)
    front_bg = build_background(front_video)
    top_ex = static_exclude_mask(top_bg, "top")
    front_ex = static_exclude_mask(front_bg, "front")

    candidates: list[tuple[float, float, float, float, float]] = []
    scores: list[float] = []
    debug_items: list[dict[str, object]] = []

    fish_debug_dir = None
    if debug_dir:
        fish_debug_dir = os.path.join(debug_dir, fish_id)
        ensure_dir(fish_debug_dir)
        if debug_save_backgrounds:
            cv2.imwrite(os.path.join(fish_debug_dir, "bg_top.jpg"), top_bg)
            cv2.imwrite(os.path.join(fish_debug_dir, "bg_front.jpg"), front_bg)
            cv2.imwrite(os.path.join(fish_debug_dir, "exclude_top.png"), top_ex)
            cv2.imwrite(os.path.join(fish_debug_dir, "exclude_front.png"), front_ex)

    i = 0
    while i < n:
        ok_t, t = top_cap.read()
        ok_f, f = front_cap.read()
        if not ok_t or not ok_f or t is None or f is None:
            break

        if (i % cfg.stride) != 0:
            i += 1
            continue

        top_m = motion_mask(t, top_bg, top_ex, diff_thresh=cfg.diff_thresh_top)
        front_m = motion_mask(f, front_bg, front_ex, diff_thresh=cfg.diff_thresh_front)

        top_m = combine_large_components(top_m, min_area_px=cfg.min_area_top)
        front_m = combine_large_components(front_m, min_area_px=cfg.min_area_front)

        if cv2.countNonZero(top_m) < cfg.min_area_top or cv2.countNonZero(front_m) < cfg.min_area_front:
            i += 1
            continue
        if not passes_border_margin(top_m, margin_px=cfg.border_margin) or not passes_border_margin(
            front_m, margin_px=cfg.border_margin
        ):
            i += 1
            continue

        L, W, A, P = measure_top(top_m, calib.cm_ground_per_px)
        H = measure_height(front_m, calib.cm_vertical_per_px)
        if not (np.isfinite(L) and np.isfinite(W) and np.isfinite(A) and np.isfinite(P) and np.isfinite(H)):
            i += 1
            continue

        if L <= 0 or W <= 0 or H <= 0:
            i += 1
            continue
        if L < W:
            i += 1
            continue
        if (L / max(W, 1e-6)) < cfg.min_aspect:
            i += 1
            continue

        bt = blur_score(t)
        bf = blur_score(f)
        if bt < cfg.min_blur or bf < cfg.min_blur:
            i += 1
            continue

        score = (bt + bf) + 0.02 * float(cv2.countNonZero(top_m))
        candidates.append((L, W, H, A, P))
        scores.append(score)
        if fish_debug_dir:
            debug_items.append(
                {
                    "frame_index": i,
                    "top_frame": t.copy(),
                    "top_mask": top_m.copy(),
                    "front_frame": f.copy(),
                    "front_mask": front_m.copy(),
                    "L": float(L),
                    "W": float(W),
                    "H": float(H),
                    "A": float(A),
                    "P": float(P),
                    "score": float(score),
                }
            )

        i += 1

    top_cap.release()
    front_cap.release()

    if not candidates:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    order = np.argsort(-np.asarray(scores))
    order = order[: cfg.max_frames]
    data = np.asarray([candidates[j] for j in order], dtype=float)
    L, W, H, A, P = np.nanmedian(data, axis=0)

    if debug_items and fish_debug_dir:
        by_score = sorted(debug_items, key=lambda d: float(d["score"]), reverse=True)
        keep = by_score[: max(1, int(debug_max_images))]
        for rank, item in enumerate(keep, start=1):
            out_path = os.path.join(fish_debug_dir, f"{rank:02d}_frame{int(item['frame_index'])}.jpg")
            save_debug_preview(
                out_path=out_path,
                frame_index=int(item["frame_index"]),
                top_frame=item["top_frame"],
                top_mask=item["top_mask"],
                front_frame=item["front_frame"],
                front_mask=item["front_mask"],
                L=float(item["L"]),
                W=float(item["W"]),
                H=float(item["H"]),
                A=float(item["A"]),
                P=float(item["P"]),
                score=float(item["score"]),
            )

    return float(L), float(W), float(H), float(A), float(P)


def process_pair_per_frame(
    top_video: str,
    front_video: str,
    calib: Calibration,
    cfg: ExtractConfig,
    fish_id: str,
    assumed_fps: float,
) -> list[dict[str, object]]:
    top_cap = cv2.VideoCapture(top_video)
    front_cap = cv2.VideoCapture(front_video)

    top_n = int(top_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    front_n = int(front_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(top_n, front_n)
    if n <= 0:
        top_cap.release()
        front_cap.release()
        return []

    fps_top = float(top_cap.get(cv2.CAP_PROP_FPS))
    fps_front = float(front_cap.get(cv2.CAP_PROP_FPS))
    fps = fps_top
    if not np.isfinite(fps) or fps <= 0:
        fps = fps_front
    if not np.isfinite(fps) or fps <= 0:
        fps = float(assumed_fps)

    top_bg = build_background(top_video)
    front_bg = build_background(front_video)
    top_ex = static_exclude_mask(top_bg, "top")
    front_ex = static_exclude_mask(front_bg, "front")

    out: list[dict[str, object]] = []
    i = 0
    while i < n:
        ok_t, t = top_cap.read()
        ok_f, f = front_cap.read()
        if not ok_t or not ok_f or t is None or f is None:
            break

        if (i % cfg.stride) != 0:
            i += 1
            continue

        top_m = motion_mask(t, top_bg, top_ex, diff_thresh=cfg.diff_thresh_top)
        front_m = motion_mask(f, front_bg, front_ex, diff_thresh=cfg.diff_thresh_front)

        top_m = combine_large_components(top_m, min_area_px=cfg.min_area_top)
        front_m = combine_large_components(front_m, min_area_px=cfg.min_area_front)

        top_px = int(cv2.countNonZero(top_m))
        front_px = int(cv2.countNonZero(front_m))

        if top_px < cfg.min_area_top or front_px < cfg.min_area_front:
            i += 1
            continue
        if not passes_border_margin(top_m, margin_px=cfg.border_margin) or not passes_border_margin(
            front_m, margin_px=cfg.border_margin
        ):
            i += 1
            continue

        L, W, A, P = measure_top(top_m, calib.cm_ground_per_px)
        H = measure_height(front_m, calib.cm_vertical_per_px)
        if not (np.isfinite(L) and np.isfinite(W) and np.isfinite(A) and np.isfinite(P) and np.isfinite(H)):
            i += 1
            continue

        if L <= 0 or W <= 0 or H <= 0:
            i += 1
            continue
        if L < W:
            i += 1
            continue
        if (L / max(W, 1e-6)) < cfg.min_aspect:
            i += 1
            continue

        bt = blur_score(t)
        bf = blur_score(f)
        if bt < cfg.min_blur or bf < cfg.min_blur:
            i += 1
            continue

        score = (bt + bf) + 0.02 * float(top_px)
        out.append(
            {
                "FishID": fish_id,
                "FrameIndex": int(i),
                "Timestamp (s)": float(i / fps) if np.isfinite(fps) and fps > 0 else float("nan"),
                "FPS_Top": float(fps_top) if np.isfinite(fps_top) else float("nan"),
                "FPS_Front": float(fps_front) if np.isfinite(fps_front) else float("nan"),
                "Length (cm)": float(L),
                "Width (cm)": float(W),
                "Height (cm)": float(H),
                "Area (cm²)": float(A),
                "Perimeter (cm)": float(P),
                "TopMaskPixels": int(top_px),
                "FrontMaskPixels": int(front_px),
                "BlurTop": float(bt),
                "BlurFront": float(bf),
                "Score": float(score),
            }
        )
        i += 1

    top_cap.release()
    front_cap.release()
    return out


def iter_fish_folders(dataset_root: str) -> list[str]:
    out: list[str] = []
    for name in os.listdir(dataset_root):
        p = os.path.join(dataset_root, name)
        if not os.path.isdir(p):
            continue
        if not name.lower().startswith("fish"):
            continue
        out.append(p)
    return sorted(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Path containing fishXX folders",
    )
    parser.add_argument("--weights-csv", default=None, help="Optional CSV with columns FishID,Weight")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--skip-aggregate", action="store_true", default=False)
    parser.add_argument(
        "--per-frame-csv",
        default=None,
        help="Optional CSV for per-frame measurements (single fish only by default)",
    )
    parser.add_argument("--per-frame-include-groups", action="store_true", default=False)
    parser.add_argument("--per-frame-stride", type=int, default=1)
    parser.add_argument("--assume-fps", type=float, default=60.0)
    parser.add_argument("--per-frame-min-blur", type=float, default=10.0)
    parser.add_argument("--per-frame-min-area-top", type=int, default=300)
    parser.add_argument("--per-frame-min-area-front", type=int, default=250)
    parser.add_argument("--per-frame-min-aspect", type=float, default=1.05)
    parser.add_argument("--per-frame-border-margin", type=int, default=8)
    parser.add_argument("--per-frame-diff-thresh-top", type=int, default=12)
    parser.add_argument("--per-frame-diff-thresh-front", type=int, default=10)
    parser.add_argument(
        "--per-frame-only-ids",
        default=None,
        help="Optional comma-separated FishID list to process (e.g., fish1+3+4+5+15)",
    )
    parser.add_argument("--debug-dir", default=None, help="Optional folder to save mask-overlay previews")
    parser.add_argument("--debug-max-images", type=int, default=5)
    parser.add_argument("--debug-save-backgrounds", action="store_true", default=False)
    parser.add_argument(
        "--truth-csv",
        default=None,
        help="Optional CSV for truth values from single image/graph paper photos",
    )
    parser.add_argument("--truth-debug-dir", default=None, help="Optional folder for truth overlay previews")
    parser.add_argument("--truth-px-per-cm", type=float, default=None, help="Optional override for truth image px/cm")
    parser.add_argument("--truth-grid-cm", type=float, default=None, help="Optional grid size in cm per cell/period")
    parser.add_argument("--correct-per-frame-in", default=None, help="Input per-frame CSV to correct using truth")
    parser.add_argument("--correct-truth-csv", default=None, help="Truth CSV used for correction")
    parser.add_argument("--correct-per-frame-out", default=None, help="Output corrected per-frame CSV")
    parser.add_argument("--correct-tol-start", type=float, default=0.2)
    parser.add_argument("--correct-tol-step", type=float, default=0.1)
    parser.add_argument("--correct-tol-max", type=float, default=1.5)
    parser.add_argument("--correct-min-rows", type=int, default=30)
    parser.add_argument("--correct-score-quantile", type=float, default=0.5)
    parser.add_argument("--best-frames-in", default=None, help="Input per-frame CSV to pick best frames by truth")
    parser.add_argument("--best-frames-truth", default=None, help="Truth CSV used for best-frame selection")
    parser.add_argument("--best-frames-out", default=None, help="Output CSV of selected best frames")
    parser.add_argument("--best-frames-ae-max", type=float, default=1.0, help="Max abs error (cm) allowed per fish")
    parser.add_argument("--train-frames-in", default=None, help="Input per-frame CSV to pick training frames by truth")
    parser.add_argument("--train-frames-truth", default=None, help="Truth CSV used for training-frame selection")
    parser.add_argument("--train-frames-out", default=None, help="Output CSV of selected training frames")
    parser.add_argument("--train-frames-per-fish", type=int, default=20, help="Max selected frames per fish")
    parser.add_argument("--train-ae-max", type=float, default=1.0, help="Max abs error (cm) for within selection")
    parser.add_argument("--train-fallback-ae-max", type=float, default=0.0, help="Fallback max abs error (cm) if none within")
    parser.add_argument("--train-fill-ae-max", type=float, default=0.0, help="Max abs error (cm) for fill-to-quota rows")
    parser.add_argument("--train-within-by", choices=["length", "composite"], default="length")
    parser.add_argument("--train-composite-max", type=float, default=0.1)
    parser.add_argument("--train-composite-fallback-max", type=float, default=0.0)
    parser.add_argument("--train-composite-fill-max", type=float, default=0.0)
    parser.add_argument("--train-apply-per-fish-scale", action="store_true", default=False)
    parser.add_argument("--train-apply-per-fish-metric-scales", action="store_true", default=False)
    parser.add_argument("--train-rank-by-composite-error", action="store_true", default=False)
    parser.add_argument("--train-groups-in", default=None, help="Input per-frame CSV to pick training frames for multi-fish groups")
    parser.add_argument("--train-groups-out", default=None, help="Output CSV of selected training frames for multi-fish groups")
    parser.add_argument("--train-frames-per-group", type=int, default=50, help="Max selected frames per group FishID")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Default: dataset_root/fish_measurements.csv",
    )
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    out_csv = args.output_csv or os.path.join(dataset_root, "fish_measurements.csv")
    per_frame_csv = (
        os.path.abspath(args.per_frame_csv)
        if args.per_frame_csv
        else (os.path.join(dataset_root, "fish_frame_measurements.csv") if args.per_frame_csv is not None else None)
    )
    truth_csv = os.path.abspath(args.truth_csv) if args.truth_csv else None
    truth_debug_dir = os.path.abspath(args.truth_debug_dir) if args.truth_debug_dir else None
    debug_dir = os.path.abspath(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        ensure_dir(debug_dir)
    if truth_debug_dir:
        ensure_dir(truth_debug_dir)

    correct_in = os.path.abspath(args.correct_per_frame_in) if args.correct_per_frame_in else None
    correct_truth = os.path.abspath(args.correct_truth_csv) if args.correct_truth_csv else None
    correct_out = (
        os.path.abspath(args.correct_per_frame_out)
        if args.correct_per_frame_out
        else (os.path.join(dataset_root, "fish_frame_measurements_groups_corrected.csv") if correct_in and correct_truth else None)
    )

    if correct_in and correct_truth and correct_out:
        corrected = correct_per_frame_csv_with_truth(
            per_frame_csv_in=correct_in,
            truth_csv=correct_truth,
            out_csv=correct_out,
            tol_start=float(args.correct_tol_start),
            tol_step=float(args.correct_tol_step),
            tol_max=float(args.correct_tol_max),
            min_rows=int(args.correct_min_rows),
            score_quantile=float(args.correct_score_quantile),
        )
        print(correct_out)
        print(corrected.groupby("FishID").size() if not corrected.empty else corrected)
        return 0

    best_in = os.path.abspath(args.best_frames_in) if args.best_frames_in else None
    best_truth = os.path.abspath(args.best_frames_truth) if args.best_frames_truth else None
    best_out = os.path.abspath(args.best_frames_out) if args.best_frames_out else None
    if best_in and best_truth and best_out:
        out = select_best_frames_by_truth(
            per_frame_csv_in=best_in,
            truth_csv=best_truth,
            out_csv=best_out,
            ae_max_cm=float(args.best_frames_ae_max),
        )
        print(best_out)
        print(out.groupby("FishID").size() if not out.empty else out)
        return 0

    train_in = os.path.abspath(args.train_frames_in) if args.train_frames_in else None
    train_truth = os.path.abspath(args.train_frames_truth) if args.train_frames_truth else None
    train_out = os.path.abspath(args.train_frames_out) if args.train_frames_out else None
    if train_in and train_truth and train_out:
        out = select_training_frames_by_truth(
            per_frame_csv_in=train_in,
            truth_csv=train_truth,
            out_csv=train_out,
            frames_per_fish=int(args.train_frames_per_fish),
            ae_max_cm=float(args.train_ae_max),
            fallback_ae_max_cm=float(args.train_fallback_ae_max),
            fill_ae_max_cm=float(args.train_fill_ae_max),
            within_by=str(args.train_within_by),
            composite_max=float(args.train_composite_max),
            composite_fallback_max=float(args.train_composite_fallback_max),
            composite_fill_max=float(args.train_composite_fill_max),
            apply_per_fish_scale=bool(args.train_apply_per_fish_scale),
            apply_per_fish_metric_scales=bool(args.train_apply_per_fish_metric_scales),
            rank_by_composite_error=bool(args.train_rank_by_composite_error),
        )
        print(train_out)
        print(out.groupby("FishID").size() if not out.empty else out)
        return 0

    train_groups_in = os.path.abspath(args.train_groups_in) if args.train_groups_in else None
    train_groups_out = os.path.abspath(args.train_groups_out) if args.train_groups_out else None
    if train_groups_in and train_groups_out:
        out = select_training_frames_for_groups(
            per_frame_csv_in=train_groups_in,
            out_csv=train_groups_out,
            frames_per_group=int(args.train_frames_per_group),
        )
        print(train_groups_out)
        print(out.groupby("FishID").size() if not out.empty else out)
        return 0

    calib = load_calibration(dataset_root)
    weights = load_weights_map(args.weights_csv)

    folders = iter_fish_folders(dataset_root)

    if truth_csv:
        truth_rows: list[dict[str, object]] = []
        for folder in tqdm([p for p in folders if is_single_fish_id(os.path.basename(p))], desc="Truth extraction (single images)"):
            fish_id = os.path.basename(folder)
            img_path = find_first_image(os.path.join(folder, "single image"))
            if not img_path:
                continue
            row = {"FishID": fish_id}
            debug_out = os.path.join(truth_debug_dir, f"{fish_id}.jpg") if truth_debug_dir else None
            row.update(
                truth_from_single_image(
                    img_path,
                    px_per_cm_override=args.truth_px_per_cm,
                    grid_cm=args.truth_grid_cm,
                    debug_out_path=debug_out,
                )
            )
            truth_rows.append(row)
        truth_df = pd.DataFrame(truth_rows)
        if not truth_df.empty:
            cols = [
                "FishID",
                "ImagePath",
                "PxPerCm",
                "Length_truth (cm)",
                "Width_truth (cm)",
                "Area_truth (cm²)",
                "Perimeter_truth (cm)",
            ]
            existing = [c for c in cols if c in truth_df.columns]
            truth_df = truth_df[existing]
        truth_df.to_csv(truth_csv, index=False)
        print(truth_csv)
        print(truth_df)

    agg_cfg = ExtractConfig(
        stride=max(1, int(args.stride)),
        max_frames=max(5, int(args.max_frames)),
        diff_thresh_top=12,
        diff_thresh_front=10,
        min_area_top=500,
        min_area_front=400,
        border_margin=8,
        min_blur=20.0,
        min_aspect=1.1,
    )

    per_frame_cfg = ExtractConfig(
        stride=max(1, int(args.per_frame_stride)),
        max_frames=max(5, int(args.max_frames)),
        diff_thresh_top=int(args.per_frame_diff_thresh_top),
        diff_thresh_front=int(args.per_frame_diff_thresh_front),
        min_area_top=max(1, int(args.per_frame_min_area_top)),
        min_area_front=max(1, int(args.per_frame_min_area_front)),
        border_margin=max(0, int(args.per_frame_border_margin)),
        min_blur=float(args.per_frame_min_blur),
        min_aspect=float(args.per_frame_min_aspect),
    )

    if per_frame_csv:
        frame_rows: list[dict[str, object]] = []
        per_frame_folders = (
            folders
            if bool(args.per_frame_include_groups)
            else [p for p in folders if is_single_fish_id(os.path.basename(p))]
        )
        if args.per_frame_only_ids:
            wanted = {s.strip() for s in str(args.per_frame_only_ids).split(",") if s.strip()}
            per_frame_folders = [p for p in per_frame_folders if os.path.basename(p) in wanted]
        for folder in tqdm(per_frame_folders, desc="Per-frame extraction"):
            fish_id = os.path.basename(folder)
            top_video = find_first_video(os.path.join(folder, "top view"))
            front_video = find_first_video(os.path.join(folder, "front view"))
            if not top_video or not front_video:
                continue
            rows = process_pair_per_frame(
                top_video=top_video,
                front_video=front_video,
                calib=calib,
                cfg=per_frame_cfg,
                fish_id=fish_id,
                assumed_fps=float(args.assume_fps),
            )
            for r in rows:
                r["Weight (g)"] = compute_weight(weights, fish_id)
            frame_rows.extend(rows)
        frame_df = pd.DataFrame(frame_rows)
        if not frame_df.empty:
            cols = [
                "FishID",
                "Weight (g)",
                "FrameIndex",
                "Timestamp (s)",
                "FPS_Top",
                "FPS_Front",
                "Length (cm)",
                "Width (cm)",
                "Height (cm)",
                "Area (cm²)",
                "Perimeter (cm)",
                "TopMaskPixels",
                "FrontMaskPixels",
                "BlurTop",
                "BlurFront",
                "Score",
            ]
            existing = [c for c in cols if c in frame_df.columns]
            frame_df = frame_df[existing]
        frame_df.to_csv(per_frame_csv, index=False)
        print(per_frame_csv)
        print(frame_df.groupby("FishID").size() if not frame_df.empty else frame_df)

    if not bool(args.skip_aggregate):
        rows: list[dict[str, object]] = []
        for folder in tqdm(folders, desc="Aggregate extraction (per folder)"):
            fish_id = os.path.basename(folder)
            top_video = find_first_video(os.path.join(folder, "top view"))
            front_video = find_first_video(os.path.join(folder, "front view"))
            if not top_video or not front_video:
                rows.append(
                    {
                        "FishID": fish_id,
                        "Weight (g)": compute_weight(weights, fish_id),
                        "Length (cm)": float("nan"),
                        "Width (cm)": float("nan"),
                        "Height (cm)": float("nan"),
                        "Area (cm²)": float("nan"),
                        "Perimeter (cm)": float("nan"),
                    }
                )
                continue

            L, W, H, A, P = process_pair(
                top_video=top_video,
                front_video=front_video,
                calib=calib,
                cfg=agg_cfg,
                debug_dir=debug_dir,
                fish_id=fish_id,
                debug_max_images=max(1, int(args.debug_max_images)),
                debug_save_backgrounds=bool(args.debug_save_backgrounds),
            )
            rows.append(
                {
                    "FishID": fish_id,
                    "Weight (g)": compute_weight(weights, fish_id),
                    "Length (cm)": L,
                    "Width (cm)": W,
                    "Height (cm)": H,
                    "Area (cm²)": A,
                    "Perimeter (cm)": P,
                }
            )

        df = pd.DataFrame(
            rows,
            columns=[
                "FishID",
                "Weight (g)",
                "Length (cm)",
                "Width (cm)",
                "Height (cm)",
                "Area (cm²)",
                "Perimeter (cm)",
            ],
        )
        df.to_csv(out_csv, index=False)
        print(out_csv)
        print(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

