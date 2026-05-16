import argparse
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.linear_model import ElasticNet, Lasso, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GroupKFold, GridSearchCV, LeaveOneGroupOut
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float
    r2: float


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def _ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    Y = _as_2d(np.asarray(Y, dtype=float))
    n, d = X.shape
    ones = np.ones((n, 1), dtype=float)
    Xa = np.hstack([ones, X])
    I = np.eye(d + 1, dtype=float)
    I[0, 0] = 0.0
    A = Xa.T @ Xa + float(alpha) * I
    B = Xa.T @ Y
    W = np.linalg.solve(A, B)
    b = W[0, :].reshape(1, -1)
    w = W[1:, :]
    return w, b


def _ridge_predict(X: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X @ w + b


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return Metrics(mae=float("nan"), rmse=float("nan"), r2=float("nan"))
    yt = y_true[m]
    yp = y_pred[m]
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(float(np.mean(err * err))))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    return Metrics(mae=mae, rmse=rmse, r2=r2)


def _zscore_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma = np.where(sigma <= 1e-12, 1.0, sigma)
    return mu, sigma


def _zscore_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return (X - mu) / sigma


def _split_by_fish_id(df: pd.DataFrame, test_frac: float, random_state: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = sorted({str(x) for x in df["FishID"].dropna().astype(str).tolist()})
    if not ids:
        return df.iloc[0:0].copy(), df.iloc[0:0].copy()
    k = max(1, int(round(len(ids) * float(test_frac))))
    if random_state is not None:
        rng = np.random.default_rng(int(random_state))
        ids = list(ids)
        rng.shuffle(ids)
    test_ids = set(ids[:k])
    train = df[~df["FishID"].astype(str).isin(test_ids)].copy()
    test = df[df["FishID"].astype(str).isin(test_ids)].copy()
    return train, test


def _clean_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _eval_weight(df: pd.DataFrame, features: list[str], target: str, alpha: float, test_frac: float, random_state: int) -> None:
    df = _clean_numeric(df, features + [target])
    df = df[np.isfinite(df[target])].copy()
    for c in features:
        df = df[np.isfinite(df[c])].copy()
    train, test = _split_by_fish_id(df, test_frac=test_frac, random_state=random_state)
    if train.empty or test.empty:
        raise RuntimeError("Not enough data after filtering to create a train/test split.")

    X_train = train[features].to_numpy(dtype=float)
    y_train = train[target].to_numpy(dtype=float)
    X_test = test[features].to_numpy(dtype=float)
    y_test = test[target].to_numpy(dtype=float)

    mu, sigma = _zscore_fit(X_train)
    X_train_s = _zscore_apply(X_train, mu, sigma)
    X_test_s = _zscore_apply(X_test, mu, sigma)

    w, b = _ridge_fit(X_train_s, y_train.reshape(-1, 1), alpha=alpha)
    pred = _ridge_predict(X_test_s, w, b).reshape(-1)
    m = _metrics(y_test, pred)

    print("Task: weight regression")
    print(f"Target: {target}")
    print(f"Features: {', '.join(features)}")
    print(f"Train fish IDs: {train['FishID'].nunique()}  rows: {len(train)}")
    print(f"Test fish IDs: {test['FishID'].nunique()}   rows: {len(test)}")
    print(f"MAE:  {m.mae:.4f}")
    print(f"RMSE: {m.rmse:.4f}")
    print(f"R2:   {m.r2:.4f}")

def _sk_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return Metrics(mae=float("nan"), rmse=float("nan"), r2=float("nan"))
    yt = y_true[m]
    yp = y_pred[m]
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(math.sqrt(float(mean_squared_error(yt, yp))))
    r2 = float(r2_score(yt, yp))
    return Metrics(mae=mae, rmse=rmse, r2=r2)


def _eval_weight_tuned_sklearn(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    test_frac: float,
    cv_splits: int,
    n_jobs: int,
    random_state: int,
    do_bagging: bool,
    do_stacking: bool,
    repeats: int,
    select_by: str,
    out_csv: str | None,
) -> None:
    if not _HAS_SKLEARN:
        raise RuntimeError("scikit-learn is not installed. Install it with: python -m pip install -U scikit-learn")

    df = _clean_numeric(df, features + [target])
    df = df[np.isfinite(df[target])].copy()
    for c in features:
        df = df[np.isfinite(df[c])].copy()

    def run_once(seed: int) -> dict[str, object]:
        train, test = _split_by_fish_id(df, test_frac=test_frac, random_state=seed)
        if train.empty or test.empty:
            raise RuntimeError("Not enough data after filtering to create a train/test split.")

        X_train = train[features].to_numpy(dtype=float)
        y_train = train[target].to_numpy(dtype=float)
        X_test = test[features].to_numpy(dtype=float)
        y_test = test[target].to_numpy(dtype=float)
        groups_train = train["FishID"].astype(str).to_numpy()

        n_fish_train = int(train["FishID"].nunique())
        local_splits = int(cv_splits)
        if local_splits < 2:
            local_splits = 2
        if n_fish_train < 2:
            raise RuntimeError("Need at least 2 FishIDs in train split.")
        local_splits = min(local_splits, n_fish_train)
        cv = GroupKFold(n_splits=local_splits)

        models: list[tuple[str, object, dict[str, list[object]]]] = []
        models.append(
            (
                "ridge",
                Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=seed))]),
                {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
            )
        )
        models.append(
            (
                "lasso",
                Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=20000, random_state=seed))]),
                {"model__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]},
            )
        )
        models.append(
            (
                "elasticnet",
                Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(max_iter=20000, random_state=seed))]),
                {"model__alpha": [1e-3, 1e-2, 1e-1, 1.0], "model__l1_ratio": [0.1, 0.5, 0.9]},
            )
        )
        models.append(
            (
                "svr_rbf",
                Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf"))]),
                {"model__C": [1.0, 10.0, 100.0], "model__epsilon": [0.1, 1.0], "model__gamma": ["scale", 0.1, 0.01]},
            )
        )
        models.append(
            (
                "knn",
                Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())]),
                {"model__n_neighbors": [3, 5, 7, 9], "model__weights": ["uniform", "distance"]},
            )
        )
        models.append(
            (
                "random_forest",
                RandomForestRegressor(random_state=seed, n_estimators=400),
                {"max_depth": [None, 6, 12], "min_samples_leaf": [1, 2, 4]},
            )
        )
        # Improved ExtraTrees with stronger regularization for small dataset
        models.append(
            (
                "extra_trees_stable",
                ExtraTreesRegressor(random_state=seed, n_estimators=200, max_features='sqrt', bootstrap=True),
                {
                    "max_depth": [3, 6, 9], 
                    "min_samples_leaf": [3, 5, 8],  # Increased minimum samples
                    "min_samples_split": [5, 10],   # Added minimum split samples
                    "max_features": ['sqrt', 'log2', 0.7]  # Feature bagging
                },
            )
        )
        # Improved GBR with stronger regularization and early stopping
        models.append(
            (
                "gbr_stable",
                GradientBoostingRegressor(random_state=seed, validation_fraction=0.2, n_iter_no_change=10),
                {
                    "n_estimators": [100, 200],  # Reduced from 500
                    "learning_rate": [0.05, 0.1, 0.15], 
                    "max_depth": [2, 3],  # Kept shallow
                    "subsample": [0.8, 0.9],  # Added subsampling
                    "min_samples_leaf": [3, 5]  # Added minimum leaf samples
                },
            )
        )
        if do_bagging:
            models.append(
                (
                    "bagging_ridge",
                    BaggingRegressor(
                        estimator=Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=seed, alpha=1.0))]),
                        random_state=seed,
                    ),
                    {"n_estimators": [25, 100], "max_samples": [0.7, 1.0]},
                )
            )

        results: list[dict[str, object]] = []
        tuned_estimators: dict[str, object] = {}
        best_by_cv = None
        for name, estimator, grid in models:
            search = GridSearchCV(
                estimator=estimator,
                param_grid=grid,
                scoring="neg_mean_absolute_error",
                cv=cv,
                n_jobs=int(n_jobs),
                refit=True,
            )
            search.fit(X_train, y_train, groups=groups_train)
            tuned = search.best_estimator_
            tuned_estimators[name] = tuned
            pred_test = tuned.predict(X_test)
            m_test = _sk_metrics(y_test, pred_test)
            cv_mae = float(-search.best_score_)
            row = {
                "model": name,
                "cv_mae": cv_mae,
                "test_mae": m_test.mae,
                "test_rmse": m_test.rmse,
                "test_r2": m_test.r2,
                "best_params": getattr(search, "best_params_", {}),
            }
            results.append(row)
            if best_by_cv is None or float(row["cv_mae"]) < float(best_by_cv["cv_mae"]):
                best_by_cv = row

        if do_stacking:
            base = []
            for base_name in ["ridge", "svr_rbf", "random_forest"]:
                if base_name in tuned_estimators:
                    base.append((base_name, tuned_estimators[base_name]))
            if len(base) >= 2:
                stack = StackingRegressor(
                    estimators=base,
                    final_estimator=Ridge(alpha=1.0, random_state=seed),
                    passthrough=False,
                    n_jobs=int(n_jobs),
                )
                stack.fit(X_train, y_train)
                pred_test = stack.predict(X_test)
                m_test = _sk_metrics(y_test, pred_test)
                results.append(
                    {
                        "model": "stacking",
                        "cv_mae": float("nan"),
                        "test_mae": m_test.mae,
                        "test_rmse": m_test.rmse,
                        "test_r2": m_test.r2,
                        "best_params": {"base": [n for n, _ in base], "final": "Ridge(alpha=1.0)"},
                    }
                )

        best_test = min(results, key=lambda r: float(r["test_mae"]))
        return {
            "seed": int(seed),
            "train_fish": int(train["FishID"].nunique()),
            "test_fish": int(test["FishID"].nunique()),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "cv_splits": int(local_splits),
            "results": results,
            "best_test": best_test,
            "best_cv": best_by_cv,
            "test_ids": sorted({str(x) for x in test["FishID"].astype(str).tolist()}),
        }

    repeats = int(repeats)
    if repeats <= 1:
        out = run_once(int(random_state))
        results = list(out["results"])
        best_by_cv = out["best_cv"]
        train_fish = out["train_fish"]
        test_fish = out["test_fish"]
        train_rows = out["train_rows"]
        test_rows = out["test_rows"]
        local_splits = out["cv_splits"]

        results_sorted = sorted(results, key=lambda r: float(r["test_mae"]))
        print("Task: weight regression (multiple models + tuning)")
        print(f"Target: {target}")
        print(f"Features: {', '.join(features)}")
        print(f"Train fish IDs: {train_fish}  rows: {train_rows}")
        print(f"Test fish IDs: {test_fish}   rows: {test_rows}")
        print(f"CV: GroupKFold splits={local_splits} (group=FishID)")
        print("")
        print("Leaderboard (sorted by test MAE):")
        for r in results_sorted:
            cv_mae = float(r["cv_mae"])
            cv_txt = f"{cv_mae:.4f}" if np.isfinite(cv_mae) else "n/a"
            print(
                f"- {r['model']}: test_MAE={float(r['test_mae']):.4f}  test_RMSE={float(r['test_rmse']):.4f}  test_R2={float(r['test_r2']):.4f}  cv_MAE={cv_txt}"
            )
        if best_by_cv is not None:
            print("")
            print(f"Best by CV MAE: {best_by_cv['model']}  cv_MAE={float(best_by_cv['cv_mae']):.4f}")
            if best_by_cv.get("best_params"):
                print(f"Best params: {best_by_cv['best_params']}")
        return

    select_by = str(select_by).strip().lower()
    if select_by not in {"cv", "test"}:
        select_by = "cv"

    outs = [run_once(int(random_state) + i) for i in range(repeats)]

    selected_models: list[str] = []
    maes: list[float] = []
    rmses: list[float] = []
    r2s: list[float] = []
    rows_out: list[dict[str, object]] = []
    for o in outs:
        seed = int(o["seed"])
        results = list(o["results"])

        chosen: dict[str, object] | None = None
        if select_by == "cv":
            cv_candidates = [r for r in results if np.isfinite(float(r.get("cv_mae", float("nan"))))]
            if cv_candidates:
                chosen = min(cv_candidates, key=lambda r: float(r["cv_mae"]))
        if chosen is None:
            chosen = min(results, key=lambda r: float(r["test_mae"]))

        selected_models.append(str(chosen["model"]))
        maes.append(float(chosen["test_mae"]))
        rmses.append(float(chosen["test_rmse"]))
        r2s.append(float(chosen["test_r2"]))
        rows_out.append(
            {
                "seed": seed,
                "train_fish_ids": int(o["train_fish"]),
                "test_fish_ids": int(o["test_fish"]),
                "train_rows": int(o["train_rows"]),
                "test_rows": int(o["test_rows"]),
                "test_ids": ",".join(list(o["test_ids"]))[:200],
                "selected_by": select_by,
                "model": str(chosen["model"]),
                "cv_mae": _safe_float(chosen.get("cv_mae", float("nan"))),
                "test_mae": float(chosen["test_mae"]),
                "test_rmse": float(chosen["test_rmse"]),
                "test_r2": float(chosen["test_r2"]),
                "best_params": str(chosen.get("best_params", {}))[:500],
            }
        )

    def mean_std(x: list[float]) -> tuple[float, float]:
        a = np.asarray(x, dtype=float)
        return float(np.mean(a)), float(np.std(a))

    mae_mu, mae_sd = mean_std(maes)
    rmse_mu, rmse_sd = mean_std(rmses)
    r2_mu, r2_sd = mean_std(r2s)

    counts: dict[str, int] = {}
    for m in selected_models:
        counts[m] = counts.get(m, 0) + 1
    winners = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    train_fish = int(outs[0]["train_fish"])
    test_fish = int(outs[0]["test_fish"])
    print("Task: weight regression (repeated FishID-holdout + tuning)")
    print(f"Target: {target}")
    print(f"Features: {', '.join(features)}")
    print(f"Split: test_frac={float(test_frac):.3f}  repeats={repeats}  seed_start={int(random_state)}")
    print(f"Per split: train fish IDs ~{train_fish}, test fish IDs ~{test_fish}")
    print(f"Selection policy: {select_by} (cv = choose best by CV MAE, then report test metrics)")
    print("")
    print("Per-split results:")
    per_df = pd.DataFrame(rows_out)
    for _, r in per_df.iterrows():
        cv_mae = float(r["cv_mae"])
        cv_txt = f"{cv_mae:.4f}" if np.isfinite(cv_mae) else "n/a"
        print(
            f"- seed={int(r['seed'])}  model={r['model']}  cv_MAE={cv_txt}  "
            f"test_MAE={float(r['test_mae']):.4f}  test_RMSE={float(r['test_rmse']):.4f}  test_R2={float(r['test_r2']):.4f}  "
            f"test_ids={r['test_ids']}"
        )
    print("")
    print("Accuracy summary (best model per split, measured in grams):")
    print(f"MAE:  mean={mae_mu:.4f}  std={mae_sd:.4f}")
    print(f"RMSE: mean={rmse_mu:.4f}  std={rmse_sd:.4f}")
    print(f"R2:   mean={r2_mu:.4f}  std={r2_sd:.4f}")
    print("")
    print("Best-model frequency (by test MAE):")
    for name, cnt in winners:
        print(f"- {name}: {cnt}/{repeats}")

    if out_csv:
        per_df.to_csv(out_csv, index=False)
        print("")
        print(f"Wrote per-split results CSV: {out_csv}")


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    L = pd.to_numeric(out.get("Length (cm)"), errors="coerce")
    W = pd.to_numeric(out.get("Width (cm)"), errors="coerce")
    H = pd.to_numeric(out.get("Height (cm)"), errors="coerce")
    A = pd.to_numeric(out.get("Area (cm²)"), errors="coerce")
    P = pd.to_numeric(out.get("Perimeter (cm)"), errors="coerce")

    out["LW"] = L * W
    out["LH"] = L * H
    out["WH"] = W * H
    out["VolumeProxy"] = L * W * H
    out["AreaHeight"] = A * H
    out["Aspect_LW"] = L / W.replace(0, np.nan)
    out["Compactness_P2_over_4piA"] = (P * P) / (4.0 * math.pi * A.replace(0, np.nan))
    return out


def _aggregate_one_row_per_fish(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keep = ["FishID"] + [c for c in cols if c in df.columns]
    x = df[keep].copy()
    for c in keep:
        if c != "FishID":
            x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.groupby("FishID", sort=False).median(numeric_only=True).reset_index()


def _eval_weight_fish_level_tuned(
    df: pd.DataFrame,
    base_features: list[str],
    target: str,
    cv_splits: int,
    n_jobs: int,
    random_state: int,
    derived_features: bool,
    export_csv: str | None,
) -> None:
    if not _HAS_SKLEARN:
        raise RuntimeError("scikit-learn is not installed. Install it with: python -m pip install -U scikit-learn")

    df = df.copy()
    if derived_features:
        df = _add_derived_features(df)

    feature_cols = list(base_features)
    if derived_features:
        feature_cols = feature_cols + ["LW", "LH", "WH", "VolumeProxy", "AreaHeight", "Aspect_LW", "Compactness_P2_over_4piA"]

    df = _clean_numeric(df, feature_cols + [target])
    df = df[np.isfinite(df[target])].copy()
    for c in feature_cols:
        df = df[np.isfinite(df[c])].copy()

    agg = _aggregate_one_row_per_fish(df, cols=feature_cols + [target])
    if int(agg["FishID"].nunique()) < 5:
        raise RuntimeError("Need at least 5 FishIDs for fish-level evaluation.")

    if export_csv:
        out_cols = ["FishID", target] + feature_cols
        existing = [c for c in out_cols if c in agg.columns]
        agg[existing].to_csv(export_csv, index=False)
        print(f"Wrote fish-level dataset CSV: {export_csv}")

    X = agg[feature_cols].to_numpy(dtype=float)
    y = pd.to_numeric(agg[target], errors="coerce").to_numpy(dtype=float)
    groups = agg["FishID"].astype(str).to_numpy()

    outer = LeaveOneGroupOut()
    inner_splits = max(2, min(int(cv_splits), max(2, int(len(np.unique(groups)) - 1))))
    inner = GroupKFold(n_splits=inner_splits)

    models: list[tuple[str, object, dict[str, list[object]]]] = [
        (
            "ridge",
            Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=random_state))]),
            {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        (
            "svr_rbf",
            Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf"))]),
            {"model__C": [1.0, 10.0, 100.0], "model__epsilon": [0.1, 1.0], "model__gamma": ["scale", 0.1, 0.01]},
        ),
        (
            "random_forest",
            RandomForestRegressor(random_state=random_state, n_estimators=800),
            {"max_depth": [None, 4, 8], "min_samples_leaf": [1, 2, 4]},
        ),
        # Improved ExtraTrees with stronger regularization
        (
            "extra_trees_stable",
            ExtraTreesRegressor(random_state=random_state, n_estimators=150, max_features='sqrt', bootstrap=True),
            {
                "max_depth": [4, 6, 8], 
                "min_samples_leaf": [3, 5, 7],
                "min_samples_split": [5, 8, 12],
                "max_features": ['sqrt', 'log2', 0.6]
            },
        ),
        # Improved GBR with early stopping and regularization
        (
            "gbr_stable",
            GradientBoostingRegressor(random_state=random_state, validation_fraction=0.15, n_iter_no_change=15),
            {
                "n_estimators": [80, 150, 200],
                "learning_rate": [0.05, 0.08, 0.12],
                "max_depth": [2, 3],
                "subsample": [0.7, 0.8, 0.9],
                "min_samples_leaf": [3, 5, 7]
            },
        ),
    ]

    preds: list[float] = []
    trues: list[float] = []
    chosen_models: list[str] = []

    for train_idx, test_idx in outer.split(X, y, groups=groups):
        X_tr, y_tr, g_tr = X[train_idx], y[train_idx], groups[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        best_cv = None
        best_est = None
        best_name = None
        for name, est, grid in models:
            search = GridSearchCV(
                estimator=est,
                param_grid=grid,
                scoring="neg_mean_absolute_error",
                cv=inner,
                n_jobs=int(n_jobs),
                refit=True,
            )
            search.fit(X_tr, y_tr, groups=g_tr)
            cv_mae = float(-search.best_score_)
            if best_cv is None or cv_mae < best_cv:
                best_cv = cv_mae
                best_est = search.best_estimator_
                best_name = name

        if best_est is None or best_name is None:
            raise RuntimeError("Unexpected: no estimator selected.")

        preds.append(float(best_est.predict(X_te)[0]))
        trues.append(float(y_te[0]))
        chosen_models.append(str(best_name))

    preds_a = np.asarray(preds, dtype=float)
    trues_a = np.asarray(trues, dtype=float)
    m = _sk_metrics(trues_a, preds_a)

    baseline = float(np.mean(np.abs(trues_a - float(np.mean(trues_a)))))
    counts: dict[str, int] = {}
    for nm in chosen_models:
        counts[nm] = counts.get(nm, 0) + 1
    winners = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    print("Task: weight regression (single fish only, one row per fish)")
    print("Evaluation: Leave-One-Fish-Out (LOFO) with inner GroupKFold tuning")
    print(f"Fish rows: {int(len(agg))}")
    print(f"Features: {', '.join(feature_cols)}")
    print("")
    print(f"MAE:  {m.mae:.4f} g")
    print(f"RMSE: {m.rmse:.4f} g")
    print(f"R2:   {m.r2:.4f}")
    print(f"Baseline MAE (predict global mean): {baseline:.4f} g")
    print("")
    print("Chosen model frequency:")
    for name, cnt in winners:
        print(f"- {name}: {cnt}/{len(agg)}")


def _eval_weight_fish_level_svr(
    df: pd.DataFrame,
    base_features: list[str],
    target: str,
    cv_splits: int,
    random_state: int,
    drop_height: bool,
    log_target: bool,
    top_score_k: int,
    include_pixels: bool,
) -> None:
    if not _HAS_SKLEARN:
        raise RuntimeError("scikit-learn is not installed. Install it with: python -m pip install -U scikit-learn")

    feature_cols = list(base_features)
    if drop_height and "Height (cm)" in feature_cols:
        feature_cols = [c for c in feature_cols if c != "Height (cm)"]
    if bool(include_pixels):
        for c in ["TopMaskPixels", "FrontMaskPixels"]:
            if c in df.columns and c not in feature_cols:
                feature_cols.append(c)

    df = _clean_numeric(df, feature_cols + [target])
    df = df[np.isfinite(df[target])].copy()
    for c in feature_cols:
        df = df[np.isfinite(df[c])].copy()

    top_score_k = int(top_score_k)
    if top_score_k < 0:
        top_score_k = 0
    if top_score_k > 0 and "Score" in df.columns:
        df = df.copy()
        df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
        df = df[np.isfinite(df["Score"])].copy()
        df = (
            df.sort_values(["FishID", "Score"], ascending=[True, False], kind="mergesort")
            .groupby("FishID", sort=False)
            .head(top_score_k)
            .reset_index(drop=True)
        )

    agg = _aggregate_one_row_per_fish(df, cols=feature_cols + [target])
    if int(agg["FishID"].nunique()) < 5:
        raise RuntimeError("Need at least 5 FishIDs for fish-level evaluation.")

    X = agg[feature_cols].to_numpy(dtype=float)
    y = pd.to_numeric(agg[target], errors="coerce").to_numpy(dtype=float)
    groups = agg["FishID"].astype(str).to_numpy()

    outer = LeaveOneGroupOut()
    preds: list[float] = []
    trues: list[float] = []
    chosen_params: list[str] = []

    base_seed = int(random_state)
    for fold_idx, (train_idx, test_idx) in enumerate(outer.split(X, y, groups=groups)):
        X_tr, y_tr, g_tr = X[train_idx], y[train_idx], groups[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        n_train = int(len(np.unique(g_tr)))
        inner_splits = max(2, min(int(cv_splits), n_train))
        inner = GroupKFold(n_splits=inner_splits)

        C_grid = [0.1, 1.0, 10.0, 100.0, 1000.0]
        eps_grid = [0.01, 0.05, 0.1, 0.5, 1.0]
        gamma_grid: list[object] = ["scale", "auto", 0.3, 0.1, 0.03, 0.01, 0.003]

        best = None
        best_params = None
        for C in C_grid:
            for eps in eps_grid:
                for gamma in gamma_grid:
                    fold_mae: list[float] = []
                    for tr2, va2 in inner.split(X_tr, y_tr, groups=g_tr):
                        X_a, y_a = X_tr[tr2], y_tr[tr2]
                        X_b, y_b = X_tr[va2], y_tr[va2]

                        if log_target:
                            y_fit = np.log1p(np.maximum(y_a, 0.0))
                        else:
                            y_fit = y_a

                        model = Pipeline(
                            [
                                ("scaler", StandardScaler()),
                                ("model", SVR(kernel="rbf", C=float(C), epsilon=float(eps), gamma=gamma)),
                            ]
                        )
                        model.fit(X_a, y_fit)
                        y_hat = model.predict(X_b)
                        if log_target:
                            y_hat = np.expm1(y_hat)
                        y_hat = np.maximum(y_hat, 0.0)
                        fold_mae.append(float(mean_absolute_error(y_b, y_hat)))

                    score = float(np.mean(np.asarray(fold_mae, dtype=float)))
                    if best is None or score < best:
                        best = score
                        best_params = (float(C), float(eps), gamma)

        if best_params is None:
            raise RuntimeError("Unexpected: tuning produced no best params.")

        C, eps, gamma = best_params
        if log_target:
            y_fit = np.log1p(np.maximum(y_tr, 0.0))
        else:
            y_fit = y_tr

        final = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=float(C), epsilon=float(eps), gamma=gamma)),
            ]
        )
        final.fit(X_tr, y_fit)
        y_pred = float(final.predict(X_te)[0])
        if log_target:
            y_pred = float(np.expm1(y_pred))
        y_pred = float(max(0.0, y_pred))

        preds.append(y_pred)
        trues.append(float(y_te[0]))
        chosen_params.append(f"C={C} eps={eps} gamma={gamma}")

    m = _sk_metrics(np.asarray(trues, dtype=float), np.asarray(preds, dtype=float))
    baseline = float(np.mean(np.abs(np.asarray(trues, dtype=float) - float(np.mean(trues)))))

    print("Task: weight regression (fish-level SVR only, LOFO)")
    print(f"Fish rows: {int(len(agg))}")
    print(f"Features: {', '.join(feature_cols)}")
    print(f"log_target: {bool(log_target)}")
    if top_score_k > 0:
        print(f"Frame selection: top_score_k={top_score_k}")
    print("")
    print(f"MAE:  {m.mae:.4f} g")
    print(f"RMSE: {m.rmse:.4f} g")
    print(f"R2:   {m.r2:.4f}")
    print(f"Baseline MAE (predict global mean): {baseline:.4f} g")
    print("")
    counts: dict[str, int] = {}
    for p in chosen_params:
        counts[p] = counts.get(p, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    print("Most common tuned params:")
    for p, cnt in top:
        print(f"- {p}: {cnt}/{len(agg)}")


def _weight_fish_level_svr_mae(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    cv_splits: int,
    random_state: int,
    log_target: bool,
    top_score_k: int,
) -> float:
    df2 = df.copy()
    df2 = _clean_numeric(df2, feature_cols + [target])
    df2 = df2[np.isfinite(df2[target])].copy()
    for c in feature_cols:
        df2 = df2[np.isfinite(df2[c])].copy()

    top_score_k = int(top_score_k)
    if top_score_k > 0 and "Score" in df2.columns:
        df2["Score"] = pd.to_numeric(df2["Score"], errors="coerce")
        df2 = df2[np.isfinite(df2["Score"])].copy()
        df2 = (
            df2.sort_values(["FishID", "Score"], ascending=[True, False], kind="mergesort")
            .groupby("FishID", sort=False)
            .head(top_score_k)
            .reset_index(drop=True)
        )

    agg = _aggregate_one_row_per_fish(df2, cols=feature_cols + [target])
    X = agg[feature_cols].to_numpy(dtype=float)
    y = pd.to_numeric(agg[target], errors="coerce").to_numpy(dtype=float)
    groups = agg["FishID"].astype(str).to_numpy()

    outer = LeaveOneGroupOut()
    preds: list[float] = []
    trues: list[float] = []

    for train_idx, test_idx in outer.split(X, y, groups=groups):
        X_tr, y_tr, g_tr = X[train_idx], y[train_idx], groups[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        n_train = int(len(np.unique(g_tr)))
        inner_splits = max(2, min(int(cv_splits), n_train))
        inner = GroupKFold(n_splits=inner_splits)

        C_grid = [0.1, 1.0, 10.0, 100.0, 1000.0]
        eps_grid = [0.01, 0.05, 0.1, 0.5, 1.0]
        gamma_grid: list[object] = ["scale", "auto", 0.3, 0.1, 0.03, 0.01, 0.003]

        best = None
        best_params = None
        for C in C_grid:
            for eps in eps_grid:
                for gamma in gamma_grid:
                    fold_mae: list[float] = []
                    for tr2, va2 in inner.split(X_tr, y_tr, groups=g_tr):
                        X_a, y_a = X_tr[tr2], y_tr[tr2]
                        X_b, y_b = X_tr[va2], y_tr[va2]

                        if log_target:
                            y_fit = np.log1p(np.maximum(y_a, 0.0))
                        else:
                            y_fit = y_a

                        model = Pipeline(
                            [
                                ("scaler", StandardScaler()),
                                ("model", SVR(kernel="rbf", C=float(C), epsilon=float(eps), gamma=gamma)),
                            ]
                        )
                        model.fit(X_a, y_fit)
                        y_hat = model.predict(X_b)
                        if log_target:
                            y_hat = np.expm1(y_hat)
                        y_hat = np.maximum(y_hat, 0.0)
                        fold_mae.append(float(mean_absolute_error(y_b, y_hat)))

                    score = float(np.mean(np.asarray(fold_mae, dtype=float)))
                    if best is None or score < best:
                        best = score
                        best_params = (float(C), float(eps), gamma)

        if best_params is None:
            raise RuntimeError("Unexpected: tuning produced no best params.")

        C, eps, gamma = best_params
        if log_target:
            y_fit = np.log1p(np.maximum(y_tr, 0.0))
        else:
            y_fit = y_tr
        final = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=float(C), epsilon=float(eps), gamma=gamma)),
            ]
        )
        final.fit(X_tr, y_fit)
        y_pred = float(final.predict(X_te)[0])
        if log_target:
            y_pred = float(np.expm1(y_pred))
        y_pred = float(max(0.0, y_pred))
        preds.append(y_pred)
        trues.append(float(y_te[0]))

    m = _sk_metrics(np.asarray(trues, dtype=float), np.asarray(preds, dtype=float))
    return float(m.mae)


def _eval_weight_fish_level_search(
    df: pd.DataFrame,
    target: str,
    cv_splits: int,
    random_state: int,
) -> None:
    base = ["Length (cm)", "Width (cm)", "Height (cm)", "Area (cm²)", "Perimeter (cm)"]
    candidates: list[tuple[str, list[str], int, bool]] = []
    for k in [0, 5, 10, 20, 50]:
        candidates.append((f"base_topScore{k}", base, k, False))
        candidates.append((f"base+pixels_topScore{k}", base + ["TopMaskPixels", "FrontMaskPixels"], k, False))

    rows: list[dict[str, object]] = []
    best = None
    best_row = None
    for name, feats, k, log_t in candidates:
        feats2 = [c for c in feats if c in df.columns]
        if any(c not in df.columns for c in feats2):
            continue
        mae = _weight_fish_level_svr_mae(
            df=df,
            feature_cols=feats2,
            target=target,
            cv_splits=cv_splits,
            random_state=random_state,
            log_target=log_t,
            top_score_k=k,
        )
        row = {"variant": name, "top_score_k": int(k), "n_features": int(len(feats2)), "mae_g": float(mae)}
        rows.append(row)
        if best is None or float(mae) < float(best):
            best = float(mae)
            best_row = row

    out = pd.DataFrame(rows).sort_values(["mae_g", "variant"], kind="mergesort").reset_index(drop=True)
    print("Task: weight regression (fish-level search, LOFO)")
    print(f"Target: {target}")
    print(f"Candidates: {len(out)}")
    print("")
    if not out.empty:
        print(out.head(10).to_string(index=False))
    print("")
    if best_row is not None:
        print(f"Best: {best_row['variant']}  MAE={float(best_row['mae_g']):.4f} g")


def _eval_truth_mapping(
    df: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    alpha: float,
    test_frac: float,
    random_state: int,
) -> None:
    df = _clean_numeric(df, x_cols + y_cols)
    for c in x_cols + y_cols:
        df = df[np.isfinite(df[c])].copy()
    train, test = _split_by_fish_id(df, test_frac=test_frac, random_state=random_state)
    if train.empty or test.empty:
        raise RuntimeError("Not enough data after filtering to create a train/test split.")

    X_train = train[x_cols].to_numpy(dtype=float)
    Y_train = train[y_cols].to_numpy(dtype=float)
    X_test = test[x_cols].to_numpy(dtype=float)
    Y_test = test[y_cols].to_numpy(dtype=float)

    mu, sigma = _zscore_fit(X_train)
    X_train_s = _zscore_apply(X_train, mu, sigma)
    X_test_s = _zscore_apply(X_test, mu, sigma)

    w, b = _ridge_fit(X_train_s, Y_train, alpha=alpha)
    pred = _ridge_predict(X_test_s, w, b)

    print("Task: truth-mapping regression (measured -> truth)")
    print(f"Inputs:  {', '.join(x_cols)}")
    print(f"Targets: {', '.join(y_cols)}")
    print(f"Train fish IDs: {train['FishID'].nunique()}  rows: {len(train)}")
    print(f"Test fish IDs: {test['FishID'].nunique()}   rows: {len(test)}")

    baseline_pred = np.zeros_like(Y_test, dtype=float)
    for j, name in enumerate(y_cols):
        base_name = name
        if base_name.startswith("_Truth_"):
            base_name = base_name.replace("_Truth_", "", 1)
        if base_name.endswith("_truth (cm)"):
            base_name = base_name.replace("_truth (cm)", " (cm)")
        if base_name.endswith("_truth (cm²)"):
            base_name = base_name.replace("_truth (cm²)", " (cm²)")
        if base_name in test.columns:
            baseline_pred[:, j] = pd.to_numeric(test[base_name], errors="coerce").to_numpy(dtype=float)
        else:
            baseline_pred[:, j] = np.nan
    print("Baseline (no model): use measured values as prediction")
    for j, name in enumerate(y_cols):
        m = _metrics(Y_test[:, j], baseline_pred[:, j])
        print(f"{name}: MAE={m.mae:.4f}  RMSE={m.rmse:.4f}  R2={m.r2:.4f}")
    print("Ridge regression (fish-ID holdout)")
    for j, name in enumerate(y_cols):
        m = _metrics(Y_test[:, j], pred[:, j])
        print(f"{name}: MAE={m.mae:.4f}  RMSE={m.rmse:.4f}  R2={m.r2:.4f}")


def _eval_truth_tuned_sklearn(
    df: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    test_frac: float,
    cv_splits: int,
    n_jobs: int,
    random_state: int,
) -> None:
    if not _HAS_SKLEARN:
        raise RuntimeError("scikit-learn is not installed. Install it with: python -m pip install -U scikit-learn")

    df = _clean_numeric(df, x_cols + y_cols)
    for c in x_cols + y_cols:
        df = df[np.isfinite(df[c])].copy()

    train, test = _split_by_fish_id(df, test_frac=test_frac, random_state=random_state)
    if train.empty or test.empty:
        raise RuntimeError("Not enough data after filtering to create a train/test split.")

    X_train = train[x_cols].to_numpy(dtype=float)
    X_test = test[x_cols].to_numpy(dtype=float)
    groups_train = train["FishID"].astype(str).to_numpy()

    n_fish_train = int(train["FishID"].nunique())
    cv_splits = max(2, min(int(cv_splits), n_fish_train))
    cv = GroupKFold(n_splits=cv_splits)

    print("Task: truth-mapping regression (tuned models, measured -> truth)")
    print(f"Inputs:  {', '.join(x_cols)}")
    print(f"Targets: {', '.join(y_cols)}")
    print(f"Train fish IDs: {train['FishID'].nunique()}  rows: {len(train)}")
    print(f"Test fish IDs: {test['FishID'].nunique()}   rows: {len(test)}")
    print(f"CV: GroupKFold splits={cv_splits} (group=FishID)")
    print("")

    for target_col in y_cols:
        y_train = train[target_col].to_numpy(dtype=float)
        y_test = test[target_col].to_numpy(dtype=float)

        baseline = None
        base_name = target_col
        if base_name.startswith("_Truth_"):
            base_name = base_name.replace("_Truth_", "", 1)
        if base_name.endswith("_truth (cm)"):
            base_name = base_name.replace("_truth (cm)", " (cm)")
        if base_name.endswith("_truth (cm²)"):
            base_name = base_name.replace("_truth (cm²)", " (cm²)")
        baseline_train = None
        if base_name in train.columns and base_name in test.columns:
            baseline_train = pd.to_numeric(train[base_name], errors="coerce").to_numpy(dtype=float)
            baseline = pd.to_numeric(test[base_name], errors="coerce").to_numpy(dtype=float)
        else:
            baseline = np.full_like(y_test, np.nan, dtype=float)

        base_m = _metrics(y_test, baseline)
        print(f"Target: {target_col}")
        print(f"Baseline MAE={base_m.mae:.4f}  RMSE={base_m.rmse:.4f}  R2={base_m.r2:.4f}")

        models: list[tuple[str, object, dict[str, list[object]]]] = [
            (
                "ridge",
                Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=random_state))]),
                {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
            ),
            (
                "svr_rbf",
                Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf"))]),
                {"model__C": [1.0, 10.0, 100.0], "model__epsilon": [0.05, 0.1, 1.0], "model__gamma": ["scale", 0.1, 0.01]},
            ),
            # Improved SVR with better regularization for small dataset
            (
                "svr_rbf_stable",
                Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf"))]),
                {
                    "model__C": [0.1, 1.0, 10.0],  # Reduced C range
                    "model__epsilon": [0.1, 0.2, 0.5],  # Increased epsilon for robustness
                    "model__gamma": ["scale", "auto", 0.3, 0.1]  # More conservative gamma
                },
            ),
            # Improved GBR with stronger regularization
            (
                "gbr_stable",
                GradientBoostingRegressor(random_state=random_state, validation_fraction=0.2, n_iter_no_change=10),
                {
                    "n_estimators": [100, 200], 
                    "learning_rate": [0.05, 0.08, 0.12], 
                    "max_depth": [2, 3],
                    "subsample": [0.8, 0.9],
                    "min_samples_leaf": [3, 5]
                },
            ),
            # Improved ExtraTrees with stronger regularization
            (
                "extra_trees_stable",
                ExtraTreesRegressor(random_state=random_state, n_estimators=150, max_features='sqrt', bootstrap=True),
                {
                    "max_depth": [4, 6, 8], 
                    "min_samples_leaf": [3, 5, 7],
                    "min_samples_split": [5, 8, 12],
                    "max_features": ['sqrt', 'log2', 0.6]
                },
            ),
            # Random Forest as more stable alternative
            (
                "random_forest_stable",
                RandomForestRegressor(random_state=random_state, n_estimators=150, max_features='sqrt'),
                {
                    "max_depth": [4, 6, 8], 
                    "min_samples_leaf": [3, 5, 7],
                    "min_samples_split": [5, 8, 12],
                    "max_features": ['sqrt', 'log2', 0.6]
                },
            ),
            # Ultra-conservative models for small dataset
            (
                "ridge_conservative",
                Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=random_state))]),
                {"model__alpha": [0.1, 0.5, 1.0, 5.0]},  # Lower alpha values
            ),
            (
                "lasso_conservative",
                Pipeline([("scaler", StandardScaler()), ("model", Lasso(random_state=random_state, max_iter=2000))]),
                {"model__alpha": [0.001, 0.01, 0.1]},  # Very low regularization
            ),
        ]

        best = None
        best_name = None
        best_params = None
        best_test = None

        for name, est, grid in models:
            search = GridSearchCV(
                estimator=est,
                param_grid=grid,
                scoring="neg_mean_absolute_error",
                cv=cv,
                n_jobs=int(n_jobs),
                refit=True,
            )
            search.fit(X_train, y_train, groups=groups_train)
            tuned = search.best_estimator_
            pred = tuned.predict(X_test)
            m = _sk_metrics(y_test, pred)
            cv_mae = float(-search.best_score_)
            print(f"- {name}: test_MAE={m.mae:.4f}  test_RMSE={m.rmse:.4f}  test_R2={m.r2:.4f}  cv_MAE={cv_mae:.4f}")
            if best is None or m.mae < best:
                best = float(m.mae)
                best_name = name
                best_params = getattr(search, "best_params_", {})
                best_test = m

        if baseline_train is not None and np.all(np.isfinite(baseline_train)) and np.all(np.isfinite(baseline)):
            y_train_delta = y_train - baseline_train
            y_test_delta = y_test - baseline

            residual_search = GridSearchCV(
                estimator=Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=random_state))]),
                param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
                scoring="neg_mean_absolute_error",
                cv=cv,
                n_jobs=int(n_jobs),
                refit=True,
            )
            residual_search.fit(X_train, y_train_delta, groups=groups_train)
            tuned = residual_search.best_estimator_
            pred_delta = tuned.predict(X_test)
            pred = baseline + pred_delta
            m = _sk_metrics(y_test, pred)
            cv_mae = float(-residual_search.best_score_)
            print(f"- residual_ridge: test_MAE={m.mae:.4f}  test_RMSE={m.rmse:.4f}  test_R2={m.r2:.4f}  cv_MAE={cv_mae:.4f}")
            if best is None or m.mae < best:
                best = float(m.mae)
                best_name = "residual_ridge"
                best_params = getattr(residual_search, "best_params_", {})
                best_test = m

        if best_name is not None and best_test is not None:
            print(f"Best: {best_name}  test_MAE={best_test.mae:.4f}  params={best_params}")
        print("")


def _find_truth_col(cols: list[str], prefix: str) -> str | None:
    for c in cols:
        if str(c).strip().lower().startswith(prefix.lower()):
            return str(c)
    return None


def _compare_truth_csv(old_path: str, new_path: str, out_csv: str | None) -> None:
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)
    old = old.copy()
    new = new.copy()
    old["FishID"] = old["FishID"].astype(str)
    new["FishID"] = new["FishID"].astype(str)

    cols_old = list(old.columns)
    cols_new = list(new.columns)
    c_len_old = _find_truth_col(cols_old, "Length_truth")
    c_wid_old = _find_truth_col(cols_old, "Width_truth")
    c_area_old = _find_truth_col(cols_old, "Area_truth")
    c_per_old = _find_truth_col(cols_old, "Perimeter_truth")

    c_len_new = _find_truth_col(cols_new, "Length_truth")
    c_wid_new = _find_truth_col(cols_new, "Width_truth")
    c_area_new = _find_truth_col(cols_new, "Area_truth")
    c_per_new = _find_truth_col(cols_new, "Perimeter_truth")

    keep_old = ["FishID"] + [c for c in [c_len_old, c_wid_old, c_area_old, c_per_old] if c]
    keep_new = ["FishID"] + [c for c in [c_len_new, c_wid_new, c_area_new, c_per_new] if c]
    old_s = old[keep_old].copy()
    new_s = new[keep_new].copy()

    merged = old_s.merge(new_s, on="FishID", how="outer", suffixes=("_old", "_new"), indicator=True)

    pairs: list[tuple[str, str, float]] = []
    if c_len_old and c_len_new:
        pairs.append((c_len_old, c_len_new, 0.5))
    if c_wid_old and c_wid_new:
        pairs.append((c_wid_old, c_wid_new, 0.5))
    if c_per_old and c_per_new:
        pairs.append((c_per_old, c_per_new, 0.5))
    if c_area_old and c_area_new:
        pairs.append((c_area_old, c_area_new, 5.0))

    for a, b, _ in pairs:
        merged[a + "_old"] = pd.to_numeric(merged.get(a + "_old"), errors="coerce")
        merged[b + "_new"] = pd.to_numeric(merged.get(b + "_new"), errors="coerce")
        merged[a.replace("_truth", "_diff") + ""] = merged[b + "_new"] - merged[a + "_old"]

    print("Truth comparison:")
    print(f"Old: {old_path}")
    print(f"New: {new_path}")
    print(merged["_merge"].value_counts(dropna=False).to_string())

    flags = np.zeros(len(merged), dtype=bool)
    diff_cols: list[str] = []
    for a, _, thr in pairs:
        dcol = a.replace("_truth", "_diff")
        if dcol in merged.columns:
            diff_cols.append(dcol)
            d = pd.to_numeric(merged[dcol], errors="coerce").abs()
            flags |= d > float(thr)

    show = ["FishID", "_merge"]
    for a, b, _ in pairs:
        show += [a + "_old", b + "_new", a.replace("_truth", "_diff")]
    out = merged.loc[flags, [c for c in show if c in merged.columns]].sort_values("FishID")

    if out.empty:
        print("No large differences found (>|0.5| cm or >|5| cm²).")
    else:
        print("Large differences found:")
        print(out.to_string(index=False))

    if out_csv:
        merged.to_csv(out_csv, index=False)
        print(f"Wrote full comparison CSV: {out_csv}")


def _truth_mae_report(csv_path: str, per_fish: bool) -> None:
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["FishID"] = df["FishID"].astype(str)
    df = df[~df["FishID"].str.contains("+", regex=False)].copy()

    need = {
        "L": ("Length (cm)", "_Truth_Length (cm)"),
        "W": ("Width (cm)", "Width_truth (cm)"),
        "A": ("Area (cm²)", "Area_truth (cm²)"),
        "P": ("Perimeter (cm)", "Perimeter_truth (cm)"),
    }
    for _, (pred_c, truth_c) in need.items():
        if pred_c not in df.columns or truth_c not in df.columns:
            raise RuntimeError(f"Missing required columns: {pred_c} and/or {truth_c}")
        df[pred_c] = pd.to_numeric(df[pred_c], errors="coerce")
        df[truth_c] = pd.to_numeric(df[truth_c], errors="coerce")

    df = df.dropna(subset=[need[k][0] for k in need] + [need[k][1] for k in need]).copy()
    if df.empty:
        raise RuntimeError("No rows with finite measured+truth values.")

    for k, (pred_c, truth_c) in need.items():
        df[f"AE_{k}"] = (df[pred_c] - df[truth_c]).abs()

    if per_fish:
        g = (
            df.groupby("FishID", sort=False)[["AE_L", "AE_W", "AE_A", "AE_P"]]
            .mean()
            .reset_index()
        )
        mae_L = float(g["AE_L"].mean())
        mae_W = float(g["AE_W"].mean())
        mae_A = float(g["AE_A"].mean())
        mae_P = float(g["AE_P"].mean())
        rows = int(len(g))
        mode = "per_fish_mean_AE_then_average"
    else:
        mae_L = float(df["AE_L"].mean())
        mae_W = float(df["AE_W"].mean())
        mae_A = float(df["AE_A"].mean())
        mae_P = float(df["AE_P"].mean())
        rows = int(len(df))
        mode = "per_row_average"

    print("Truth MAE report (measured vs truth)")
    print(f"CSV: {csv_path}")
    print(f"Mode: {mode}")
    print(f"Items: {rows}")
    print(f"MAE Length (cm):    {mae_L:.4f}")
    print(f"MAE Width (cm):     {mae_W:.4f}")
    print(f"MAE Area (cm²):     {mae_A:.4f}")
    print(f"MAE Perimeter (cm): {mae_P:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to fish_frames.csv or combined training CSV")
    parser.add_argument(
        "--mode",
        choices=[
            "weight",
            "weight_tuned",
            "weight_fish_level",
            "weight_fish_level_svr",
            "weight_fish_level_search",
            "truth",
            "truth_tuned",
            "truth_compare",
            "truth_report",
        ],
        default="weight",
    )
    parser.add_argument("--include-groups", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength")
    parser.add_argument("--test-frac", type=float, default=0.2, help="FishID-based test fraction")
    parser.add_argument("--cv-splits", type=int, default=5, help="GroupKFold splits for tuning (FishID groups)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for GridSearchCV")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--bagging", action="store_true", default=False)
    parser.add_argument("--stacking", action="store_true", default=False)
    parser.add_argument("--repeats", type=int, default=1, help="Repeat FishID-holdout evaluation (weight_tuned only)")
    parser.add_argument("--select-by", choices=["cv", "test"], default="cv", help="Model selection policy (weight_tuned only)")
    parser.add_argument("--out-csv", default=None, help="Optional output CSV for per-split results (weight_tuned only)")
    parser.add_argument("--derived-features", action="store_true", default=False, help="Add physics-inspired derived features (weight_fish_level only)")
    parser.add_argument("--export-csv", default=None, help="Export fish-level dataset CSV (weight_fish_level only)")
    parser.add_argument("--drop-height", action="store_true", default=False, help="Drop Height feature (fish-level modes)")
    parser.add_argument("--log-target", action="store_true", default=False, help="Predict log1p(weight), then expm1 (fish-level SVR mode)")
    parser.add_argument("--top-score-k", type=int, default=0, help="Use top-K frames by Score per fish (SVR mode)")
    parser.add_argument("--include-pixels", action="store_true", default=False, help="Include mask pixel counts as features (SVR mode)")
    parser.add_argument("--truth-old", default="truth_values.csv", help="Old truth CSV (truth_compare only)")
    parser.add_argument("--truth-new", default="truth_values_reextracted.csv", help="New truth CSV (truth_compare only)")
    parser.add_argument("--truth-compare-out", default=None, help="Optional output CSV for truth comparison (truth_compare only)")
    parser.add_argument("--truth-report-per-fish", action="store_true", default=False, help="Compute truth MAE per-fish (truth_report only)")
    parser.add_argument("--truth-include-height", action="store_true", default=False, help="Include Height as input (truth modes)")
    args = parser.parse_args()

    if args.mode == "truth_compare":
        _compare_truth_csv(
            old_path=str(args.truth_old),
            new_path=str(args.truth_new),
            out_csv=(str(args.truth_compare_out) if args.truth_compare_out else None),
        )
        return 0
    if args.mode == "truth_report":
        _truth_mae_report(csv_path=str(args.csv), per_fish=bool(args.truth_report_per_fish))
        return 0

    df = pd.read_csv(args.csv)
    df = df.copy()
    df["FishID"] = df["FishID"].astype(str)
    if not bool(args.include_groups):
        df = df[~df["FishID"].str.contains("+", regex=False)].copy()

    if args.mode == "weight":
        features = ["Length (cm)", "Width (cm)", "Height (cm)", "Area (cm²)", "Perimeter (cm)"]
        _eval_weight(
            df=df,
            features=features,
            target="Weight (g)",
            alpha=float(args.alpha),
            test_frac=float(args.test_frac),
            random_state=int(args.random_state),
        )
        return 0
    if args.mode == "weight_tuned":
        features = ["Length (cm)", "Width (cm)", "Height (cm)", "Area (cm²)", "Perimeter (cm)"]
        _eval_weight_tuned_sklearn(
            df=df,
            features=features,
            target="Weight (g)",
            test_frac=float(args.test_frac),
            cv_splits=int(args.cv_splits),
            n_jobs=int(args.n_jobs),
            random_state=int(args.random_state),
            do_bagging=bool(args.bagging),
            do_stacking=bool(args.stacking),
            repeats=int(args.repeats),
            select_by=str(args.select_by),
            out_csv=(str(args.out_csv) if args.out_csv else None),
        )
        return 0

    if args.mode == "weight_fish_level":
        features = ["Length (cm)", "Width (cm)", "Height (cm)", "Area (cm²)", "Perimeter (cm)"]
        _eval_weight_fish_level_tuned(
            df=df,
            base_features=features,
            target="Weight (g)",
            cv_splits=int(args.cv_splits),
            n_jobs=int(args.n_jobs),
            random_state=int(args.random_state),
            derived_features=bool(args.derived_features),
            export_csv=(str(args.export_csv) if args.export_csv else None),
        )
        return 0
    if args.mode == "weight_fish_level_svr":
        features = ["Length (cm)", "Width (cm)", "Height (cm)", "Area (cm²)", "Perimeter (cm)"]
        _eval_weight_fish_level_svr(
            df=df,
            base_features=features,
            target="Weight (g)",
            cv_splits=int(args.cv_splits),
            random_state=int(args.random_state),
            drop_height=bool(args.drop_height),
            log_target=bool(args.log_target),
            top_score_k=int(args.top_score_k),
            include_pixels=bool(args.include_pixels),
        )
        return 0
    if args.mode == "weight_fish_level_search":
        _eval_weight_fish_level_search(
            df=df,
            target="Weight (g)",
            cv_splits=int(args.cv_splits),
            random_state=int(args.random_state),
        )
        return 0

    x_cols = ["Length (cm)", "Width (cm)", "Area (cm²)", "Perimeter (cm)"]
    if bool(args.truth_include_height):
        x_cols = ["Length (cm)", "Width (cm)", "Height (cm)", "Area (cm²)", "Perimeter (cm)"]
    y_cols = ["_Truth_Length (cm)", "Width_truth (cm)", "Area_truth (cm²)", "Perimeter_truth (cm)"]
    missing = [c for c in (x_cols + y_cols) if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {', '.join(missing)}")
    if args.mode == "truth_tuned":
        _eval_truth_tuned_sklearn(
            df=df,
            x_cols=x_cols,
            y_cols=y_cols,
            test_frac=float(args.test_frac),
            cv_splits=int(args.cv_splits),
            n_jobs=int(args.n_jobs),
            random_state=int(args.random_state),
        )
        return 0
    _eval_truth_mapping(df=df, x_cols=x_cols, y_cols=y_cols, alpha=float(args.alpha), test_frac=float(args.test_frac), random_state=int(args.random_state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

