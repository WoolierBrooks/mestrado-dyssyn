#!/usr/bin/env python3
"""Multiclass density comparison (real vs MoSS) in script form.

This script ports the notebook workflow from `02_density_comparison_m.ipynb`
to a runnable Python file and writes artifacts to disk:
- CSV summary
- PNG plots (top/bottom matches)
- HTML reports (interactive when plotly is available)
"""

import ast
import os
import pickle
import warnings
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from mlquantify.model_selection import UPP

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

DATASETS_ROOT = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/multiclass"
MOSS_ROOT = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/moss/multiclass"

BATCH_SIZE = 100
GRID_SIZE = 200
TOPK = 5
MAX_CURVES_ENV = os.getenv("DENSITY_MAX_CURVES")
MAX_CURVES = int(MAX_CURVES_ENV) if MAX_CURVES_ENV not in (None, "") else None  # None = use all curves
SAMPLE_FRAC = float(os.getenv("DENSITY_SAMPLE_FRAC", "0.35"))
MAX_SAMPLES_PER_DATASET = int(os.getenv("DENSITY_MAX_SAMPLES", "12000"))
RF_N_ESTIMATORS = int(os.getenv("DENSITY_RF_TREES", "200"))

OUTPUT_DIR = os.path.join(os.getcwd(), "results", "exp_015_density_multiclass")

grid = np.linspace(0, 1, GRID_SIZE)


try:
    import plotly.express as px
except Exception:  # optional dependency
    px = None


def kde_curve(scores: np.ndarray, kde_grid: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if len(s) < 2 or np.allclose(np.std(s), 0.0):
        return np.zeros_like(kde_grid)
    try:
        return gaussian_kde(s)(kde_grid)
    except Exception:
        return np.zeros_like(kde_grid)


def mean_l1_multiclass(real_kdes: List[np.ndarray], moss_kdes: List[np.ndarray]) -> float:
    return float(np.mean([np.mean(np.abs(r - m)) for r, m in zip(real_kdes, moss_kdes)]))


def load_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    classes = np.unique(y)
    mapper = {c: i for i, c in enumerate(classes)}
    y = np.array([mapper[v] for v in y], dtype=int)
    return X, y


def sample_dataset(
    X: np.ndarray,
    y: np.ndarray,
    frac: float = SAMPLE_FRAC,
    max_samples: int = MAX_SAMPLES_PER_DATASET,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(y)
    if n == 0:
        return X, y

    frac = max(0.0, min(1.0, float(frac)))
    if frac <= 0:
        frac = 1.0

    target = int(np.ceil(n * frac))
    if max_samples > 0:
        target = min(target, max_samples)

    n_classes = len(np.unique(y))
    target = max(target, n_classes * 2)

    if target >= n:
        return X, y

    idx_all = np.arange(n)
    try:
        idx_sample, _ = train_test_split(
            idx_all,
            train_size=target,
            stratify=y,
            random_state=seed,
        )
    except ValueError:
        rng = np.random.default_rng(seed)
        idx_sample = rng.choice(idx_all, size=target, replace=False)

    return X[idx_sample], y[idx_sample]


def random_upp_batch_idx(Xte: np.ndarray, yte: np.ndarray, seed: int = SEED) -> np.ndarray:
    protocol = UPP(batch_size=min(BATCH_SIZE, len(yte)), n_prevalences=10, repeats=3, random_state=seed)
    batches = list(protocol.split(Xte, yte))
    if not batches:
        raise RuntimeError("UPP não gerou batches")
    return batches[np.random.randint(0, len(batches))]


def moss_candidates(n_classes: int) -> List[str]:
    return [
        os.path.join(MOSS_ROOT, f"moss_d_lite_{n_classes}.pkl"),
        os.path.join(MOSS_ROOT, f"moss_d_{n_classes}.pkl"),
        os.path.join(MOSS_ROOT, f"moss_d_lite_{n_classes}.csv"),
        os.path.join(MOSS_ROOT, f"moss_d_{n_classes}.csv"),
    ]


def iter_curves_from_pkl(path: str) -> Iterator[Dict]:
    with open(path, "rb") as f:
        moss = pickle.load(f)
    for (alpha_prev, merge), curves in moss.items():
        for curve_id, scores in enumerate(curves):
            yield {
                "alpha_prev": tuple(float(x) for x in alpha_prev),
                "merge": float(merge),
                "curve_id": int(curve_id),
                "scores": np.asarray(scores),
            }


def iter_curves_from_csv(path: str, n_classes: int) -> Iterator[Dict]:
    df = pd.read_csv(path)
    if not {"prev", "merge", "curve_id"}.issubset(df.columns):
        return iter(())

    score_cols = [f"scores_{c}" for c in range(n_classes)]
    if not all(c in df.columns for c in score_cols):
        return iter(())

    def _gen() -> Iterator[Dict]:
        for _, row in df.iterrows():
            try:
                alpha_prev = tuple(float(x) for x in ast.literal_eval(str(row["prev"])))
                merge = float(row["merge"])
                curve_id = int(row["curve_id"])
                per_class = [np.asarray(ast.literal_eval(str(row[c])), dtype=float) for c in score_cols]
                n = min(len(a) for a in per_class)
                if n < 2:
                    continue
                scores = np.column_stack([a[:n] for a in per_class])
                yield {
                    "alpha_prev": alpha_prev,
                    "merge": merge,
                    "curve_id": curve_id,
                    "scores": scores,
                }
            except Exception:
                continue

    return _gen()


def load_curve_iterator(n_classes: int) -> Tuple[Optional[Iterable[Dict]], Optional[str]]:
    for path in moss_candidates(n_classes):
        if not os.path.exists(path):
            continue
        if path.endswith(".pkl"):
            return iter_curves_from_pkl(path), path
        return iter_curves_from_csv(path, n_classes), path
    return None, None


def best_moss_match(real_kdes: List[np.ndarray], n_classes: int) -> Optional[Dict]:
    it, used = load_curve_iterator(n_classes)
    if it is None:
        return None

    best = {
        "best_l1": np.inf,
        "best_prev": None,
        "best_merge": None,
        "best_curve_id": None,
        "best_kdes": None,
        "moss_file": used,
    }

    seen = 0
    for item in it:
        scores = np.asarray(item["scores"])
        if scores.ndim != 2 or scores.shape[1] != n_classes:
            continue
        if MAX_CURVES is not None and seen >= MAX_CURVES:
            break
        seen += 1

        moss_kdes = [kde_curve(scores[:, c], grid) for c in range(n_classes)]
        l1 = mean_l1_multiclass(real_kdes, moss_kdes)

        if l1 < best["best_l1"]:
            best.update(
                {
                    "best_l1": float(l1),
                    "best_prev": item["alpha_prev"],
                    "best_merge": float(item["merge"]),
                    "best_curve_id": int(item["curve_id"]),
                    "best_kdes": moss_kdes,
                }
            )

    return best if np.isfinite(best["best_l1"]) else None


def save_group_png(group: List[Dict], title: str, out_path: str) -> None:
    n = len(group)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for row_ax, row in zip(axes, group):
        row_ax.plot([], [])  # keep axis alive
        row_ax.axis("off")
        txt = (
            f"{row['dataset']} | classes={row['n_classes']} | L1={row['best_l1']:.4f} | "
            f"prev={row['best_prev']} | merge={row['best_merge']:.4f} | curve={row['best_curve_id']}"
        )
        row_ax.text(0.01, 0.92, txt, fontsize=10, transform=row_ax.transAxes)

        n_classes = max(1, int(row["n_classes"]))
        available_h = 0.75
        gap = 0.01
        inset_h = max(0.02, (available_h - gap * (n_classes - 1)) / n_classes)

        for c in range(n_classes):
            y0 = 0.12 + (n_classes - 1 - c) * (inset_h + gap)
            ax_in = row_ax.inset_axes([0.03, y0, 0.94, inset_h])
            ax_in.plot(grid, row["real_kdes"][c], label=f"Real c={c}", linewidth=1.8)
            ax_in.plot(grid, row["moss_kdes"][c], label=f"MoSS c={c}", linewidth=1.8)
            ax_in.set_ylabel("dens")
            if c == n_classes - 1:
                ax_in.set_xlabel("score")
            ax_in.legend(loc="upper right", fontsize=8)

    fig.suptitle(title, y=0.995)
    plt.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_html_reports(summary_best: pd.DataFrame) -> None:
    html_table = os.path.join(OUTPUT_DIR, "density_multiclass_summary.html")
    summary_best.to_html(html_table, index=False)

    if px is None:
        return

    ok = summary_best[summary_best["status"] == "ok"].copy()
    if ok.empty:
        return

    fig = px.scatter(
        ok,
        x="n_classes",
        y="best_kde_l1",
        hover_name="dataset",
        color="n_classes",
        title="KDE-L1 por dataset (multiclasse)",
    )
    fig.write_html(os.path.join(OUTPUT_DIR, "kde_l1_scatter.html"), include_plotlyjs="cdn")

    agg = (
        ok.groupby("n_classes", as_index=False)
        .agg(
            n_datasets=("dataset", "count"),
            mean_kde_l1=("best_kde_l1", "mean"),
            median_kde_l1=("best_kde_l1", "median"),
        )
        .sort_values("n_classes")
    )
    fig2 = px.bar(
        agg,
        x="n_classes",
        y="mean_kde_l1",
        hover_data=["median_kde_l1", "n_datasets"],
        title="Média de KDE-L1 por número de classes",
    )
    fig2.write_html(os.path.join(OUTPUT_DIR, "kde_l1_by_nclasses.html"), include_plotlyjs="cdn")


def run() -> None:
    if not os.path.isdir(DATASETS_ROOT):
        raise FileNotFoundError(f"Diretório não encontrado: {DATASETS_ROOT}")
    if not os.path.isdir(MOSS_ROOT):
        raise FileNotFoundError(f"Diretório não encontrado: {MOSS_ROOT}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows: List[Dict] = []
    plot_cache: List[Dict] = []

    datasets = sorted([f for f in os.listdir(DATASETS_ROOT) if f.endswith(".csv")])
    if not datasets:
        raise RuntimeError(f"Nenhum CSV encontrado em {DATASETS_ROOT}")

    print(
        "⚙️ Config: "
        f"sample_frac={SAMPLE_FRAC}, max_samples={MAX_SAMPLES_PER_DATASET}, "
        f"rf_trees={RF_N_ESTIMATORS}, max_curves={MAX_CURVES}"
    )

    for ds in tqdm(datasets, desc="Processando datasets", unit="dataset"):
        X, y = load_dataset(os.path.join(DATASETS_ROOT, ds))
        original_size = len(y)
        X, y = sample_dataset(X, y, frac=SAMPLE_FRAC, max_samples=MAX_SAMPLES_PER_DATASET, seed=SEED)
        sampled_size = len(y)
        n_classes = len(np.unique(y))

        try:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=SEED)
        except ValueError:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=SEED)
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

        clf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=SEED, n_jobs=-1)
        clf.fit(Xtr, ytr)

        idx = random_upp_batch_idx(Xte, yte)
        scores_batch = clf.predict_proba(Xte[idx])
        real_kdes = [kde_curve(scores_batch[:, c], grid) for c in range(n_classes)]

        best = best_moss_match(real_kdes, n_classes)
        if best is None:
            rows.append(
                {
                    "dataset": ds,
                    "n_classes": n_classes,
                    "best_moss_prev": None,
                    "best_moss_merge": None,
                    "best_curve_id": None,
                    "best_kde_l1": np.nan,
                    "moss_file": None,
                    "n_samples_original": original_size,
                    "n_samples_used": sampled_size,
                    "status": "moss_not_found_or_invalid",
                }
            )
            continue

        rows.append(
            {
                "dataset": ds,
                "n_classes": n_classes,
                "best_moss_prev": best["best_prev"],
                "best_moss_merge": best["best_merge"],
                "best_curve_id": best["best_curve_id"],
                "best_kde_l1": best["best_l1"],
                "moss_file": best["moss_file"],
                "n_samples_original": original_size,
                "n_samples_used": sampled_size,
                "status": "ok",
            }
        )
        plot_cache.append(
            {
                "dataset": ds,
                "n_classes": n_classes,
                "real_kdes": real_kdes,
                "moss_kdes": best["best_kdes"],
                "best_l1": best["best_l1"],
                "best_prev": best["best_prev"],
                "best_merge": best["best_merge"],
                "best_curve_id": best["best_curve_id"],
            }
        )

    summary_best = pd.DataFrame(rows).sort_values("best_kde_l1", na_position="last")

    out_csv = os.path.join(OUTPUT_DIR, "density_multiclass_summary.csv")
    summary_best.to_csv(out_csv, index=False)

    valid = [r for r in plot_cache if np.isfinite(r["best_l1"])]
    if valid:
        valid_sorted = sorted(valid, key=lambda r: r["best_l1"])
        best_group = valid_sorted[:TOPK]
        worst_group = valid_sorted[-TOPK:]

        save_group_png(best_group, "Mais próximos do MoSS", os.path.join(OUTPUT_DIR, "top_matches.png"))
        save_group_png(worst_group, "Mais distantes do MoSS", os.path.join(OUTPUT_DIR, "worst_matches.png"))

    save_html_reports(summary_best)

    print(f"✅ Resumo CSV: {out_csv}")
    print(f"✅ PNG: {os.path.join(OUTPUT_DIR, 'top_matches.png')}")
    print(f"✅ PNG: {os.path.join(OUTPUT_DIR, 'worst_matches.png')}")
    print(f"✅ HTML: {os.path.join(OUTPUT_DIR, 'density_multiclass_summary.html')}")
    if px is not None:
        print(f"✅ HTML: {os.path.join(OUTPUT_DIR, 'kde_l1_scatter.html')}")
        print(f"✅ HTML: {os.path.join(OUTPUT_DIR, 'kde_l1_by_nclasses.html')}")


if __name__ == "__main__":
    run()
