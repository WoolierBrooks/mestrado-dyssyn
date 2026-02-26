#!/usr/bin/env python3
"""Visual comparison: clusterbase vs exp_014 methods (baseline/quadapt).

Builds a per-dataset boxplot dashboard (Plotly) where models are locally ordered
(best median error -> worst) in each dataset, following the style requested by user.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot clusterbase vs exp_014 results by dataset")
    parser.add_argument(
        "--clusterbase-csv",
        default="results/exp_022_clusterbase/clusterbase_batches.csv",
        help="Path to clusterbase_batches.csv",
    )
    parser.add_argument(
        "--quadapt-csv",
        default="experiments/exp_014/results/quadapt.csv",
        help="Path to exp_014 quadapt.csv",
    )
    parser.add_argument(
        "--baseline-csv",
        default="experiments/exp_014/results/baseline.csv",
        help="Path to exp_014 baseline.csv",
    )
    parser.add_argument(
        "--output-html",
        default="results/exp_022_clusterbase/clusterbase_vs_exp014.html",
        help="Where to save the interactive HTML",
    )
    parser.add_argument("--height-per-dataset", type=int, default=320)
    parser.add_argument("--title", default="Comparação local por dataset: Clusterbase vs exp_014")
    return parser.parse_args()


def _find_col(df: pd.DataFrame, candidates: Iterable[str], csv_name: str) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise ValueError(f"Could not find any of {list(candidates)} in {csv_name}. Columns: {list(df.columns)}")


def load_clusterbase(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    dataset_col = _find_col(df, ["dataset"], path.name)
    error_col = _find_col(df, ["abs_error", "mae", "error", "erro"], path.name)

    out = pd.DataFrame(
        {
            "dataset": df[dataset_col].astype(str),
            "modelo": "clusterbase",
            "erro": pd.to_numeric(df[error_col], errors="coerce"),
        }
    ).dropna(subset=["erro"])
    return out


def load_generic_method(path: Path, fallback_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    dataset_col = _find_col(df, ["dataset", "dataset_name"], path.name)
    error_col = _find_col(df, ["erro", "error", "mae", "abs_error"], path.name)

    # If no explicit model column exists, use file name as model id.
    if any(c.lower() in {"modelo", "model", "method"} for c in df.columns):
        model_col = _find_col(df, ["modelo", "model", "method"], path.name)
        modelo = df[model_col].astype(str)
    else:
        modelo = pd.Series([fallback_name] * len(df), index=df.index)

    out = pd.DataFrame(
        {
            "dataset": df[dataset_col].astype(str),
            "modelo": modelo,
            "erro": pd.to_numeric(df[error_col], errors="coerce"),
        }
    ).dropna(subset=["erro"])
    return out


def main() -> None:
    args = parse_args()

    clusterbase_path = Path(args.clusterbase_csv)
    quadapt_path = Path(args.quadapt_csv)
    baseline_path = Path(args.baseline_csv)

    frames: list[pd.DataFrame] = []

    if clusterbase_path.exists():
        frames.append(load_clusterbase(clusterbase_path))
    else:
        print(f"[WARN] Missing clusterbase CSV: {clusterbase_path}")

    if quadapt_path.exists():
        frames.append(load_generic_method(quadapt_path, fallback_name="quadapt"))
    else:
        print(f"[WARN] Missing quadapt CSV: {quadapt_path}")

    if baseline_path.exists():
        frames.append(load_generic_method(baseline_path, fallback_name="baseline"))
    else:
        print(f"[WARN] Missing baseline CSV: {baseline_path}")

    if not frames:
        raise FileNotFoundError("None of the input CSV files were found.")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["dataset", "modelo", "erro"])

    datasets = sorted(df["dataset"].unique())
    n_datasets = len(datasets)

    fig = make_subplots(
        rows=n_datasets,
        cols=1,
        subplot_titles=[f"<b>{ds}</b>" for ds in datasets],
        vertical_spacing=max(0.001, min(0.04, 0.5 / max(n_datasets, 1))),
    )

    for i, ds in enumerate(datasets, start=1):
        df_ds = df[df["dataset"] == ds].copy()
        ordem_local = df_ds.groupby("modelo")["erro"].median().sort_values().index.tolist()

        for modelo in ordem_local:
            df_mod = df_ds[df_ds["modelo"] == modelo]
            fig.add_trace(
                go.Box(
                    y=df_mod["erro"],
                    name=modelo,
                    boxpoints="outliers",
                    legendgroup=modelo,
                    showlegend=(i == 1),
                ),
                row=i,
                col=1,
            )

    fig.update_layout(
        height=max(600, n_datasets * args.height_per_dataset),
        template="plotly_white",
        title_text=args.title,
        margin=dict(t=100, b=50, l=50, r=50),
        showlegend=True,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(title_text="Erro")

    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html)
    print(f"Saved: {output_html}")


if __name__ == "__main__":
    main()
