#!/usr/bin/env python3
"""Cluster-based binary quantification over Quapy UCI datasets.

Uses APP prevalence protocol:
    prevalences = np.linspace(0.05, 0.95, n_prev)
    [[1-p, p] for p in prevalences] * repeats
Pipeline:
1) Load each binary dataset from Quapy.
2) Fit StandardScaler + PCA(2) on training split.
3) Compute one centroid for class 0 and one centroid for class 1 in PCA space.
4) For each target prevalence (5..95 by 5), sample a batch from test (size=100).
5) Classify each sample by nearest centroid (no classifier/regressor).
6) Save per-instance and per-batch CSVs for future visualization/analysis.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import quapy as qp
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

SEED = 42
BATCH_SIZE = 100
N_PREV = 19
REPEATS = 1


QUAPY_BINARY_DATASETS = [
    "balance.1",
    "balance.3",
    "breast-cancer",
    "cmc.1",
    "cmc.2",
    "cmc.3",
    "ctg.1",
    "ctg.2",
    "ctg.3",
    "german",
    "haberman",
    "ionosphere",
    "iris.1",
    "iris.2",
    "iris.3",
    "mammographic",
    "pageblocks.5",
    "semeion",
    "sonar",
    "spambase",
    "spectf",
    "tictactoe",
    "transfusion",
    "wdbc",
    "wine.1",
    "wine.2",
    "wine.3",
    "wine-q-red",
    "wine-q-white",
    "yeast",
]


@dataclass
class PCAClusterModel:
    scaler: StandardScaler
    pca: PCA
    centroid_0: np.ndarray
    centroid_1: np.ndarray


def generate_prevalences(n_prev: int, repeats: int) -> List[List[float]]:
    prevalences = np.linspace(0.05, 0.95, n_prev)
    return [[1 - float(p), float(p)] for p in prevalences] * repeats


def load_binary_quapy_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = qp.datasets.fetch_UCIBinaryDataset(name, verbose=False)
    Xtr = np.asarray(ds.training.X)
    ytr_raw = np.asarray(ds.training.y)
    Xte = np.asarray(ds.test.X)
    yte_raw = np.asarray(ds.test.y)

    classes = np.unique(np.concatenate([ytr_raw, yte_raw]))
    if len(classes) != 2:
        raise ValueError(f"Dataset {name} is not binary after loading")

    pos_class = classes.max()
    ytr = (ytr_raw == pos_class).astype(int)
    yte = (yte_raw == pos_class).astype(int)
    return Xtr, ytr, Xte, yte


def fit_pca_cluster_model(Xtr: np.ndarray, ytr: np.ndarray, seed: int) -> PCAClusterModel:
    scaler = StandardScaler().fit(Xtr)
    Xtr_scaled = scaler.transform(Xtr)

    pca = PCA(n_components=2, random_state=seed)
    Xtr_pca = pca.fit_transform(Xtr_scaled)

    if len(np.unique(ytr)) < 2:
        raise ValueError("Training split has <2 classes; cannot compute centroids")

    centroid_0 = Xtr_pca[ytr == 0].mean(axis=0)
    centroid_1 = Xtr_pca[ytr == 1].mean(axis=0)
    return PCAClusterModel(scaler=scaler, pca=pca, centroid_0=centroid_0, centroid_1=centroid_1)


def sample_batch_indices(y: np.ndarray, prevalence_pos: float, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Test split needs both classes for APP sampling")

    n_pos = int(batch_size * prevalence_pos)
    n_neg = batch_size - n_pos

    idx = np.concatenate(
        [
            rng.choice(pos, size=n_pos, replace=True),
            rng.choice(neg, size=n_neg, replace=True),
        ]
    )
    rng.shuffle(idx)
    return idx


def predict_by_centroid(X_pca: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d0 = np.linalg.norm(X_pca - c0.reshape(1, -1), axis=1)
    d1 = np.linalg.norm(X_pca - c1.reshape(1, -1), axis=1)
    yhat = (d1 < d0).astype(int)
    return yhat, d0, d1


def run_experiment(
    datasets: Iterable[str],
    batch_size: int,
    app_prevalences: List[List[float]],
    n_prev: int,
    seed: int,
    out_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    rows_points: List[Dict] = []
    rows_batches: List[Dict] = []

    for ds_name in tqdm(list(datasets), desc="Datasets", unit="dataset"):
        try:
            Xtr, ytr, Xte, yte = load_binary_quapy_dataset(ds_name)
        except Exception as e:
            rows_batches.append({"dataset": ds_name, "status": "load_error", "error": str(e)})
            continue

        try:
            model = fit_pca_cluster_model(Xtr, ytr, seed=seed)
        except Exception:
            Xtr2, _, ytr2, _ = train_test_split(Xtr, ytr, test_size=0.2, random_state=seed, stratify=ytr)
            model = fit_pca_cluster_model(Xtr2, ytr2, seed=seed)

        for batch_idx, prev_pair in enumerate(app_prevalences, start=1):
            p_neg, p_pos = float(prev_pair[0]), float(prev_pair[1])
            repeat_id = (batch_idx - 1) // n_prev

            idx = sample_batch_indices(yte, prevalence_pos=p_pos, batch_size=batch_size, rng=rng)
            Xb = Xte[idx]
            yb = yte[idx]

            Xb_pca = model.pca.transform(model.scaler.transform(Xb))
            yhat, d0, d1 = predict_by_centroid(Xb_pca, model.centroid_0, model.centroid_1)

            prev_true = float(yb.mean())
            prev_pred = float(yhat.mean())
            abs_err = float(abs(prev_true - prev_pred))
            acc = float(accuracy_score(yb, yhat))

            rows_batches.append(
                {
                    "dataset": ds_name,
                    "repeat": repeat_id,
                    "batch_id": batch_idx,
                    "batch_size": batch_size,
                    "target_prev_neg": p_neg,
                    "target_prev_pos": p_pos,
                    "true_prev_pos": prev_true,
                    "pred_prev_pos": prev_pred,
                    "abs_error": abs_err,
                    "instance_accuracy": acc,
                    "centroid0_x": float(model.centroid_0[0]),
                    "centroid0_y": float(model.centroid_0[1]),
                    "centroid1_x": float(model.centroid_1[0]),
                    "centroid1_y": float(model.centroid_1[1]),
                    "status": "ok",
                    "error": None,
                }
            )

            for i in range(len(yb)):
                rows_points.append(
                    {
                        "dataset": ds_name,
                        "repeat": repeat_id,
                        "batch_id": batch_idx,
                        "sample_idx_in_batch": i,
                        "target_prev_neg": p_neg,
                        "target_prev_pos": p_pos,
                        "x_pca": float(Xb_pca[i, 0]),
                        "y_pca": float(Xb_pca[i, 1]),
                        "true_label": int(yb[i]),
                        "pred_label": int(yhat[i]),
                        "dist_to_centroid0": float(d0[i]),
                        "dist_to_centroid1": float(d1[i]),
                        "centroid0_x": float(model.centroid_0[0]),
                        "centroid0_y": float(model.centroid_0[1]),
                        "centroid1_x": float(model.centroid_1[0]),
                        "centroid1_y": float(model.centroid_1[1]),
                    }
                )

    points_df = pd.DataFrame(rows_points)
    batches_df = pd.DataFrame(rows_batches)

    points_path = os.path.join(out_dir, "clusterbase_points.csv")
    batches_path = os.path.join(out_dir, "clusterbase_batches.csv")
    points_df.to_csv(points_path, index=False)
    batches_df.to_csv(batches_path, index=False)

    ok_batches = batches_df[batches_df.get("status", "ok") == "ok"].copy()
    if not ok_batches.empty:
        summary = (
            ok_batches.groupby("dataset", as_index=False)
            .agg(
                n_batches=("batch_id", "count"),
                mean_abs_error=("abs_error", "mean"),
                median_abs_error=("abs_error", "median"),
                mean_instance_acc=("instance_accuracy", "mean"),
            )
            .sort_values("mean_abs_error")
        )
        summary.to_csv(os.path.join(out_dir, "clusterbase_summary_by_dataset.csv"), index=False)

    return points_df, batches_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="exp_022 - Cluster-based quantification with PCA centroids")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n-prev", type=int, default=N_PREV)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default=os.path.join(os.getcwd(), "results", "exp_022_clusterbase"))
    parser.add_argument("--datasets", nargs="*", default=QUAPY_BINARY_DATASETS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_prevalences = generate_prevalences(args.n_prev, args.repeats)

    points_df, batches_df = run_experiment(
        datasets=args.datasets,
        batch_size=args.batch_size,
        app_prevalences=app_prevalences,
        n_prev=args.n_prev,
        seed=args.seed,
        out_dir=args.output_dir,
    )

    print("✅ Experimento finalizado")
    print(f"✅ Pontos salvos: {os.path.join(args.output_dir, 'clusterbase_points.csv')}")
    print(f"✅ Batches salvos: {os.path.join(args.output_dir, 'clusterbase_batches.csv')}")
    print(f"   N pontos: {len(points_df)}")
    print(f"   N batches: {len(batches_df)}")


if __name__ == "__main__":
    main()
