#!/usr/bin/env python3
"""exp_023 - unified comparison: DDM-style cluster methods vs DyS(Topsoe).

Models compared (binary quantification):
- cluster_ddm_centroid: nearest centroid in RF probability space
- cluster_ddm_gmm: class-conditional GMM on RF probabilities
- cluster_ddm_kmeans: KMeans on RF probabilities + cluster->label mapping
- dys_topsoe: DyS quantifier with Topsoe distance

All models are evaluated on exactly the same APP batches and random states per dataset.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import quapy as qp
from mlquantify.mixture import DyS
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

SEED = 42
BATCH_SIZE = 100
N_PREV = 19
REPEATS = 30

QUAPY_BINARY_DATASETS = [
    "balance.1","balance.3","breast-cancer","cmc.1","cmc.2","cmc.3",
    "ctg.1","ctg.2","ctg.3","german","haberman","ionosphere",
    "iris.1","iris.2","iris.3","mammographic","pageblocks.5",
    "semeion","sonar","spambase","spectf","tictactoe",
    "transfusion","wdbc","wine.1","wine.2","wine.3",
    "wine-q-red","wine-q-white","yeast",
]

# =========================================================
# Batch spec
# =========================================================

@dataclass
class BatchSpec:
    batch_id: int
    repeat: int
    target_prev_neg: float
    target_prev_pos: float
    indices: np.ndarray

# =========================================================
# DDM-STYLE CLUSTER MODELS (probability space)
# =========================================================

@dataclass
class ClusterDDMCentroid:
    rf: RandomForestClassifier
    centroid_0: float
    centroid_1: float

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        probs = self.rf.predict_proba(X)[:, 1]
        d0 = np.abs(probs - self.centroid_0)
        d1 = np.abs(probs - self.centroid_1)
        return (d1 < d0).astype(int)


@dataclass
class ClusterDDMGMM:
    rf: RandomForestClassifier
    gmm0: GaussianMixture
    gmm1: GaussianMixture
    prior0: float
    prior1: float

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        probs = self.rf.predict_proba(X)[:, 1].reshape(-1, 1)

        s0 = self.gmm0.score_samples(probs) + np.log(max(self.prior0, 1e-12))
        s1 = self.gmm1.score_samples(probs) + np.log(max(self.prior1, 1e-12))

        return (s1 > s0).astype(int)


@dataclass
class ClusterDDMKMeans:
    rf: RandomForestClassifier
    kmeans: KMeans
    cluster_to_label: Dict[int, int]

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        probs = self.rf.predict_proba(X)[:, 1].reshape(-1, 1)
        clusters = self.kmeans.predict(probs)
        return np.array([self.cluster_to_label[int(c)] for c in clusters], dtype=int)

# =========================================================
# Utility
# =========================================================

def to_pos_prev(pred) -> float:
    # Caso DyS retorne dict {0: p0, 1: p1}
    if isinstance(pred, dict):
        # tenta pegar classe positiva = 1
        if 1 in pred:
            return float(pred[1])
        # caso as chaves sejam string
        if "1" in pred:
            return float(pred["1"])
        # fallback: pega maior chave
        return float(pred[max(pred.keys())])

    # Caso retorne array-like
    arr = np.asarray(pred)

    if arr.ndim == 0:
        return float(arr)

    arr = arr.ravel()

    if arr.size == 1:
        return float(arr[0])

    # assume formato [p0, p1]
    return float(arr[1])

def generate_prevalences(n_prev: int, repeats: int):
    prevalences = np.linspace(0.05, 0.95, n_prev)
    return [[1 - float(p), float(p)] for p in prevalences] * repeats


def sample_batch_indices(y, prevalence_pos, batch_size, rng):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    n_pos = int(batch_size * prevalence_pos)
    n_neg = batch_size - n_pos

    idx = np.concatenate([
        rng.choice(pos, size=n_pos, replace=True),
        rng.choice(neg, size=n_neg, replace=True),
    ])
    rng.shuffle(idx)
    return idx


def build_batches(yte, app_prevalences, n_prev, batch_size, seed):
    rng = np.random.default_rng(seed)
    batches = []
    for batch_id, prev_pair in enumerate(app_prevalences, start=1):
        p_neg, p_pos = float(prev_pair[0]), float(prev_pair[1])
        idx = sample_batch_indices(yte, p_pos, batch_size, rng)
        repeat = (batch_id - 1) // n_prev
        batches.append(
            BatchSpec(batch_id, repeat, p_neg, p_pos, idx)
        )
    return batches


def load_binary_quapy_dataset(name):
    ds = qp.datasets.fetch_UCIBinaryDataset(name, verbose=False)
    Xtr = np.asarray(ds.training.X)
    ytr_raw = np.asarray(ds.training.y)
    Xte = np.asarray(ds.test.X)
    yte_raw = np.asarray(ds.test.y)

    classes = np.unique(np.concatenate([ytr_raw, yte_raw]))
    pos_class = classes.max()

    ytr = (ytr_raw == pos_class).astype(int)
    yte = (yte_raw == pos_class).astype(int)
    return Xtr, ytr, Xte, yte

# =========================================================
# Fit DDM cluster models
# =========================================================

def fit_cluster_ddm_centroid(Xtr, ytr, seed):
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    rf.fit(Xtr, ytr)

    probs = rf.predict_proba(Xtr)[:, 1]
    c0 = probs[ytr == 0].mean()
    c1 = probs[ytr == 1].mean()

    return ClusterDDMCentroid(rf, float(c0), float(c1))


def fit_cluster_ddm_gmm(Xtr, ytr, seed):
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    rf.fit(Xtr, ytr)

    probs = rf.predict_proba(Xtr)[:, 1].reshape(-1, 1)
    X0 = probs[ytr == 0]
    X1 = probs[ytr == 1]

    gmm0 = GaussianMixture(n_components=1, random_state=seed).fit(X0)
    gmm1 = GaussianMixture(n_components=1, random_state=seed).fit(X1)

    prior1 = float(np.mean(ytr == 1))
    prior0 = 1.0 - prior1

    return ClusterDDMGMM(rf, gmm0, gmm1, prior0, prior1)


def fit_cluster_ddm_kmeans(Xtr, ytr, seed):
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    rf.fit(Xtr, ytr)

    probs = rf.predict_proba(Xtr)[:, 1].reshape(-1, 1)

    kmeans = KMeans(n_clusters=2, random_state=seed, n_init=10)
    clusters = kmeans.fit_predict(probs)

    cluster_to_label = {}
    for c in [0, 1]:
        yc = ytr[clusters == c]
        cluster_to_label[c] = int(np.mean(yc) >= 0.5) if len(yc) > 0 else 0

    return ClusterDDMKMeans(rf, kmeans, cluster_to_label)

# =========================================================
# Evaluation
# =========================================================

def evaluate_label_model(model_name, model, Xte, yte, batches):
    rows = []
    for b in batches:
        Xb = Xte[b.indices]
        yb = yte[b.indices]
        prev_true = float(np.mean(yb))

        t0 = time.perf_counter()
        yhat = model.predict_labels(Xb)
        elapsed = time.perf_counter() - t0

        prev_pred = float(np.mean(yhat))

        rows.append({
            "model": model_name,
            "dataset": None,
            "repeat": b.repeat,
            "batch_id": b.batch_id,
            "batch_size": len(yb),
            "target_prev_pos": b.target_prev_pos,
            "true_prev_pos": prev_true,
            "pred_prev_pos": prev_pred,
            "abs_error": abs(prev_pred - prev_true),
            "time_per_sample": elapsed / len(yb),
            "status": "ok",
        })
    return rows


def evaluate_dys_topsoe(Xtr, ytr, Xte, yte, batches, seed):
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    q_model = DyS(learner=rf, measure="topsoe")
    q_model.fit(Xtr, ytr)

    rows = []
    for b in batches:
        Xb = Xte[b.indices]
        yb = yte[b.indices]
        prev_true = float(np.mean(yb))

        t0 = time.perf_counter()
        prev_pred = to_pos_prev(q_model.predict(Xb))
        elapsed = time.perf_counter() - t0

        rows.append({
            "model": "dys_topsoe",
            "dataset": None,
            "repeat": b.repeat,
            "batch_id": b.batch_id,
            "batch_size": len(yb),
            "target_prev_pos": b.target_prev_pos,
            "true_prev_pos": prev_true,
            "pred_prev_pos": prev_pred,
            "abs_error": abs(prev_pred - prev_true),
            "time_per_sample": elapsed / len(yb),
            "status": "ok",
        })
    return rows

# =========================================================
# Main experiment loop
# =========================================================

def run_experiment(datasets, batch_size, n_prev, repeats, seed, out_csv):

    app_prevalences = generate_prevalences(n_prev, repeats)
    rows = []

    model_names = [
        "cluster_ddm_centroid",
        "cluster_ddm_gmm",
        "cluster_ddm_kmeans",
        "dys_topsoe",
    ]

    total_steps = len(list(datasets)) * len(app_prevalences) * len(model_names)

    with tqdm(total=total_steps, desc="exp_023", unit="batch") as pbar:

        for ds_i, ds_name in enumerate(datasets):

            Xtr, ytr, Xte, yte = load_binary_quapy_dataset(ds_name)

            ds_seed = seed + ds_i * 10000

            batches = build_batches(
                yte, app_prevalences, n_prev, batch_size, ds_seed
            )

            # DDM Centroid
            model = fit_cluster_ddm_centroid(Xtr, ytr, seed)
            ds_rows = evaluate_label_model("cluster_ddm_centroid", model, Xte, yte, batches)
            for r in ds_rows: r["dataset"] = ds_name
            rows.extend(ds_rows)
            pbar.update(len(app_prevalences))

            # DDM GMM
            model = fit_cluster_ddm_gmm(Xtr, ytr, seed)
            ds_rows = evaluate_label_model("cluster_ddm_gmm", model, Xte, yte, batches)
            for r in ds_rows: r["dataset"] = ds_name
            rows.extend(ds_rows)
            pbar.update(len(app_prevalences))

            # DDM KMeans
            model = fit_cluster_ddm_kmeans(Xtr, ytr, seed)
            ds_rows = evaluate_label_model("cluster_ddm_kmeans", model, Xte, yte, batches)
            for r in ds_rows: r["dataset"] = ds_name
            rows.extend(ds_rows)
            pbar.update(len(app_prevalences))

            # DyS
            scaler = StandardScaler().fit(Xtr)
            Xtr_s = scaler.transform(Xtr)
            Xte_s = scaler.transform(Xte)

            ds_rows = evaluate_dys_topsoe(Xtr_s, ytr, Xte_s, yte, batches, seed)
            for r in ds_rows: r["dataset"] = ds_name
            rows.extend(ds_rows)
            pbar.update(len(app_prevalences))

            pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    return out_df

# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n-prev", type=int, default=N_PREV)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--datasets", nargs="*", default=QUAPY_BINARY_DATASETS)
    parser.add_argument("--output-csv", default="results/exp_023_results.csv")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    res = run_experiment(
        args.datasets,
        args.batch_size,
        args.n_prev,
        args.repeats,
        args.seed,
        args.output_csv,
    )

    print("✅ Results saved:", args.output_csv)
    print("Rows:", len(res))

if __name__ == "__main__":
    main()