#!/usr/bin/env python3
"""exp_027_multiclass - unified comparison with cluster models (multiclass, UPP protocol)

Models compared:
- cluster_pca
- cluster_gmm
- cluster_kmeans
- EMQ_BCTS (baseline)

Protocol:
- UPP (Unnatural Prevalence Protocol)
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from tqdm.auto import tqdm

# QuaPy / mlquantify
import quapy as qp
from quapy.method.aggregative import EMQ
from mlquantify.model_selection import UPP
from mlquantify.utils import get_prev_from_labels

import numpy as np

# FIX numpy >= 2.0 compatibility with QuaPy
if not hasattr(np, "in1d"):
    np.in1d = np.isin

# ============================================================
# CONFIG
# ============================================================

SEED = 42
BATCH_SIZE = 100
REPEATS = 30
N_PREV = 10

QUAPY_MULTICLASS_DATASETS = [
    'dry-bean', 'wine-quality', 'academic-success', 'digits', 'letter', 'abalone', 'obesity', 'nursery', 'yeast', 'hand_digits', 'satellite', 'shuttle', 'cmc', 'isolet', 'waveform-v1', 'molecular', 'poker_hand', 'connect-4', 'mhr', 'chess', 'page_block', 'phishing', 'image_seg', 'hcv',
]

# ============================================================
# UTILITIES
# ============================================================

def mae_multiclass(p_true, p_pred):
    return float(np.mean(np.abs(p_true - p_pred)))


def prevalence_vector(p_dict, K):
    return np.array([p_dict[k] for k in range(K)])


# ============================================================
# MODELS
# ============================================================

@dataclass
class ClusterPCAModel:
    scaler: StandardScaler
    pca: PCA
    centroids: List[np.ndarray]

    def predict_labels(self, X):
        Xp = self.pca.transform(self.scaler.transform(X))
        dists = np.stack(
            [np.linalg.norm(Xp - c, axis=1) for c in self.centroids],
            axis=1
        )
        return np.argmin(dists, axis=1)


@dataclass
class ClusterGMMModel:
    scaler: StandardScaler
    pca: PCA
    gmms: List[GaussianMixture]
    priors: List[float]

    def predict_labels(self, X):
        Xp = self.pca.transform(self.scaler.transform(X))

        scores = []
        for gmm, prior in zip(self.gmms, self.priors):
            s = gmm.score_samples(Xp) + np.log(prior + 1e-12)
            scores.append(s)

        scores = np.stack(scores, axis=1)
        return np.argmax(scores, axis=1)


@dataclass
class ClusterKMeansModel:
    scaler: StandardScaler
    pca: PCA
    kmeans: KMeans
    cluster_to_label: Dict[int, int]

    def predict_labels(self, X):
        Xp = self.pca.transform(self.scaler.transform(X))
        clusters = self.kmeans.predict(Xp)
        return np.array([self.cluster_to_label[c] for c in clusters])


# ============================================================
# FIT
# ============================================================

def fit_cluster_pca(Xtr, ytr, K):
    scaler = StandardScaler().fit(Xtr)
    Xt = scaler.transform(Xtr)
    pca = PCA(n_components=2, random_state=SEED).fit(Xt)
    Xp = pca.transform(Xt)

    centroids = [Xp[ytr == k].mean(axis=0) for k in range(K)]
    return ClusterPCAModel(scaler, pca, centroids)


def fit_cluster_gmm(Xtr, ytr, K):
    scaler = StandardScaler().fit(Xtr)
    Xt = scaler.transform(Xtr)
    pca = PCA(n_components=2, random_state=SEED).fit(Xt)
    Xp = pca.transform(Xt)

    gmms = []
    priors = []

    for k in range(K):
        Xk = Xp[ytr == k]
        gmm = GaussianMixture(n_components=1, random_state=SEED).fit(Xk)
        gmms.append(gmm)
        priors.append(np.mean(ytr == k))

    return ClusterGMMModel(scaler, pca, gmms, priors)


def fit_cluster_kmeans(Xtr, ytr, K):
    scaler = StandardScaler().fit(Xtr)
    Xt = scaler.transform(Xtr)
    pca = PCA(n_components=2, random_state=SEED).fit(Xt)
    Xp = pca.transform(Xt)

    kmeans = KMeans(n_clusters=K, random_state=SEED, n_init=10)
    clusters = kmeans.fit_predict(Xp)

    cluster_to_label = {}
    for c in range(K):
        yc = ytr[clusters == c]
        cluster_to_label[c] = np.bincount(yc).argmax() if len(yc) > 0 else 0

    return ClusterKMeansModel(scaler, pca, kmeans, cluster_to_label)


# ============================================================
# DATASET
# ============================================================

def load_multiclass_dataset(name: str):
    ds = qp.datasets.fetch_UCIMulticlassDataset(name, verbose=False)

    Xtr = np.asarray(ds.training.X)
    ytr = np.asarray(ds.training.y)
    Xte = np.asarray(ds.test.X)
    yte = np.asarray(ds.test.y)

    classes = np.unique(np.concatenate([ytr, yte]))
    mapping = {c: i for i, c in enumerate(classes)}

    ytr = np.array([mapping[y] for y in ytr])
    yte = np.array([mapping[y] for y in yte])

    return Xtr, ytr, Xte, yte, len(classes)


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment(
    datasets: Iterable[str],
    batch_size: int,
    repeats: int,
    n_prev: int,
    seed: int,
    out_csv: str,
):

    rows = []

    for ds_i, ds_name in enumerate(datasets):

        print(f"\n📂 Dataset: {ds_name}")

        Xtr, ytr, Xte, yte, K = load_multiclass_dataset(ds_name)

        # Models
        cluster_models = {
            "cluster_pca": fit_cluster_pca(Xtr, ytr, K),
            "cluster_gmm": fit_cluster_gmm(Xtr, ytr, K),
            "cluster_kmeans": fit_cluster_kmeans(Xtr, ytr, K),
        }

        clf = RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=seed
        )

        emq = EMQ(
            clf,
            calib="bcts",
            val_split=0.2,
            exact_train_prev=True,
            on_calib_error="backup"
        )

        emq.fit(Xtr, ytr)

        # UPP protocol
        protocol = UPP(
            batch_size=batch_size,
            n_prevalences=n_prev,
            repeats=repeats,
            random_state=seed
        )

        for idx_batch in tqdm(
            protocol.split(Xte, yte),
            total=protocol.get_n_combinations(),
            desc=ds_name
        ):

            yb = yte[idx_batch]

            p_real = prevalence_vector(
                get_prev_from_labels(yb, classes=np.arange(K)), K
            )

            # Clusters
            for name, model in cluster_models.items():

                t0 = time.perf_counter()
                yhat = model.predict_labels(Xte[idx_batch])
                elapsed = time.perf_counter() - t0

                p_pred = prevalence_vector(
                    get_prev_from_labels(yhat, classes=np.arange(K)), K
                )

                rows.append({
                    "dataset": ds_name,
                    "model": name,
                    "n_classes": K,
                    "abs_error": mae_multiclass(p_real, p_pred),
                    "time_per_sample": elapsed / len(idx_batch),
                    "status": "ok",
                })

            # EMQ_BCTS
            t0 = time.perf_counter()
            p_emq = emq.predict(Xte[idx_batch])
            elapsed = time.perf_counter() - t0

            rows.append({
                "dataset": ds_name,
                "model": "EMQ_BCTS",
                "n_classes": K,
                "abs_error": mae_multiclass(p_real, p_emq),
                "time_per_sample": elapsed / len(idx_batch),
                "status": "ok",
            })

        pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby("model", as_index=False)
        .agg(mean_abs_error=("abs_error", "mean"))
        .sort_values("mean_abs_error")
    )

    summary_path = out_csv.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\n✅ Results: {out_csv}")
    print(f"✅ Summary: {summary_path}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--n-prev", type=int, default=N_PREV)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--datasets", nargs="*", default=QUAPY_MULTICLASS_DATASETS)
    parser.add_argument("--output-csv", default="exp_027_multiclass/results.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    run_experiment(
        datasets=args.datasets,
        batch_size=args.batch_size,
        repeats=args.repeats,
        n_prev=args.n_prev,
        seed=args.seed,
        out_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()