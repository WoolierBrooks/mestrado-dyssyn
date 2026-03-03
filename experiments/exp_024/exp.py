#!/usr/bin/env python3
"""
exp_023_multiclass
Clusters (PCA, GMM, KMeans) vs EMQ_BCTS
UPP protocol (multiclass)
"""

import os
import numpy as np
import pandas as pd
import warnings
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm

# QuaPy
import quapy as qp
from quapy.method.aggregative import EMQ

# mlquantify
from mlquantify.model_selection import UPP
from mlquantify.utils import get_prev_from_labels

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
SEED = 42
BATCH_SIZE = 100
REPEATS = 30
N_PREVALENCES = 10

DATASETS_ROOT = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/multiclass"
OUTPUT_CSV = "results/exp_023_multiclass/results.csv"

np.random.seed(SEED)


# ============================================================
# UTILITIES
# ============================================================

def mae_multiclass(p_true, p_pred):
    return float(np.mean(np.abs(p_true - p_pred)))


def prevalence_vector_from_dict(p_dict, K):
    return np.array([p_dict[k] for k in range(K)])


# ============================================================
# CLUSTER MODELS
# ============================================================

class ClusterPCAModel:

    def __init__(self, scaler, pca, centroids):
        self.scaler = scaler
        self.pca = pca
        self.centroids = centroids

    def predict(self, X):
        Xp = self.pca.transform(self.scaler.transform(X))
        dists = np.stack(
            [np.linalg.norm(Xp - c, axis=1) for c in self.centroids],
            axis=1
        )
        return np.argmin(dists, axis=1)


class ClusterGMMModel:

    def __init__(self, scaler, pca, gmms, priors):
        self.scaler = scaler
        self.pca = pca
        self.gmms = gmms
        self.priors = priors

    def predict(self, X):
        Xp = self.pca.transform(self.scaler.transform(X))

        scores = []
        for gmm, prior in zip(self.gmms, self.priors):
            s = gmm.score_samples(Xp) + np.log(prior + 1e-12)
            scores.append(s)

        scores = np.stack(scores, axis=1)
        return np.argmax(scores, axis=1)


class ClusterKMeansModel:

    def __init__(self, scaler, pca, kmeans, cluster_to_label):
        self.scaler = scaler
        self.pca = pca
        self.kmeans = kmeans
        self.cluster_to_label = cluster_to_label

    def predict(self, X):
        Xp = self.pca.transform(self.scaler.transform(X))
        clusters = self.kmeans.predict(Xp)
        return np.array([self.cluster_to_label[c] for c in clusters])


# ============================================================
# FIT FUNCTIONS
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
        if len(yc) == 0:
            cluster_to_label[c] = 0
        else:
            cluster_to_label[c] = np.bincount(yc).argmax()

    return ClusterKMeansModel(scaler, pca, kmeans, cluster_to_label)


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment():

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    datasets = sorted(
        f for f in os.listdir(DATASETS_ROOT)
        if f.endswith(".csv")
    )

    rows = []

    for ds_name in datasets:

        print(f"\n📂 Dataset: {ds_name}")

        df = pd.read_csv(os.path.join(DATASETS_ROOT, ds_name))
        y = df.iloc[:, -1].values
        if y.min() > 0:
            y -= y.min()

        K = len(np.unique(y))
        print(f"   🔢 Classes: {K}")

        X = StandardScaler().fit_transform(df.iloc[:, :-1].values)

        Xtr, Xte, ytr, yte = train_test_split(
            X, y,
            test_size=0.5,
            stratify=y,
            random_state=SEED
        )

        # --------------------------
        # Fit Models
        # --------------------------
        cluster_models = {
            "cluster_pca": fit_cluster_pca(Xtr, ytr, K),
            "cluster_gmm": fit_cluster_gmm(Xtr, ytr, K),
            "cluster_kmeans": fit_cluster_kmeans(Xtr, ytr, K),
        }

        clf = RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=SEED
        )

        emq = EMQ(
            clf,
            calib="bcts",
            val_split=0.2,
            exact_train_prev=True,
            on_calib_error="backup"
        )

        emq.fit(Xtr, ytr)

        # --------------------------
        # UPP Protocol
        # --------------------------
        protocol = UPP(
            batch_size=BATCH_SIZE,
            n_prevalences=N_PREVALENCES,
            repeats=REPEATS,
            random_state=SEED
        )

        for idx_batch in tqdm(
            protocol.split(Xte, yte),
            total=protocol.get_n_combinations(),
            desc="UPP"
        ):

            p_real_dict = get_prev_from_labels(
                yte[idx_batch],
                classes=np.arange(K)
            )

            p_real = prevalence_vector_from_dict(p_real_dict, K)

            # -------- Clusters --------
            for name, model in cluster_models.items():

                t0 = time.perf_counter()
                yhat = model.predict(Xte[idx_batch])
                elapsed = time.perf_counter() - t0

                p_pred = get_prev_from_labels(
                    yhat,
                    classes=np.arange(K)
                )

                p_pred = prevalence_vector_from_dict(p_pred, K)

                rows.append({
                    "dataset": ds_name,
                    "model": name,
                    "n_classes": K,
                    "abs_error": mae_multiclass(p_real, p_pred),
                    "time_per_sample": elapsed / len(idx_batch),
                    "status": "ok"
                })

            # -------- EMQ_BCTS --------
            t0 = time.perf_counter()
            p_emq = emq.predict(Xte[idx_batch])
            elapsed = time.perf_counter() - t0

            rows.append({
                "dataset": ds_name,
                "model": "EMQ_BCTS",
                "n_classes": K,
                "abs_error": mae_multiclass(p_real, p_emq),
                "time_per_sample": elapsed / len(idx_batch),
                "status": "ok"
            })

        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Experimento concluído! Resultados salvos em {OUTPUT_CSV}")


if __name__ == "__main__":
    run_experiment()
