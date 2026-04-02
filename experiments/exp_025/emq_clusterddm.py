#!/usr/bin/env python3
"""
exp_025_multiclass
Direct Distribution Matching (PCA, GMM, KMeans) vs EMQ_BCTS
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
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm

# QuaPy
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
OUTPUT_CSV = "results/exp_025_multiclass/results.csv"

np.random.seed(SEED)


# ============================================================
# UTILITIES
# ============================================================

def mae_multiclass(p_true, p_pred):
    return float(np.mean(np.abs(p_true - p_pred)))


def prevalence_vector_from_dict(p_dict, K):
    return np.array([p_dict[k] for k in range(K)])


def project_to_simplex(v):
    """Projection onto probability simplex."""
    v = np.asarray(v)
    n = len(v)

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w


# ============================================================
# DIRECT DISTRIBUTION MATCHING MODELS
# ============================================================

class DDMModel:
    """
    Base class for Direct Distribution Matching.
    """

    def __init__(self, scaler, pca, centroids_matrix):
        self.scaler = scaler
        self.pca = pca
        self.M = centroids_matrix  # shape (d, K)

    def predict_prevalence(self, X):

        Xp = self.pca.transform(self.scaler.transform(X))

        mu_test = Xp.mean(axis=0)  # (d,)

        # solve M p ≈ mu_test
        p_hat, *_ = np.linalg.lstsq(self.M, mu_test, rcond=None)

        p_hat = project_to_simplex(p_hat)

        return p_hat


# ============================================================
# FIT FUNCTIONS
# ============================================================

def fit_ddm_pca(Xtr, ytr, K):

    scaler = StandardScaler().fit(Xtr)
    Xt = scaler.transform(Xtr)

    pca = PCA(n_components=2, random_state=SEED).fit(Xt)
    Xp = pca.transform(Xt)

    centroids = [Xp[ytr == k].mean(axis=0) for k in range(K)]
    M = np.stack(centroids, axis=1)  # (d, K)

    return DDMModel(scaler, pca, M)


def fit_ddm_gmm(Xtr, ytr, K):

    scaler = StandardScaler().fit(Xtr)
    Xt = scaler.transform(Xtr)

    pca = PCA(n_components=2, random_state=SEED).fit(Xt)
    Xp = pca.transform(Xt)

    means = []

    for k in range(K):
        Xk = Xp[ytr == k]
        gmm = GaussianMixture(n_components=1, random_state=SEED).fit(Xk)
        means.append(gmm.means_[0])

    M = np.stack(means, axis=1)

    return DDMModel(scaler, pca, M)


def fit_ddm_kmeans_supervised(Xtr, ytr, K):
    """
    Supervised centroid version (class means).
    """

    scaler = StandardScaler().fit(Xtr)
    Xt = scaler.transform(Xtr)

    pca = PCA(n_components=2, random_state=SEED).fit(Xt)
    Xp = pca.transform(Xt)

    centroids = [Xp[ytr == k].mean(axis=0) for k in range(K)]
    M = np.stack(centroids, axis=1)

    return DDMModel(scaler, pca, M)


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

        X = df.iloc[:, :-1].values

        Xtr, Xte, ytr, yte = train_test_split(
            X, y,
            test_size=0.5,
            stratify=y,
            random_state=SEED
        )

        # --------------------------
        # Fit DDM models
        # --------------------------
        ddm_models = {
            "DDM_PCA": fit_ddm_pca(Xtr, ytr, K),
            "DDM_GMM": fit_ddm_gmm(Xtr, ytr, K),
            "DDM_KMeansSup": fit_ddm_kmeans_supervised(Xtr, ytr, K),
        }

        # --------------------------
        # EMQ + BCTS
        # --------------------------
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

            # -------- DDM --------
            for name, model in ddm_models.items():

                t0 = time.perf_counter()
                p_pred = model.predict_prevalence(Xte[idx_batch])
                elapsed = time.perf_counter() - t0

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