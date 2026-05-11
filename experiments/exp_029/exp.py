import os
import pickle
import time
import numpy as np
import pandas as pd
import warnings

import quapy as qp

from scipy.stats import skew, kurtosis, entropy
from scipy.stats import wasserstein_distance, ks_2samp

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_predict

from scipy.signal import find_peaks

from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

SEED = 42

BATCH_SIZE = 100
N_PREV = 19
REPEATS = 30

MOSS_PKL = "moss_binario_lite.pkl"
MOSS_FOLDER = "."

OUTPUT_CSV = "moss_calibrated_results.csv"

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

np.random.seed(SEED)

# ============================================================
# HELPERS
# ============================================================

def to_scalar_prev_pred(prev_pred):
    arr = np.asarray(prev_pred)

    if arr.ndim == 0:
        return float(arr)

    if arr.size == 1:
        return float(arr.item())

    if arr.size == 2:
        return float(arr.ravel()[1])

    return float(arr.ravel()[0])

# ============================================================
# APP
# ============================================================

def generate_prevalences(n_prev, repeats):
    prevalences = np.linspace(0.05, 0.95, n_prev)
    return [[1 - p, p] for p in prevalences] * repeats

def sample_batch(y, prevalence, batch_size):

    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    n_pos = int(batch_size * prevalence[1])

    idx = np.concatenate([
        np.random.choice(pos, n_pos, replace=True),
        np.random.choice(neg, batch_size - n_pos, replace=True)
    ])

    np.random.shuffle(idx)

    return idx

class APP:

    def __init__(self):
        self.prevs = generate_prevalences(
            N_PREV,
            REPEATS
        )

    def split(self, X, y):

        for p in self.prevs:
            yield sample_batch(y, p, BATCH_SIZE), p

# ============================================================
# FEATURE EXTRACTORS
# ============================================================

def baseline_features(scores):

    hist = np.histogram(
        scores,
        bins=20,
        density=True
    )[0]

    return np.array([
        np.mean(scores),
        np.var(scores),
        skew(scores),
        kurtosis(scores),

        *np.quantile(
            scores,
            [0.1, 0.25, 0.5, 0.75, 0.9]
        ),

        entropy(hist + 1e-12)
    ])

def shape_features(scores):

    peaks, _ = find_peaks(scores)

    return np.array([
        np.max(scores),
        np.min(scores),
        np.ptp(scores),
        len(peaks),
        np.mean(np.diff(np.sort(scores)))
    ])

def tail_features(scores):

    q95 = np.quantile(scores, 0.95)
    q05 = np.quantile(scores, 0.05)

    return np.array([
        np.quantile(scores, 0.99),
        np.quantile(scores, 0.01),

        np.mean(scores[scores > q95])
        if np.any(scores > q95) else 0,

        np.mean(scores[scores < q05])
        if np.any(scores < q05) else 0
    ])

def divergence_features(scores):

    uni = np.linspace(0, 1, len(scores))

    return np.array([
        wasserstein_distance(scores, uni),
        ks_2samp(scores, uni).statistic
    ])

def quantile_derivative_features(scores):

    q = np.quantile(
        scores,
        np.linspace(0.1, 0.9, 9)
    )

    return np.diff(q)

# ============================================================
# 🔥 ONE STEP EM FEATURES
# ============================================================

def one_step_em(scores, train_prev):

    eps = 1e-12

    train_prev = np.array([
        1 - train_prev,
        train_prev
    ])

    probs = np.vstack([
        1 - scores,
        scores
    ]).T

    weighted = probs * train_prev

    denom = weighted.sum(axis=1, keepdims=True) + eps

    posterior = weighted / denom

    p1 = posterior.mean(axis=0)

    return p1[1]

def em_features(scores, train_prev):

    eps = 1e-12

    mean_score = np.mean(scores)

    p1 = one_step_em(scores, train_prev)

    return np.array([

        p1,

        p1 - train_prev,

        p1 - mean_score,

        np.abs(p1 - train_prev),

        np.abs(p1 - mean_score),

        entropy([
            p1 + eps,
            1 - p1 + eps
        ])
    ])

# ============================================================
# FEATURE SETS
# ============================================================

FEATURE_SETS = {
    "Baseline": baseline_features,
    "A_Shape": shape_features,
    "B_Tail": tail_features,
    "C_Divergence": divergence_features,
    "D_QDeriv": quantile_derivative_features,
}

# ============================================================
# LOADERS
# ============================================================

def load_moss_train():

    with open("moss_binario_lite.pkl", "rb") as f:
        return pickle.load(f)

def load_quapy_dataset(name):

    data = qp.datasets.fetch_UCIBinaryDataset(
        name,
        verbose=False
    )

    return (
        data.training.X,
        data.training.y,
        data.test.X,
        data.test.y
    )

# ============================================================
# FEATURE COMBINER
# ============================================================

def extract_all_features(
    scores,
    extractor,
    train_prev
):

    base_feat = extractor(scores)

    em_feat = em_features(
        scores,
        train_prev
    )

    return np.concatenate([
        base_feat,
        em_feat
    ])

# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment():

    moss_data = load_moss_train()

    app = APP()

    rows = []

    total_steps = (
        len(FEATURE_SETS)
        * len(QUAPY_BINARY_DATASETS)
        * len(app.prevs)
    )

    with tqdm(
        total=total_steps,
        desc="⏳ Total",
        unit="exp"
    ) as pbar:

        for feat_name, extractor in FEATURE_SETS.items():

            print(f"\n🔹 Feature set: {feat_name}")

            # ====================================================
            # MOSS SYNTHETIC PRETRAIN
            # ====================================================

            X_moss = []
            y_moss = []

            for prev, curves in moss_data.items():

                scores = np.vstack(curves)[:, 0]

                if isinstance(prev, tuple):

                    if isinstance(prev[0], (list, tuple, np.ndarray)):
                        train_prev = float(prev[0][1])
                    else:
                        train_prev = float(prev[1])

                else:

                    train_prev = float(prev)

                feat = extract_all_features(
                    scores,
                    extractor,
                    train_prev
                )

                X_moss.append(feat)
                y_moss.append(train_prev)

            X_moss = np.vstack(X_moss)
            y_moss = np.array(y_moss)

            # ====================================================
            # DATASETS
            # ====================================================

            for ds in QUAPY_BINARY_DATASETS:

                print(f"   📂 {ds}")

                Xtr, ytr, Xte, yte = load_quapy_dataset(ds)

                scaler = StandardScaler()

                Xtr = scaler.fit_transform(Xtr)
                Xte = scaler.transform(Xte)

                # ================================================
                # BASE CLASSIFIER
                # ================================================

                clf = RandomForestClassifier(
                    n_estimators=300,
                    random_state=SEED,
                    n_jobs=-1
                )

                clf.fit(Xtr, ytr)

                # ================================================
                # 🔥 OOF TRAIN SCORES
                # ================================================

                print("      🔄 Generating OOF probabilities...")

                oof_scores = cross_val_predict(
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=SEED,
                        n_jobs=-1
                    ),
                    Xtr,
                    ytr,
                    cv=5,
                    method="predict_proba",
                    n_jobs=-1
                )[:, 1]

                train_prev_global = np.mean(ytr)

                # ================================================
                # REAL CALIBRATION SET
                # ================================================

                X_real = []
                y_real = []

                for idx, _ in app.split(Xtr, ytr):

                    scores_batch = oof_scores[idx]

                    prev_real = np.mean(ytr[idx])

                    feat = extract_all_features(
                        scores_batch,
                        extractor,
                        train_prev_global
                    )

                    X_real.append(feat)
                    y_real.append(prev_real)

                X_real = np.vstack(X_real)
                y_real = np.array(y_real)

                # ================================================
                # 🔥 HYBRID TRAINING
                # ================================================

                X_final = np.vstack([
                    X_moss,
                    X_real
                ])

                y_final = np.concatenate([
                    y_moss,
                    y_real
                ])

                # Peso maior para dados reais
                weights = np.concatenate([
                    np.ones(len(X_moss)),
                    np.ones(len(X_real)) * 5
                ])

                reg = RandomForestRegressor(
                    n_estimators=500,
                    random_state=SEED,
                    n_jobs=-1,
                    min_samples_leaf=2
                )

                reg.fit(
                    X_final,
                    y_final,
                    sample_weight=weights
                )

                # ================================================
                # TEST
                # ================================================

                for idx, _ in app.split(Xte, yte):

                    scores = clf.predict_proba(
                        Xte[idx]
                    )[:, 1]

                    prev_real = np.mean(
                        yte[idx]
                    )

                    feat = extract_all_features(
                        scores,
                        extractor,
                        train_prev_global
                    )

                    t0 = time.perf_counter()

                    prev_pred = reg.predict(
                        feat.reshape(1, -1)
                    )[0]

                    prev_pred = to_scalar_prev_pred(
                        prev_pred
                    )

                    prev_pred = np.clip(
                        prev_pred,
                        0,
                        1
                    )

                    t = (
                        time.perf_counter() - t0
                    ) / len(idx)

                    rows.append({
                        "modelo": f"{feat_name}_CALIBRATED",
                        "dataset": ds,
                        "prev_real": prev_real,
                        "prev_pred": prev_pred,
                        "erro": abs(
                            prev_pred - prev_real
                        ),
                        "tempo_por_amostra": t
                    })

                    pbar.update(1)

    pd.DataFrame(rows).to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        f"\n✅ Finalizado → {OUTPUT_CSV}"
    )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_experiment()