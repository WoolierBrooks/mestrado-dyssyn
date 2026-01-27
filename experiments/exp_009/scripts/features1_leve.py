import os
import pickle
import time
import numpy as np
import pandas as pd
import warnings

from scipy.stats import skew, kurtosis, entropy, wasserstein_distance, ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
from pymfe.mfe import MFE

from tsfresh.feature_extraction import extract_features as tsfresh_extract
from tsfresh.feature_extraction import EfficientFCParameters

from pycatch22 import catch22_all
from sktime.transformations.panel.rocket import MiniRocket

from tqdm import tqdm  # ← PROGRESS BAR

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

SEED = 42
BATCH_SIZE = 100
N_PREV = 19
REPEATS = 1

MOSS_PKL = "moss_ns100_np15_nm15_nc20.pkl"
MOSS_FOLDER = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/moss/hold"
DATASETS_ROOT = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/binary"

np.random.seed(SEED)

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
        self.prevs = generate_prevalences(N_PREV, REPEATS)

    def split(self, X, y):
        for p in self.prevs:
            yield sample_batch(y, p, BATCH_SIZE), p

# ============================================================
# FEATURE EXTRACTORS
# ============================================================

def baseline_features(scores):
    return np.array([
        np.mean(scores),
        np.var(scores),
        skew(scores),
        kurtosis(scores),
        *np.quantile(scores, [0.1, 0.25, 0.5, 0.75, 0.9]),
        entropy(np.histogram(scores, bins=20, density=True)[0] + 1e-12)
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
        np.mean(scores[scores > q95]),
        np.mean(scores[scores < q05])
    ])

def divergence_features(scores):
    uni = np.linspace(0, 1, len(scores))
    return np.array([
        wasserstein_distance(scores, uni),
        ks_2samp(scores, uni).statistic
    ])

def quantile_derivative_features(scores):
    q = np.quantile(scores, np.linspace(0.1, 0.9, 9))
    return np.diff(q)

def tsfresh_features(scores):
    df = pd.DataFrame({
        "id": 0,
        "time": np.arange(len(scores)),
        "value": scores
    })
    feats = tsfresh_extract(
        df,
        column_id="id",
        column_sort="time",
        column_value="value",
        default_fc_parameters=EfficientFCParameters(),
        disable_progressbar=True
    )
    return feats.values.flatten()

def catch22_features(scores):
    return np.array(catch22_all(scores)["values"])

def pymfe_features(scores):
    X = scores.reshape(-1, 1)
    y = (scores > np.median(scores)).astype(int)

    mfe = MFE(
        groups=["statistical", "general", "info-theory"],
        summary=["mean", "sd"]
    )
    mfe.fit(X, y)
    _, values = mfe.extract()
    return np.array(values, dtype=float)

minirocket = MiniRocket(random_state=SEED)

def sktime_features(scores):
    X = scores.reshape(1, -1)
    return minirocket.transform(X).flatten()

FEATURE_SETS = {
    #"G_sktime": sktime_features,
    #"H_pymfe": pymfe_features,
    "Baseline": baseline_features,
    "A_Shape": shape_features,
    "B_Tail": tail_features,
    "C_Divergence": divergence_features,
    "D_QDeriv": quantile_derivative_features
    #"E_tsfresh": tsfresh_features,
    #"F_catch22": catch22_features,
}

# ============================================================
# LOADERS
# ============================================================

def load_moss_train():
    with open(os.path.join(MOSS_FOLDER, MOSS_PKL), "rb") as f:
        return pickle.load(f)

def load_dataset(path):
    df = pd.read_csv(path)
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values
    if len(np.unique(y)) > 2:
        y = (y == np.max(y)).astype(int)
    return X, y

# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment():
    moss_data = load_moss_train()
    app = APP()

    SERIES_LEN = BATCH_SIZE
    rocket_series = []

    for curves in moss_data.values():
        scores = np.vstack(curves)[:, 0]
        if len(scores) > SERIES_LEN:
            scores = scores[:SERIES_LEN]
        elif len(scores) < SERIES_LEN:
            scores = np.pad(scores, (0, SERIES_LEN - len(scores)), constant_values=scores.mean())
        rocket_series.append(scores)

    minirocket.fit(np.vstack(rocket_series))

    datasets = sorted(f for f in os.listdir(DATASETS_ROOT) if f.endswith(".csv"))

    total_steps = (
        len(FEATURE_SETS)
        * len(datasets)
        * len(app.prevs)
    )

    rows = []

    with tqdm(total=total_steps, desc="⏳ Progresso total", unit="exp") as pbar:

        for name, extractor in FEATURE_SETS.items():
            print(f"\n🔹 Modelo {name}")

            Xtr_feat, ytr_feat = [], []
            for prev, curves in moss_data.items():
                scores = np.vstack(curves)[:, 0]
                Xtr_feat.append(extractor(scores))
                ytr_feat.append(prev[0] if isinstance(prev, tuple) else prev)

            reg = RandomForestRegressor(
                n_estimators=300,
                random_state=SEED,
                n_jobs=-1
            ).fit(np.vstack(Xtr_feat), np.array(ytr_feat))

            for ds in datasets:
                X, y = load_dataset(os.path.join(DATASETS_ROOT, ds))
                X = StandardScaler().fit_transform(X)

                Xtr, Xte, ytr, yte = train_test_split(
                    X, y,
                    test_size=0.5,
                    stratify=y,
                    random_state=SEED
                )

                clf = RandomForestClassifier(
                    n_estimators=300,
                    random_state=SEED,
                    n_jobs=-1
                ).fit(Xtr, ytr)

                for idx, _ in app.split(Xte, yte):
                    scores = clf.predict_proba(Xte[idx])[:, 1]
                    prev_real = np.mean(yte[idx])

                    t0 = time.perf_counter()
                    prev_pred = reg.predict(extractor(scores).reshape(1, -1))[0]
                    t = (time.perf_counter() - t0) / len(idx)

                    rows.append({
                        "modelo": name,
                        "dataset": ds,
                        "prev_real": prev_real,
                        "prev_pred": np.clip(prev_pred, 0, 1),
                        "erro": abs(prev_pred - prev_real),
                        "tempo_por_amostra": t
                    })

                    pbar.update(1)

    pd.DataFrame(rows).to_csv("comparacao_moss_feature_sets.csv", index=False)
    print("\n✅ Finalizado → comparacao_moss_feature_sets.csv")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_experiment()