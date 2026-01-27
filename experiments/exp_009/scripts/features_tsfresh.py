import os
import pickle
import time
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from tsfresh.feature_extraction import extract_features as tsfresh_extract
from tsfresh.feature_extraction import EfficientFCParameters

from tqdm import tqdm

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
# TSFRESH FEATURES
# ============================================================

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

FEATURE_SETS = {
    "E_tsfresh": tsfresh_features
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

    datasets = sorted(
        f for f in os.listdir(DATASETS_ROOT) if f.endswith(".csv")
    )

    total_steps = (
        len(FEATURE_SETS)
        * len(datasets)
        * len(app.prevs)
    )

    rows = []

    with tqdm(total=total_steps, desc="⏳ Progresso total", unit="exp") as pbar:

        for name, extractor in FEATURE_SETS.items():
            print(f"\n🔹 Modelo {name}")

            # ------------------------
            # Train regressor (MOSS)
            # ------------------------
            Xtr_feat, ytr_feat = [], []

            for prev, curves in moss_data.items():
                scores = np.vstack(curves)[:, 0]
                Xtr_feat.append(extractor(scores))
                ytr_feat.append(prev[0] if isinstance(prev, tuple) else prev)

            reg = RandomForestRegressor(
                n_estimators=300,
                random_state=SEED,
                n_jobs=-1
            ).fit(
                np.vstack(Xtr_feat),
                np.array(ytr_feat)
            )

            # ------------------------
            # Evaluate on datasets
            # ------------------------
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
                    prev_pred = reg.predict(
                        extractor(scores).reshape(1, -1)
                    )[0]
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

    pd.DataFrame(rows).to_csv(
        "comparacao_moss_tsfresh.csv",
        index=False
    )

    print("\n✅ Finalizado → comparacao_moss_tsfresh.csv")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_experiment()