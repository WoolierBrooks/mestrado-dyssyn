import os
import pickle
import time
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import MiniRocket
from tqdm import tqdm

warnings.filterwarnings("ignore")

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
# CONFIG
# ============================================================

SEED = 42
BATCH_SIZE = 100
N_PREV = 19
REPEATS = 30

SERIES_LEN = BATCH_SIZE

MOSS_PKL = "moss_binario_lite.pkl"
MOSS_FOLDER = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/moss/lite"
DATASETS_ROOT = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/binary"

np.random.seed(SEED)

# ============================================================
# UTILS
# ============================================================

def normalize_series(scores):
    scores = np.asarray(scores, dtype=np.float32)

    if len(scores) > SERIES_LEN:
        scores = scores[:SERIES_LEN]
    elif len(scores) < SERIES_LEN:
        scores = np.pad(
            scores,
            (0, SERIES_LEN - len(scores)),
            constant_values=float(scores.mean())
        )

    return scores

def to_3d(scores):
    """(T,) → (1, 1, T)"""
    return scores.reshape(1, 1, -1)

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
# MINIROCKET
# ============================================================

minirocket = MiniRocket(
    random_state=SEED,
    n_jobs=1   # 🔒 evita paralelismo interno do numba
)

def minirocket_features(scores):
    scores = normalize_series(scores)
    X = to_3d(scores)                 # (1, 1, T)
    Xt = minirocket.transform(X)      # DataFrame
    return Xt.to_numpy().ravel()      # ✅ agora sim

# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment():
    moss_data = load_moss_train()
    app = APP()

    # --------------------------------------------------------
    # FIT MINIROCKET (3D!)
    # --------------------------------------------------------
    rocket_series = []

    for curves in moss_data.values():
        scores = np.vstack(curves)[:, 0]
        scores = normalize_series(scores)
        rocket_series.append(to_3d(scores))

    rocket_series = np.vstack(rocket_series)   # (N, 1, T)
    minirocket.fit(rocket_series)

    # --------------------------------------------------------
    # TRAIN REGRESSOR
    # --------------------------------------------------------
    Xtr_feat, ytr_feat = [], []

    for prev, curves in moss_data.items():
        scores = np.vstack(curves)[:, 0]
        feats = minirocket_features(scores)

        Xtr_feat.append(feats)
        ytr_feat.append(prev[0] if isinstance(prev, tuple) else prev)

    reg = RandomForestRegressor(
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1
    ).fit(np.vstack(Xtr_feat), np.array(ytr_feat))

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------
    datasets = sorted(f for f in os.listdir(DATASETS_ROOT) if f.endswith(".csv"))
    rows = []

    total_steps = len(datasets) * len(app.prevs)

    with tqdm(total=total_steps, desc="🚀 MiniRocket", unit="exp") as pbar:
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
                scores = normalize_series(scores)

                prev_real = np.mean(yte[idx])

                t0 = time.perf_counter()
                prev_pred = reg.predict(
                    minirocket_features(scores).reshape(1, -1)
                )[0]
                prev_pred = to_scalar_prev_pred(prev_pred)
                t = (time.perf_counter() - t0) / len(idx)

                rows.append({
                    "modelo": "MiniRocket",
                    "dataset": ds,
                    "prev_real": prev_real,
                    "prev_pred": np.clip(prev_pred, 0, 1),
                    "erro": abs(prev_pred - prev_real),
                    "tempo_por_amostra": t
                })

                pbar.update(1)

    pd.DataFrame(rows).to_csv("comparacao_moss_minirocket.csv", index=False)
    print("\n✅ Finalizado → comparacao_moss_minirocket.csv")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_experiment()
