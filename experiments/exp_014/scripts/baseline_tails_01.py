import os
import pickle
import time
import numpy as np
import pandas as pd
import warnings

from scipy.stats import skew, kurtosis, entropy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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

MOSS_PKL = "moss_binario_lite.pkl"
MOSS_FOLDER = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/moss/lite"
DATASETS_ROOT = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/binary"
OUTPUT_CSV = "comparacao_moss_baseline_tails.csv"

np.random.seed(SEED)

# ============================================================
# APP & LOADERS
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
# FEATURE EXTRACTOR (MODIFICADO)
# ============================================================
def baseline_tails_extractor(scores):
    # Estatísticas básicas
    stats = [
        np.mean(scores), 
        np.var(scores), 
        skew(scores), 
        kurtosis(scores),
        entropy(np.histogram(scores, bins=20, density=True)[0] + 1e-12)
    ]
    
    # Lógica QDeriv_Tails_01 integrada
    q_points = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    q_values = np.quantile(scores, q_points)
    q_deriv = np.diff(q_values)
    
    return np.concatenate([stats, q_deriv])

FEATURE_SETS = {
    "Baseline_Tails_01": baseline_tails_extractor
}

# ============================================================
# EXPERIMENT
# ============================================================
def run_experiment():
    print(f"📂 Carregando MOSS...")
    moss_data = load_moss_train()
    app = APP()
    datasets = sorted(f for f in os.listdir(DATASETS_ROOT) if f.endswith(".csv"))

    total_steps = len(FEATURE_SETS) * len(datasets) * len(app.prevs)
    rows = []

    with tqdm(total=total_steps, desc="⏳ Executando Baseline", unit="exp") as pbar:
        for name, extractor in FEATURE_SETS.items():
            
            # --- Treino do Regressor Meta-Learning ---
            Xtr_feat, ytr_feat = [], []
            for prev, curves in moss_data.items():
                scores = np.vstack(curves)[:, 0]
                Xtr_feat.append(extractor(scores))
                ytr_feat.append(prev[0] if isinstance(prev, tuple) else prev)

            reg = RandomForestRegressor(
                n_estimators=300, random_state=SEED, n_jobs=-1
            ).fit(np.vstack(Xtr_feat), np.array(ytr_feat))

            # --- Teste nos Datasets Reais ---
            for ds in datasets:
                X, y = load_dataset(os.path.join(DATASETS_ROOT, ds))
                X = StandardScaler().fit_transform(X)

                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.5, stratify=y, random_state=SEED
                )

                clf = RandomForestClassifier(
                    n_estimators=300, random_state=SEED, n_jobs=-1
                ).fit(Xtr, ytr)

                for idx, _ in app.split(Xte, yte):
                    scores = clf.predict_proba(Xte[idx])[:, 1]
                    prev_real = np.mean(yte[idx])

                    t0 = time.perf_counter()
                    feat_vector = extractor(scores).reshape(1, -1)
                    prev_pred = reg.predict(feat_vector)[0]
                    prev_pred = to_scalar_prev_pred(prev_pred)
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
                
                # Salvamento incremental
                pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Finalizado! Resultados em: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_experiment()
