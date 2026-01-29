import os
import time
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Importando mlquantify
from mlquantify.mixture import DyS
from mlquantify.meta import QuaDapt

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
SEED = 42
BATCH_SIZE = 100
N_PREV = 19
REPEATS = 30

DATASETS_ROOT = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/binary"
OUTPUT_CSV = "comparacao_quadapt_results.csv"

np.random.seed(SEED)

# ============================================================
# APP (Protocolo de amostragem)
# ============================================================
def generate_prevalences(n_prev, repeats):
    prevalences = np.linspace(0.05, 0.95, n_prev)
    return [[1 - p, p] for p in prevalences] * repeats

def sample_batch(y, prevalence, batch_size):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_pos = int(batch_size * prevalence[1])
    n_neg = batch_size - n_pos
    
    idx = np.concatenate([
        np.random.choice(pos, n_pos, replace=True),
        np.random.choice(neg, n_neg, replace=True)
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
# LOADER
# ============================================================
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
    app = APP()
    datasets = sorted(f for f in os.listdir(DATASETS_ROOT) if f.endswith(".csv"))

    # Configuração dos modelos: DyS puro vs QuaDapt(DyS)
    # O QuaDapt precisa de um quantificador base.
    base_rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
    
    # Lista de modelos para comparar
    model_configs = {
        "DyS_Topsoe": DyS(learner=base_rf, measure='topsoe'),
        "QuaDapt_DyS": QuaDapt(
            quantifier=DyS(learner=base_rf, measure='topsoe'),
            measure='topsoe',
            merging_factors=np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        )
    }

    total_steps = len(model_configs) * len(datasets) * len(app.prevs)
    rows = []

    with tqdm(total=total_steps, desc="⏳ Executando Experimento", unit="exp") as pbar:
        for ds in datasets:
            X, y = load_dataset(os.path.join(DATASETS_ROOT, ds))
            X = StandardScaler().fit_transform(X)

            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.5, stratify=y, random_state=SEED
            )

            # Treinar e testar cada configuração
            for name, q_model in model_configs.items():
                # O Fit do QuaDapt treina o classificador e prepara a simulação MoSS
                q_model.fit(Xtr, ytr)

                for idx, p in app.split(Xte, yte):
                    X_batch = Xte[idx]
                    prev_real = np.mean(yte[idx])

                    t0 = time.perf_counter()
                    
                    # Predição de prevalência
                    prev_pred_array = q_model.predict(X_batch)
                    
                    # QuaDapt e DyS retornam vetores de prevalência [neg, pos]
                    prev_pred = prev_pred_array[1]
                    
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
                
            # Salvamento incremental por dataset para segurança
            pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Finalizado! Resultados em: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_experiment()