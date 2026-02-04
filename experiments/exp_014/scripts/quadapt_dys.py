import time
import numpy as np
import pandas as pd
import warnings

import quapy as qp

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

OUTPUT_CSV = "comparacao_quadapt_results.csv"

QUAPY_BINARY_DATASETS = [
    "acute.a",
    "acute.b",
    "balance.1",
    "balance.2",
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
def load_quapy_dataset(name):
    data = qp.datasets.fetch_UCIBinaryDataset(name, verbose=False)
    return data.training.X, data.training.y, data.test.X, data.test.y

# ============================================================
# EXPERIMENT
# ============================================================
def run_experiment():
    app = APP()
    datasets = QUAPY_BINARY_DATASETS

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
            Xtr, ytr, Xte, yte = load_quapy_dataset(ds)
            scaler = StandardScaler().fit(Xtr)
            Xtr = scaler.transform(Xtr)
            Xte = scaler.transform(Xte)

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
