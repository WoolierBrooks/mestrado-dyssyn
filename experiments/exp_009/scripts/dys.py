import os
import time
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Importando o DyS da mlquantify
from mlquantify.mixture import DyS

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
SEED = 42
BATCH_SIZE = 100
N_PREV = 19
REPEATS = 1

# Nota: O DyS treina diretamente nos datasets reais, 
# então o MOSS pode ser usado como validação ou ignorado.
DATASETS_ROOT = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/binary"
OUTPUT_CSV = "comparacao_dys_results.csv"

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
    
    # Garantir que não tentamos amostrar mais do que existe sem replace
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

    # Vamos testar o DyS com diferentes medidas de distância
    measures = ['hellinger', 'topsoe']
    total_steps = len(measures) * len(datasets) * len(app.prevs)
    rows = []

    with tqdm(total=total_steps, desc="⏳ Executando DyS", unit="exp") as pbar:
        for measure_name in measures:
            for ds in datasets:
                X, y = load_dataset(os.path.join(DATASETS_ROOT, ds))
                X = StandardScaler().fit_transform(X)

                # Divide em treino (para o classificador base do DyS) e teste
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.5, stratify=y, random_state=SEED
                )

                # Inicializa o DyS com um RandomForest como learner base
                # O DyS vai dar o fit no RF e usar os scores para criar os histogramas de treino
                q = DyS(
                    learner=RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
                    measure=measure_name,
                    bins_size=[10, 20, 30] # Tamanhos de histograma para testar
                )
                
                # O DyS treina o classificador e armazena as distribuições de score positivas/negativas
                q.fit(Xtr, ytr)

                for idx, p in app.split(Xte, yte):
                    X_batch = Xte[idx]
                    prev_real = np.mean(yte[idx])

                    t0 = time.perf_counter()
                    
                    # O DyS prediz a prevalência baseando-se na similaridade de distribuição
                    # Retorna um array [prob_classe_0, prob_classe_1]
                    prev_pred_array = q.predict(X_batch)
                    prev_pred = prev_pred_array[1] # Prevalência da classe positiva
                    
                    t = (time.perf_counter() - t0) / len(idx)

                    rows.append({
                        "modelo": f"DyS_{measure_name}",
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