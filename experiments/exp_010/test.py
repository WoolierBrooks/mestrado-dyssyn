import os
import pickle
import numpy as np
import pandas as pd
import warnings

from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Importes oficiais do mlquantify
from mlquantify.likelihood import EMQ
from mlquantify.model_selection import UPP
from mlquantify.utils import get_prev_from_labels

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
SEED = 42
BATCH_SIZE = 100
MOSS_DIR = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/moss/moss_m"
DATASETS_ROOT = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/multiclass"
OUTPUT_CSV = "cross_nclasses_experiment.csv"

np.random.seed(SEED)

def baseline_features(scores_matrix, target_n_classes):
    feats = []
    for c in range(scores_matrix.shape[1]):
        s = scores_matrix[:, c]
        feats.extend([np.mean(s), np.var(s), skew(s), kurtosis(s)])
    
    total_expected = 4 * target_n_classes
    if len(feats) < total_expected:
        feats.extend([0] * (total_expected - len(feats)))
    return np.array(feats[:total_expected])

# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment():
    datasets = sorted(f for f in os.listdir(DATASETS_ROOT) if f.endswith(".csv"))
    moss_files = [3, 4, 7]
    regressores_treinados = {}

    # --- FASE 1: Treinar os modelos MoSS ---
    # --- FASE 1: Treinar os modelos MoSS ---
# --- FASE 1: Treinar os modelos MoSS ---
    # --- FASE 1: Treinar os modelos MoSS ---
# --- FASE 1: Treinar os modelos MoSS ---
# --- FASE 1: Treinar os modelos MoSS ---
    print("🧠 Treinando regressores MoSS...")
    for n_m in moss_files:
        path = os.path.join(MOSS_DIR, f"moss_m_lite_{n_m}.pkl")
        
        if not os.path.exists(path):
            print(f"⚠️ Arquivo {path} não encontrado. Pulando MoSS_{n_m}")
            continue
            
        with open(path, 'rb') as f:
            # O seu arquivo é um dict: {(prev_tuple, merge): [lista_de_scores]}
            synthetic_distributions = pickle.load(f)
            
        X_train_moss = []
        y_train_moss = []

        print(f"  📊 Extraindo features do arquivo MoSS_{n_m}...")
        for (alpha_prev, merge_val), curves in synthetic_distributions.items():
            for scores_matrix in curves:
                # Extrai as 4 estatísticas por classe (mean, var, skew, kurt)
                # O scores_matrix gerado pelo seu código tem shape (n_samples, n_classes)
                feats = baseline_features(scores_matrix, n_m)
                
                X_train_moss.append(feats)
                y_train_moss.append(list(alpha_prev))

        X_m = np.array(X_train_moss)
        y_m = np.array(y_train_moss)

        if len(X_m) == 0:
            print(f"❌ Erro: Não foi possível extrair dados de {path}")
            continue

        # Treinando o regressor meta-modelo
        model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED)
        model.fit(X_m, y_m)
        regressores_treinados[n_m] = model
        print(f"✅ MoSS_{n_m} treinado com sucesso! (Amostras: {len(X_m)})")


    # --- FASE 2: Teste Cruzado com Protocolo UPP ---
    rows = []
    for ds_name in datasets:
        print(f"\n📂 Dataset: {ds_name}")
        df = pd.read_csv(os.path.join(DATASETS_ROOT, ds_name))
        y = df.iloc[:, -1].values
        if y.min() > 0: y -= y.min()
        n_classes_real = len(np.unique(y))
        
        X = StandardScaler().fit_transform(df.iloc[:, :-1].values)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=SEED)
        
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
        emq = EMQ(learner=clf).fit(Xtr, ytr)

        # Usando n_prevalences=10 para não explodir em datasets com muitas classes
        protocol = UPP(batch_size=BATCH_SIZE, n_prevalences=10, repeats=30, random_state=SEED)

        for idx_batch in tqdm(protocol.split(Xte, yte), desc="Protocolo UPP", total=protocol.get_n_combinations()):
            
            # CORREÇÃO: parâmetro 'classes' em vez de 'n_classes'
            # No loop do protocolo UPP:
            p_real_raw = get_prev_from_labels(yte[idx_batch], classes=np.arange(n_classes_real))

            # CONVERSÃO: Se for dicionário, transforma em array ordenado pelas chaves
            if isinstance(p_real_raw, dict):
                p_real = np.array([p_real_raw[k] for k in sorted(p_real_raw.keys())])
            else:
                p_real = np.array(p_real_raw)
            scores_batch = clf.predict_proba(Xte[idx_batch])

            # 3. Testar MoSS (Verifica se a chave existe antes)
            for n_m in moss_files:
                if n_m not in regressores_treinados:
                    continue
                    
                f_vec = baseline_features(scores_batch, n_m).reshape(1, -1)
                p_pred = regressores_treinados[n_m].predict(f_vec)[0]
                
                # Ajuste de dimensões
                p_pred_trimmed = p_pred[:n_classes_real]
                if len(p_pred_trimmed) < n_classes_real:
                    p_pred_trimmed = np.pad(p_pred_trimmed, (0, n_classes_real - len(p_pred_trimmed)))
                
                p_pred_trimmed /= (p_pred_trimmed.sum() + 1e-12)

                rows.append({
                    "dataset": ds_name,
                    "modelo": f"MoSS_{n_m}",
                    "n_classes_original": n_classes_real,
                    "erro": np.mean(np.abs(p_pred_trimmed - p_real))
                })

            # 4. Testar EMQ
            try:
                p_emq_raw = emq.predict(Xte[idx_batch])
                if isinstance(p_emq_raw, dict):
                    p_emq = np.array([p_emq_raw[k] for k in sorted(p_emq_raw.keys())])
                else:
                    p_emq = np.array(p_emq_raw).flatten()

                rows.append({
                    "dataset": ds_name,
                    "modelo": "EMQ",
                    "n_classes_original": n_classes_real,
                    "erro": np.mean(np.abs(p_emq - p_real))
                })
            except:
                continue

        # Salva o progresso por dataset
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Experimento concluído! Resultados em: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_experiment()