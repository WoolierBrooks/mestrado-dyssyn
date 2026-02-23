import os
import pickle
import numpy as np
import pandas as pd
import warnings

from scipy.stats import skew, kurtosis
from scipy.special import rel_entr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ==============================
# Quantificação
# ==============================
from quapy.method.aggregative import EMQ
from mlquantify.model_selection import UPP
from mlquantify.utils import get_prev_from_labels

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
SEED = 42
BATCH_SIZE = 100

MOSS_DIR = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/moss/multiclass"
DATASETS_ROOT = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/multiclass"
OUTPUT_CSV = "moss_em1_dynamic.csv"

np.random.seed(SEED)

# ============================================================
# 1️⃣ ONE STEP EM
# ============================================================
def one_step_em(scores_matrix, train_prev):
    eps = 1e-12
    train_prev = np.array(train_prev, dtype=float)
    train_prev /= (train_prev.sum() + eps)

    weighted = scores_matrix * train_prev
    denom = weighted.sum(axis=1, keepdims=True) + eps
    posterior = weighted / denom

    p1 = posterior.mean(axis=0)
    return p1


# ============================================================
# 2️⃣ FEATURES
# ============================================================
def moss_features(scores_matrix, train_prev=None):
    eps = 1e-12
    feats = []
    n_classes = scores_matrix.shape[1]

    # ---------------------------------------
    # Estatísticas por classe
    # ---------------------------------------
    for c in range(n_classes):
        s = scores_matrix[:, c]
        feats.extend([
            np.mean(s),
            np.var(s),
            skew(s),
            kurtosis(s),
            -np.mean(s * np.log(s + eps))
        ])

    # ---------------------------------------
    # Estatísticas globais
    # ---------------------------------------
    mean_probs = np.mean(scores_matrix, axis=0)

    entropy_mean = -np.sum(mean_probs * np.log(mean_probs + eps))
    entropy_instances = -np.mean(
        np.sum(scores_matrix * np.log(scores_matrix + eps), axis=1)
    )
    entropy_gap = entropy_mean - entropy_instances

    feats.extend([
        entropy_mean,
        entropy_instances,
        entropy_gap
    ])

    # ---------------------------------------
    # 🔥 1-STEP EM DYNAMIC FEATURES
    # ---------------------------------------
    if train_prev is not None:

        train_prev = np.array(train_prev, dtype=float)
        train_prev /= (train_prev.sum() + eps)

        p1 = one_step_em(scores_matrix, train_prev)

        # movement relative to train prior
        feats.extend((p1 - train_prev).tolist())

        # movement relative to raw mean
        feats.extend((p1 - mean_probs).tolist())

        # KL divergences
        kl_p1_train = np.sum(rel_entr(p1 + eps, train_prev + eps))
        kl_p1_mean = np.sum(rel_entr(p1 + eps, mean_probs + eps))

        feats.append(kl_p1_train)
        feats.append(kl_p1_mean)

        # entropy of p1
        entropy_p1 = -np.sum(p1 * np.log(p1 + eps))
        feats.append(entropy_p1)

    return np.array(feats)


# ============================================================
# 3️⃣ EXPERIMENT
# ============================================================
def run_experiment():

    datasets = sorted(
        f for f in os.listdir(DATASETS_ROOT) if f.endswith(".csv")
    )

    regressores_treinados = {}
    rows = []

    for ds_name in datasets:

        print(f"\n📂 Dataset: {ds_name}")

        df = pd.read_csv(os.path.join(DATASETS_ROOT, ds_name))

        y = df.iloc[:, -1].values
        if y.min() > 0:
            y -= y.min()

        n_classes = len(np.unique(y))
        print(f"   🔢 Número de classes: {n_classes}")

        X = StandardScaler().fit_transform(df.iloc[:, :-1].values)

        Xtr, Xte, ytr, yte = train_test_split(
            X, y,
            test_size=0.5,
            stratify=y,
            random_state=SEED
        )

        # -----------------------------
        # Prior do treino
        # -----------------------------
        train_prev = np.bincount(ytr) / len(ytr)

        # -----------------------------
        # Classificador base
        # -----------------------------
        clf = RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=SEED
        )

        # -----------------------------
        # EMQ baseline
        # -----------------------------
        emq = EMQ(
            clf,
            calib="bcts",
            val_split=0.2,
            exact_train_prev=True,
            on_calib_error="backup"
        )

        emq.fit(Xtr, ytr)

        # =====================================================
        # Treinar MoSS (se necessário)
        # =====================================================
        if n_classes not in regressores_treinados:

            moss_path = os.path.join(
                MOSS_DIR,
                f"moss_d_lite_{n_classes}.pkl"
            )

            if not os.path.exists(moss_path):
                print("⚠️ MoSS sintético não encontrado.")
                regressores_treinados[n_classes] = None
            else:
                print(f"🧠 Treinando MoSS_{n_classes}...")

                with open(moss_path, "rb") as f:
                    synthetic_distributions = pickle.load(f)

                X_m, y_m = [], []

                for (alpha_prev, _), curves in synthetic_distributions.items():
                    for scores_matrix in curves:

                        X_m.append(
                            moss_features(scores_matrix, train_prev)
                        )
                        y_m.append(list(alpha_prev))

                X_m = np.array(X_m)
                y_m = np.array(y_m)

                if len(X_m) == 0:
                    regressores_treinados[n_classes] = None
                else:
                    model = RandomForestRegressor(
                        n_estimators=200,
                        n_jobs=-1,
                        random_state=SEED
                    )
                    model.fit(X_m, y_m)
                    regressores_treinados[n_classes] = model
                    print(f"✅ MoSS_{n_classes} treinado!")

        moss_model = regressores_treinados[n_classes]

        # =====================================================
        # Protocolo UPP
        # =====================================================
        protocol = UPP(
            batch_size=BATCH_SIZE,
            n_prevalences=10,
            repeats=30,
            random_state=SEED
        )

        for idx_batch in tqdm(
            protocol.split(Xte, yte),
            total=protocol.get_n_combinations(),
            desc="UPP"
        ):

            p_real_dict = get_prev_from_labels(
                yte[idx_batch],
                classes=np.arange(n_classes)
            )

            p_real = np.array([p_real_dict[k] for k in sorted(p_real_dict)])

            scores_batch = emq.classifier.predict_proba(
                Xte[idx_batch]
            )

            # ---------- MoSS EM-informed ----------
            if moss_model is not None:

                f_vec = moss_features(
                    scores_batch,
                    train_prev
                ).reshape(1, -1)

                p_pred = moss_model.predict(f_vec)[0][:n_classes]
                p_pred = np.maximum(p_pred, 0)
                p_pred /= (p_pred.sum() + 1e-12)

                rows.append({
                    "dataset": ds_name,
                    "modelo": f"MoSS_EM1_{n_classes}",
                    "erro": np.mean(np.abs(p_pred - p_real))
                })

            # ---------- EMQ ----------
            p_emq = emq.predict(Xte[idx_batch])

            rows.append({
                "dataset": ds_name,
                "modelo": "EMQ_BCTS",
                "erro": np.mean(np.abs(p_emq - p_real))
            })

        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Experimento concluído! Resultados em {OUTPUT_CSV}")


# ============================================================
if __name__ == "__main__":
    run_experiment()