import os
import pickle
import numpy as np
import pandas as pd
import warnings

from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm

# ==============================
# Quantificação (QuaPy)
# ==============================
import quapy as qp
from quapy.method.aggregative import EMQ

from mlquantify.model_selection import UPP
from mlquantify.utils import get_prev_from_labels
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

def estimate_prevalence_from_clusters(q, M):
    """
    Resolve:
        min || M pi - q ||^2
    s.t. pi >= 0, sum(pi)=1
    """
    n_classes = M.shape[1]

    def objective(pi):
        return np.linalg.norm(M @ pi - q)**2

    constraints = [
        {"type": "eq", "fun": lambda pi: np.sum(pi) - 1}
    ]

    bounds = [(0, 1)] * n_classes

    pi0 = np.ones(n_classes) / n_classes

    result = minimize(
        objective,
        pi0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP"
    )

    if result.success:
        return result.x
    else:
        return pi0
    


# ============================================================
# CONFIG
# ============================================================
SEED = 42
BATCH_SIZE = 100

MOSS_DIR = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/moss/multiclass"
DATASETS_ROOT = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/multiclass"
OUTPUT_CSV = "m_30_cluster_vs_emq.csv"

np.random.seed(SEED)

# ============================================================
# FEATURES (MoSS)
# ============================================================
def baseline_features(scores_matrix, target_n_classes):
    feats = []
    for c in range(scores_matrix.shape[1]):
        s = scores_matrix[:, c]
        feats.extend([
            np.mean(s),
            np.var(s),
            skew(s),
            kurtosis(s)
        ])

    total_expected = 4 * target_n_classes
    if len(feats) < total_expected:
        feats.extend([0] * (total_expected - len(feats)))

    return np.array(feats[:total_expected])


# ============================================================
# EXPERIMENT
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

        n_classes_real = len(np.unique(y))
        print(f"   🔢 Número de classes: {n_classes_real}")

        X = StandardScaler().fit_transform(df.iloc[:, :-1].values)

        Xtr, Xte, ytr, yte = train_test_split(
            X, y,
            test_size=0.5,
            stratify=y,
            random_state=SEED
        )

        # ======================================================
        # 1️⃣ CLASSIFICADOR BASE + EMQ
        # ======================================================
        clf = RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=SEED
        )

        emq = EMQ(
            clf,
            calib="bcts",
            val_split=0.2,
            exact_train_prev=True,
            on_calib_error="backup"
        )

        emq.fit(Xtr, ytr)

        # ======================================================
        # 2️⃣ MoSS
        # ======================================================
        if n_classes_real not in regressores_treinados:

            moss_path = os.path.join(
                MOSS_DIR,
                f"moss_d_lite_{n_classes_real}.pkl"
            )

            if not os.path.exists(moss_path):
                print(f"⚠️ moss_d_lite_{n_classes_real}.pkl não encontrado.")
                regressores_treinados[n_classes_real] = None
            else:
                print(f"🧠 Treinando MoSS_{n_classes_real}...")

                with open(moss_path, "rb") as f:
                    synthetic_distributions = pickle.load(f)

                X_m, y_m = [], []

                for (alpha_prev, _), curves in synthetic_distributions.items():
                    for scores_matrix in curves:
                        X_m.append(
                            baseline_features(scores_matrix, n_classes_real)
                        )
                        y_m.append(list(alpha_prev))

                X_m = np.array(X_m)
                y_m = np.array(y_m)

                model = RandomForestRegressor(
                    n_estimators=100,
                    n_jobs=-1,
                    random_state=SEED
                )
                model.fit(X_m, y_m)

                regressores_treinados[n_classes_real] = model
                print(f"✅ MoSS_{n_classes_real} treinado!")

        moss_model = regressores_treinados[n_classes_real]

        # ============================
        # CLUSTERING BASE
        # ============================
        n_clusters = n_classes_real * 3

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=SEED,
            n_init=10
        )

        kmeans.fit(Xtr)

        cluster_assign_tr = kmeans.predict(Xtr)

        # M[k,c] = P(cluster=k | y=c)
        M = np.zeros((n_clusters, n_classes_real))

        for c in range(n_classes_real):
            idx_c = ytr == c
            clusters_c = cluster_assign_tr[idx_c]

            counts = np.bincount(clusters_c, minlength=n_clusters)
            if counts.sum() > 0:
                M[:, c] = counts / counts.sum()
            else:
                M[:, c] = np.ones(n_clusters) / n_clusters

        # ======================================================
        # 4️⃣ PROTOCOLO UPP
        # ======================================================
        protocol = UPP(
            batch_size=BATCH_SIZE,
            n_prevalences=10,
            repeats=30,
            random_state=SEED
        )

        for idx_batch in tqdm(
            protocol.split(Xte, yte),
            total=protocol.get_n_combinations(),
            desc="Protocolo UPP"
        ):

            p_real_raw = get_prev_from_labels(
                yte[idx_batch],
                classes=np.arange(n_classes_real)
            )

            if isinstance(p_real_raw, dict):
                p_real = np.array(
                    [p_real_raw[k] for k in sorted(p_real_raw)]
                )
            else:
                p_real = np.array(p_real_raw)

            # ==================================================
            # EMQ
            # ==================================================
            p_emq = emq.predict(Xte[idx_batch])

            rows.append({
                "dataset": ds_name,
                "modelo": "EMQ_BCTS",
                "n_classes_original": n_classes_real,
                "erro": np.mean(np.abs(p_emq - p_real))
            })

            # ==================================================
            # MoSS
            # ==================================================
            if moss_model is not None:

                scores_batch = clf.predict_proba(Xte[idx_batch])

                f_vec = baseline_features(
                    scores_batch,
                    n_classes_real
                ).reshape(1, -1)

                p_moss = moss_model.predict(f_vec)[0][:n_classes_real]
                p_moss /= (p_moss.sum() + 1e-12)

                rows.append({
                    "dataset": ds_name,
                    "modelo": f"MoSS_{n_classes_real}",
                    "n_classes_original": n_classes_real,
                    "erro": np.mean(np.abs(p_moss - p_real))
                })

            # ==============================
            # ClusterQuant-Mixture
            # ==============================
            cluster_assign_batch = kmeans.predict(Xte[idx_batch])

            q = np.bincount(cluster_assign_batch, minlength=n_clusters)
            q = q / (q.sum() + 1e-12)

            p_cluster = estimate_prevalence_from_clusters(q, M)

            rows.append({
                "dataset": ds_name,
                "modelo": "ClusterQuant_Mixture",
                "n_classes_original": n_classes_real,
                "erro": np.mean(np.abs(p_cluster - p_real))
            })

        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Experimento concluído! Resultados salvos em {OUTPUT_CSV}")


# ============================================================
if __name__ == "__main__":
    run_experiment()