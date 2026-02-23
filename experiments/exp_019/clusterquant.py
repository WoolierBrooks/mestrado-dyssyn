import os
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.optimize import nnls
from tqdm import tqdm

import quapy as qp
from quapy.method.aggregative import EMQ

from mlquantify.model_selection import UPP
from mlquantify.utils import get_prev_from_labels

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
SEED = 42
BATCH_SIZE = 100

DATASETS_ROOT = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/multiclass"
OUTPUT_CSV = "clusterquant2_vs_emq.csv"

np.random.seed(SEED)

# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================
def run_experiment():

    datasets = sorted(
        f for f in os.listdir(DATASETS_ROOT) if f.endswith(".csv")
    )

    rows = []

    for ds_name in datasets:

        print(f"\n📂 Dataset: {ds_name}")

        df = pd.read_csv(os.path.join(DATASETS_ROOT, ds_name))

        y = df.iloc[:, -1].values
        if y.min() > 0:
            y -= y.min()

        n_classes = len(np.unique(y))
        print(f"   🔢 Classes: {n_classes}")

        X = StandardScaler().fit_transform(df.iloc[:, :-1].values)

        Xtr, Xte, ytr, yte = train_test_split(
            X, y,
            test_size=0.5,
            stratify=y,
            random_state=SEED
        )

        # ======================================================
        # EMQ (baseline forte)
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
        # CLUSTERQUANT 2.0
        # ======================================================

        n_clusters = n_classes * 3

        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=SEED
        )

        gmm.fit(Xtr)

        # Soft assignments treino
        gamma_tr = gmm.predict_proba(Xtr)

        # ======================================================
        # Estimar A[k,c] = P(z=k | y=c)
        # ======================================================
        A = np.zeros((n_clusters, n_classes))

        for c in range(n_classes):
            idx = (ytr == c)
            if np.sum(idx) > 0:
                A[:, c] = gamma_tr[idx].mean(axis=0)

        # Normalizar colunas
        A = A / (A.sum(axis=0, keepdims=True) + 1e-12)

        # ======================================================
        # PROTOCOLO UPP
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
            desc="UPP"
        ):

            p_real_raw = get_prev_from_labels(
                yte[idx_batch],
                classes=np.arange(n_classes)
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
                "erro": np.mean(np.abs(p_emq - p_real))
            })

            # ==================================================
            # ClusterQuant 2.0
            # ==================================================

            gamma_batch = gmm.predict_proba(Xte[idx_batch])
            q = gamma_batch.mean(axis=0)  # P(z=k)

            # Resolver:  q ≈ A π
            pi_hat, _ = nnls(A, q)

            if pi_hat.sum() > 0:
                pi_hat = pi_hat / pi_hat.sum()
            else:
                pi_hat = np.ones(n_classes) / n_classes

            rows.append({
                "dataset": ds_name,
                "modelo": "ClusterQuant2",
                "erro": np.mean(np.abs(pi_hat - p_real))
            })

        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Finalizado! Resultados salvos em {OUTPUT_CSV}")


# ============================================================
if __name__ == "__main__":
    run_experiment()