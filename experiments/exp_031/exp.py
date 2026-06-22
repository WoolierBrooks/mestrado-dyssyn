import os
import pickle
import numpy as np
import pandas as pd
import warnings

from scipy.stats import skew, kurtosis

from sklearn.model_selection import (
    train_test_split,
    cross_val_predict
)

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from tqdm import tqdm

# ============================================================
# QuaPy
# ============================================================

from quapy.method.aggregative import EMQ

# ============================================================
# MLQuantify
# ============================================================

from mlquantify.model_selection import UPP
from mlquantify.utils import get_prev_from_labels

from mlquantify.adjust_counting import (
    CC,
    PCC,
    AC,
    PAC,
    FM
)

from mlquantify.neighbors import (
    PWK,
    KDEyML,
    KDEyHD,
    KDEyCS
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

SEED = 42
BATCH_SIZE = 100

MOSS_DIR = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/moss/multiclass"

DATASETS_ROOT = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/multiclass"

OUTPUT_CSV = "all_quantifiers_vs_moss.csv"

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

        feats.extend(
            [0] * (total_expected - len(feats))
        )

    return np.array(
        feats[:total_expected]
    )

# ============================================================
# SAFE PREVALENCE
# ============================================================

def safe_prevalence_vector(p, n_classes):

    # --------------------------------------------------------
    # dict -> array
    # --------------------------------------------------------

    if isinstance(p, dict):

        vec = np.zeros(n_classes)

        for k, v in p.items():

            if k < n_classes:

                vec[int(k)] = v

        p = vec

    else:

        p = np.array(p, dtype=float)

    # --------------------------------------------------------
    # PAD if missing classes
    # --------------------------------------------------------

    if len(p) < n_classes:

        padded = np.zeros(n_classes)

        padded[:len(p)] = p

        p = padded

    # --------------------------------------------------------
    # CUT if larger
    # --------------------------------------------------------

    p = p[:n_classes]

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    p = np.nan_to_num(
        p,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    p = np.clip(p, 0, 1)

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    s = p.sum()

    if s <= 0:

        p = np.ones(n_classes) / n_classes

    else:

        p = p / s

    return p

# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment():

    datasets = sorted([
        f for f in os.listdir(DATASETS_ROOT)
        if f.endswith(".csv")
    ])

    regressores_treinados = {}

    rows = []

    for ds_name in datasets:

        print(f"\n📂 Dataset: {ds_name}")

        # ====================================================
        # LOAD DATASET
        # ====================================================

        df = pd.read_csv(
            os.path.join(DATASETS_ROOT, ds_name)
        )

        y = df.iloc[:, -1].values

        if y.min() > 0:
            y -= y.min()

        n_classes_real = len(np.unique(y))

        print(f"   🔢 Número de classes: {n_classes_real}")

        X = StandardScaler().fit_transform(
            df.iloc[:, :-1].values
        )

        # ====================================================
        # TRAIN / TEST SPLIT
        # ====================================================

        Xtr, Xte, ytr, yte = train_test_split(
            X,
            y,
            test_size=0.5,
            stratify=y,
            random_state=SEED
        )

        # ====================================================
        # QUANTIFIERS
        # ====================================================

        print("      🔥 Building quantifiers...")

        quantifiers = {}

        # ----------------------------------------------------
        # EMQ
        # ----------------------------------------------------

        quantifiers["EMQ"] = EMQ(
            RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                random_state=SEED
            ),
            calib=None,
            exact_train_prev=True
        )

        quantifiers["EMQ_BCTS"] = EMQ(
            RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                random_state=SEED
            ),
            calib="bcts",
            val_split=0.2,
            exact_train_prev=True,
            on_calib_error="backup"
        )

        # ----------------------------------------------------
        # Counting
        # ----------------------------------------------------

        quantifiers["CC"] = CC(
            learner=RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                random_state=SEED
            )
        )

        quantifiers["PCC"] = PCC(
            learner=RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                random_state=SEED
            )
        )

        # ----------------------------------------------------
        # Adjusted Counting
        # ----------------------------------------------------

        quantifiers["AC"] = AC(
            learner=RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                random_state=SEED
            )
        )

        quantifiers["PAC"] = PAC(
            learner=RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                random_state=SEED
            )
        )

        quantifiers["FM"] = FM(
            learner=RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                random_state=SEED
            )
        )

        # ----------------------------------------------------
        # KDEy
        # ----------------------------------------------------

        quantifiers["KDEyML"] = KDEyML(
            learner=RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                random_state=SEED
            ),
            bandwidth=0.1
        )

        quantifiers["KDEyHD"] = KDEyHD(
            learner=RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                random_state=SEED
            ),
            montecarlo_trials=500,
            random_state=SEED
        )

        quantifiers["KDEyCS"] = KDEyCS(
            learner=RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                random_state=SEED
            ),
            bandwidth=0.1
        )

        # ----------------------------------------------------
        # PWK
        # ----------------------------------------------------

        quantifiers["PWK"] = PWK(
            n_neighbors=10
        )

        # ====================================================
        # TRAIN QUANTIFIERS
        # ====================================================

        trained_quantifiers = {}

        print("      🚀 Training quantifiers...")

        for q_name, q in quantifiers.items():

            try:

                print(f"         → {q_name}")

                q.fit(Xtr, ytr)

                trained_quantifiers[q_name] = q

            except Exception as e:

                print(f"❌ Erro em {q_name}: {e}")

        # ====================================================
        # CLASSIFIER FOR MoSS
        # ====================================================

        if "EMQ_BCTS" in trained_quantifiers:

            clf = trained_quantifiers["EMQ_BCTS"].classifier

        else:

            clf = RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                random_state=SEED
            )

            clf.fit(Xtr, ytr)

        # ====================================================
        # OOF PROBABILITIES
        # ====================================================

        print("      🔄 Generating OOF probabilities...")

        oof_scores = cross_val_predict(
            RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                random_state=SEED
            ),
            Xtr,
            ytr,
            cv=5,
            method="predict_proba",
            n_jobs=-1
        )

        # ====================================================
        # MoSS + CALIBRATION
        # ====================================================

        if n_classes_real not in regressores_treinados:

            moss_path = os.path.join(
                MOSS_DIR,
                f"moss_d_lite_{n_classes_real}.pkl"
            )

            if not os.path.exists(moss_path):

                print(
                    f"⚠️ moss_d_lite_{n_classes_real}.pkl não encontrado."
                )

                regressores_treinados[n_classes_real] = None

            else:

                print(
                    f"🧠 Treinando MoSS_{n_classes_real}_CALIBRATED..."
                )

                with open(moss_path, "rb") as f:

                    synthetic_distributions = pickle.load(f)

                # ================================================
                # SYNTHETIC
                # ================================================

                X_m = []
                y_m = []

                for (alpha_prev, _), curves in synthetic_distributions.items():

                    for scores_matrix in curves:

                        feat = baseline_features(
                            scores_matrix,
                            n_classes_real
                        )

                        X_m.append(feat)

                        y_m.append(
                            list(alpha_prev)
                        )

                X_m = np.array(X_m)
                y_m = np.array(y_m)

                print(
                    f"      🧪 Synthetic samples: {len(X_m)}"
                )

                # ================================================
                # REAL CALIBRATION
                # ================================================

                X_real = []
                y_real = []

                protocol_calib = UPP(
                    batch_size=BATCH_SIZE,
                    n_prevalences=10,
                    repeats=15,
                    random_state=SEED
                )

                for idx_batch in protocol_calib.split(Xtr, ytr):

                    scores_batch = oof_scores[idx_batch]

                    feat = baseline_features(
                        scores_batch,
                        n_classes_real
                    )

                    prev_real = get_prev_from_labels(
                        ytr[idx_batch],
                        classes=np.arange(n_classes_real)
                    )

                    prev_real = safe_prevalence_vector(
                        prev_real,
                        n_classes_real
                    )

                    X_real.append(feat)
                    y_real.append(prev_real)

                X_real = np.array(X_real)
                y_real = np.array(y_real)

                print(
                    f"      🔥 Real calibration samples: {len(X_real)}"
                )

                # ================================================
                # HYBRID TRAINING
                # ================================================

                X_final = np.vstack([
                    X_m,
                    X_real
                ])

                y_final = np.vstack([
                    y_m,
                    y_real
                ])

                weights = np.concatenate([
                    np.ones(len(X_m)),
                    np.ones(len(X_real)) * 5
                ])

                model = RandomForestRegressor(
                    n_estimators=300,
                    n_jobs=-1,
                    random_state=SEED,
                    min_samples_leaf=2
                )

                model.fit(
                    X_final,
                    y_final,
                    sample_weight=weights
                )

                regressores_treinados[n_classes_real] = model

                print(
                    f"✅ MoSS_{n_classes_real}_CALIBRATED treinado!"
                )

        moss_model = regressores_treinados[n_classes_real]

        # ====================================================
        # CLUSTERQUANT
        # ====================================================

        n_clusters = n_classes_real * 3

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=SEED,
            n_init=10
        )

        kmeans.fit(Xtr)

        cluster_assign_tr = kmeans.predict(Xtr)

        M = np.zeros(
            (n_clusters, n_classes_real)
        )

        for k in range(n_clusters):

            idx = cluster_assign_tr == k

            if np.sum(idx) > 0:

                prev = get_prev_from_labels(
                    ytr[idx],
                    classes=np.arange(n_classes_real)
                )

                prev = safe_prevalence_vector(
                    prev,
                    n_classes_real
                )

                M[k] = prev

            else:

                M[k] = (
                    np.ones(n_classes_real)
                    / n_classes_real
                )

        # ====================================================
        # PROTOCOL
        # ====================================================

        protocol = UPP(
            batch_size=BATCH_SIZE,
            n_prevalences=10,
            repeats=30,
            random_state=SEED
        )

        total_batches = protocol.get_n_combinations()

        for idx_batch in tqdm(
            protocol.split(Xte, yte),
            total=total_batches,
            desc="Protocolo UPP"
        ):

            # ================================================
            # REAL PREVALENCE
            # ================================================

            p_real = get_prev_from_labels(
                yte[idx_batch],
                classes=np.arange(n_classes_real)
            )

            p_real = safe_prevalence_vector(
                p_real,
                n_classes_real
            )

            # ================================================
            # ALL QUANTIFIERS
            # ================================================

            for q_name, q in trained_quantifiers.items():

                try:

                    p_pred = q.predict(
                        Xte[idx_batch]
                    )

                    p_pred = safe_prevalence_vector(
                        p_pred,
                        n_classes_real
                    )

                    rows.append({
                        "dataset": ds_name,
                        "modelo": q_name,
                        "n_classes_original": n_classes_real,
                        "erro": np.mean(
                            np.abs(
                                p_pred - p_real
                            )
                        )
                    })

                except Exception as e:

                    print(
                        f"❌ Erro em {q_name}: {e}"
                    )

            # ================================================
            # MoSS
            # ================================================

            if moss_model is not None:

                try:

                    scores_batch = clf.predict_proba(
                        Xte[idx_batch]
                    )

                    f_vec = baseline_features(
                        scores_batch,
                        n_classes_real
                    ).reshape(1, -1)

                    p_moss = moss_model.predict(
                        f_vec
                    )[0][:n_classes_real]

                    p_moss = safe_prevalence_vector(
                        p_moss,
                        n_classes_real
                    )

                    rows.append({
                        "dataset": ds_name,
                        "modelo": f"MoSS_{n_classes_real}_CALIBRATED",
                        "n_classes_original": n_classes_real,
                        "erro": np.mean(
                            np.abs(
                                p_moss - p_real
                            )
                        )
                    })

                except Exception as e:

                    print(f"❌ Erro no MoSS: {e}")

            # ================================================
            # ClusterQuant
            # ================================================

            try:

                cluster_assign_batch = kmeans.predict(
                    Xte[idx_batch]
                )

                q_cluster = np.bincount(
                    cluster_assign_batch,
                    minlength=n_clusters
                )

                q_cluster = (
                    q_cluster
                    / (q_cluster.sum() + 1e-12)
                )

                p_cluster = M.T @ q_cluster

                p_cluster = safe_prevalence_vector(
                    p_cluster,
                    n_classes_real
                )

                rows.append({
                    "dataset": ds_name,
                    "modelo": "ClusterQuant",
                    "n_classes_original": n_classes_real,
                    "erro": np.mean(
                        np.abs(
                            p_cluster - p_real
                        )
                    )
                })

            except Exception as e:

                print(
                    f"❌ Erro no ClusterQuant: {e}"
                )

        # ====================================================
        # SAVE PARTIAL
        # ====================================================

        pd.DataFrame(rows).to_csv(
            OUTPUT_CSV,
            index=False
        )

        print(
            f"💾 Parcial salva em {OUTPUT_CSV}"
        )

    # ========================================================
    # END
    # ========================================================

    print(
        f"\n✅ Experimento concluído!"
    )

    print(
        f"📁 Resultados salvos em:"
    )

    print(
        f"   {OUTPUT_CSV}"
    )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_experiment()