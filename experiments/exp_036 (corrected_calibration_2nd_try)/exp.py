### CÓDIGO B

import os
import pickle
import numpy as np
import pandas as pd
import warnings
import quapy as qp
from sklearn.isotonic import IsotonicRegression
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

    QUAPY_MULTICLASS_DATASETS = [
        'dry-bean',
        'wine-quality',
        'academic-success',
        'digits',
        'letter',
        'abalone',
        'obesity',
        'nursery',
        'yeast',
        'hand_digits',
        'satellite',
        'shuttle',
        'cmc',
        'isolet',
        'waveform-v1',
        'molecular',
        'poker_hand',
        'connect-4',
        'mhr',
        'chess',
        'page_block',
        'phishing',
        'image_seg',
        'hcv',
    ]

    datasets = QUAPY_MULTICLASS_DATASETS

    print("\n===== DATASETS =====")

    for d in datasets:
        print(d)

    print(f"\nTotal datasets: {len(datasets)}")

    qmoss_models = {}
    hybrid_models = {}
    isotonic_models = {}

    rows = []

    for ds_name in datasets:

        print(f"\n📂 Dataset: {ds_name}")

        # ====================================================
        # LOAD DATASET FROM QUAPY
        # ====================================================

        print(f"      📥 Loading QuaPy dataset: {ds_name}")

        dataset = qp.datasets.fetch_UCIMulticlassDataset(
            ds_name,
            min_test_split=0.5,
            verbose=False
        )

        train, test = dataset.train_test

        Xtr = train.instances
        ytr = train.labels

        Xte = test.instances
        yte = test.labels

        # ----------------------------------------------------
        # label normalization
        # ----------------------------------------------------

        all_labels = np.unique(
            np.concatenate([ytr, yte])
        )

        mapping = {
            lab:i
            for i, lab in enumerate(all_labels)
        }

        ytr = np.array([mapping[x] for x in ytr])
        yte = np.array([mapping[x] for x in yte])

        n_classes_real = len(np.unique(ytr))

        print(f"   🔢 Número de classes: {n_classes_real}")

        # ----------------------------------------------------
        # scaling
        # ----------------------------------------------------

        scaler = StandardScaler()

        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

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

        # ================================================
        # REAL CALIBRATION
        # ================================================

        X_real = []
        y_real = []

        protocol_calib = UPP(
            batch_size=500,
            n_prevalences=10,
            repeats=25,
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

        # ====================================================
        # QMOSS + HYBRID + CALIBRATED
        # ====================================================

        if n_classes_real not in qmoss_models:

            moss_path = os.path.join(
                MOSS_DIR,
                f"moss_d_lite_{n_classes_real}.pkl"
            )

            if not os.path.exists(moss_path):

                print(
                    f"⚠️ moss_d_lite_{n_classes_real}.pkl não encontrado."
                )

                qmoss_models[n_classes_real] = None
                hybrid_models[n_classes_real] = None
                isotonic_models[ds_name] = None

            else:

                print(
                    f"🧠 Treinando QMOSS_{n_classes_real}"
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

                # ==================================================
                # QMOSS BASE
                # ==================================================

                qmoss = RandomForestRegressor(
                    n_estimators=300,
                    n_jobs=-1,
                    random_state=SEED,
                    min_samples_leaf=2
                )

                qmoss.fit(
                    X_m,
                    y_m
                )

                qmoss_models[n_classes_real] = qmoss

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

                hybrid = RandomForestRegressor(
                    n_estimators=300,
                    n_jobs=-1,
                    random_state=SEED,
                    min_samples_leaf=2
                )

                hybrid.fit(
                    X_final,
                    y_final,
                    sample_weight=weights
                )

                hybrid_models[n_classes_real] = hybrid

        if ds_name not in isotonic_models:

            pred_matrix = []
            real_matrix = []

            qmoss = qmoss_models[n_classes_real]

            for feat, prev_real in zip(X_real, y_real):

                p_pred = qmoss.predict(
                    feat.reshape(1, -1)
                )[0]

                p_pred = safe_prevalence_vector(
                    p_pred,
                    n_classes_real
                )

                pred_matrix.append(p_pred)
                real_matrix.append(prev_real)

            pred_matrix = np.array(pred_matrix)
            real_matrix = np.array(real_matrix)

            dataset_isotonic = {}

            for c in range(n_classes_real):

                iso = IsotonicRegression(
                    y_min=0,
                    y_max=1,
                    out_of_bounds="clip"
                )

                iso.fit(
                    pred_matrix[:, c],
                    real_matrix[:, c]
                )

                dataset_isotonic[c] = iso

            isotonic_models[ds_name] = dataset_isotonic

        qmoss_model = qmoss_models.get(n_classes_real)
        hybrid_model = hybrid_models.get(n_classes_real)
        dataset_isotonic = isotonic_models.get(ds_name)

        # ====================================================
        # PROTOCOL
        # ====================================================

        protocol = UPP(
            batch_size=500,
            n_prevalences=20,
            repeats=50,
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

            if qmoss_model is not None:

                try:

                    scores_batch = clf.predict_proba(
                        Xte[idx_batch]
                    )

                    f_vec = baseline_features(
                        scores_batch,
                        n_classes_real
                    ).reshape(1, -1)

                    p_qmoss = qmoss_model.predict(
                        f_vec
                    )[0][:n_classes_real]

                    p_qmoss = safe_prevalence_vector(
                        p_qmoss,
                        n_classes_real
                    )

                    rows.append({
                        "dataset": ds_name,
                        "modelo": f"QMOSS_{n_classes_real}",
                        "n_classes_original": n_classes_real,
                        "erro": np.mean(
                            np.abs(
                                p_qmoss - p_real
                            )
                        )
                    })

                    p_cal = np.zeros(
                        n_classes_real
                    )

                    for c in range(n_classes_real):

                        p_cal[c] = dataset_isotonic[c].predict(
                            [p_qmoss[c]]
                        )[0]

                    p_cal = safe_prevalence_vector(
                        p_cal,
                        n_classes_real
                    )

                    rows.append({
                        "dataset": ds_name,
                        "modelo": f"QMOSS_{n_classes_real}_CALIBRATED",
                        "n_classes_original": n_classes_real,
                        "erro": np.mean(
                            np.abs(
                                p_cal - p_real
                            )
                        )
                    })

                    p_hybrid = hybrid_model.predict(
                        f_vec
                    )[0][:n_classes_real]

                    p_hybrid = safe_prevalence_vector(
                        p_hybrid,
                        n_classes_real
                    )

                    rows.append({
                        "dataset": ds_name,
                        "modelo": f"QMOSS_{n_classes_real}_HYBRID",
                        "n_classes_original": n_classes_real,
                        "erro": np.mean(
                            np.abs(
                                p_hybrid - p_real
                            )
                        )
                    })

                except Exception as e:

                    print(f"❌ Erro no MoSS: {e}")

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