import os
import pickle
import numpy as np
import pandas as pd
import warnings
import quapy as qp

from scipy.stats import skew, kurtosis

from sklearn.model_selection import (
    train_test_split,
    cross_val_predict,
    KFold
)

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)

from sklearn.isotonic import IsotonicRegression

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

OUTPUT_CSV = "results.csv"

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

    if isinstance(p, dict):

        vec = np.zeros(n_classes)

        for k, v in p.items():

            if k < n_classes:

                vec[int(k)] = v

        p = vec

    else:

        p = np.array(p, dtype=float)

    if len(p) < n_classes:

        padded = np.zeros(n_classes)

        padded[:len(p)] = p

        p = padded

    p = p[:n_classes]

    p = np.nan_to_num(
        p,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    p = np.clip(p, 0, 1)

    s = p.sum()

    if s <= 0:

        p = np.ones(n_classes) / n_classes

    else:

        p = p / s

    return p

# ============================================================
# CALIBRATION VIA CV5 ON TRAIN
# ============================================================

def build_calibration_table(
    moss_base_model,
    X_m,
    y_m,
    Xtr,
    ytr,
    oof_scores,
    n_classes_real,
    protocol_calib,
    seed=42
):
    """
    Gera uma tabela de calibração com:
    - Colunas 0..n-1   : prevalências PREDITAS pelo MoSS base (por batch)
    - Colunas n..2n-1  : prevalências REAIS do batch

    O MoSS base é avaliado via CV5 nas amostras sintéticas para
    estimar o bias, e depois aplicado nos batches reais do treino.

    Retorna:
        pred_all  : np.array shape (N_batches, n_classes)
        real_all  : np.array shape (N_batches, n_classes)
    """

    pred_all = []
    real_all = []

    # ----------------------------------------------------------
    # CV5 nas amostras SINTÉTICAS para ver o bias do modelo base
    # ----------------------------------------------------------

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    synth_pred = np.zeros_like(y_m)

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_m)):

        fold_model = RandomForestRegressor(
            n_estimators=300,
            n_jobs=-1,
            random_state=seed,
            min_samples_leaf=2
        )

        fold_model.fit(X_m[tr_idx], y_m[tr_idx])

        synth_pred[val_idx] = fold_model.predict(X_m[val_idx])

    # (Opcional) poderíamos acumular synth_pred/y_m para calibração
    # mas a calibração principal é feita nos batches reais abaixo

    # ----------------------------------------------------------
    # Batches REAIS do treino (OOF scores)
    # ----------------------------------------------------------

    for idx_batch in protocol_calib.split(Xtr, ytr):

        scores_batch = oof_scores[idx_batch]

        feat = baseline_features(
            scores_batch,
            n_classes_real
        )

        # predição do MoSS base
        p_pred_raw = moss_base_model.predict(
            feat.reshape(1, -1)
        )[0][:n_classes_real]

        p_pred = safe_prevalence_vector(p_pred_raw, n_classes_real)

        # prevalência real
        p_real = get_prev_from_labels(
            ytr[idx_batch],
            classes=np.arange(n_classes_real)
        )

        p_real = safe_prevalence_vector(p_real, n_classes_real)

        pred_all.append(p_pred)
        real_all.append(p_real)

    return np.array(pred_all), np.array(real_all)


# ============================================================
# ISOTONIC CALIBRATOR
# ============================================================

def train_isotonic_calibrator(pred_train, real_train, n_classes):
    """
    Treina um IsotonicRegressor por classe.
    X = prevalência predita pelo MoSS base
    y = prevalência real

    Retorna lista de modelos (um por classe).
    """

    calibrators = []

    for c in range(n_classes):

        iso = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing='auto',
            out_of_bounds='clip'
        )

        iso.fit(pred_train[:, c], real_train[:, c])

        calibrators.append(iso)

    return calibrators


def apply_isotonic_calibrator(calibrators, p_pred_raw, n_classes):
    """
    Aplica os calibradores isotônicos e normaliza o vetor resultante.
    """

    p_cal = np.array([
        calibrators[c].predict([p_pred_raw[c]])[0]
        for c in range(n_classes)
    ])

    return safe_prevalence_vector(p_cal, n_classes)


# ============================================================
# RF CALIBRATOR
# ============================================================

def train_rf_calibrator(pred_train, real_train):
    """
    Treina um RandomForestRegressor usando as predições do MoSS base
    como features e as prevalências reais como target.

    pred_train : (N, n_classes) - predições do MoSS base
    real_train : (N, n_classes) - prevalências reais
    """

    rf_cal = RandomForestRegressor(
        n_estimators=300,
        n_jobs=-1,
        random_state=42,
        min_samples_leaf=2
    )

    rf_cal.fit(pred_train, real_train)

    return rf_cal


def apply_rf_calibrator(rf_cal, p_pred_raw, n_classes):
    """
    Aplica o RF calibrador e normaliza o vetor resultante.
    """

    p_cal = rf_cal.predict(
        p_pred_raw.reshape(1, -1)
    )[0][:n_classes]

    return safe_prevalence_vector(p_cal, n_classes)


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

    # Cache de modelos MoSS base por n_classes
    # (sintético puro, carregado do .pkl)
    moss_base_cache = {}

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

        ytr = pd.factorize(ytr)[0]

        label_map = {
            old: new
            for new, old in enumerate(np.unique(test.labels))
        }

        yte = pd.factorize(yte)[0]

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
        # CLASSIFIER FOR MoSS (scores nas amostras de teste)
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
        # OOF PROBABILITIES (para calibração no treino)
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
        # LOAD MoSS BASE (sintético puro) — cache por n_classes
        # ====================================================

        if n_classes_real not in moss_base_cache:

            moss_path = os.path.join(
                MOSS_DIR,
                f"moss_d_lite_{n_classes_real}.pkl"
            )

            if not os.path.exists(moss_path):

                print(
                    f"⚠️ moss_d_lite_{n_classes_real}.pkl não encontrado."
                )

                moss_base_cache[n_classes_real] = {
                    "model": None,
                    "X_m": None,
                    "y_m": None
                }

            else:

                print(
                    f"🧠 Carregando MoSS base sintético para {n_classes_real} classes..."
                )

                with open(moss_path, "rb") as f:

                    synthetic_distributions = pickle.load(f)

                # ------------------------------------------------
                # Montar X_m / y_m a partir das curvas sintéticas
                # ------------------------------------------------

                X_m = []
                y_m = []

                for (alpha_prev, _), curves in synthetic_distributions.items():

                    for scores_matrix in curves:

                        feat = baseline_features(
                            scores_matrix,
                            n_classes_real
                        )

                        X_m.append(feat)

                        y_m.append(list(alpha_prev))

                X_m = np.array(X_m)
                y_m = np.array(y_m)

                print(
                    f"      🧪 Synthetic samples: {len(X_m)}"
                )

                # ------------------------------------------------
                # Treina MoSS BASE (só sintético)
                # ------------------------------------------------

                moss_base_model = RandomForestRegressor(
                    n_estimators=300,
                    n_jobs=-1,
                    random_state=SEED,
                    min_samples_leaf=2
                )

                moss_base_model.fit(X_m, y_m)

                print(
                    f"✅ MoSS_BASE_{n_classes_real} treinado!"
                )

                moss_base_cache[n_classes_real] = {
                    "model": moss_base_model,
                    "X_m": X_m,
                    "y_m": y_m
                }

        moss_entry = moss_base_cache[n_classes_real]
        moss_base_model = moss_entry["model"]
        X_m = moss_entry["X_m"]
        y_m = moss_entry["y_m"]

        # ====================================================
        # CALIBRAÇÃO POR DATASET
        # (só se o MoSS base estiver disponível)
        # ====================================================

        iso_calibrators = None   # lista de IsotonicRegression (um por classe)
        rf_calibrator   = None   # RandomForestRegressor

        if moss_base_model is not None:

            print("      📐 Gerando tabela de calibração (CV5 no treino)...")

            protocol_calib = UPP(
                batch_size=500,
                n_prevalences=10,
                repeats=25,
                random_state=SEED
            )

            # --------------------------------------------------
            # Tabela: pred (MoSS base) x real
            # --------------------------------------------------

            pred_calib, real_calib = build_calibration_table(
                moss_base_model=moss_base_model,
                X_m=X_m,
                y_m=y_m,
                Xtr=Xtr,
                ytr=ytr,
                oof_scores=oof_scores,
                n_classes_real=n_classes_real,
                protocol_calib=protocol_calib,
                seed=SEED
            )

            print(
                f"      📊 Calibration table: {pred_calib.shape[0]} amostras "
                f"× {n_classes_real * 2} colunas "
                f"(pred[0..{n_classes_real-1}] | real[{n_classes_real}..{n_classes_real*2-1}])"
            )

            # --------------------------------------------------
            # ISOTONIC calibrator (por classe)
            # --------------------------------------------------

            print("      🔧 Treinando MoSS_ISO_CALIBRATED...")

            iso_calibrators = train_isotonic_calibrator(
                pred_calib,
                real_calib,
                n_classes_real
            )

            print(f"✅ MoSS_ISO_CALIBRATED_{n_classes_real} treinado!")

            # --------------------------------------------------
            # RF calibrator
            # --------------------------------------------------

            print("      🌲 Treinando MoSS_RF_CALIBRATED...")

            rf_calibrator = train_rf_calibrator(
                pred_calib,
                real_calib
            )

            print(f"✅ MoSS_RF_CALIBRATED_{n_classes_real} treinado!")

        # ====================================================
        # PROTOCOL DE TESTE
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
            # QUANTIFIERS CLÁSSICOS
            # ================================================

            for q_name, q in trained_quantifiers.items():

                try:

                    p_pred = q.predict(Xte[idx_batch])

                    p_pred = safe_prevalence_vector(
                        p_pred,
                        n_classes_real
                    )

                    rows.append({
                        "dataset": ds_name,
                        "modelo": q_name,
                        "n_classes": n_classes_real,
                        "erro": np.mean(
                            np.abs(p_pred - p_real)
                        )
                    })

                except Exception as e:

                    print(f"❌ Erro em {q_name}: {e}")

            # ================================================
            # MoSS (3 versões)
            # ================================================

            if moss_base_model is not None:

                try:

                    scores_batch = clf.predict_proba(
                        Xte[idx_batch]
                    )

                    f_vec = baseline_features(
                        scores_batch,
                        n_classes_real
                    ).reshape(1, -1)

                    # ----------------------------------------
                    # 1) MoSS BASE (só sintético)
                    # ----------------------------------------

                    p_moss_raw = moss_base_model.predict(
                        f_vec
                    )[0][:n_classes_real]

                    p_moss_base = safe_prevalence_vector(
                        p_moss_raw,
                        n_classes_real
                    )

                    rows.append({
                        "dataset": ds_name,
                        "modelo": f"MoSS_BASE_{n_classes_real}",
                        "n_classes_original": n_classes_real,
                        "erro": np.mean(
                            np.abs(p_moss_base - p_real)
                        )
                    })

                    # ----------------------------------------
                    # 2) MoSS ISO CALIBRATED
                    # ----------------------------------------

                    if iso_calibrators is not None:

                        p_moss_iso = apply_isotonic_calibrator(
                            iso_calibrators,
                            p_moss_base,
                            n_classes_real
                        )

                        rows.append({
                            "dataset": ds_name,
                            "modelo": f"MoSS_ISO_CALIBRATED_{n_classes_real}",
                            "n_classes_original": n_classes_real,
                            "erro": np.mean(
                                np.abs(p_moss_iso - p_real)
                            )
                        })

                    # ----------------------------------------
                    # 3) MoSS RF CALIBRATED
                    # ----------------------------------------

                    if rf_calibrator is not None:

                        p_moss_rf = apply_rf_calibrator(
                            rf_calibrator,
                            p_moss_base,
                            n_classes_real
                        )

                        rows.append({
                            "dataset": ds_name,
                            "modelo": f"MoSS_RF_CALIBRATED_{n_classes_real}",
                            "n_classes_original": n_classes_real,
                            "erro": np.mean(
                                np.abs(p_moss_rf - p_real)
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

        print(f"💾 Parcial salva em {OUTPUT_CSV}")

    # ========================================================
    # END
    # ========================================================

    print(f"\n✅ Experimento concluído!")
    print(f"📁 Resultados salvos em: {OUTPUT_CSV}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_experiment()