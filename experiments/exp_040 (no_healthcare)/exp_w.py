import os
import pickle
import numpy as np
import pandas as pd
import warnings
import quapy as qp

from scipy.stats import skew, kurtosis

from sklearn.model_selection import (
    train_test_split,
    cross_val_predict
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

from quapy.method.aggregative import EMQ as EMQ_QUAPY

# ============================================================
# MLQuantify
# ============================================================

from mlquantify.likelihood import EMQ as EMQ_MLQ

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

# Datasets locais (10-50 classes), filtrados da lista CSV
LOCAL_DATASETS_DIR = "/var/new_homes/julio/mestrado/mestrado-dyssyn/experiments/datasets_multiclass"

OUTPUT_CSV = "results_w.csv"

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
# CALIBRATION TABLE
# Aplica o MoSS BASE (treinado só em sintéticas) nos batches
# reais do treino e retorna pred vs real para calibração.
# Também retorna os f_vecs originais para o RF calibrador.
# ============================================================

def build_calibration_table(
    moss_base_model,
    Xtr,
    ytr,
    oof_scores,
    n_classes_real,
    protocol_calib
):
    """
    Retorna:
        pred_all : (N, n_classes)   — predições do MoSS base nos batches reais
        real_all : (N, n_classes)   — prevalências reais dos batches
        feat_all : (N, 4*n_classes) — f_vecs originais (para RF calibrador)
    """

    pred_all = []
    real_all = []
    feat_all = []

    for idx_batch in protocol_calib.split(Xtr, ytr):

        scores_batch = oof_scores[idx_batch]

        feat = baseline_features(
            scores_batch,
            n_classes_real
        )

        p_pred_raw = moss_base_model.predict(
            feat.reshape(1, -1)
        )[0][:n_classes_real]

        p_pred = safe_prevalence_vector(p_pred_raw, n_classes_real)

        p_real = get_prev_from_labels(
            ytr[idx_batch],
            classes=np.arange(n_classes_real)
        )

        p_real = safe_prevalence_vector(p_real, n_classes_real)

        pred_all.append(p_pred)
        real_all.append(p_real)
        feat_all.append(feat)

    return (
        np.array(pred_all),
        np.array(real_all),
        np.array(feat_all)
    )


# ============================================================
# ISOTONIC CALIBRATOR
# Melhoria 2: aceita sample_weight
# ============================================================

def train_isotonic_calibrator(pred_train, real_train, n_classes):
    """
    Treina um IsotonicRegressor por classe.
    X = predição do MoSS base nos dados reais
    y = prevalência real
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

    p_cal = np.array([
        calibrators[c].predict([p_pred_raw[c]])[0]
        for c in range(n_classes)
    ])

    return safe_prevalence_vector(p_cal, n_classes)


# ============================================================
# RF CALIBRATOR
# Melhoria 3: usa f_vec original concatenado com pred do MoSS
# ============================================================

def train_rf_calibrator(pred_train, real_train, feat_train):
    """
    RF calibrador com features enriquecidas:
    Input = [pred_moss | f_vec_original]
    O modelo acessa tanto a estimativa do MoSS quanto as
    estatísticas brutas dos scores (média, var, skew, kurtosis).
    """

    X_input = np.hstack([pred_train, feat_train])

    rf_cal = RandomForestRegressor(
        n_estimators=300,
        n_jobs=-1,
        random_state=42,
        min_samples_leaf=2
    )

    rf_cal.fit(X_input, real_train)

    return rf_cal


def apply_rf_calibrator(rf_cal, p_pred_raw, f_vec, n_classes):
    """
    Aplica o RF calibrador com features enriquecidas.
    """

    X_input = np.hstack([
        p_pred_raw.reshape(1, -1),
        f_vec.reshape(1, -1)
    ])

    p_cal = rf_cal.predict(X_input)[0][:n_classes]

    return safe_prevalence_vector(p_cal, n_classes)


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment():

    # --------------------------------------------------------
    # QUAPY datasets (já têm split definido)
    # --------------------------------------------------------

    QUAPY_MULTICLASS_DATASETS = [
    ]

    # --------------------------------------------------------
    # Datasets locais com 10-50 classes (split 70/30)
    # --------------------------------------------------------

    LOCAL_DATASETS = [
        ("Walking.csv",                                     "Class"),
    ]

    print("\n===== QUAPY DATASETS =====")
    for d in QUAPY_MULTICLASS_DATASETS:
        print(f"  [QuaPy] {d}")

    print("\n===== LOCAL DATASETS (10-50 classes) =====")
    for fname, _ in LOCAL_DATASETS:
        print(f"  [Local] {fname}")

    print(f"\nTotal QuaPy: {len(QUAPY_MULTICLASS_DATASETS)}")
    print(f"Total Local:  {len(LOCAL_DATASETS)}")
    print(f"Total geral:  {len(QUAPY_MULTICLASS_DATASETS) + len(LOCAL_DATASETS)}")

    moss_base_cache = {}

    rows = []

    # ========================================================
    # LOOP UNIFICADO: cada entrada é (ds_name, source, meta)
    # source = "quapy" ou "local"
    # meta   = None (quapy) ou (filepath, target_col) (local)
    # ========================================================

    all_entries = (
        [(ds, "quapy", None) for ds in QUAPY_MULTICLASS_DATASETS] +
        [(fname, "local", (os.path.join(LOCAL_DATASETS_DIR, fname), col))
         for fname, col in LOCAL_DATASETS]
    )

    for ds_name, source, meta in all_entries:

        print(f"\n📂 Dataset: {ds_name} [{source}]")

        # ====================================================
        # LOAD DATASET
        # ====================================================

        try:

            if source == "quapy":

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

                ytr = pd.factorize(ytr)[0]
                yte = pd.factorize(yte)[0]

            else:

                filepath, target_col = meta

                print(f"      📥 Loading local CSV: {filepath}")

                df = pd.read_csv(filepath)

                if target_col not in df.columns:
                    print(f"⚠️ Coluna '{target_col}' não encontrada em {ds_name}, pulando...")
                    continue

                # Remover colunas não numéricas exceto target
                X_cols = [c for c in df.columns if c != target_col]
                df_X = df[X_cols].select_dtypes(include=[np.number])
                df_X = df_X.dropna(axis=1, how="all")
                y_raw = df[target_col].values

                X_all = df_X.values
                y_all, _ = pd.factorize(y_raw)

                # Filtrar classes com menos de 2 amostras
                classes, counts = np.unique(y_all, return_counts=True)
                valid_classes = classes[counts >= 2]
                mask = np.isin(y_all, valid_classes)
                X_all = X_all[mask]
                y_all = y_all[mask]

                # Re-mapear para 0..n-1 contíguo
                y_all, _ = pd.factorize(y_all)

                n_cls = len(np.unique(y_all))

                if not (10 <= n_cls <= 50):
                    print(f"⚠️ {ds_name} tem {n_cls} classes após filtro, pulando...")
                    continue

                # Split 70/30 estratificado
                Xtr, Xte, ytr, yte = train_test_split(
                    X_all, y_all,
                    test_size=0.3,
                    random_state=SEED,
                    stratify=y_all
                )

        except Exception as e:

            print(f"❌ Erro ao carregar {ds_name}: {e}")
            continue

        # ----------------------------------------------------
        # Verificar n_classes após split
        # ----------------------------------------------------

        n_classes_real = len(np.unique(ytr))

        # Garantir que teste tem as mesmas classes que treino
        yte = np.array([y if y < n_classes_real else 0 for y in yte])

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
        # EMQ (QuaPy)
        # ----------------------------------------------------

        quantifiers["EMQ_QUAPY"] = EMQ_QUAPY(
            RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                random_state=SEED
            ),
            calib=None,
            exact_train_prev=True
        )

        quantifiers["EMQ_BCTS_QUAPY"] = EMQ_QUAPY(
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
        # EMQ (MLQuantify)
        # ----------------------------------------------------

        quantifiers["EMQ_MLQ"] = EMQ_MLQ(
            learner=RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                random_state=SEED
            ),
            calib_function=None,
            on_calib_error="backup"
        )

        quantifiers["EMQ_BCTS_MLQ"] = EMQ_MLQ(
            learner=RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                random_state=SEED
            ),
            calib_function="bcts",
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
        # CLASSIFIER FOR MoSS
        # ====================================================

        if "EMQ_BCTS_QUAPY" in trained_quantifiers:

            clf = trained_quantifiers["EMQ_BCTS_QUAPY"].classifier

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
        # LOAD / TRAIN MoSS BASE — cache por n_classes
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

                print(f"      🧪 Synthetic samples: {len(X_m)}")

                moss_base_model = RandomForestRegressor(
                    n_estimators=300,
                    n_jobs=-1,
                    random_state=SEED,
                    min_samples_leaf=2
                )

                moss_base_model.fit(X_m, y_m)

                print(f"✅ MoSS_BASE_{n_classes_real} treinado!")

                moss_base_cache[n_classes_real] = {
                    "model": moss_base_model,
                    "X_m": X_m,
                    "y_m": y_m
                }

        moss_entry      = moss_base_cache[n_classes_real]
        moss_base_model = moss_entry["model"]
        X_m             = moss_entry["X_m"]
        y_m             = moss_entry["y_m"]

        # ====================================================
        # CALIBRAÇÃO POR DATASET
        # ====================================================

        iso_calibrators = None
        rf_calibrator   = None

        if moss_base_model is not None:

            print("      📐 Gerando tabela de calibração...")

            protocol_calib = UPP(
                batch_size=500,
                n_prevalences=10,
                repeats=25,
                random_state=SEED
            )

            # --------------------------------------------------
            # Tabela enriquecida: sintéticas (OOF) + reais
            # --------------------------------------------------

            pred_calib, real_calib, feat_calib = build_calibration_table(
                moss_base_model=moss_base_model,
                Xtr=Xtr,
                ytr=ytr,
                oof_scores=oof_scores,
                n_classes_real=n_classes_real,
                protocol_calib=protocol_calib
            )

            print(
                f"      📊 Calibration table: {pred_calib.shape[0]} amostras reais"
            )

            # --------------------------------------------------
            # ISOTONIC: por classe com pesos
            # --------------------------------------------------

            print("      🔧 Treinando MoSS_ISO_CALIBRATED...")

            iso_calibrators = train_isotonic_calibrator(
                pred_calib,
                real_calib,
                n_classes_real
            )

            print(f"✅ MoSS_ISO_CALIBRATED_{n_classes_real} treinado!")

            # --------------------------------------------------
            # RF: pred_moss + f_vec original como features
            # --------------------------------------------------

            print("      🌲 Treinando MoSS_RF_CALIBRATED...")

            rf_calibrator = train_rf_calibrator(
                pred_calib,
                real_calib,
                feat_calib
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
                        "n_classes_original": n_classes_real,
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
                    )

                    f_vec_2d = f_vec.reshape(1, -1)

                    # ----------------------------------------
                    # 1) MoSS BASE (só sintético)
                    # ----------------------------------------

                    p_moss_raw = moss_base_model.predict(
                        f_vec_2d
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
                    # 2) MoSS ISO CALIBRATED (com pesos)
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
                    # 3) MoSS RF CALIBRATED (pred + f_vec)
                    # ----------------------------------------

                    if rf_calibrator is not None:

                        p_moss_rf = apply_rf_calibrator(
                            rf_calibrator,
                            p_moss_base,
                            f_vec,
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

    print(f"\n✅ Experimento concluído!")
    print(f"📁 Resultados salvos em: {OUTPUT_CSV}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_experiment()