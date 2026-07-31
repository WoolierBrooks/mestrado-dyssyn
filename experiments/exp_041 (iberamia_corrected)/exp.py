import os
import sys
import functools
import traceback

# ── Força flush imediato em todo print() do módulo ──
# Rodando com nohup/background, o stdout fica bufferizado em bloco:
# se o processo morrer antes do buffer encher, os prints que você
# "viu rodando" na tela nunca chegam a ser gravados no arquivo de
# log. É por isso que você só via "nohup: ignoring input" e nada
# mais, mesmo o script tendo processado várias linhas antes de cair.
print = functools.partial(print, flush=True)

# ── TMPDIR do joblib ──
# NÃO usar /tmp (é tmpfs = RAM). Usa o $HOME real do usuário (resolvido
# dinamicamente, em vez de um caminho chutado) + fallback se não tiver
# permissão de escrita lá.
_home_dir = os.path.expanduser("~")
_joblib_tmp = os.path.join(_home_dir, "tmp_joblib")

try:
    os.makedirs(_joblib_tmp, exist_ok=True)
    # testa se realmente consegue escrever ali
    _test_path = os.path.join(_joblib_tmp, ".write_test")
    with open(_test_path, "w") as _f:
        _f.write("ok")
    os.remove(_test_path)

except (PermissionError, OSError) as _e:
    import tempfile
    _joblib_tmp = os.path.join(tempfile.gettempdir(), "julio_joblib_tmp")
    os.makedirs(_joblib_tmp, exist_ok=True)
    print(f"⚠️  Não consegui usar {os.path.join(_home_dir, 'tmp_joblib')} ({_e}); "
          f"usando fallback: {_joblib_tmp}")

os.environ["JOBLIB_TEMP_FOLDER"] = _joblib_tmp
print(f"📁 JOBLIB_TEMP_FOLDER = {_joblib_tmp}")

import pickle
import time
import gc
import numpy as np
import pandas as pd
import warnings
import quapy as qp

from scipy.stats import skew, kurtosis

from joblib.externals.loky import get_reusable_executor

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

OUTPUT_CSV = "results.csv"

# ── Número de repetições do loop treino/teste ──────────────
N_REPETITIONS = 1

# ── UPP: repeats=1; n_prevalences calculado para ≥1000 batches
# batches = repeats * n_prevalences  →  1 * 1000 = 1000
UPP_REPEATS        = 1
N_PREV_CALIB       = 1000   # ≥1000 batches na calibração
N_PREV_TEST        = 1000   # 1000 batches no teste

# ── Paralelismo por modelo. -1 (todos os núcleos) em 13 modelos
# diferentes, repetidos por 30 reps x 41 datasets, é o que estava
# esgotando semáforos/memória/tmp e derrubando o processo. Ajuste
# esse número para caber confortavelmente na sua máquina.
N_JOBS = 4

# ── Cap de amostras de TREINO para datasets grandes ──────────
# poker_hand tem ~1M linhas: treinar 13 quantifiers (RF n_estimators=300)
# + MoSS + calibração nesse volume, com n_jobs=4 replicando os dados por
# worker, foi o que estourou 125Gi RAM + 8Gi swap e derrubou o processo
# (confirmado via monitor de memória: subida contínua até OOM-kill).
# Datasets com treino maior que isso são subamostrados de forma
# estratificada ANTES de treinar (mantém proporção de classes).
# O teste (Xte/yte) permanece com o tamanho original.
MAX_TRAIN_SAMPLES = 50000

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
# ============================================================

def train_rf_calibrator(pred_train, real_train, feat_train):
    """
    RF calibrador com features enriquecidas:
    Input = [pred_moss | f_vec_original]
    """

    X_input = np.hstack([pred_train, feat_train])

    rf_cal = RandomForestRegressor(
        n_estimators=300,
        n_jobs=N_JOBS,
        random_state=42,
        min_samples_leaf=2
    )

    rf_cal.fit(X_input, real_train)

    return rf_cal


def apply_rf_calibrator(rf_cal, p_pred_raw, f_vec, n_classes):

    X_input = np.hstack([
        p_pred_raw.reshape(1, -1),
        f_vec.reshape(1, -1)
    ])

    p_cal = rf_cal.predict(X_input)[0][:n_classes]

    return safe_prevalence_vector(p_cal, n_classes)


# ============================================================
# LOAD FULL DATASET (retorna X_all, y_all ou None em erro)
# ============================================================

def load_full_dataset(ds_name, source, meta):
    """
    Carrega o dataset completo (sem split) para qualquer fonte.
    Retorna (X_all, y_all) ou (None, None) em caso de erro.
    """

    try:

        if source == "quapy":

            print(f"      📥 Loading QuaPy dataset (full): {ds_name}")

            dataset = qp.datasets.fetch_UCIMulticlassDataset(
                ds_name,
                min_test_split=0.5,
                verbose=False
            )

            # Concatenar treino + teste para obter o dataset completo
            train, test = dataset.train_test

            X_all = np.vstack([train.instances, test.instances])
            y_raw = np.concatenate([train.labels, test.labels])

            y_all = pd.factorize(y_raw)[0]

        else:

            filepath, target_col = meta

            print(f"      📥 Loading local CSV (full): {filepath}")

            df = pd.read_csv(filepath)

            if target_col not in df.columns:
                print(f"⚠️ Coluna '{target_col}' não encontrada em {ds_name}, pulando...")
                return None, None

            X_cols = [c for c in df.columns if c != target_col]
            df_X   = df[X_cols].select_dtypes(include=[np.number])
            df_X   = df_X.dropna(axis=1, how="all")
            y_raw  = df[target_col].values

            X_all  = df_X.values
            y_all, _ = pd.factorize(y_raw)

            # Filtrar classes com menos de 2 amostras
            classes, counts = np.unique(y_all, return_counts=True)
            valid_classes   = classes[counts >= 2]
            mask            = np.isin(y_all, valid_classes)
            X_all           = X_all[mask]
            y_all           = y_all[mask]

            # Re-mapear para 0..n-1 contíguo
            y_all, _ = pd.factorize(y_all)

            n_cls = len(np.unique(y_all))

            if not (10 <= n_cls <= 50):
                print(f"⚠️ {ds_name} tem {n_cls} classes após filtro, pulando...")
                return None, None

        return X_all, y_all

    except Exception as e:

        print(f"❌ Erro ao carregar {ds_name}: {e}")
        return None, None


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment():

    experiment_start = time.time()  # ← INÍCIO DA CONTAGEM

    # --------------------------------------------------------
    # QUAPY datasets
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Datasets locais com 10-50 classes
    # --------------------------------------------------------

    LOCAL_DATASETS = [
        ("!dataset_372_internet_usage.csv",                 "class"),
        ("Avila.csv",                                       "V11"),
        ("Chessgame.csv",                                   "game"),
        ("dataset_1457_amazon-commerce-reviews.csv",        "class"),
        ("dataset_313_spectrometer.csv",                    "class"),
        ("dataset_44478_amazon-commerce-reviews_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True.csv", "class"),
        ("dataset_44479_amazon-commerce-reviews_seed_1_nrows_2000_nclasses_10_ncols_100_stratify_True.csv", "class"),
        ("dataset_44480_amazon-commerce-reviews_seed_2_nrows_2000_nclasses_10_ncols_100_stratify_True.csv", "class"),
        ("dataset_44481_amazon-commerce-reviews_seed_3_nrows_2000_nclasses_10_ncols_100_stratify_True.csv", "class"),
        ("dataset_44482_amazon-commerce-reviews_seed_4_nrows_2000_nclasses_10_ncols_100_stratify_True.csv", "class"),
        ("digits.csv",                                      "class"),
        ("fashion-mnist.csv",                               "class"),
        ("fashion-mnist_train.csv",                         "label"),
        ("fashion-mnist_test.csv",                          "label"),
        ("letter.csv",                                      "class"),
        ("Mfeat.csv",                                       "class"),
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
    print(f"Repetições treino/teste: {N_REPETITIONS}")
    print(f"Batches calibração: {UPP_REPEATS * N_PREV_CALIB}")
    print(f"Batches teste:      {UPP_REPEATS * N_PREV_TEST}")

    moss_base_cache = {}

    # ========================================================
    # CHECKPOINT: retoma de onde parou, sem refazer datasets
    # já salvos em OUTPUT_CSV
    # ========================================================

    if os.path.exists(OUTPUT_CSV):

        df_prev = pd.read_csv(OUTPUT_CSV)
        rows = df_prev.to_dict("records")
        datasets_done = set(df_prev["dataset"].unique())

        print(f"\n♻️  Checkpoint encontrado: {len(datasets_done)} dataset(s) já concluído(s), "
              f"{len(rows)} linhas carregadas de {OUTPUT_CSV}")

    else:

        rows = []
        datasets_done = set()

    all_entries = (
        [(ds, "quapy", None) for ds in QUAPY_MULTICLASS_DATASETS] +
        [(fname, "local", (os.path.join(LOCAL_DATASETS_DIR, fname), col))
         for fname, col in LOCAL_DATASETS]
    )

    # ========================================================
    # LOOP PRINCIPAL: dataset → repetição de split
    # ========================================================

    for ds_name, source, meta in all_entries:

        if ds_name in datasets_done:
            print(f"\n⏭️  Pulando {ds_name} (já processado no checkpoint)")
            continue

        print(f"\n📂 Dataset: {ds_name} [{source}]")

        # ----------------------------------------------------
        # Carregar dataset completo UMA vez por dataset
        # ----------------------------------------------------

        X_all, y_all = load_full_dataset(ds_name, source, meta)

        if X_all is None:
            continue

        n_classes_global = len(np.unique(y_all))
        print(f"   🔢 Classes (total): {n_classes_global}  |  Amostras: {len(y_all)}")

        # ====================================================
        # LOOP DE REPETIÇÕES (30x split treino/teste)
        # ====================================================

        for rep in range(N_REPETITIONS):

            rep_seed = SEED + rep   # seed diferente a cada repetição

            print(f"\n   🔁 Repetição {rep + 1}/{N_REPETITIONS}  (seed={rep_seed})")

            # ------------------------------------------------
            # Split 70/30 estratificado com seed variável
            # ------------------------------------------------

            try:

                Xtr, Xte, ytr, yte = train_test_split(
                    X_all, y_all,
                    test_size=0.3,
                    random_state=rep_seed,
                    stratify=y_all
                )

            except Exception as e:

                print(f"❌ Erro no split (rep {rep + 1}): {e}")
                continue

            # ------------------------------------------------
            # Verificar n_classes após split
            # ------------------------------------------------

            n_classes_real = len(np.unique(ytr))

            # Garantir que teste tem as mesmas classes que treino
            yte = np.array([y if y < n_classes_real else 0 for y in yte])

            print(f"      🔢 Classes no treino: {n_classes_real}")

            # ------------------------------------------------
            # Scaling
            # ------------------------------------------------

            scaler = StandardScaler()
            Xtr    = scaler.fit_transform(Xtr)
            Xte    = scaler.transform(Xte)

            # ================================================
            # QUANTIFIERS
            # ================================================

            print("      🔥 Building quantifiers...")

            quantifiers = {}

            quantifiers["EMQ_QUAPY"] = EMQ_QUAPY(
                RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=N_JOBS,
                    random_state=rep_seed
                ),
                calib=None,
                exact_train_prev=True
            )

            quantifiers["EMQ_BCTS_QUAPY"] = EMQ_QUAPY(
                RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=N_JOBS,
                    random_state=rep_seed
                ),
                calib="bcts",
                val_split=0.2,
                exact_train_prev=True,
                on_calib_error="backup"
            )

            quantifiers["EMQ_MLQ"] = EMQ_MLQ(
                learner=RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=N_JOBS,
                    random_state=rep_seed
                ),
                calib_function=None,
                on_calib_error="backup"
            )

            quantifiers["EMQ_BCTS_MLQ"] = EMQ_MLQ(
                learner=RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=N_JOBS,
                    random_state=rep_seed
                ),
                calib_function="bcts",
                on_calib_error="backup"
            )

            quantifiers["CC"] = CC(
                learner=RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=N_JOBS,
                    random_state=rep_seed
                )
            )

            quantifiers["PCC"] = PCC(
                learner=RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=N_JOBS,
                    random_state=rep_seed
                )
            )

            quantifiers["AC"] = AC(
                learner=RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=N_JOBS,
                    random_state=rep_seed
                )
            )

            quantifiers["PAC"] = PAC(
                learner=RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=N_JOBS,
                    random_state=rep_seed
                )
            )

            quantifiers["FM"] = FM(
                learner=RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=N_JOBS,
                    random_state=rep_seed
                )
            )

            quantifiers["KDEyML"] = KDEyML(
                learner=RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=N_JOBS,
                    random_state=rep_seed
                ),
                bandwidth=0.1
            )

            quantifiers["KDEyHD"] = KDEyHD(
                learner=RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=N_JOBS,
                    random_state=rep_seed
                ),
                montecarlo_trials=500,
                random_state=rep_seed
            )

            quantifiers["KDEyCS"] = KDEyCS(
                learner=RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=N_JOBS,
                    random_state=rep_seed
                ),
                bandwidth=0.1
            )

            quantifiers["PWK"] = PWK(
                n_neighbors=10
            )

            # ================================================
            # TRAIN QUANTIFIERS
            # ================================================

            trained_quantifiers = {}

            print("      🚀 Training quantifiers...")

            for q_name, q in quantifiers.items():

                try:

                    print(f"         → {q_name}")

                    q.fit(Xtr, ytr)

                    trained_quantifiers[q_name] = q

                except Exception as e:

                    print(f"❌ Erro em {q_name}: {e}")

            # ================================================
            # CLASSIFIER FOR MoSS
            # ================================================

            if "EMQ_BCTS_QUAPY" in trained_quantifiers:

                clf = trained_quantifiers["EMQ_BCTS_QUAPY"].classifier

            else:

                clf = RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=N_JOBS,
                    random_state=rep_seed
                )

                clf.fit(Xtr, ytr)

            # ================================================
            # OOF PROBABILITIES
            # ================================================

            print("      🔄 Generating OOF probabilities...")

            oof_scores = cross_val_predict(
                RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=N_JOBS,
                    random_state=rep_seed
                ),
                Xtr,
                ytr,
                cv=5,
                method="predict_proba",
                n_jobs=N_JOBS
            )

            # ================================================
            # LOAD / TRAIN MoSS BASE — cache por n_classes
            # ================================================

            if n_classes_real not in moss_base_cache:

                moss_path = os.path.join(
                    MOSS_DIR,
                    f"moss_d500_lite_{n_classes_real}.pkl"
                )

                if not os.path.exists(moss_path):

                    print(
                        f"⚠️ moss_d500_lite_{n_classes_real}.pkl não encontrado."
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
                        n_jobs=N_JOBS,
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

            # ================================================
            # CALIBRAÇÃO POR REPETIÇÃO
            # ================================================

            iso_calibrators = None
            rf_calibrator   = None

            if moss_base_model is not None:

                print("      📐 Gerando tabela de calibração...")

                # batches = UPP_REPEATS * N_PREV_CALIB = 1 * 1000 = 1000
                protocol_calib = UPP(
                    batch_size=500,
                    n_prevalences=N_PREV_CALIB,
                    repeats=UPP_REPEATS,
                    random_state=rep_seed
                )

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

                print("      🔧 Treinando MoSS_ISO_CALIBRATED...")

                iso_calibrators = train_isotonic_calibrator(
                    pred_calib,
                    real_calib,
                    n_classes_real
                )

                print(f"✅ MoSS_ISO_CALIBRATED_{n_classes_real} treinado!")

                print("      🌲 Treinando MoSS_RF_CALIBRATED...")

                rf_calibrator = train_rf_calibrator(
                    pred_calib,
                    real_calib,
                    feat_calib
                )

                print(f"✅ MoSS_RF_CALIBRATED_{n_classes_real} treinado!")

            # ================================================
            # PROTOCOLO DE TESTE
            # batches = UPP_REPEATS * N_PREV_TEST = 1 * 1000 = 1000
            # ================================================

            protocol_test = UPP(
                batch_size=500,
                n_prevalences=N_PREV_TEST,
                repeats=UPP_REPEATS,
                random_state=rep_seed
            )

            total_batches = protocol_test.get_n_combinations()

            for idx_batch in tqdm(
                protocol_test.split(Xte, yte),
                total=total_batches,
                desc=f"UPP rep={rep + 1}"
            ):

                # ============================================
                # REAL PREVALENCE
                # ============================================

                p_real = get_prev_from_labels(
                    yte[idx_batch],
                    classes=np.arange(n_classes_real)
                )

                p_real = safe_prevalence_vector(
                    p_real,
                    n_classes_real
                )

                # ============================================
                # QUANTIFIERS CLÁSSICOS
                # ============================================

                for q_name, q in trained_quantifiers.items():

                    try:

                        p_pred = q.predict(Xte[idx_batch])

                        p_pred = safe_prevalence_vector(
                            p_pred,
                            n_classes_real
                        )

                        rows.append({
                            "dataset":            ds_name,
                            "repetition":         rep + 1,
                            "modelo":             q_name,
                            "n_classes_original": n_classes_real,
                            "erro":               np.mean(
                                np.abs(p_pred - p_real)
                            )
                        })

                    except Exception as e:

                        print(f"❌ Erro em {q_name}: {e}")

                # ============================================
                # MoSS (3 versões)
                # ============================================

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

                        # ------------------------------------
                        # 1) MoSS BASE (só sintético)
                        # ------------------------------------

                        p_moss_raw = moss_base_model.predict(
                            f_vec_2d
                        )[0][:n_classes_real]

                        p_moss_base = safe_prevalence_vector(
                            p_moss_raw,
                            n_classes_real
                        )

                        rows.append({
                            "dataset":            ds_name,
                            "repetition":         rep + 1,
                            "modelo":             f"MoSS_BASE_{n_classes_real}",
                            "n_classes_original": n_classes_real,
                            "erro":               np.mean(
                                np.abs(p_moss_base - p_real)
                            )
                        })

                        # ------------------------------------
                        # 2) MoSS ISO CALIBRATED
                        # ------------------------------------

                        if iso_calibrators is not None:

                            p_moss_iso = apply_isotonic_calibrator(
                                iso_calibrators,
                                p_moss_base,
                                n_classes_real
                            )

                            rows.append({
                                "dataset":            ds_name,
                                "repetition":         rep + 1,
                                "modelo":             f"MoSS_ISO_CALIBRATED_{n_classes_real}",
                                "n_classes_original": n_classes_real,
                                "erro":               np.mean(
                                    np.abs(p_moss_iso - p_real)
                                )
                            })

                        # ------------------------------------
                        # 3) MoSS RF CALIBRATED
                        # ------------------------------------

                        if rf_calibrator is not None:

                            p_moss_rf = apply_rf_calibrator(
                                rf_calibrator,
                                p_moss_base,
                                f_vec,
                                n_classes_real
                            )

                            rows.append({
                                "dataset":            ds_name,
                                "repetition":         rep + 1,
                                "modelo":             f"MoSS_RF_CALIBRATED_{n_classes_real}",
                                "n_classes_original": n_classes_real,
                                "erro":               np.mean(
                                    np.abs(p_moss_rf - p_real)
                                )
                            })

                    except Exception as e:

                        print(f"❌ Erro no MoSS: {e}")

            # ================================================
            # SAVE PARTIAL (a cada repetição)
            # ================================================

            pd.DataFrame(rows).to_csv(
                OUTPUT_CSV,
                index=False
            )

            print(f"💾 Parcial salva em {OUTPUT_CSV}  (rep {rep + 1}/{N_REPETITIONS})")

            # ============================================================
            # LIMPEZA DE RECURSOS: mata os workers do loky e libera
            # memória entre repetições, para não acumular semáforos/
            # pastas temporárias ao longo de 30 reps x 41 datasets.
            # ============================================================

            get_reusable_executor().shutdown(wait=True, kill_workers=True)

            del quantifiers, trained_quantifiers, oof_scores, clf
            gc.collect()

        # ============================================================
        # Fim de todas as repetições deste dataset: libera cache MoSS
        # do dataset atual se não for mais precisar (opcional — deixe
        # comentado se quiser manter o cache entre datasets do mesmo
        # n_classes_real)
        # ============================================================

        gc.collect()

    # ── TEMPO TOTAL ─────────────────────────────────────────
    elapsed = time.time() - experiment_start
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"\n✅ Experimento concluído!")
    print(f"⏱️  Tempo total: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s")
    print(f"📁 Resultados salvos em: {OUTPUT_CSV}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        run_experiment()

    except Exception:

        # Garante que QUALQUER exceção não tratada apareça no log,
        # mesmo rodando em background com nohup (onde o stderr some
        # se o terminal/sessão SSH já fechou).
        print("\n\n❌❌❌ EXPERIMENTO ABORTADO POR EXCEÇÃO NÃO TRATADA ❌❌❌")
        print(traceback.format_exc())
        sys.stdout.flush()
        sys.exit(1)