import os
import sys
import functools
import traceback

# ── Força flush imediato em todo print() do módulo ──
print = functools.partial(print, flush=True)

# ── TMPDIR do joblib ──
_home_dir = os.path.expanduser("~")
_joblib_tmp = os.path.join(_home_dir, "tmp_joblib")

try:
    os.makedirs(_joblib_tmp, exist_ok=True)
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

LOCAL_DATASETS_DIR = "/var/new_homes/julio/mestrado/mestrado-dyssyn/experiments/datasets_multiclass"

OUTPUT_CSV = "results.csv"

# ── DIAGNÓSTICO: desliga MoSS_BASE, ISO, RF_CALIBRATED e HYBRID
# inteiros (treino + predição), incluindo o custo de OOF/clf que
# só existe para alimentar o MoSS. Com False, o CSV vai ter só os
# quantifiers clássicos — compare o tempo por repetição/dataset
# contra uma rodada anterior com True para isolar o gargalo.
# Lembre de voltar para True depois do teste, ou vai faltar MoSS
# nos seus resultados finais.
RUN_MOSS_MODELS = False

# ── Número de repetições do loop treino/teste ──────────────
N_REPETITIONS = 30

# ── UPP: repeats=1; n_prevalences calculado para ≥1000 batches
UPP_REPEATS        = 1
N_PREV_CALIB       = 1000
N_PREV_TEST        = 1000

# ── Paralelismo por modelo. ──
N_JOBS = 4

# ── Cap de amostras de TREINO para datasets grandes ──────────
MAX_TRAIN_SAMPLES = 50000

HYBRID_REAL_WEIGHT = 5

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
# HYBRID CALIBRATOR
# ============================================================

def train_hybrid_calibrator(X_m, y_m, feat_calib, real_calib, rep_seed):

    if X_m is None or len(X_m) == 0:
        return None

    X_final = np.vstack([X_m, feat_calib])
    y_final = np.vstack([y_m, real_calib])

    weights = np.concatenate([
        np.ones(len(X_m)),
        np.ones(len(feat_calib)) * HYBRID_REAL_WEIGHT
    ])

    hybrid = RandomForestRegressor(
        n_estimators=300,
        n_jobs=N_JOBS,
        random_state=rep_seed,
        min_samples_leaf=2
    )

    hybrid.fit(X_final, y_final, sample_weight=weights)

    return hybrid


def apply_hybrid_calibrator(hybrid, f_vec, n_classes):

    p_hybrid = hybrid.predict(
        f_vec.reshape(1, -1)
    )[0][:n_classes]

    return safe_prevalence_vector(p_hybrid, n_classes)


# ============================================================
# LOAD FULL DATASET
# ============================================================

def load_full_dataset(ds_name, source, meta):

    try:

        if source == "quapy":

            print(f"      📥 Loading QuaPy dataset (full): {ds_name}")

            dataset = qp.datasets.fetch_UCIMulticlassDataset(
                ds_name,
                min_test_split=0.5,
                verbose=False
            )

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

            classes, counts = np.unique(y_all, return_counts=True)
            valid_classes   = classes[counts >= 2]
            mask            = np.isin(y_all, valid_classes)
            X_all           = X_all[mask]
            y_all           = y_all[mask]

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
# GET DATASET SIZE
# ============================================================

def get_dataset_size(ds_name, source, meta):

    try:

        if source == "quapy":

            dataset = qp.datasets.fetch_UCIMulticlassDataset(
                ds_name,
                min_test_split=0.5,
                verbose=False
            )

            train, test = dataset.train_test
            n = len(train.labels) + len(test.labels)

            del dataset, train, test
            gc.collect()

            return n

        else:

            filepath, _target_col = meta

            if not os.path.exists(filepath):
                return None

            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                n = sum(1 for _ in f) - 1

            return max(n, 0)

    except Exception as e:

        print(f"⚠️ Não consegui medir tamanho de {ds_name}: {e}")
        return None


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment():

    experiment_start = time.time()

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
    print(f"🔬 RUN_MOSS_MODELS = {RUN_MOSS_MODELS} (diagnóstico de lentidão)")

    moss_base_cache = {}

    # ========================================================
    # CHECKPOINT
    # ========================================================

    if os.path.exists(OUTPUT_CSV):

        df_prev = pd.read_csv(OUTPUT_CSV)

        reps_done_por_dataset = df_prev.groupby("dataset")["repetition"].nunique()

        datasets_done = set(
            reps_done_por_dataset[reps_done_por_dataset >= N_REPETITIONS].index
        )

        reps_completed = set(zip(df_prev["dataset"], df_prev["repetition"]))

        print(f"\n♻️  Checkpoint encontrado: {len(datasets_done)} dataset(s) completo(s) "
            f"({N_REPETITIONS} reps), {len(df_prev)} linhas já salvas em {OUTPUT_CSV}")

        header_written = True
        del df_prev

    else:

        datasets_done = set()
        reps_completed = set()
        header_written = False

    # ========================================================
    # ORDENAR DATASETS POR TAMANHO (menor → maior)
    # ========================================================

    all_entries_unsorted = (
        [(ds, "quapy", None) for ds in QUAPY_MULTICLASS_DATASETS] +
        [(fname, "local", (os.path.join(LOCAL_DATASETS_DIR, fname), col))
         for fname, col in LOCAL_DATASETS]
    )

    print("\n📏 Medindo tamanho de cada dataset (para ordenar do menor pro maior)...")

    sized_entries = []

    for ds_name, source, meta in all_entries_unsorted:

        if ds_name in datasets_done:
            sized_entries.append((-1, ds_name, source, meta))
            continue

        n = get_dataset_size(ds_name, source, meta)

        if n is None:
            print(f"   ⚠️  {ds_name}: tamanho desconhecido, será processado por último")
            n = float("inf")
        else:
            print(f"   📐 {ds_name}: {n} amostras")

        sized_entries.append((n, ds_name, source, meta))

    sized_entries.sort(key=lambda x: x[0])

    all_entries = [(ds_name, source, meta) for (_, ds_name, source, meta) in sized_entries]

    print("\n===== ORDEM DE PROCESSAMENTO (menor → maior) =====")
    for n, ds_name, source, meta in sized_entries:
        if ds_name in datasets_done:
            tamanho_str = "já concluído no checkpoint"
        elif n == float("inf"):
            tamanho_str = "desconhecido"
        else:
            tamanho_str = f"{n} amostras"
        print(f"   {ds_name} [{source}] — {tamanho_str}")

    # ========================================================
    # LOOP PRINCIPAL
    # ========================================================

    for ds_name, source, meta in all_entries:

        if ds_name in datasets_done:
            print(f"\n⏭️  Pulando {ds_name} (já processado no checkpoint)")
            continue

        print(f"\n📂 Dataset: {ds_name} [{source}]")

        X_all, y_all = load_full_dataset(ds_name, source, meta)

        if X_all is None:
            continue

        n_classes_global = len(np.unique(y_all))
        print(f"   🔢 Classes (total): {n_classes_global}  |  Amostras: {len(y_all)}")

        for rep in range(N_REPETITIONS):

            rep_seed = SEED + rep

            if (ds_name, rep + 1) in reps_completed:
                print(f"      ⏭️  Rep {rep + 1} de {ds_name} já no checkpoint, pulando")
                continue

            print(f"\n   🔁 Repetição {rep + 1}/{N_REPETITIONS}  (seed={rep_seed})")

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

            n_classes_real = len(np.unique(ytr))

            yte = np.array([y if y < n_classes_real else 0 for y in yte])

            print(f"      🔢 Classes no treino: {n_classes_real}")

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
            # BLOCO INTEIRO DO MoSS (BASE/ISO/RF/HYBRID) —
            # só roda se RUN_MOSS_MODELS=True
            # ================================================

            clf               = None
            oof_scores        = None
            moss_base_model   = None
            X_m               = None
            y_m               = None
            iso_calibrators   = None
            rf_calibrator     = None
            hybrid_calibrator = None

            if RUN_MOSS_MODELS:

                # ------------------------------------------------
                # CLASSIFIER FOR MoSS
                # ------------------------------------------------

                if "EMQ_BCTS_QUAPY" in trained_quantifiers:

                    clf = trained_quantifiers["EMQ_BCTS_QUAPY"].classifier

                else:

                    clf = RandomForestClassifier(
                        n_estimators=300,
                        n_jobs=N_JOBS,
                        random_state=rep_seed
                    )

                    clf.fit(Xtr, ytr)

                # ------------------------------------------------
                # OOF PROBABILITIES
                # ------------------------------------------------

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

                # ------------------------------------------------
                # LOAD / TRAIN MoSS BASE — cache por n_classes
                # ------------------------------------------------

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

                        X_m_local = []
                        y_m_local = []

                        for (alpha_prev, _), curves in synthetic_distributions.items():

                            for scores_matrix in curves:

                                feat = baseline_features(
                                    scores_matrix,
                                    n_classes_real
                                )

                                X_m_local.append(feat)
                                y_m_local.append(list(alpha_prev))

                        X_m_local = np.array(X_m_local)
                        y_m_local = np.array(y_m_local)

                        print(f"      🧪 Synthetic samples: {len(X_m_local)}")

                        moss_base_model_local = RandomForestRegressor(
                            n_estimators=300,
                            n_jobs=N_JOBS,
                            random_state=SEED,
                            min_samples_leaf=2
                        )

                        moss_base_model_local.fit(X_m_local, y_m_local)

                        print(f"✅ MoSS_BASE_{n_classes_real} treinado!")

                        moss_base_cache[n_classes_real] = {
                            "model": moss_base_model_local,
                            "X_m": X_m_local,
                            "y_m": y_m_local
                        }

                moss_entry      = moss_base_cache[n_classes_real]
                moss_base_model = moss_entry["model"]
                X_m             = moss_entry["X_m"]
                y_m             = moss_entry["y_m"]

                # ------------------------------------------------
                # CALIBRAÇÃO POR REPETIÇÃO
                # ------------------------------------------------

                if moss_base_model is not None:

                    print("      📐 Gerando tabela de calibração...")

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

                    print("      🌿 Treinando MoSS_HYBRID...")

                    hybrid_calibrator = train_hybrid_calibrator(
                        X_m,
                        y_m,
                        feat_calib,
                        real_calib,
                        rep_seed
                    )

                    if hybrid_calibrator is not None:
                        print(f"✅ MoSS_HYBRID_{n_classes_real} treinado!")
                    else:
                        print(f"⚠️ MoSS_HYBRID_{n_classes_real} não treinado (sem X_m).")

            else:

                print("      ⏭️  RUN_MOSS_MODELS=False — pulando classificador extra, OOF e todo o bloco MoSS")

            # ================================================
            # PROTOCOLO DE TESTE
            # ================================================

            rep_rows = []

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

                p_real = get_prev_from_labels(
                    yte[idx_batch],
                    classes=np.arange(n_classes_real)
                )

                p_real = safe_prevalence_vector(
                    p_real,
                    n_classes_real
                )

                for q_name, q in trained_quantifiers.items():

                    try:

                        p_pred = q.predict(Xte[idx_batch])

                        p_pred = safe_prevalence_vector(
                            p_pred,
                            n_classes_real
                        )

                        rep_rows.append({
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
                # MoSS (4 versões) — só se RUN_MOSS_MODELS=True
                # ============================================

                if RUN_MOSS_MODELS and moss_base_model is not None:

                    try:

                        scores_batch = clf.predict_proba(
                            Xte[idx_batch]
                        )

                        f_vec = baseline_features(
                            scores_batch,
                            n_classes_real
                        )

                        f_vec_2d = f_vec.reshape(1, -1)

                        p_moss_raw = moss_base_model.predict(
                            f_vec_2d
                        )[0][:n_classes_real]

                        p_moss_base = safe_prevalence_vector(
                            p_moss_raw,
                            n_classes_real
                        )

                        rep_rows.append({
                            "dataset":            ds_name,
                            "repetition":         rep + 1,
                            "modelo":             f"MoSS_BASE_{n_classes_real}",
                            "n_classes_original": n_classes_real,
                            "erro":               np.mean(
                                np.abs(p_moss_base - p_real)
                            )
                        })

                        if iso_calibrators is not None:

                            p_moss_iso = apply_isotonic_calibrator(
                                iso_calibrators,
                                p_moss_base,
                                n_classes_real
                            )

                            rep_rows.append({
                                "dataset":            ds_name,
                                "repetition":         rep + 1,
                                "modelo":             f"MoSS_ISO_CALIBRATED_{n_classes_real}",
                                "n_classes_original": n_classes_real,
                                "erro":               np.mean(
                                    np.abs(p_moss_iso - p_real)
                                )
                            })

                        if rf_calibrator is not None:

                            p_moss_rf = apply_rf_calibrator(
                                rf_calibrator,
                                p_moss_base,
                                f_vec,
                                n_classes_real
                            )

                            rep_rows.append({
                                "dataset":            ds_name,
                                "repetition":         rep + 1,
                                "modelo":             f"MoSS_RF_CALIBRATED_{n_classes_real}",
                                "n_classes_original": n_classes_real,
                                "erro":               np.mean(
                                    np.abs(p_moss_rf - p_real)
                                )
                            })

                        if hybrid_calibrator is not None:

                            p_moss_hybrid = apply_hybrid_calibrator(
                                hybrid_calibrator,
                                f_vec,
                                n_classes_real
                            )

                            rep_rows.append({
                                "dataset":            ds_name,
                                "repetition":         rep + 1,
                                "modelo":             f"MoSS_HYBRID_{n_classes_real}",
                                "n_classes_original": n_classes_real,
                                "erro":               np.mean(
                                    np.abs(p_moss_hybrid - p_real)
                                )
                            })

                    except Exception as e:

                        print(f"❌ Erro no MoSS: {e}")

            # ================================================
            # SAVE PARTIAL
            # ================================================

            pd.DataFrame(rep_rows).to_csv(
                OUTPUT_CSV,
                mode="a",
                header=not header_written,
                index=False
            )
            header_written = True

            reps_completed.add((ds_name, rep + 1))

            print(f"💾 Parcial salva em {OUTPUT_CSV}  (rep {rep + 1}/{N_REPETITIONS})")

            get_reusable_executor().shutdown(wait=True, kill_workers=True)

            del quantifiers, trained_quantifiers, oof_scores, clf
            del iso_calibrators, rf_calibrator, hybrid_calibrator
            gc.collect()

        gc.collect()

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

        print("\n\n❌❌❌ EXPERIMENTO ABORTADO POR EXCEÇÃO NÃO TRATADA ❌❌❌")
        print(traceback.format_exc())
        sys.stdout.flush()
        sys.exit(1)