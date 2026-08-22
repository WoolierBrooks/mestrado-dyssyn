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
    RandomForestClassifier,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

# ============================================================
# MLQuantify — mantido só pelo protocolo UPP e utilitário de
# prevalência, que são usados para gerar os batches (não mexemos
# nos batches, então essas duas dependências continuam).
# ============================================================

from mlquantify.model_selection import UPP
from mlquantify.utils import get_prev_from_labels

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

SEED = 42

MOSS_DIR = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/moss/multiclass"

DATASETS_ROOT = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/multiclass"

LOCAL_DATASETS_DIR = "/var/new_homes/julio/mestrado/mestrado-dyssyn/experiments/datasets_multiclass"

OUTPUT_CSV = "results_moss_regressors.csv"

# ── Número de repetições do loop treino/teste (inalterado) ──
N_REPETITIONS = 30

# ── UPP: mesmos parâmetros de sempre, batches inalterados ──
UPP_REPEATS        = 1
N_PREV_CALIB       = 1000
N_PREV_TEST        = 1000

N_JOBS = 4

# ── Cap de amostras de TREINO para datasets grandes. Antes essa
# constante existia mas nunca era de fato aplicada — o código de
# subamostragem tinha se perdido. Agora ela é aplicada logo após
# o split (ver bloco "SUBAMOSTRAGEM" abaixo).
MAX_TRAIN_SAMPLES = 50000

# ── Estimators do MoSS base (etapa 1, sintética) e do calibrador
# RF (etapa 2) — mantidos em 300, como no script original.
QUANTIFIER_N_ESTIMATORS = 300

# ============================================================
# REGRESSORES A COMPARAR NO CALIBRADOR (etapa 2 do MoSS)
# ============================================================
# Cada entrada é uma função (rep_seed) -> estimator NÃO treinado.
# RF é o seu baseline atual; os demais são o que o orientador
# pediu para testar. Ajuste hiperparâmetros à vontade — estes são
# pontos de partida razoáveis, não valores "oficiais" de protocolo.
#
# Nota sobre multi-output: o target da calibração é um vetor de
# n_classes por amostra. RF, ExtraTrees, KNN, Ridge e MLP suportam
# multi-output nativamente no sklearn. GradientBoosting e SVR não
# suportam — por isso vêm envolvidos em MultiOutputRegressor, que
# treina um regressor independente por classe.
# ============================================================

def make_rf(seed):
    return RandomForestRegressor(
        n_estimators=QUANTIFIER_N_ESTIMATORS,
        n_jobs=N_JOBS,
        random_state=seed,
        min_samples_leaf=2
    )

def make_extratrees(seed):
    return ExtraTreesRegressor(
        n_estimators=QUANTIFIER_N_ESTIMATORS,
        n_jobs=N_JOBS,
        random_state=seed,
        min_samples_leaf=2
    )

def make_gbr(seed):
    return MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=200,
            random_state=seed
        ),
        n_jobs=N_JOBS
    )

def make_knn(seed):
    return KNeighborsRegressor(
        n_neighbors=10,
        n_jobs=N_JOBS
    )

def make_ridge(seed):
    return Ridge(
        alpha=1.0,
        random_state=seed
    )

def make_mlp(seed):
    return MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=seed
    )

def make_svr(seed):
    return MultiOutputRegressor(
        SVR(kernel="rbf", C=1.0),
        n_jobs=N_JOBS
    )

# Nome do modelo → factory. O nome vira parte da coluna "modelo"
# no CSV, no formato "MoSS_<nome>_CALIBRATED_<n_classes>", igual
# ao padrão que seu script de plot já espera (só precisa adicionar
# os novos nomes no regex de unificação do plot_boxplots.py depois).
REGRESSOR_FACTORIES = {
    "RF":         make_rf,
    "ExtraTrees": make_extratrees,
    "GBR":        make_gbr,
    "KNN":        make_knn,
    "Ridge":      make_ridge,
    "MLP":        make_mlp,
    "SVR":        make_svr,
}

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
# CALIBRATION TABLE (etapa que gera os dados reais para o
# calibrador aprender a corrigir o MoSS base — inalterado)
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
# CALIBRADOR GENÉRICO (etapa 2 do MoSS_RF, parametrizável por
# tipo de regressor)
#
# FIX: normalização do input. O input concatena p_pred (0-1) com
# f_vec (médias/variâncias/skew/kurtosis, escalas bem diferentes
# entre si). RF/ExtraTrees/GBR são invariantes a escala, mas
# Ridge/KNN/SVR/MLP não são — sem normalizar, esses regressores
# ficam artificialmente prejudicados e a comparação deixa de ser
# justa. O StandardScaler é treinado só nos dados de CALIBRAÇÃO
# (nunca no teste, para não vazar informação) e reaplicado dentro
# de apply_calibrator na hora de prever.
# ============================================================

def train_calibrator(regressor_factory, pred_train, real_train, feat_train, rep_seed):
    """
    Input:  [p_pred_raw (predição bruta do MoSS base) | f_vec (estatísticas originais)]
    Target: prevalência real do batch de calibração
    """

    X_input = np.hstack([pred_train, feat_train])

    input_scaler = StandardScaler()
    X_input_scaled = input_scaler.fit_transform(X_input)

    model = regressor_factory(rep_seed)

    model.fit(X_input_scaled, real_train)

    return {
        "model": model,
        "scaler": input_scaler
    }


def apply_calibrator(calibrator_entry, p_pred_raw, f_vec, n_classes):

    model  = calibrator_entry["model"]
    scaler = calibrator_entry["scaler"]

    X_input = np.hstack([
        p_pred_raw.reshape(1, -1),
        f_vec.reshape(1, -1)
    ])

    X_input_scaled = scaler.transform(X_input)

    p_cal = np.array(model.predict(X_input_scaled)[0])[:n_classes]

    return safe_prevalence_vector(p_cal, n_classes)


# ============================================================
# LOAD FULL DATASET (inalterado)
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
# GET DATASET SIZE (inalterado)
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
    print(f"Regressores testados no calibrador: {list(REGRESSOR_FACTORIES.keys())}")
    print(f"MAX_TRAIN_SAMPLES (agora aplicado de fato): {MAX_TRAIN_SAMPLES}")

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
    # ORDENAR DATASETS POR TAMANHO (menor → maior) — inalterado
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

            # ------------------------------------------------
            # Split 70/30 — MESMA seed, MESMO resultado do
            # script original (batches de TESTE preservados)
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

            n_classes_real = len(np.unique(ytr))

            yte = np.array([y if y < n_classes_real else 0 for y in yte])

            print(f"      🔢 Classes no treino: {n_classes_real}")

            # ------------------------------------------------
            # FIX: SUBAMOSTRAGEM DE TREINO (MAX_TRAIN_SAMPLES)
            # A constante existia mas nunca era aplicada — o cap
            # contra OOM em datasets grandes (poker_hand, ~1M
            # linhas) era só decorativo. Aplicado aqui, ANTES do
            # scaling e de qualquer treino, sobre Xtr/ytr apenas
            # (Xte/yte permanecem no tamanho original, como o
            # comentário original já dizia que deveria ser).
            # Estratificado para preservar proporção de classes,
            # com random_state=rep_seed para reprodutibilidade.
            # ------------------------------------------------

            if len(Xtr) > MAX_TRAIN_SAMPLES:

                print(f"      ✂️  Subamostrando treino: {len(Xtr)} → {MAX_TRAIN_SAMPLES} amostras")

                Xtr, _, ytr, _ = train_test_split(
                    Xtr, ytr,
                    train_size=MAX_TRAIN_SAMPLES,
                    random_state=rep_seed,
                    stratify=ytr
                )

            # ------------------------------------------------
            # Scaling
            # ------------------------------------------------

            scaler = StandardScaler()
            Xtr    = scaler.fit_transform(Xtr)
            Xte    = scaler.transform(Xte)

            # ================================================
            # CLASSIFIER FOR MoSS
            # ================================================

            print("      🔥 Treinando classificador base (RF) para o MoSS...")

            clf = RandomForestClassifier(
                n_estimators=QUANTIFIER_N_ESTIMATORS,
                n_jobs=N_JOBS,
                random_state=rep_seed
            )

            clf.fit(Xtr, ytr)

            # ================================================
            # OOF PROBABILITIES (inalterado: cv=5, 300 árvores)
            # ================================================

            print("      🔄 Generating OOF probabilities...")

            oof_scores = cross_val_predict(
                RandomForestClassifier(
                    n_estimators=QUANTIFIER_N_ESTIMATORS,
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
            # (etapa 1, inalterada: RF sobre dados sintéticos)
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
                        "model": None
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
                        n_estimators=QUANTIFIER_N_ESTIMATORS,
                        n_jobs=N_JOBS,
                        random_state=SEED,
                        min_samples_leaf=2
                    )

                    moss_base_model.fit(X_m, y_m)

                    print(f"✅ MoSS base treinado para {n_classes_real} classes!")

                    moss_base_cache[n_classes_real] = {
                        "model": moss_base_model
                    }

            moss_entry      = moss_base_cache[n_classes_real]
            moss_base_model = moss_entry["model"]

            # ================================================
            # CALIBRAÇÃO POR REPETIÇÃO — um calibrador treinado
            # por regressor candidato, todos na MESMA tabela de
            # calibração (mesmos batches de calibração, protocol
            # UPP idêntico ao original)
            # ================================================

            calibrators = {}

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

                for reg_name, reg_factory in REGRESSOR_FACTORIES.items():

                    print(f"      🌲 Treinando calibrador: {reg_name}...")

                    try:

                        calibrators[reg_name] = train_calibrator(
                            reg_factory,
                            pred_calib,
                            real_calib,
                            feat_calib,
                            rep_seed
                        )

                        print(f"✅ MoSS_{reg_name}_CALIBRATED_{n_classes_real} treinado!")

                    except Exception as e:

                        print(f"❌ Erro treinando calibrador {reg_name}: {e}")

            # ================================================
            # PROTOCOLO DE TESTE — MESMOS batches (UPP com o
            # mesmo random_state=rep_seed) do script original
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

                if moss_base_model is not None and len(calibrators) > 0:

                    try:

                        scores_batch = clf.predict_proba(
                            Xte[idx_batch]
                        )

                        f_vec = baseline_features(
                            scores_batch,
                            n_classes_real
                        )

                        p_moss_raw = moss_base_model.predict(
                            f_vec.reshape(1, -1)
                        )[0][:n_classes_real]

                        p_moss_base = safe_prevalence_vector(
                            p_moss_raw,
                            n_classes_real
                        )

                        # ------------------------------------
                        # MoSS BASE (sem calibração) — referência
                        # ------------------------------------

                        rep_rows.append({
                            "dataset":            ds_name,
                            "repetition":         rep + 1,
                            "modelo":             f"MoSS_BASE_{n_classes_real}",
                            "n_classes_original": n_classes_real,
                            "erro":               np.mean(
                                np.abs(p_moss_base - p_real)
                            )
                        })

                        for reg_name, calibrator_entry in calibrators.items():

                            try:

                                p_final = apply_calibrator(
                                    calibrator_entry,
                                    p_moss_base,
                                    f_vec,
                                    n_classes_real
                                )

                                rep_rows.append({
                                    "dataset":            ds_name,
                                    "repetition":         rep + 1,
                                    "modelo":             f"MoSS_{reg_name}_CALIBRATED_{n_classes_real}",
                                    "n_classes_original": n_classes_real,
                                    "erro":               np.mean(
                                        np.abs(p_final - p_real)
                                    )
                                })

                            except Exception as e:

                                print(f"❌ Erro aplicando calibrador {reg_name}: {e}")

                    except Exception as e:

                        print(f"❌ Erro no MoSS (features/base): {e}")

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

            del oof_scores, clf, calibrators
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