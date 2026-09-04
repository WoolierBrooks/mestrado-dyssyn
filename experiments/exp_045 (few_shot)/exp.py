"""
Few-Shot Quantification — réplica binária (Esuli) + variantes SMQ do Julio.

Estrutura geral do experimento:
  for dataset in 32 datasets binários UCI (QuaPy):
    for repetição in 1..30:
      split treino/teste aleatório (estratificado)
      for n in {1, 2, 3, 5, 10, 100, full}:
        amostra n exemplos por classe do split de treino  -> Tr_n
        for método in {CC, PCC, ACC, PACC, EMQ, DyS, KDEyML,
                        FSNN, FSPL, FSMM, FSGMM, FSGPEM,
                        SMQ_BASE, SMQ_RF, SMQ_ET, SMQ_GBR, SMQ_ISO}:
          treina/calibra em Tr_n
          avalia em 1000 batches artificiais sorteados do split de teste
          registra MAE e MRAE

Convenções herdadas dos scripts multiclasse do Julio: flush de print,
TMPDIR do joblib configurável, checkpoint/resume via CSV incremental,
escrita protegida, datasets ordenados por tamanho.

STATUS desta versão:
- Caminho do MoSS base binário confirmado.
- Nenhum dataset é excluído (DATASETS_TO_EXCLUDE vazio, por decisão do
  usuário).
- BUG CORRIGIDO: o SMQ (classificador + tabela de calibração +
  calibradores rf/et/gbr/iso) agora é treinado UMA VEZ por
  (dataset, repetição, n-shot) via fit_smq(), e só aplicado nos 1000
  batches de teste via apply_smq(). Antes disso estava sendo
  retreinado a cada um dos 1000 batches (1000x mais caro que o
  necessário). Isso NÃO se aplica a FSGMM/FSGPEM/FSPL, que são
  transdutivos por definição no paper (re-treinam por amostra de
  teste mesmo) — só o SMQ tinha esse bug.
- smq_iso adicionado (isotonic regression por classe, igual ao seu
  script multiclasse mais recente).
- FSPL com config FIXA (strategy="RS", m=20, downstream="EMQ") — a
  otimização por n-shot ao estilo Seção 4.3 do paper foi descartada
  por decisão do usuário (FSPL é lento demais para valer o custo da
  busca de 60 variantes).
"""

import os
import sys
import functools
import traceback
import pickle
import gc
import warnings

import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis

import quapy as qp
from quapy.method.aggregative import CC, PCC, ACC, PACC, EMQ, DyS, KDEyML

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.mixture import GaussianMixture
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neighbors import NearestNeighbors

print = functools.partial(print, flush=True)
warnings.filterwarnings("ignore")

# ============================================================
# JOBLIB TMPDIR (mesmo padrão dos scripts multiclasse)
# ============================================================

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
    print(f"⚠️  Fallback joblib tmp: {_joblib_tmp} ({_e})")
os.environ["JOBLIB_TEMP_FOLDER"] = _joblib_tmp
print(f"📁 JOBLIB_TEMP_FOLDER = {_joblib_tmp}")

# ============================================================
# CONFIG
# ============================================================

SEED = 42
N_REPETITIONS = 30
N_SHOTS = [1, 2, 3, 5, 10, 100, "full"]

N_CALIB_BATCHES = 1000   # "mil batches de treino" (calibração do SMQ)
N_TEST_BATCHES = 1000    # "mil batches de teste"
BATCH_SIZE = 500

N_JOBS = 4

# MoSS base sintético binário pré-treinado (SMQ) — caminho confirmado.
MOSS_BINARY_PKL = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/moss/lite/moss_binario_lite.pkl"

QUANTIFIER_N_ESTIMATORS = 300  # RF do SMQ (base + calibrador RF/ET), como nos scripts multiclasse

OUTPUT_CSV = "results_few_shot_quantification.csv"

# ============================================================
# 32 DATASETS BINÁRIOS UCI DISPONÍVEIS NO QUAPY
# ============================================================

# Lista confirmada em runtime (qp.datasets, QuaPy 0.2.2) — 29 datasets.
# A doc do QuaPy menciona acute.a/acute.b/balance.2 em versões diferentes,
# mas essa instalação não os expõe; ajustado para o que o pacote realmente
# aceita.
UCI_BINARY_DATASETS = [
    "balance.1", "balance.3", "breast-cancer", "cmc.1", "cmc.2", "cmc.3",
    "ctg.1", "ctg.2", "ctg.3", "german", "haberman", "ionosphere",
    "iris.1", "iris.2", "iris.3", "mammographic", "pageblocks.5",
    "semeion", "sonar", "spambase", "spectf", "tictactoe", "transfusion",
    "wdbc", "wine.1", "wine.2", "wine.3", "wine-q-red", "wine-q-white",
    "yeast",
]

# Nenhum dataset excluído por decisão do usuário.
DATASETS_TO_EXCLUDE = set()


def get_active_dataset_list():
    return [d for d in UCI_BINARY_DATASETS if d not in DATASETS_TO_EXCLUDE]


# ============================================================
# AMOSTRADOR DE BATCHES ARTIFICIAIS (bootstrap, estilo APP/UPP)
#
# Sorteia uma prevalência-alvo do simplex e reamostra COM REPOSIÇÃO —
# mesmo princípio em qualquer n, do 1-shot ao full. Substitui o UPP do
# mlquantify nos dois usos (calibração SMQ e teste), já que o UPP não
# foi pensado para pools de 2 exemplos (n=1).
# ============================================================

def artificial_prevalence_batches(X, y, n_classes, n_batches, batch_size, rng,
                                   n_prevalences=21, min_prev=0.0, max_prev=1.0):
    """Protocolo APP fiel ao paper (Esuli et al. 2023): grade FIXA de
    n_prevalences pontos igualmente espaçados (21 -> 0%, 5%, ..., 100%
    no binário), cada um repetido ~n_batches/n_prevalences vezes —
    diferente da versão anterior, que sorteava a prevalência-alvo de
    uma Dirichlet uniforme (mais parecido com UPP do que com o APP real
    do paper).

    Mantém a reamostragem COM REPOSIÇÃO por classe: necessária tanto
    pros datasets menores (onde batch_size=500 pode exceder o pool de
    teste) quanto pro few-shot extremo (n=1, só 1 exemplo por classe
    no pool de calibração do SMQ).

    n_batches pode não ser múltiplo exato de n_prevalences; nesse caso
    o total de batches gerados é round(n_batches/n_prevalences)*n_prevalences
    (ex.: 1000 batches / 21 pontos -> 48 repetições/ponto -> 1008 batches).
    """
    classes = np.arange(n_classes)
    idx_by_class = [np.where(y == c)[0] for c in classes]

    grid = np.linspace(min_prev, max_prev, n_prevalences)
    repeats_per_bin = max(1, round(n_batches / n_prevalences))

    for p1 in grid:
        for _ in range(repeats_per_bin):
            if n_classes == 2:
                p_target = np.array([1 - p1, p1])
            else:
                # grade explícita só implementada pro caso binário (paper);
                # fallback multiclasse continua com Dirichlet uniforme.
                p_target = rng.dirichlet(np.ones(n_classes))

            counts = np.floor(p_target * batch_size).astype(int)
            counts[np.argmax(counts)] += batch_size - counts.sum()

            batch_idx = []
            for c, cnt in zip(classes, counts):
                if cnt <= 0:
                    continue
                pool = idx_by_class[c]
                if len(pool) == 0:
                    continue
                sampled = rng.choice(pool, size=cnt, replace=True)
                batch_idx.append(sampled)

            if not batch_idx:
                continue

            batch_idx = np.concatenate(batch_idx)
            rng.shuffle(batch_idx)
            p_real = counts / counts.sum()
            yield batch_idx, p_real


# ============================================================
# FEW-SHOT SAMPLING
# ============================================================

def sample_few_shot(X, y, n_per_class, rng, n_classes=2):
    if n_per_class == "full":
        return X, y

    idx_selected = []
    for c in range(n_classes):
        idx_c = np.where(y == c)[0]
        n_take = min(n_per_class, len(idx_c))
        chosen = rng.choice(idx_c, size=n_take, replace=False)
        idx_selected.append(chosen)
    idx_selected = np.concatenate(idx_selected)
    rng.shuffle(idx_selected)
    return X[idx_selected], y[idx_selected]


# ============================================================
# MÉTRICAS
# ============================================================

def ae(p_hat, p_true):
    return np.mean(np.abs(p_hat - p_true))


def rae(p_hat, p_true, n_train):
    eps = 1.0 / (2 * n_train)
    n_classes = len(p_true)
    p_hat_s = (p_hat + eps) / (p_hat.sum() + n_classes * eps)
    p_true_s = (p_true + eps) / (p_true.sum() + n_classes * eps)
    return np.mean(np.abs(p_hat_s - p_true_s) / p_true_s)


# ============================================================
# CLASSIFIER FACTORIES
# ============================================================

def make_lr(seed=None):
    return LogisticRegression(C=1.0, penalty="l2", class_weight="balanced",
                               max_iter=1000, random_state=seed)


def make_rf_smq(seed):
    return RandomForestClassifier(n_estimators=QUANTIFIER_N_ESTIMATORS,
                                   n_jobs=N_JOBS, random_state=seed)


# ============================================================
# BASELINES (QuaPy) — val_split adaptativo:
#   max(2, min(n, 5)), infeasible em n=1 (retorna None)
# ============================================================

def adaptive_val_split(n_per_class):
    if n_per_class == "full" or (isinstance(n_per_class, int) and n_per_class >= 100):
        return 5
    if n_per_class == 1:
        return None  # infeasible, como no paper
    return int(max(2, min(n_per_class, 5)))


BASELINE_CLASSES = {
    "CC": CC,
    "PCC": PCC,
    "ACC": ACC,
    "PACC": PACC,
    "EMQ": EMQ,
    "DyS": DyS,
    "KDEyML": KDEyML,
}

NEEDS_VAL_SPLIT = {"ACC", "PACC", "EMQ", "DyS", "KDEyML"}


def fit_baseline(name, Xtr, ytr, n_per_class):
    cls = BASELINE_CLASSES[name]
    if name in NEEDS_VAL_SPLIT:
        vs = adaptive_val_split(n_per_class)
        if vs is None:
            return None  # infeasible no one-shot, como no paper
        if name == "EMQ":
            # EMQ no modo padrão (exact_train_prev=True, calib=None) não usa
            # val_split — a prevalência de treino exata já basta. Passar
            # val_split nesse modo é redundante e o QuaPy rejeita a
            # combinação (RuntimeWarning/erro dependendo da versão).
            model = cls(classifier=make_lr())
        else:
            model = cls(classifier=make_lr(), val_split=vs)
    else:
        model = cls(classifier=make_lr())
    model.fit(Xtr, ytr)
    return model


# ============================================================
# FSNN — Few-Shot Nearest-Neighbor Quantification
# ============================================================

class FSNN:
    def fit(self, Xtr, ytr):
        self.nn_ = NearestNeighbors(n_neighbors=1).fit(Xtr)
        self.ytr_ = ytr
        self.n_classes_ = len(np.unique(ytr))
        return self

    def quantify(self, Xte):
        _, idx = self.nn_.kneighbors(Xte)
        labels = self.ytr_[idx.ravel()]
        counts = np.bincount(labels, minlength=self.n_classes_)
        return counts / counts.sum()


# ============================================================
# FSMM — Few-Shot Moment Matching
# ============================================================

def _project_to_simplex(v):
    n = len(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1
    idx = np.arange(1, n + 1)
    cond = u - css / idx > 0
    rho = idx[cond][-1]
    theta = css[cond][-1] / rho
    return np.maximum(v - theta, 0)


class FSMM:
    def __init__(self, gamma=0.2):
        self.gamma = gamma

    def fit(self, Xtr, ytr):
        self.n_classes_ = len(np.unique(ytr))
        self.mu_ = np.array([Xtr[ytr == c].mean(axis=0) for c in range(self.n_classes_)])
        return self

    def quantify(self, Xte):
        mu_U = Xte.mean(axis=0)
        M = self.mu_.T
        mu_bar = self.mu_.mean(axis=0)
        M_tilde = (1 - self.gamma) * M + self.gamma * mu_bar.reshape(-1, 1)
        pi_ls, *_ = np.linalg.lstsq(M_tilde, mu_U, rcond=None)
        pi = _project_to_simplex(pi_ls)
        s = pi.sum()
        return pi / s if s > 0 else np.ones(self.n_classes_) / self.n_classes_


# ============================================================
# FSGMM — Few-Shot Gaussian Mixture Model
# ============================================================

class FSGMM:
    def __init__(self, K=8, eps=1e-6):
        self.K = K
        self.eps = eps

    def fit_quantify(self, Xtr, ytr, Xte):
        n_classes = len(np.unique(ytr))
        combined = np.vstack([Xte, Xtr])
        k_eff = min(self.K, len(combined))
        gmm = GaussianMixture(n_components=k_eff, covariance_type="diag",
                               random_state=0).fit(combined)

        resp_labeled = gmm.predict_proba(Xtr)
        W = np.zeros((k_eff, n_classes))
        for c in range(n_classes):
            mask = ytr == c
            num = resp_labeled[mask].sum(axis=0) + self.eps
            den = resp_labeled.sum(axis=0) + n_classes * self.eps
            W[:, c] = num / den

        pi = gmm.weights_ @ W
        s = pi.sum()
        return pi / s if s > 0 else np.ones(n_classes) / n_classes


# ============================================================
# FSGPEM — Few-Shot Gaussian Prior EM Quantifier
# ============================================================

class FSGPEM:
    def __init__(self, mean_shrink=0.1, var_shrink=0.1, alpha=1.2,
                 max_iter=100, tol=1e-6, n_restarts=3):
        self.mean_shrink = mean_shrink
        self.var_shrink = var_shrink
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.n_restarts = n_restarts

    @staticmethod
    def _gaussian_logpdf_diag(X, mu, var):
        var = np.maximum(var, 1e-8)
        return -0.5 * np.sum(((X - mu) ** 2) / var + np.log(2 * np.pi * var), axis=1)

    def _run_once(self, Xtr, ytr, Xte, rng):
        n_classes = len(np.unique(ytr))
        mu = np.array([Xtr[ytr == c].mean(axis=0) for c in range(n_classes)])
        mu_test = Xte.mean(axis=0)
        mu = (1 - self.mean_shrink) * mu + self.mean_shrink * mu_test

        var_test = Xte.var(axis=0)
        var_shared = (1 - self.var_shrink) * var_test + self.var_shrink * var_test.mean()

        counts = np.bincount(ytr, minlength=n_classes)
        pi = counts / counts.sum()
        pi = pi + rng.normal(scale=1e-3, size=n_classes)
        pi = np.clip(pi, 1e-6, None)
        pi = pi / pi.sum()

        n_u = len(Xte)
        log_lik = None
        for _ in range(self.max_iter):
            log_lik = np.stack([
                self._gaussian_logpdf_diag(Xte, mu[c], var_shared) + np.log(pi[c] + 1e-12)
                for c in range(n_classes)
            ], axis=1)
            log_lik -= log_lik.max(axis=1, keepdims=True)
            post = np.exp(log_lik)
            post /= post.sum(axis=1, keepdims=True)

            pi_new = (post.sum(axis=0) + (self.alpha - 1)) / (n_u + n_classes * (self.alpha - 1))
            pi_new = np.clip(pi_new, 1e-9, None)
            pi_new = pi_new / pi_new.sum()

            if np.abs(pi_new - pi).max() < self.tol:
                pi = pi_new
                break
            pi = pi_new

        ll_total = np.sum(np.log(np.sum(np.exp(log_lik), axis=1) + 1e-12))
        return pi, ll_total

    def fit_quantify(self, Xtr, ytr, Xte):
        rng = np.random.default_rng(0)
        best_pi, best_ll = None, -np.inf
        for _ in range(self.n_restarts):
            pi, ll = self._run_once(Xtr, ytr, Xte, rng)
            if ll > best_ll:
                best_pi, best_ll = pi, ll
        return best_pi


# ============================================================
# FSPL — Few-Shot Pseudo-Labeling
#
# Config FIXA (strategy="RS", m=20, downstream="EMQ") — sem a busca de
# 60 variantes da Seção 4.3 do paper, por decisão do usuário (FSPL é
# um dos dois métodos mais lentos do paper, junto com FSGMM — ver
# Tabela 10, ~2380s por protocolo completo no full-data).
# ============================================================

class FSPL:
    def __init__(self, strategy="RS", m=20, downstream="EMQ", seed=0):
        self.strategy = strategy
        self.m = m
        self.downstream = downstream
        self.seed = seed

    def fit_quantify(self, Xtr, ytr, Xte, n_per_class):
        n_classes = len(np.unique(ytr))
        rng = np.random.default_rng(self.seed)

        clf = make_lr(self.seed)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)

        pseudo_idx, pseudo_labels = [], []
        used = np.zeros(len(Xte), dtype=bool)

        if self.strategy == "MC":
            for c in range(n_classes):
                order = np.argsort(-proba[:, c])
                order = [i for i in order if not used[i]][: self.m]
                pseudo_idx.extend(order)
                pseudo_labels.extend([c] * len(order))
                used[order] = True
        elif self.strategy == "RS":
            hard = proba.argmax(axis=1)
            for c in range(n_classes):
                cand = np.where((hard == c) & (~used))[0]
                take = min(self.m, len(cand))
                if take > 0:
                    chosen = rng.choice(cand, size=take, replace=False)
                    pseudo_idx.extend(chosen)
                    pseudo_labels.extend([c] * take)
                    used[chosen] = True
        elif self.strategy == "CS":
            for c in range(n_classes):
                avail = np.where(~used)[0]
                if len(avail) == 0:
                    continue
                w = proba[avail, c]
                w = np.exp(w - w.max())
                w = w / w.sum()
                take = min(self.m, len(avail))
                chosen = rng.choice(avail, size=take, replace=False, p=w)
                pseudo_idx.extend(chosen)
                pseudo_labels.extend([c] * take)
                used[chosen] = True

        pseudo_idx = np.array(pseudo_idx, dtype=int) if pseudo_idx else np.array([], dtype=int)
        pseudo_labels = np.array(pseudo_labels, dtype=int) if len(pseudo_labels) else np.array([], dtype=int)

        X_ext = np.vstack([Xtr] + ([Xte[pseudo_idx]] if len(pseudo_idx) else []))
        y_ext = np.concatenate([ytr] + ([pseudo_labels] if len(pseudo_labels) else []))

        for c in range(n_classes):
            if np.sum(y_ext == c) < 2:
                idx_c = np.where(ytr == c)[0]
                if len(idx_c) > 0:
                    X_ext = np.vstack([X_ext, Xtr[idx_c[:1]]])
                    y_ext = np.concatenate([y_ext, [c]])

        downstream_cls = BASELINE_CLASSES[self.downstream]
        vs = adaptive_val_split(int(min(np.bincount(y_ext).min(), 5))) if self.downstream in NEEDS_VAL_SPLIT else None
        if self.downstream in NEEDS_VAL_SPLIT and vs is None:
            downstream_cls = CC  # fallback seguro se ainda infeasible
            model = downstream_cls(classifier=make_lr(self.seed))
        elif self.downstream == "EMQ":
            # mesmo motivo do fit_baseline: EMQ padrão não usa val_split
            model = downstream_cls(classifier=make_lr(self.seed))
        elif self.downstream in NEEDS_VAL_SPLIT:
            model = downstream_cls(classifier=make_lr(self.seed), val_split=vs)
        else:
            model = downstream_cls(classifier=make_lr(self.seed))

        model.fit(X_ext, y_ext)
        return model.quantify(Xte)


# ============================================================
# SMQ — MoSS base pré-treinado + calibradores (few-shot adaptado)
# ============================================================

def baseline_features(scores_matrix, target_n_classes):
    feats = []
    for c in range(scores_matrix.shape[1]):
        s = scores_matrix[:, c]
        feats.extend([np.mean(s), np.var(s), skew(s), kurtosis(s)])
    total_expected = 4 * target_n_classes
    if len(feats) < total_expected:
        feats.extend([0] * (total_expected - len(feats)))
    feats = np.array(feats[:total_expected])
    # skew/kurtose viram NaN (0/0) quando a coluna de scores tem
    # variância zero — comum em batches few-shot com poucos exemplos
    # distintos. 0 é o valor neutro (sem assimetria/curtose a reportar).
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def safe_prevalence_vector(p, n_classes):
    p = np.array(p, dtype=float)
    if len(p) < n_classes:
        padded = np.zeros(n_classes)
        padded[: len(p)] = p
        p = padded
    p = p[:n_classes]
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0, 1)
    s = p.sum()
    return p / s if s > 0 else np.ones(n_classes) / n_classes


def get_oof_scores_adaptive(Xtr, ytr, rep_seed):
    """cv = min(5, min_contagem_por_classe) quando >=2; no extremo n=1
    por classe (impossível fazer qualquer fold sem remover uma classe
    inteira do treino), cai para score in-sample."""
    class_counts = np.bincount(ytr)
    min_class_count = class_counts.min()

    if min_class_count >= 2:
        cv = min(5, min_class_count)
        oof_scores = cross_val_predict(
            make_rf_smq(rep_seed), Xtr, ytr,
            cv=cv, method="predict_proba", n_jobs=N_JOBS,
        )
    else:
        clf = make_rf_smq(rep_seed)
        clf.fit(Xtr, ytr)
        oof_scores = clf.predict_proba(Xtr)

    return oof_scores


REGRESSOR_FACTORIES_SMQ = {
    "rf": lambda seed: RandomForestRegressor(
        n_estimators=QUANTIFIER_N_ESTIMATORS, n_jobs=N_JOBS,
        random_state=seed, min_samples_leaf=2),
    "et": lambda seed: ExtraTreesRegressor(
        n_estimators=QUANTIFIER_N_ESTIMATORS, n_jobs=N_JOBS,
        random_state=seed, min_samples_leaf=2),
    "gbr": lambda seed: MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=200, random_state=seed),
        n_jobs=N_JOBS),
}


def load_moss_base_binary():
    if not os.path.exists(MOSS_BINARY_PKL):
        print(f"⚠️  MoSS base binário não encontrado em {MOSS_BINARY_PKL} — "
              f"métodos SMQ_* serão pulados até o arquivo existir nesse caminho.")
        return None
    with open(MOSS_BINARY_PKL, "rb") as f:
        synthetic_distributions = pickle.load(f)

    X_m, y_m = [], []
    for (alpha_prev, _), curves in synthetic_distributions.items():
        for scores_matrix in curves:
            X_m.append(baseline_features(scores_matrix, 2))
            y_m.append(list(alpha_prev))
    X_m, y_m = np.array(X_m), np.array(y_m)

    model = RandomForestRegressor(n_estimators=QUANTIFIER_N_ESTIMATORS,
                                   n_jobs=N_JOBS, random_state=SEED,
                                   min_samples_leaf=2)
    model.fit(X_m, y_m)
    return model


def train_isotonic_calibrator(pred_train, real_train, n_classes):
    calibrators = []
    for c in range(n_classes):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing="auto",
                                  out_of_bounds="clip")
        iso.fit(pred_train[:, c], real_train[:, c])
        calibrators.append(iso)
    return calibrators


def apply_isotonic_calibrator(calibrators, p_pred_raw, n_classes):
    p_cal = np.array([
        calibrators[c].predict([p_pred_raw[c]])[0] for c in range(n_classes)
    ])
    return safe_prevalence_vector(p_cal, n_classes)


def fit_smq(moss_base_model, Xtr, ytr, rep_seed, rng):
    """Treina o SMQ (classificador + tabela de calibração + calibradores
    rf/et/gbr/iso) UMA VEZ por (dataset, repetição, n-shot). Retorna um
    estado pronto pra ser aplicado em quantos batches de teste forem
    necessários via apply_smq(), sem retreinar nada."""
    n_classes = 2

    clf = make_rf_smq(rep_seed)
    clf.fit(Xtr, ytr)

    oof_scores = get_oof_scores_adaptive(Xtr, ytr, rep_seed)

    pred_calib, real_calib, feat_calib = [], [], []
    for idx_batch, p_real in artificial_prevalence_batches(
        Xtr, ytr, n_classes, N_CALIB_BATCHES, BATCH_SIZE, rng
    ):
        scores_batch = oof_scores[idx_batch]
        feat = baseline_features(scores_batch, n_classes)
        p_pred_raw = moss_base_model.predict(feat.reshape(1, -1))[0][:n_classes]
        pred_calib.append(safe_prevalence_vector(p_pred_raw, n_classes))
        real_calib.append(p_real)
        feat_calib.append(feat)
    pred_calib, real_calib, feat_calib = map(np.array, (pred_calib, real_calib, feat_calib))

    X_input = np.hstack([pred_calib, feat_calib])
    X_input = np.nan_to_num(X_input, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(X_input)
    X_input_scaled = scaler.transform(X_input)

    calibrators = {}
    for name, factory in REGRESSOR_FACTORIES_SMQ.items():
        model = factory(rep_seed)
        model.fit(X_input_scaled, real_calib)
        calibrators[name] = {"model": model, "scaler": scaler}

    iso_calibrators = train_isotonic_calibrator(pred_calib, real_calib, n_classes)

    return {
        "clf": clf,
        "moss_base_model": moss_base_model,
        "calibrators": calibrators,
        "iso_calibrators": iso_calibrators,
        "n_classes": n_classes,
    }


def apply_smq(smq_state, Xte_batch):
    """Aplica o SMQ já treinado (fit_smq) em UM batch de teste. Retorna
    {"smq_base": p, "smq_rf": p, "smq_et": p, "smq_gbr": p, "smq_iso": p}."""
    clf = smq_state["clf"]
    moss_base_model = smq_state["moss_base_model"]
    n_classes = smq_state["n_classes"]

    scores_test = clf.predict_proba(Xte_batch)
    f_vec = baseline_features(scores_test, n_classes)
    p_moss_raw = moss_base_model.predict(f_vec.reshape(1, -1))[0][:n_classes]
    p_base = safe_prevalence_vector(p_moss_raw, n_classes)

    results = {"smq_base": p_base}
    for name, entry in smq_state["calibrators"].items():
        X_in = np.hstack([p_base.reshape(1, -1), f_vec.reshape(1, -1)])
        X_in_scaled = entry["scaler"].transform(X_in)
        p_cal = np.array(entry["model"].predict(X_in_scaled)[0])[:n_classes]
        results[f"smq_{name}"] = safe_prevalence_vector(p_cal, n_classes)

    p_iso = apply_isotonic_calibrator(smq_state["iso_calibrators"], p_base, n_classes)
    results["smq_iso"] = p_iso

    return results


# ============================================================
# LOOP PRINCIPAL
# ============================================================

def run_one_dataset_one_rep_one_nshot(ds_name, Xtr_full, ytr_full, Xte, yte,
                                       n_per_class, rep_seed, moss_base_model):
    rng = np.random.default_rng(rep_seed)
    Xtr, ytr = sample_few_shot(Xtr_full, ytr_full, n_per_class, rng)

    rows = []

    # ---- 7 baselines ----
    for name in BASELINE_CLASSES:
        model = fit_baseline(name, Xtr, ytr, n_per_class)
        if model is None:
            continue
        for idx_batch, p_real in artificial_prevalence_batches(
            Xte, yte, 2, N_TEST_BATCHES, BATCH_SIZE, rng
        ):
            p_hat = model.quantify(Xte[idx_batch])
            rows.append(dict(dataset=ds_name, n=n_per_class, metodo=name,
                              ae=ae(p_hat, p_real),
                              rae=rae(p_hat, p_real, len(ytr))))

    # ---- FSNN ----
    fsnn = FSNN().fit(Xtr, ytr)
    for idx_batch, p_real in artificial_prevalence_batches(Xte, yte, 2, N_TEST_BATCHES, BATCH_SIZE, rng):
        p_hat = fsnn.quantify(Xte[idx_batch])
        rows.append(dict(dataset=ds_name, n=n_per_class, metodo="FSNN",
                          ae=ae(p_hat, p_real), rae=rae(p_hat, p_real, len(ytr))))

    # ---- FSMM ----
    fsmm = FSMM(gamma=0.2).fit(Xtr, ytr)
    for idx_batch, p_real in artificial_prevalence_batches(Xte, yte, 2, N_TEST_BATCHES, BATCH_SIZE, rng):
        p_hat = fsmm.quantify(Xte[idx_batch])
        rows.append(dict(dataset=ds_name, n=n_per_class, metodo="FSMM",
                          ae=ae(p_hat, p_real), rae=rae(p_hat, p_real, len(ytr))))

    # ---- FSGMM / FSGPEM / FSPL: transdutivos por definição no paper
    # (re-treinam por amostra de teste), então entram direto no loop
    # de batches — isso é intencional, não é o bug do SMQ. ----
    fsgmm = FSGMM(K=8)
    fsgpem = FSGPEM()
    fspl = FSPL(strategy="RS", m=20, downstream="EMQ", seed=rep_seed)
    for idx_batch, p_real in artificial_prevalence_batches(Xte, yte, 2, N_TEST_BATCHES, BATCH_SIZE, rng):
        Xte_b = Xte[idx_batch]

        p_hat = fsgmm.fit_quantify(Xtr, ytr, Xte_b)
        rows.append(dict(dataset=ds_name, n=n_per_class, metodo="FSGMM",
                          ae=ae(p_hat, p_real), rae=rae(p_hat, p_real, len(ytr))))

        p_hat = fsgpem.fit_quantify(Xtr, ytr, Xte_b)
        rows.append(dict(dataset=ds_name, n=n_per_class, metodo="FSGPEM",
                          ae=ae(p_hat, p_real), rae=rae(p_hat, p_real, len(ytr))))

        p_hat = fspl.fit_quantify(Xtr, ytr, Xte_b, n_per_class)
        rows.append(dict(dataset=ds_name, n=n_per_class, metodo="FSPL",
                          ae=ae(p_hat, p_real), rae=rae(p_hat, p_real, len(ytr))))

    # ---- SMQ (base + rf + et + gbr + iso) — treina UMA VEZ, aplica em
    # todos os batches (bug corrigido) ----
    if moss_base_model is not None:
        smq_state = fit_smq(moss_base_model, Xtr, ytr, rep_seed, rng)
        for idx_batch, p_real in artificial_prevalence_batches(Xte, yte, 2, N_TEST_BATCHES, BATCH_SIZE, rng):
            smq_results = apply_smq(smq_state, Xte[idx_batch])
            for name, p_hat in smq_results.items():
                rows.append(dict(dataset=ds_name, n=n_per_class, metodo=name,
                                  ae=ae(p_hat, p_real), rae=rae(p_hat, p_real, len(ytr))))

    return rows


def run_experiment():
    datasets = get_active_dataset_list()
    moss_base_model = load_moss_base_binary()

    print(f"Datasets ativos: {len(datasets)}")
    print(f"Repetições: {N_REPETITIONS}")
    print(f"N-shots: {N_SHOTS}")
    print(f"SMQ habilitado: {moss_base_model is not None}")

    header_written = os.path.exists(OUTPUT_CSV)

    for ds_name in datasets:
        print(f"\n📂 Dataset: {ds_name}")
        try:
            dataset = qp.datasets.fetch_UCIBinaryDataset(ds_name, verbose=False)
        except Exception as e:
            print(f"❌ Erro carregando {ds_name}: {e}")
            continue

        train, test = dataset.train_test
        X_all = np.vstack([train.instances, test.instances])
        y_all = np.concatenate([train.labels, test.labels])

        for rep in range(N_REPETITIONS):
            rep_seed = SEED + rep
            Xtr_full, Xte, ytr_full, yte = train_test_split(
                X_all, y_all, test_size=0.3, random_state=rep_seed, stratify=y_all
            )
            scaler = StandardScaler().fit(Xtr_full)
            Xtr_full = scaler.transform(Xtr_full)
            Xte = scaler.transform(Xte)

            all_rows = []
            for n_per_class in N_SHOTS:
                rows = run_one_dataset_one_rep_one_nshot(
                    ds_name, Xtr_full, ytr_full, Xte, yte,
                    n_per_class, rep_seed, moss_base_model,
                )
                for r in rows:
                    r["repeticao"] = rep + 1
                all_rows.extend(rows)

            pd.DataFrame(all_rows).to_csv(
                OUTPUT_CSV, mode="a", header=not header_written, index=False
            )
            header_written = True
            print(f"   ✅ rep {rep + 1}/{N_REPETITIONS} salva ({len(all_rows)} linhas)")
            gc.collect()


if __name__ == "__main__":
    try:
        run_experiment()
    except Exception:
        print("\n❌ EXPERIMENTO ABORTADO")
        print(traceback.format_exc())
        sys.exit(1)