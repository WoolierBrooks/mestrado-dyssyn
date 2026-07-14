import os
import time
import numpy as np
import pandas as pd
import warnings
import quapy as qp

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import (
    MiniBatchKMeans,
    AffinityPropagation,
    MeanShift,
    SpectralClustering,
    AgglomerativeClustering,
    DBSCAN,
    OPTICS,
    Birch
)

try:
    from sklearn.cluster import HDBSCAN
    _HAS_HDBSCAN = True
except ImportError:
    _HAS_HDBSCAN = False

from sklearn.mixture import GaussianMixture

from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm

# ============================================================
# QuaPy
# ============================================================

from quapy.method.aggregative import EMQ as EMQ_QUAPY

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

DATASETS_ROOT = "/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/multiclass"

# Datasets locais (10-50 classes), filtrados da lista CSV
LOCAL_DATASETS_DIR = "/var/new_homes/julio/mestrado/mestrado-dyssyn/experiments/datasets_multiclass"

OUTPUT_CSV = "results.csv"

# ── Número de repetições do loop treino/teste ──────────────
N_REPETITIONS = 1

# ── UPP: repeats=1; n_prevalences calculado para ≥1000 batches
UPP_REPEATS  = 1
N_PREV_TEST  = 1000

# ── Tamanho máximo de treino usado para ajustar os algoritmos de
#    clustering computacionalmente mais caros (Affinity Propagation,
#    Spectral, Ward/Agglomerative, MeanShift, DBSCAN, OPTICS têm custo
#    ~O(n²) ou pior). Acima desse tamanho, faz-se uma subamostragem
#    estratificada apenas para o fit do clustering.
CLUSTER_SUBSAMPLE_CAP = 3000

np.random.seed(SEED)

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
# CLUSTERING COMO CLASSIFICADOR (cluster-then-label + Classify & Count)
# ============================================================

def estimate_eps(X, k=5, percentile=90):
    """
    Heurística do 'k-distance graph' para estimar o eps do DBSCAN/OPTICS:
    calcula a distância ao k-ésimo vizinho mais próximo de cada ponto e
    usa um percentil alto dessa distribuição como eps (aproxima o 'cotovelo'
    do gráfico k-distance sem precisar inspecionar visualmente).
    """

    if len(X) <= k:
        return 0.5

    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X)
    dist, _ = nn.kneighbors(X)
    kth_dist = np.sort(dist[:, -1])

    return float(np.percentile(kth_dist, percentile))


class ClusterQuantifier:
    """
    Usa um algoritmo de clustering como um classificador agregativo:

      1) Ajusta o clustering no treino (com subamostragem estratificada
         opcional para algoritmos caros);
      2) Rotula cada cluster pela classe majoritária (voto majoritário)
         entre os pontos de treino que caíram nele;
      3) Em teste, cada amostra recebe o cluster mais próximo — via
         `.predict()` nativo quando o algoritmo suporta dado novo
         (KMeans, GaussianMixture, Birch, MeanShift, AffinityPropagation),
         ou via centróide mais próximo como fallback para os algoritmos
         que só suportam `fit_predict` (DBSCAN, OPTICS, HDBSCAN,
         Spectral, Agglomerative/Ward);
      4) A prevalência prevista do lote é a contagem normalizada das
         classes atribuídas — equivalente a um Classify & Count (CC)
         usando clustering no lugar de um classificador supervisionado.

    Pontos de ruído do DBSCAN/OPTICS/HDBSCAN (rótulo -1) são ignorados
    na etapa de rotulação de clusters; se um ponto de teste cair no
    "cluster" de ruído (-1) ou não houver nenhum centróide válido, ele
    recebe a classe majoritária global como fallback.
    """

    def __init__(self, model_factory, n_classes=None, max_train_size=None,
                 random_state=42):

        self.model_factory = model_factory
        self.n_classes = n_classes
        self.max_train_size = max_train_size
        self.random_state = random_state

    def fit(self, X, y):

        if self.n_classes is None:
            self.n_classes = len(np.unique(y))

        if self.max_train_size is not None and len(X) > self.max_train_size:

            idx_sub, _ = train_test_split(
                np.arange(len(X)),
                train_size=self.max_train_size,
                stratify=y,
                random_state=self.random_state
            )

            X_fit, y_fit = X[idx_sub], y[idx_sub]

        else:

            X_fit, y_fit = X, y

        self.model = self.model_factory()

        if hasattr(self.model, "fit_predict"):
            train_labels = self.model.fit_predict(X_fit)
        else:
            self.model.fit(X_fit)
            train_labels = self.model.predict(X_fit)

        classes_, counts_ = np.unique(y_fit, return_counts=True)
        self.default_class_ = classes_[np.argmax(counts_)]

        self.cluster_to_class_ = {}
        self.centroids_ = {}

        for c in np.unique(train_labels):

            if c == -1:  # ruído (DBSCAN/OPTICS/HDBSCAN)
                continue

            mask = train_labels == c
            cls_, cnt_ = np.unique(y_fit[mask], return_counts=True)

            self.cluster_to_class_[c] = cls_[np.argmax(cnt_)]
            self.centroids_[c] = X_fit[mask].mean(axis=0)

        # só usamos o .predict() nativo se ele realmente existir e não
        # for herdado sem suporte a dado novo
        self._has_native_predict = hasattr(self.model, "predict")

        return self

    def _assign_clusters(self, X):

        if self._has_native_predict:
            try:
                return self.model.predict(X)
            except Exception:
                pass  # cai no fallback de centróide mais próximo

        if len(self.centroids_) == 0:
            return np.full(X.shape[0], -1)

        ids = list(self.centroids_.keys())
        C = np.array([self.centroids_[i] for i in ids])

        d = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        nearest = np.array(ids)[np.argmin(d, axis=1)]

        return nearest

    def predict(self, X):

        cluster_ids = self._assign_clusters(X)

        y_pred = np.array([
            self.cluster_to_class_.get(c, self.default_class_)
            for c in cluster_ids
        ])

        counts = np.bincount(y_pred, minlength=self.n_classes)

        if counts.sum() == 0:
            return np.ones(self.n_classes) / self.n_classes

        return counts / counts.sum()


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
    print(f"Batches teste:      {UPP_REPEATS * N_PREV_TEST}")

    if not _HAS_HDBSCAN:
        print("⚠️ sklearn.cluster.HDBSCAN não disponível nesta versão do "
              "scikit-learn (precisa >= 1.3). CLUST_HDBSCAN será pulado.")

    rows = []

    all_entries = (
        [(ds, "quapy", None) for ds in QUAPY_MULTICLASS_DATASETS] +
        [(fname, "local", (os.path.join(LOCAL_DATASETS_DIR, fname), col))
         for fname, col in LOCAL_DATASETS]
    )

    # ========================================================
    # LOOP PRINCIPAL: dataset → repetição de split
    # ========================================================

    for ds_name, source, meta in all_entries:

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
            # QUANTIFIERS CLÁSSICOS
            # ================================================

            print("      🔥 Building quantifiers...")

            quantifiers = {}

            quantifiers["EMQ_QUAPY"] = EMQ_QUAPY(
                RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=-1,
                    random_state=rep_seed
                ),
                calib=None,
                exact_train_prev=True
            )

            quantifiers["EMQ_BCTS_QUAPY"] = EMQ_QUAPY(
                RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=-1,
                    random_state=rep_seed
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
                    random_state=rep_seed
                )
            )

            quantifiers["PCC"] = PCC(
                learner=RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=-1,
                    random_state=rep_seed
                )
            )

            quantifiers["AC"] = AC(
                learner=RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=-1,
                    random_state=rep_seed
                )
            )

            quantifiers["PAC"] = PAC(
                learner=RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=-1,
                    random_state=rep_seed
                )
            )

            quantifiers["FM"] = FM(
                learner=RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=-1,
                    random_state=rep_seed
                )
            )

            quantifiers["KDEyML"] = KDEyML(
                learner=RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=-1,
                    random_state=rep_seed
                ),
                bandwidth=0.1
            )

            quantifiers["KDEyHD"] = KDEyHD(
                learner=RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=-1,
                    random_state=rep_seed
                ),
                montecarlo_trials=500,
                random_state=rep_seed
            )

            quantifiers["KDEyCS"] = KDEyCS(
                learner=RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=-1,
                    random_state=rep_seed
                ),
                bandwidth=0.1
            )

            quantifiers["PWK"] = PWK(
                n_neighbors=10
            )

            # ================================================
            # QUANTIFIERS BASEADOS EM CLUSTERING
            # (clustering usado como classificador — ver docstring
            # de ClusterQuantifier)
            # ================================================

            n_clusters = n_classes_real

            k_eps = min(5, max(1, len(Xtr) - 1))
            eps_est = estimate_eps(Xtr, k=k_eps)

            cluster_configs = {
                "CLUST_MiniBatchKMeans": lambda: MiniBatchKMeans(
                    n_clusters=n_clusters, random_state=rep_seed, n_init=10
                ),
                "CLUST_AffinityPropagation": lambda: AffinityPropagation(
                    random_state=rep_seed
                ),
                "CLUST_MeanShift": lambda: MeanShift(n_jobs=-1),
                "CLUST_Spectral": lambda: SpectralClustering(
                    n_clusters=n_clusters,
                    affinity="nearest_neighbors",
                    n_neighbors=10,
                    random_state=rep_seed,
                    n_jobs=-1
                ),
                "CLUST_Ward": lambda: AgglomerativeClustering(
                    n_clusters=n_clusters, linkage="ward"
                ),
                "CLUST_Agglomerative": lambda: AgglomerativeClustering(
                    n_clusters=n_clusters, linkage="average"
                ),
                "CLUST_DBSCAN": lambda: DBSCAN(
                    eps=eps_est, min_samples=k_eps, n_jobs=-1
                ),
                "CLUST_OPTICS": lambda: OPTICS(
                    min_samples=k_eps, n_jobs=-1
                ),
                "CLUST_BIRCH": lambda: Birch(n_clusters=n_clusters),
                "CLUST_GaussianMixture": lambda: GaussianMixture(
                    n_components=n_clusters, random_state=rep_seed
                ),
            }

            if _HAS_HDBSCAN:
                cluster_configs["CLUST_HDBSCAN"] = lambda: HDBSCAN(
                    min_cluster_size=max(5, len(Xtr) // (n_clusters * 4))
                )

            # algoritmos caros (O(n²) ou pior) usam subamostragem no fit
            expensive = {
                "CLUST_AffinityPropagation", "CLUST_MeanShift",
                "CLUST_Spectral", "CLUST_Ward", "CLUST_Agglomerative",
                "CLUST_DBSCAN", "CLUST_OPTICS"
            }

            for cname, factory in cluster_configs.items():
                quantifiers[cname] = ClusterQuantifier(
                    model_factory=factory,
                    n_classes=n_classes_real,
                    max_train_size=(
                        CLUSTER_SUBSAMPLE_CAP if cname in expensive else None
                    ),
                    random_state=rep_seed
                )

            # ================================================
            # TRAIN QUANTIFIERS (clássicos + clustering)
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
                # QUANTIFIERS (clássicos + clustering)
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

            # ================================================
            # SAVE PARTIAL (a cada repetição)
            # ================================================

            pd.DataFrame(rows).to_csv(
                OUTPUT_CSV,
                index=False
            )

            print(f"💾 Parcial salva em {OUTPUT_CSV}  (rep {rep + 1}/{N_REPETITIONS})")

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

    run_experiment()