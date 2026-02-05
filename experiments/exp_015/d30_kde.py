import os
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, kurtosis, skew
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from mlquantify.model_selection import UPP
from mlquantify.utils import get_prev_from_labels

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
SEED = 42
BATCH_SIZE = 100

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MOSS_DIR = os.path.join(REPO_ROOT, "datasets", "multiclass")
DATASETS_ROOT = os.path.join(REPO_ROOT, "datasets", "moss", "multiclass")
OUTPUT_CSV = "d30_kde.csv"

HIST_BINS = (10, 20, 30, 50)
KDE_GRID_SIZE = 50

np.random.seed(SEED)


# ============================================================
# FEATURES
# ============================================================
def baseline_features(scores_matrix, target_n_classes):
    feats = []
    for c in range(scores_matrix.shape[1]):
        s = scores_matrix[:, c]
        feats.extend([
            np.mean(s),
            np.var(s),
            skew(s),
            kurtosis(s),
        ])

    total_expected = 4 * target_n_classes
    if len(feats) < total_expected:
        feats.extend([0.0] * (total_expected - len(feats)))

    return np.array(feats[:total_expected], dtype=float)


def _kde_values(scores):
    grid = np.linspace(0, 1, KDE_GRID_SIZE)
    scores = np.asarray(scores, dtype=float)

    if len(scores) < 2 or np.allclose(scores.std(), 0.0):
        return np.zeros_like(grid)

    try:
        kde = gaussian_kde(scores)
        return kde(grid)
    except Exception:
        return np.zeros_like(grid)


def _histogram_pyramid(scores):
    blocks = []
    for bins in HIST_BINS:
        h, _ = np.histogram(scores, bins=bins, range=(0, 1), density=True)
        blocks.append(h)
    return np.concatenate(blocks)


def kde_hist_features(scores_matrix, target_n_classes):
    feats = []

    n_present_classes = scores_matrix.shape[1]
    for c in range(n_present_classes):
        s = scores_matrix[:, c]
        feats.extend(_histogram_pyramid(s))
        feats.extend(_kde_values(s))

    per_class_size = sum(HIST_BINS) + KDE_GRID_SIZE
    total_expected = per_class_size * target_n_classes

    if len(feats) < total_expected:
        feats.extend([0.0] * (total_expected - len(feats)))

    return np.array(feats[:total_expected], dtype=float)


FEATURE_EXTRACTORS = {
    "MoSS_Baseline": baseline_features,
    "MoSS_KDEHist": kde_hist_features,
}


# ============================================================
# EXPERIMENT
# ============================================================
def run_experiment():
    datasets = sorted(f for f in os.listdir(DATASETS_ROOT) if f.endswith(".csv"))

    models_by_nclasses = {}
    rows = []

    for ds_name in datasets:
        print(f"\n📂 Dataset: {ds_name}")

        df = pd.read_csv(os.path.join(DATASETS_ROOT, ds_name))
        y = df.iloc[:, -1].values
        if y.min() > 0:
            y -= y.min()

        n_classes_real = len(np.unique(y))
        print(f"   🔢 Número de classes: {n_classes_real}")

        X = StandardScaler().fit_transform(df.iloc[:, :-1].values)

        Xtr, Xte, ytr, yte = train_test_split(
            X,
            y,
            test_size=0.5,
            stratify=y,
            random_state=SEED,
        )

        clf = RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=SEED,
        )
        clf.fit(Xtr, ytr)

        if n_classes_real not in models_by_nclasses:
            moss_path = os.path.join(MOSS_DIR, f"moss_d_lite_{n_classes_real}.pkl")

            if not os.path.exists(moss_path):
                print(f"⚠️ {os.path.basename(moss_path)} não encontrado.")
                models_by_nclasses[n_classes_real] = None
            else:
                print(f"🧠 Treinando modelos MoSS_{n_classes_real}...")
                with open(moss_path, "rb") as f:
                    synthetic_distributions = pickle.load(f)

                reg_models = {}

                for model_name, extractor in FEATURE_EXTRACTORS.items():
                    X_m, y_m = [], []

                    for (alpha_prev, _), curves in synthetic_distributions.items():
                        for scores_matrix in curves:
                            X_m.append(extractor(scores_matrix, n_classes_real))
                            y_m.append(list(alpha_prev))

                    X_m = np.array(X_m)
                    y_m = np.array(y_m)

                    if len(X_m) == 0:
                        reg_models[model_name] = None
                        continue

                    model = RandomForestRegressor(
                        n_estimators=100,
                        n_jobs=-1,
                        random_state=SEED,
                    )
                    model.fit(X_m, y_m)
                    reg_models[model_name] = model

                models_by_nclasses[n_classes_real] = reg_models
                print(f"✅ Modelos MoSS_{n_classes_real} treinados!")

        moss_models = models_by_nclasses[n_classes_real]

        protocol = UPP(
            batch_size=BATCH_SIZE,
            n_prevalences=10,
            repeats=30,
            random_state=SEED,
        )

        for idx_batch in tqdm(
            protocol.split(Xte, yte),
            total=protocol.get_n_combinations(),
            desc="Protocolo UPP",
        ):
            p_real_raw = get_prev_from_labels(yte[idx_batch], classes=np.arange(n_classes_real))
            if isinstance(p_real_raw, dict):
                p_real = np.array([p_real_raw[k] for k in sorted(p_real_raw)])
            else:
                p_real = np.array(p_real_raw)

            scores_batch = clf.predict_proba(Xte[idx_batch])

            if moss_models is None:
                continue

            for model_name, reg in moss_models.items():
                if reg is None:
                    continue

                feats = FEATURE_EXTRACTORS[model_name](scores_batch, n_classes_real).reshape(1, -1)
                p_pred = reg.predict(feats)[0][:n_classes_real]
                p_pred = np.clip(p_pred, 0, None)
                p_pred /= (p_pred.sum() + 1e-12)

                rows.append(
                    {
                        "dataset": ds_name,
                        "modelo": model_name,
                        "n_classes_original": n_classes_real,
                        "erro": np.mean(np.abs(p_pred - p_real)),
                    }
                )

        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Experimento concluído! Resultados salvos em {OUTPUT_CSV}")


if __name__ == "__main__":
    run_experiment()
