import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import quapy as qp
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42
BATCH_SIZE = 200
TARGET_POS_PREV = 0.30
N_REPEATS = 30

QUAPY_BINARY_DATASETS = [
    "balance.1", "balance.3", "breast-cancer", "cmc.1", "cmc.2", "cmc.3",
    "ctg.1", "ctg.2", "ctg.3", "german", "haberman", "ionosphere", "iris.1",
    "iris.2", "iris.3", "mammographic", "pageblocks.5", "semeion", "sonar",
    "spambase", "spectf", "tictactoe", "transfusion", "wdbc", "wine.1",
    "wine.2", "wine.3", "wine-q-red", "wine-q-white", "yeast",
]


def _scalar_prev_from_moss_key(key):
    if isinstance(key, (int, float, np.floating)):
        return float(key)

    if isinstance(key, (list, tuple)):
        first = key[0]
        if isinstance(first, (int, float, np.floating)):
            return float(first)
        if isinstance(first, (list, tuple, np.ndarray)) and len(first) >= 2:
            return float(first[0])

    raise ValueError(f"Formato de chave MoSS não suportado: {key}")


def _score_vector_from_curve(curve):
    arr = np.asarray(curve)
    if arr.ndim == 1:
        return arr.astype(float)
    return arr[:, 0].astype(float)


def _sample_with_prevalence(y, pos_prev, size, rng):
    y = np.asarray(y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos = int(round(size * pos_prev))
    n_neg = size - n_pos

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Conjunto não binário ou sem exemplos de uma das classes")

    idx = np.concatenate([
        rng.choice(pos_idx, n_pos, replace=True),
        rng.choice(neg_idx, n_neg, replace=True),
    ])
    rng.shuffle(idx)
    return idx


def _kde_curve(scores, grid):
    scores = np.asarray(scores, dtype=float)
    if len(scores) < 2 or np.allclose(scores.std(), 0):
        return np.zeros_like(grid)
    try:
        return gaussian_kde(scores)(grid)
    except Exception:
        return np.zeros_like(grid)


def _load_moss_reference_scores(moss_data, target_neg_prev=0.7):
    prev_to_scores = {}
    for key, curves in moss_data.items():
        neg_prev = _scalar_prev_from_moss_key(key)
        all_scores = np.concatenate([_score_vector_from_curve(c) for c in curves])
        prev_to_scores[neg_prev] = all_scores

    closest_prev = min(prev_to_scores.keys(), key=lambda p: abs(p - target_neg_prev))
    return closest_prev, prev_to_scores[closest_prev]


def _load_binary_dataset(ds_name, local_binary_root=None):
    if local_binary_root is not None:
        import pandas as pd

        path = Path(local_binary_root) / f"{ds_name}.csv"
        if not path.exists():
            path = Path(local_binary_root) / ds_name
        if not path.exists():
            raise FileNotFoundError(f"Dataset local não encontrado: {path}")

        df = pd.read_csv(path)
        y = df.iloc[:, -1].to_numpy()
        X = df.iloc[:, :-1].to_numpy()

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"Dataset local {ds_name} não é binário")

        y = (y == classes.max()).astype(int)

        from sklearn.model_selection import train_test_split

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=SEED
        )
        return Xtr, ytr, Xte, yte

    data = qp.datasets.fetch_UCIBinaryDataset(ds_name, verbose=False)
    return data.training.X, data.training.y, data.test.X, data.test.y


def analyze_dataset(ds_name, moss_ref_scores, out_dir, local_binary_root=None):
    Xtr, ytr, Xte, yte = _load_binary_dataset(ds_name, local_binary_root=local_binary_root)

    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)

    clf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    clf.fit(Xtr, ytr)

    rng = np.random.default_rng(SEED)
    batches = []
    for _ in range(N_REPEATS):
        idx = _sample_with_prevalence(yte, TARGET_POS_PREV, BATCH_SIZE, rng)
        scores = clf.predict_proba(Xte[idx])[:, 1]
        batches.append(scores)

    real_scores = np.concatenate(batches)

    grid = np.linspace(0, 1, 300)
    real_kde = _kde_curve(real_scores, grid)
    moss_kde = _kde_curve(moss_ref_scores, grid)

    plt.figure(figsize=(8, 4))
    plt.plot(grid, real_kde, label=f"Real ({ds_name})", linewidth=2)
    plt.plot(grid, moss_kde, label="MoSS referência (~70/30)", linewidth=2)
    plt.title(f"Densidade de scores — {ds_name}")
    plt.xlabel("Score da classe positiva")
    plt.ylabel("Densidade")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / f"density_{ds_name.replace('.', '_')}.png"
    plt.savefig(out_path, dpi=180)
    plt.close()

    l1 = float(np.mean(np.abs(real_kde - moss_kde)))
    return {
        "dataset": ds_name,
        "real_scores_n": int(len(real_scores)),
        "moss_scores_n": int(len(moss_ref_scores)),
        "kde_l1": l1,
        "plot": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Compara densidades de scores reais vs MoSS (binário)")
    parser.add_argument("--moss-pkl", required=True, help="Caminho para moss_binario_lite.pkl")
    parser.add_argument("--output-dir", default="results/exp_015_density", help="Pasta de saída")
    parser.add_argument("--datasets", nargs="*", default=QUAPY_BINARY_DATASETS, help="Lista de datasets binários")
    parser.add_argument("--local-binary-root", default=None, help="Se informado, carrega datasets locais CSV desta pasta")
    args = parser.parse_args()

    moss_path = Path(args.moss_pkl)
    if not moss_path.exists():
        raise FileNotFoundError(f"Arquivo MoSS não encontrado: {moss_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(moss_path, "rb") as f:
        moss_data = pickle.load(f)

    chosen_prev, moss_ref_scores = _load_moss_reference_scores(moss_data, target_neg_prev=0.7)
    print(f"Usando MoSS com prevalência negativa mais próxima de 0.70: {chosen_prev:.3f}")

    rows = []
    for ds_name in args.datasets:
        print(f"Analisando {ds_name}...")
        rows.append(analyze_dataset(ds_name, moss_ref_scores, out_dir, local_binary_root=args.local_binary_root))

    import pandas as pd

    df = pd.DataFrame(rows).sort_values("kde_l1")
    csv_path = out_dir / "density_comparison_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Resumo salvo em: {csv_path}")


if __name__ == "__main__":
    main()
