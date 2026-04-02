#!/usr/bin/env python3
"""exp_027 - unified comparison with cluster_pca using test-time reclustering.

Models compared (binary quantification):
- cluster_pca: train supervised centroids + test reclustering (KMeans) + centroid matching
- cluster_gmm: class-conditional GMMs in PCA(2)
- cluster_kmeans: unsupervised KMeans in PCA(2) + cluster->label mapping
- dys_topsoe: DyS quantifier with Topsoe distance

All models are evaluated on exactly the same APP batches and random states per dataset.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import quapy as qp
from mlquantify.mixture import DyS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

SEED = 42
BATCH_SIZE = 100
N_PREV = 19
REPEATS = 30

QUAPY_BINARY_DATASETS = [
    "balance.1",
    "balance.3",
    "breast-cancer",
    "cmc.1",
    "cmc.2",
    "cmc.3",
    "ctg.1",
    "ctg.2",
    "ctg.3",
    "german",
    "haberman",
    "ionosphere",
    "iris.1",
    "iris.2",
    "iris.3",
    "mammographic",
    "pageblocks.5",
    "semeion",
    "sonar",
    "spambase",
    "spectf",
    "tictactoe",
    "transfusion",
    "wdbc",
    "wine.1",
    "wine.2",
    "wine.3",
    "wine-q-red",
    "wine-q-white",
    "yeast",
]


@dataclass
class BatchSpec:
    batch_id: int
    repeat: int
    target_prev_neg: float
    target_prev_pos: float
    indices: np.ndarray


def _assign_test_clusters_to_train_labels(test_centers: np.ndarray, train_c0: np.ndarray, train_c1: np.ndarray) -> Dict[int, int]:
    """Map cluster id -> label {0,1} minimizing total train/test centroid distance."""
    d00 = np.linalg.norm(test_centers[0] - train_c0)
    d01 = np.linalg.norm(test_centers[0] - train_c1)
    d10 = np.linalg.norm(test_centers[1] - train_c0)
    d11 = np.linalg.norm(test_centers[1] - train_c1)

    cost_a = d00 + d11  # c0->0, c1->1
    cost_b = d01 + d10  # c0->1, c1->0
    return {0: 0, 1: 1} if cost_a <= cost_b else {0: 1, 1: 0}


@dataclass
class ClusterPCAModel:
    scaler: StandardScaler
    pca: PCA
    centroid_0: np.ndarray
    centroid_1: np.ndarray
    base_seed: int

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Recompute test centroids (unsupervised), align with train centroids, then label."""
        Xp = self.pca.transform(self.scaler.transform(X))

        km = KMeans(n_clusters=2, random_state=self.base_seed + len(X), n_init=10)
        test_cluster_ids = km.fit_predict(Xp)
        test_centers = km.cluster_centers_

        mapping = _assign_test_clusters_to_train_labels(test_centers, self.centroid_0, self.centroid_1)
        return np.array([mapping[int(cid)] for cid in test_cluster_ids], dtype=int)


@dataclass
class ClusterGMMModel:
    scaler: StandardScaler
    pca: PCA
    centroid_0: np.ndarray
    centroid_1: np.ndarray
    base_seed: int

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Recompute test clusters with GMM and align them to train centroids."""
        Xp = self.pca.transform(self.scaler.transform(X))
        gmm = GaussianMixture(n_components=2, random_state=self.base_seed + len(X), covariance_type="full")
        test_cluster_ids = gmm.fit_predict(Xp)
        test_centers = gmm.means_
        mapping = _assign_test_clusters_to_train_labels(test_centers, self.centroid_0, self.centroid_1)
        return np.array([mapping[int(cid)] for cid in test_cluster_ids], dtype=int)


@dataclass
class ClusterKMeansModel:
    scaler: StandardScaler
    pca: PCA
    centroid_0: np.ndarray
    centroid_1: np.ndarray
    base_seed: int

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Recompute test clusters with KMeans and align them to train centroids."""
        Xp = self.pca.transform(self.scaler.transform(X))
        kmeans = KMeans(n_clusters=2, random_state=self.base_seed + len(X), n_init=10)
        test_cluster_ids = kmeans.fit_predict(Xp)
        test_centers = kmeans.cluster_centers_
        mapping = _assign_test_clusters_to_train_labels(test_centers, self.centroid_0, self.centroid_1)
        return np.array([mapping[int(cid)] for cid in test_cluster_ids], dtype=int)


def to_pos_prev(pred) -> float:
    if isinstance(pred, dict):
        for key in (1, "1", True):
            if key in pred:
                return float(pred[key])
        vals = list(pred.values())
        if len(vals) == 1:
            return float(vals[0])
        if len(vals) >= 2:
            return float(vals[1])
        raise ValueError("Empty dict prevalence output")

    arr = np.asarray(pred)
    if arr.ndim == 0:
        return float(arr)
    arr = arr.ravel()
    if arr.size == 1:
        return float(arr[0])
    if arr.size >= 2:
        return float(arr[1])
    raise ValueError("Empty prevalence output")


def generate_prevalences(n_prev: int, repeats: int) -> List[List[float]]:
    prevalences = np.linspace(0.05, 0.95, n_prev)
    return [[1 - float(p), float(p)] for p in prevalences] * repeats


def sample_batch_indices(y: np.ndarray, prevalence_pos: float, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Test set must contain both classes for APP sampling")

    n_pos = int(batch_size * prevalence_pos)
    n_neg = batch_size - n_pos
    idx = np.concatenate(
        [
            rng.choice(pos, size=n_pos, replace=True),
            rng.choice(neg, size=n_neg, replace=True),
        ]
    )
    rng.shuffle(idx)
    return idx


def build_batches(yte: np.ndarray, app_prevalences: Sequence[Sequence[float]], n_prev: int, batch_size: int, seed: int) -> List[BatchSpec]:
    rng = np.random.default_rng(seed)
    batches: List[BatchSpec] = []
    for batch_id, prev_pair in enumerate(app_prevalences, start=1):
        p_neg, p_pos = float(prev_pair[0]), float(prev_pair[1])
        idx = sample_batch_indices(yte, prevalence_pos=p_pos, batch_size=batch_size, rng=rng)
        repeat = (batch_id - 1) // n_prev
        batches.append(
            BatchSpec(
                batch_id=batch_id,
                repeat=repeat,
                target_prev_neg=p_neg,
                target_prev_pos=p_pos,
                indices=idx,
            )
        )
    return batches


def load_binary_quapy_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = qp.datasets.fetch_UCIBinaryDataset(name, verbose=False)
    Xtr = np.asarray(ds.training.X)
    ytr_raw = np.asarray(ds.training.y)
    Xte = np.asarray(ds.test.X)
    yte_raw = np.asarray(ds.test.y)

    classes = np.unique(np.concatenate([ytr_raw, yte_raw]))
    if len(classes) != 2:
        raise ValueError(f"Dataset {name} is not binary")
    pos_class = classes.max()
    ytr = (ytr_raw == pos_class).astype(int)
    yte = (yte_raw == pos_class).astype(int)
    return Xtr, ytr, Xte, yte


def fit_cluster_pca(Xtr: np.ndarray, ytr: np.ndarray, seed: int) -> ClusterPCAModel:
    if len(np.unique(ytr)) < 2:
        raise ValueError("Training set must contain both classes")
    scaler = StandardScaler().fit(Xtr)
    Xt = scaler.transform(Xtr)
    pca = PCA(n_components=2, random_state=seed).fit(Xt)
    Xp = pca.transform(Xt)
    c0 = Xp[ytr == 0].mean(axis=0)
    c1 = Xp[ytr == 1].mean(axis=0)
    return ClusterPCAModel(scaler=scaler, pca=pca, centroid_0=c0, centroid_1=c1, base_seed=seed)


def fit_cluster_gmm(Xtr: np.ndarray, ytr: np.ndarray, seed: int, n_components: int = 1) -> ClusterGMMModel:
    if len(np.unique(ytr)) < 2:
        raise ValueError("Training set must contain both classes")
    scaler = StandardScaler().fit(Xtr)
    Xt = scaler.transform(Xtr)
    pca = PCA(n_components=2, random_state=seed).fit(Xt)
    Xp = pca.transform(Xt)

    c0 = Xp[ytr == 0].mean(axis=0)
    c1 = Xp[ytr == 1].mean(axis=0)
    return ClusterGMMModel(scaler=scaler, pca=pca, centroid_0=c0, centroid_1=c1, base_seed=seed)


def fit_cluster_kmeans(Xtr: np.ndarray, ytr: np.ndarray, seed: int) -> ClusterKMeansModel:
    if len(np.unique(ytr)) < 2:
        raise ValueError("Training set must contain both classes")
    scaler = StandardScaler().fit(Xtr)
    Xt = scaler.transform(Xtr)
    pca = PCA(n_components=2, random_state=seed).fit(Xt)
    Xp = pca.transform(Xt)

    c0 = Xp[ytr == 0].mean(axis=0)
    c1 = Xp[ytr == 1].mean(axis=0)
    return ClusterKMeansModel(scaler=scaler, pca=pca, centroid_0=c0, centroid_1=c1, base_seed=seed)


def evaluate_label_model(model_name: str, model, Xte: np.ndarray, yte: np.ndarray, batches: Sequence[BatchSpec]) -> List[Dict]:
    rows: List[Dict] = []
    for b in batches:
        Xb = Xte[b.indices]
        yb = yte[b.indices]
        prev_true = float(np.mean(yb))

        t0 = time.perf_counter()
        yhat = model.predict_labels(Xb)
        elapsed = time.perf_counter() - t0

        prev_pred = float(np.mean(yhat))
        rows.append(
            {
                "model": model_name,
                "dataset": None,
                "repeat": b.repeat,
                "batch_id": b.batch_id,
                "batch_size": len(yb),
                "target_prev_neg": b.target_prev_neg,
                "target_prev_pos": b.target_prev_pos,
                "true_prev_pos": prev_true,
                "pred_prev_pos": float(np.clip(prev_pred, 0, 1)),
                "abs_error": float(abs(prev_pred - prev_true)),
                "time_per_sample": elapsed / max(1, len(yb)),
                "status": "ok",
                "error": None,
            }
        )
    return rows


def evaluate_dys_topsoe(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray, batches: Sequence[BatchSpec], seed: int) -> List[Dict]:
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    q_model = DyS(learner=rf, measure="topsoe")
    q_model.fit(Xtr, ytr)

    rows: List[Dict] = []
    for b in batches:
        Xb = Xte[b.indices]
        yb = yte[b.indices]
        prev_true = float(np.mean(yb))

        t0 = time.perf_counter()
        prev_pred = to_pos_prev(q_model.predict(Xb))
        elapsed = time.perf_counter() - t0

        rows.append(
            {
                "model": "dys_topsoe",
                "dataset": None,
                "repeat": b.repeat,
                "batch_id": b.batch_id,
                "batch_size": len(yb),
                "target_prev_neg": b.target_prev_neg,
                "target_prev_pos": b.target_prev_pos,
                "true_prev_pos": prev_true,
                "pred_prev_pos": float(np.clip(prev_pred, 0, 1)),
                "abs_error": float(abs(prev_pred - prev_true)),
                "time_per_sample": elapsed / max(1, len(yb)),
                "status": "ok",
                "error": None,
            }
        )
    return rows


def run_experiment(
    datasets: Iterable[str],
    batch_size: int,
    n_prev: int,
    repeats: int,
    seed: int,
    out_csv: str,
) -> pd.DataFrame:
    app_prevalences = generate_prevalences(n_prev=n_prev, repeats=repeats)
    rows: List[Dict] = []

    model_names = ["cluster_pca", "cluster_gmm", "cluster_kmeans", "dys_topsoe"]
    total_steps = len(list(datasets)) * len(app_prevalences) * len(model_names)

    with tqdm(total=total_steps, desc="exp_027", unit="batch") as pbar:
        for ds_i, ds_name in enumerate(datasets):
            try:
                Xtr, ytr, Xte, yte = load_binary_quapy_dataset(ds_name)
            except Exception as e:
                for m in model_names:
                    rows.append(
                        {
                            "model": m,
                            "dataset": ds_name,
                            "repeat": None,
                            "batch_id": None,
                            "batch_size": batch_size,
                            "target_prev_neg": None,
                            "target_prev_pos": None,
                            "true_prev_pos": None,
                            "pred_prev_pos": None,
                            "abs_error": None,
                            "time_per_sample": None,
                            "status": "load_error",
                            "error": str(e),
                        }
                    )
                pbar.update(len(app_prevalences) * len(model_names))
                continue

            ds_seed = seed + ds_i * 10_000
            batches = build_batches(
                yte=yte,
                app_prevalences=app_prevalences,
                n_prev=n_prev,
                batch_size=batch_size,
                seed=ds_seed,
            )

            try:
                pca_model = fit_cluster_pca(Xtr, ytr, seed=seed + ds_i)
                ds_rows = evaluate_label_model("cluster_pca", pca_model, Xte, yte, batches)
                for r in ds_rows:
                    r["dataset"] = ds_name
                rows.extend(ds_rows)
            except Exception as e:
                for b in batches:
                    rows.append(
                        {
                            "model": "cluster_pca",
                            "dataset": ds_name,
                            "repeat": b.repeat,
                            "batch_id": b.batch_id,
                            "batch_size": batch_size,
                            "target_prev_neg": b.target_prev_neg,
                            "target_prev_pos": b.target_prev_pos,
                            "true_prev_pos": None,
                            "pred_prev_pos": None,
                            "abs_error": None,
                            "time_per_sample": None,
                            "status": "model_error",
                            "error": str(e),
                        }
                    )
            pbar.update(len(app_prevalences))

            try:
                gmm_model = fit_cluster_gmm(Xtr, ytr, seed=seed, n_components=1)
                ds_rows = evaluate_label_model("cluster_gmm", gmm_model, Xte, yte, batches)
                for r in ds_rows:
                    r["dataset"] = ds_name
                rows.extend(ds_rows)
            except Exception as e:
                for b in batches:
                    rows.append(
                        {
                            "model": "cluster_gmm",
                            "dataset": ds_name,
                            "repeat": b.repeat,
                            "batch_id": b.batch_id,
                            "batch_size": batch_size,
                            "target_prev_neg": b.target_prev_neg,
                            "target_prev_pos": b.target_prev_pos,
                            "true_prev_pos": None,
                            "pred_prev_pos": None,
                            "abs_error": None,
                            "time_per_sample": None,
                            "status": "model_error",
                            "error": str(e),
                        }
                    )
            pbar.update(len(app_prevalences))

            try:
                kmeans_model = fit_cluster_kmeans(Xtr, ytr, seed=seed)
                ds_rows = evaluate_label_model("cluster_kmeans", kmeans_model, Xte, yte, batches)
                for r in ds_rows:
                    r["dataset"] = ds_name
                rows.extend(ds_rows)
            except Exception as e:
                for b in batches:
                    rows.append(
                        {
                            "model": "cluster_kmeans",
                            "dataset": ds_name,
                            "repeat": b.repeat,
                            "batch_id": b.batch_id,
                            "batch_size": batch_size,
                            "target_prev_neg": b.target_prev_neg,
                            "target_prev_pos": b.target_prev_pos,
                            "true_prev_pos": None,
                            "pred_prev_pos": None,
                            "abs_error": None,
                            "time_per_sample": None,
                            "status": "model_error",
                            "error": str(e),
                        }
                    )
            pbar.update(len(app_prevalences))

            try:
                scaler = StandardScaler().fit(Xtr)
                Xtr_scaled = scaler.transform(Xtr)
                Xte_scaled = scaler.transform(Xte)
                ds_rows = evaluate_dys_topsoe(Xtr_scaled, ytr, Xte_scaled, yte, batches, seed=seed)
                for r in ds_rows:
                    r["dataset"] = ds_name
                rows.extend(ds_rows)
            except Exception as e:
                for b in batches:
                    rows.append(
                        {
                            "model": "dys_topsoe",
                            "dataset": ds_name,
                            "repeat": b.repeat,
                            "batch_id": b.batch_id,
                            "batch_size": batch_size,
                            "target_prev_neg": b.target_prev_neg,
                            "target_prev_pos": b.target_prev_pos,
                            "true_prev_pos": None,
                            "pred_prev_pos": None,
                            "abs_error": None,
                            "time_per_sample": None,
                            "status": "model_error",
                            "error": str(e),
                        }
                    )
            pbar.update(len(app_prevalences))

            pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="exp_027 - compare cluster models vs DyS(Topsoe)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n-prev", type=int, default=N_PREV)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--datasets", nargs="*", default=QUAPY_BINARY_DATASETS)
    parser.add_argument(
        "--output-csv",
        default=os.path.join(os.getcwd(), "exp_027", "exp_027_results.csv"),
    )
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    res = run_experiment(
        datasets=args.datasets,
        batch_size=args.batch_size,
        n_prev=args.n_prev,
        repeats=args.repeats,
        seed=args.seed,
        out_csv=args.output_csv,
    )

    ok = res[res["status"] == "ok"] if "status" in res.columns else res
    if not ok.empty:
        summary = (
            ok.groupby("model", as_index=False)
            .agg(mean_abs_error=("abs_error", "mean"), median_abs_error=("abs_error", "median"))
            .sort_values("mean_abs_error")
        )
        summary_path = os.path.splitext(args.output_csv)[0] + "_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"✅ Summary saved: {summary_path}")

    print(f"✅ Results saved: {args.output_csv}")
    print(f"Rows: {len(res)}")


if __name__ == "__main__":
    main()