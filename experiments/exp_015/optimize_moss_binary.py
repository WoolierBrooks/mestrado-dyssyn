import argparse
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class RealDatasetKDE:
    name: str
    kde: np.ndarray
    n_scores: int


# ============================================================
# APP-LIKE BATCH SAMPLING (TARGET PREVALENCE)
# ============================================================

def sample_batch(y, pos_prevalence, batch_size, rng):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    n_pos = int(batch_size * pos_prevalence)
    n_neg = batch_size - n_pos

    idx = np.concatenate([
        rng.choice(pos, n_pos, replace=True),
        rng.choice(neg, n_neg, replace=True),
    ])
    rng.shuffle(idx)
    return idx


def kde_curve(scores, grid):
    scores = np.asarray(scores, dtype=float)
    if len(scores) < 2 or np.allclose(scores.std(), 0.0):
        return np.zeros_like(grid)
    try:
        return gaussian_kde(scores)(grid)
    except Exception:
        return np.zeros_like(grid)


# ============================================================
# REAL DATA REFERENCE KDES
# ============================================================

def load_binary_csv(path):
    df = pd.read_csv(path)
    y = df.iloc[:, -1].to_numpy()
    X = df.iloc[:, :-1].to_numpy()

    classes = np.unique(y)
    if len(classes) != 2:
        return None, None

    y = (y == classes.max()).astype(int)
    return X, y


def build_real_reference_kdes(
    datasets_root,
    grid,
    target_pos_prev=0.30,
    batch_size=100,
    repeats=30,
    seed=42,
    max_datasets=None,
):
    rng = np.random.default_rng(seed)

    files = sorted([f for f in os.listdir(datasets_root) if f.endswith(".csv")])
    if max_datasets is not None:
        files = files[:max_datasets]

    refs = []
    for fname in tqdm(files, desc="KDE real (datasets)", unit="dataset"):
        X, y = load_binary_csv(os.path.join(datasets_root, fname))
        if X is None:
            continue

        try:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.5, stratify=y, random_state=seed
            )
        except Exception:
            continue

        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue

        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xte = scaler.transform(Xte)

        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(Xtr, ytr)

        batches = []
        for _ in tqdm(range(repeats), desc=f"Batches {fname}", unit="batch", leave=False):
            idx = sample_batch(yte, target_pos_prev, batch_size, rng)
            batches.append(clf.predict_proba(Xte[idx])[:, 1])

        scores = np.concatenate(batches)
        refs.append(RealDatasetKDE(name=fname, kde=kde_curve(scores, grid), n_scores=len(scores)))

    if not refs:
        raise RuntimeError("Nenhuma referência real KDE foi gerada. Verifique datasets binários.")

    return refs


# ============================================================
# MOSS GENERATION
# ============================================================

def beta_params_from_mu_phi(mu, phi, eps=1e-6):
    mu = float(np.clip(mu, eps, 1 - eps))
    phi = float(max(phi, 1e-3))
    a = mu * phi
    b = (1 - mu) * phi
    return max(a, eps), max(b, eps)


def generate_moss_candidate(
    trial,
    n_samples,
    n_prevalences,
    n_merges,
    n_curves,
    seed,
):
    rng = np.random.default_rng(seed + trial.number)

    # searchable params
    mu_neg_base = trial.suggest_float("mu_neg_base", 0.02, 0.35)
    mu_pos_base = trial.suggest_float("mu_pos_base", 0.65, 0.98)
    phi_base = trial.suggest_float("phi_base", 8.0, 180.0, log=True)
    phi_decay = trial.suggest_float("phi_decay", 0.5, 8.0)
    mu_pull = trial.suggest_float("mu_pull_to_half", 0.0, 0.85)

    prevalences = np.linspace(0.05, 0.95, n_prevalences)
    merges = np.linspace(0.0, 0.95, n_merges)

    moss = {}
    for p_pos in prevalences:
        alpha_key = (round(1 - p_pos, 4), round(p_pos, 4))

        for merge in merges:
            # move means toward 0.5 as merge increases
            mu_neg = (1 - mu_pull * merge) * mu_neg_base + (mu_pull * merge) * 0.5
            mu_pos = (1 - mu_pull * merge) * mu_pos_base + (mu_pull * merge) * 0.5

            # reduce concentration with merge -> broader distributions
            phi = phi_base / (1.0 + phi_decay * merge)

            a_neg, b_neg = beta_params_from_mu_phi(mu_neg, phi)
            a_pos, b_pos = beta_params_from_mu_phi(mu_pos, phi)

            curves = []
            n_pos = int(np.floor(n_samples * p_pos))
            n_neg = n_samples - n_pos

            for _ in range(n_curves):
                scores = np.zeros((n_samples, 2), dtype=float)

                if n_neg > 0:
                    s_neg = rng.beta(a_neg, b_neg, size=n_neg)
                    scores[:n_neg, 1] = s_neg
                    scores[:n_neg, 0] = 1 - s_neg

                if n_pos > 0:
                    s_pos = rng.beta(a_pos, b_pos, size=n_pos)
                    scores[n_neg:, 1] = s_pos
                    scores[n_neg:, 0] = 1 - s_pos

                rng.shuffle(scores)
                curves.append(scores)

            moss[(alpha_key, round(float(merge), 4))] = curves

    return moss


# ============================================================
# OBJECTIVE
# ============================================================

def make_objective(
    refs,
    grid,
    target_pos_prev,
    n_samples,
    n_prevalences,
    n_merges,
    n_curves,
    seed,
):
    target_neg = 1.0 - target_pos_prev

    def objective(trial):
        moss = generate_moss_candidate(
            trial=trial,
            n_samples=n_samples,
            n_prevalences=n_prevalences,
            n_merges=n_merges,
            n_curves=n_curves,
            seed=seed,
        )

        # keep only prevalence nearest target (compare merges for same prevalence)
        keys = list(moss.keys())
        prev_diffs = [abs(k[0][0] - target_neg) for k in keys]
        min_diff = min(prev_diffs)
        target_keys = [k for k in keys if abs(k[0][0] - target_neg) == min_diff]

        # precompute moss KDE for each candidate key
        moss_kdes = {}
        for k in target_keys:
            all_scores = np.concatenate([np.asarray(c)[:, 1] for c in moss[k]])
            moss_kdes[k] = kde_curve(all_scores, grid)

        losses = []
        for i, ref in enumerate(refs):
            l1s = [float(np.mean(np.abs(ref.kde - mkde))) for mkde in moss_kdes.values()]
            best_l1 = min(l1s)
            losses.append(best_l1)

            running = float(np.mean(losses))
            trial.report(running, step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_l1 = float(np.mean(losses))
        q90_l1 = float(np.quantile(losses, 0.90))

        # weighted objective
        score = 0.75 * mean_l1 + 0.25 * q90_l1

        trial.set_user_attr("mean_l1", mean_l1)
        trial.set_user_attr("q90_l1", q90_l1)
        trial.set_user_attr("target_neg_prev_used", float(target_neg))
        trial.set_user_attr("candidate_moss", moss)

        return score

    return objective


class SaveBestMossCallback:
    def __init__(self, out_pkl):
        self.out_pkl = Path(out_pkl)
        self.best_value = np.inf

    def __call__(self, study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        if trial.value >= self.best_value:
            return

        moss = trial.user_attrs.get("candidate_moss")
        if moss is None:
            return

        self.out_pkl.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_pkl, "wb") as f:
            pickle.dump(moss, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.best_value = trial.value

        meta_path = self.out_pkl.with_suffix(".best.json")
        best_info = {
            "trial_number": trial.number,
            "objective": float(trial.value),
            "mean_l1": float(trial.user_attrs.get("mean_l1", np.nan)),
            "q90_l1": float(trial.user_attrs.get("q90_l1", np.nan)),
            "params": trial.params,
        }
        import json

        with open(meta_path, "w") as f:
            json.dump(best_info, f, indent=2)

        print(f"\n✅ Novo melhor trial={trial.number} objective={trial.value:.6f}")
        print(f"   Salvo: {self.out_pkl}")



def estimate_trials_from_baseline(
    baseline_trials: int,
    baseline_minutes: float,
    deadline_hours: float,
    safety_factor: float = 0.85,
) -> int:
    if baseline_trials <= 0 or baseline_minutes <= 0 or deadline_hours <= 0:
        raise ValueError("Parâmetros inválidos para estimativa de trials")

    trials_per_min = baseline_trials / baseline_minutes
    total_minutes = deadline_hours * 60.0
    estimated = int(trials_per_min * total_minutes * safety_factor)
    return max(1, estimated)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Otimiza MoSS binário com Optuna (KDE-L1)")
    p.add_argument("--datasets-root", default="/var/new_homes/julio/mestrado/mestrado-dyssyn/datasets/binary")
    p.add_argument("--output-pkl", default="moss_outputs/best_moss_binario.pkl")

    p.add_argument("--n-trials", type=int, default=120)
    p.add_argument("--deadline-hours", type=float, default=16.0, help="tempo alvo em horas para execução")
    p.add_argument("--baseline-trials", type=int, default=120, help="trials medidos no benchmark")
    p.add_argument("--baseline-minutes", type=float, default=1.33, help="tempo (min) do benchmark")
    p.add_argument("--safety-factor", type=float, default=0.85, help="fator de segurança para estimativa de trials")
    p.add_argument("--timeout", type=int, default=None, help="segundos")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--target-pos-prev", type=float, default=0.30)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--max-datasets", type=int, default=None)

    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--n-prevalences", type=int, default=19)
    p.add_argument("--n-merges", type=int, default=13)
    p.add_argument("--n-curves", type=int, default=30)

    p.add_argument("--study-name", default="optimize_moss_binary")
    p.add_argument("--storage", default=None, help="ex: sqlite:///optuna_moss.db")

    return p.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    grid = np.linspace(0, 1, 300)

    print("🔎 Preparando referências reais KDE...")
    refs = build_real_reference_kdes(
        datasets_root=args.datasets_root,
        grid=grid,
        target_pos_prev=args.target_pos_prev,
        batch_size=args.batch_size,
        repeats=args.repeats,
        seed=args.seed,
        max_datasets=args.max_datasets,
    )
    print(f"   Referências construídas: {len(refs)} datasets")

    objective = make_objective(
        refs=refs,
        grid=grid,
        target_pos_prev=args.target_pos_prev,
        n_samples=args.n_samples,
        n_prevalences=args.n_prevalences,
        n_merges=args.n_merges,
        n_curves=args.n_curves,
        seed=args.seed,
    )

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=8)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )

    callback = SaveBestMossCallback(args.output_pkl)

    effective_n_trials = args.n_trials
    effective_timeout = args.timeout

    if args.deadline_hours is not None and args.deadline_hours > 0:
        estimated_trials = estimate_trials_from_baseline(
            baseline_trials=args.baseline_trials,
            baseline_minutes=args.baseline_minutes,
            deadline_hours=args.deadline_hours,
            safety_factor=args.safety_factor,
        )
        effective_n_trials = max(effective_n_trials, estimated_trials)
        if effective_timeout is None:
            effective_timeout = int(args.deadline_hours * 3600)
        print(
            "⏱️ Planejamento de execução: "
            f"deadline={args.deadline_hours}h, "
            f"estimativa_trials={estimated_trials}, "
            f"n_trials_final={effective_n_trials}, "
            f"timeout_s={effective_timeout}"
        )

    t0 = time.perf_counter()
    study.optimize(
        objective,
        n_trials=effective_n_trials,
        timeout=effective_timeout,
        callbacks=[callback],
        gc_after_trial=True,
        show_progress_bar=True,
    )
    dt = time.perf_counter() - t0

    print("\n================ RESULTADO ================")
    print(f"Trials: {len(study.trials)}")
    print(f"Tempo: {dt/60:.2f} min")
    print(f"Best objective: {study.best_value:.6f}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best params: {study.best_trial.params}")
    print(f"Best mean_l1: {study.best_trial.user_attrs.get('mean_l1')}")
    print(f"Best q90_l1: {study.best_trial.user_attrs.get('q90_l1')}")
    print(f"Best pkl path: {args.output_pkl}")


if __name__ == "__main__":
    main()
