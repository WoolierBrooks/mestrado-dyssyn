# 📚 Research Project — Master's in Computer Science

This repository contains the complete research workflow for my Master's project in Computer Science, including:

```
src/         → Full source code (models, utilities, dataset loaders)
experiments/ → Reproducible experiment runs with configuration logging
results/     → Final figures, tables, and evaluation metrics
docs/        → Thesis, papers, presentations, and meeting notes
data/        → Raw and processed datasets (mostly ignored by Git)
project/     → Project management and administration (timeline, planning, references)
```

This structure ensures that:

* Every experiment is fully reproducible
* Environments are version-controlled
* No results or data are accidentally lost

---

## 🚀 Quick Start

### 1️⃣ Clone this repository

```bash
git clone <repo-url>
cd <repo-folder>
```

---

### 2️⃣ Create Environment

#### ✔️ Option A — Standard Lightweight Install (Recommended)

Recreates only the core dependencies explicitly installed:

```bash
conda env create -f env/environment.yml
conda activate masters
```

#### 🧠 Option B — Fully Reproducible Scientific Environment

Installs **all dependencies**, including low-level and indirect packages:

```bash
conda env create -f env/environment_full.yaml
conda activate masters
```

Use this for audits, publication reviews, results validation, or long-term reproducibility.

---

### 3️⃣ Running Experiments

Experiments follow the structure:

```
experiments/
└── exp_001/
    ├── config.yaml
    ├── environment.yml
    ├── environment_full.yaml
    ├── logs/
    ├── results.csv
    └── figures/
```

To execute an experiment:

```bash
python src/experiments/run.py --config config/model_A.yaml
```

All logs, results, and figures are automatically stored in a new experiment directory.

---

## 🧠 Repository Overview

### 🧩 `src/` — Source Code

```
src/
├── data/        → Dataset loaders and preprocessors
├── models/      → ML models, statistical methods, or algorithms
├── utils/       → Helper functions (metrics, plotting, I/O)
├── experiments/ → Scripts to execute full experiments
└── notebooks/   → Exploratory analysis notebooks
```

---

### 📊 `experiments/` — Reproducible Experiments

Each experiment folder stores:

* `config.yaml` — hyperparameters and settings
* `results.csv` — metrics and experiment output
* `logs/` — runtime logs
* `figures/` — generated plots and visualizations

This guarantees that experiments can be **reproduced exactly as originally executed**, even years later.

---

### 🧪 `data/` — Datasets

```
raw/        → Original data (immutable)
processed/  → Cleaned or transformed datasets
external/   → Datasets from external collaborators
```

Most of this directory is **excluded from Git**, keeping the repository size manageable.

---

### 📁 `docs/` — Thesis & Publications

Includes:

* Drafts and manuscript versions
* Figures and plot exports
* Bibliographies
* Submission-ready versions
* Meeting notes
* Slide decks and research presentations

---

### 📅 `project/` — Research Planning

Common files:

* `roadmap.md` — long-term milestones
* `timeline.md` — Gantt-style execution plan
* `todo.md` — weekly progress tracking
* `references.bib` — central BibTeX citation database
* `admin/` — official project paperwork, funding forms, etc.

---

## 🔧 Environment Management

### Export a clean environment (recommended)

```bash
conda env export --from-history > env/environment.yml
```

### Export a fully pinned, bit-for-bit reproducible environment

```bash
conda env export > env/environment_full.yaml
```

---

## 🧾 License

Add a license here when applicable (MIT, Apache 2.0, GPL, etc.).

---

## 🤝 Contributions

As this is an academic research project, contributions are typically limited to:

* Official collaborators
* Supervisors
* Approved co-authors

---

## 📣 Contact

**Email:** [julio.cardus@unioeste.br](mailto:julio.cardus@unioeste.br)
**Institution:** State University of Western Paraná (UNIOESTE)
**Advisor:** Dr. Willian Zalewski
**Lab:** [Laboratory of Applied Computing (LACA)](https://divulga.unila.edu.br/laca/)

---

## 🔰 Shields

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![Status](https://img.shields.io/badge/Project-In_Progress-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---