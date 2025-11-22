import yaml
from src.data.loaders import load_csv_dataset
from src.models.model_example import build_model
from src.utils.metrics import accuracy
import pandas as pd
import argparse

def run_experiment(config_path):
    cfg = yaml.safe_load(open(config_path))

    df = load_csv_dataset(cfg['dataset'])
    model = build_model()

    X = df.drop(cfg['target'], axis=1)
    y = df[cfg['target']]

    model.fit(X, y)
    preds = model.predict(X)
    acc = accuracy(y, preds)

    print(f"Accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    run_experiment(args.config)
