"""Entrena el RandomForest de la línea base (semana 1) como pipeline automatizado.

1. Valida el CSV contra el contrato de datos; si hay errores, se detiene.
2. Ejecuta las mismas 4 configuraciones de la semana 1 (Baseline + 3 runs),
   registrando parámetros y métricas en MLflow (SQLite, experimento
   "Proyecto_Final_Grado").
3. Conserva el modelo con menor sobreajuste (criterio de la semana 1) y lo guarda
   en models/model.joblib junto con metrics/metrics.json.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_data  # noqa: E402

ROOT = Path(__file__).resolve().parent
SEED = 42

CONFIGS = [
    {"run_name": "Baseline",           "n_estimators": 100, "max_depth": None},
    {"run_name": "Run_1_n50_depth3",   "n_estimators": 50,  "max_depth": 3},
    {"run_name": "Run_2_n150_depth5",  "n_estimators": 150, "max_depth": 5},
    {"run_name": "Run_3_n200_depth10", "n_estimators": 200, "max_depth": 10},
]


def load_xy(path, features, target):
    X, y = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            X.append([float(row[c]) for c in features])
            y.append(int(row[target]))
    return X, y


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data" / "iris.csv"))
    args = p.parse_args()

    contract = validate_data.load_contract()
    n, problems = validate_data.validate_csv(args.data, contract)
    if problems:
        print(f"[train] datos inválidos ({len(problems)} filas con errores); entrenamiento cancelado.")
        return 1
    print(f"[train] datos válidos: {n} filas")

    features = [f["name"] for f in contract["features"]]
    X, y = load_xy(args.data, features, contract["model"]["target"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("Proyecto_Final_Grado")

    best = None
    for cfg in CONFIGS:
        with mlflow.start_run(run_name=cfg["run_name"]):
            params = {"n_estimators": cfg["n_estimators"], "max_depth": cfg["max_depth"], "random_state": SEED}
            model = RandomForestClassifier(**params).fit(X_train, y_train)

            accuracy = model.score(X_test, y_test)
            train_accuracy = model.score(X_train, y_train)
            gap = train_accuracy - accuracy
            cv = cross_val_score(model, X, y, cv=5)
            metrics = {
                "accuracy": round(float(accuracy), 4),
                "train_accuracy": round(float(train_accuracy), 4),
                "overfitting_gap": round(float(gap), 4),
                "cv_mean_accuracy": round(float(cv.mean()), 4),
                "cv_std_accuracy": round(float(cv.std()), 4),
            }
            for k, v in params.items():
                mlflow.log_param(k, v)
            mlflow.log_metrics(metrics)
            print(f"[train] {cfg['run_name']}: {params} -> {metrics}")

            # Criterio de selección: menor brecha de sobreajuste; desempate por mayor CV.
            key = (metrics["overfitting_gap"], -metrics["cv_mean_accuracy"])
            if best is None or key < best["key"]:
                best = {"key": key, "run_name": cfg["run_name"], "params": params,
                        "metrics": metrics, "model": model}

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "metrics").mkdir(exist_ok=True)
    joblib.dump({"model": best["model"], "features": features}, ROOT / "models" / "model.joblib")
    (ROOT / "metrics" / "metrics.json").write_text(
        json.dumps({"selected_run": best["run_name"], "params": best["params"], "metrics": best["metrics"]},
                   indent=2),
        encoding="utf-8",
    )
    print(f"[train] modelo seleccionado: {best['run_name']} (menor sobreajuste)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
