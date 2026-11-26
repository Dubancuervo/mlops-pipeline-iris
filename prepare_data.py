"""Genera data/iris.csv a partir del dataset Iris incluido en scikit-learn.

Es el mismo dataset usado en la línea base de la semana 1 (load_iris). El paso
es determinista: no requiere descargas ni credenciales.
"""
import csv
import sys
from pathlib import Path

from sklearn.datasets import load_iris

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "iris.csv"

# Nombres de columna sin espacios ni paréntesis.
COLUMNS = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm", "target"]


def main() -> int:
    bunch = load_iris()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for row, label in zip(bunch.data, bunch.target):
            writer.writerow([*map(float, row), int(label)])
    print(f"[prepare_data] {len(bunch.target)} filas escritas en {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
