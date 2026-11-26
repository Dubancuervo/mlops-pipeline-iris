"""Validación del modelo entrenado.

Comprueba y falla (código 1) si alguna condición no se cumple:
  1. La exactitud en prueba supera el umbral mínimo.
  2. La brecha de sobreajuste (train - test) está acotada.
  3. La inferencia respeta el contrato: acepta el ejemplo válido y rechaza el
     inválido ANTES de llamar al modelo.
"""
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_data  # noqa: E402

ROOT = Path(__file__).resolve().parent
MIN_TEST_ACCURACY = 0.90
MAX_OVERFIT_GAP = 0.10


def predict(record: dict, artifact: dict, contract: dict) -> int:
    """Inferencia protegida por el contrato de datos."""
    errors = validate_data.validate_record(record, contract)
    if errors:
        raise ValueError("entrada rechazada por el contrato: " + "; ".join(errors))
    row = [[float(record[k]) for k in artifact["features"]]]
    return int(artifact["model"].predict(row)[0])


def main() -> int:
    contract = validate_data.load_contract()
    saved = json.loads((ROOT / "metrics" / "metrics.json").read_text(encoding="utf-8"))
    metrics = saved["metrics"]
    artifact = joblib.load(ROOT / "models" / "model.joblib")
    failed = False

    print(f"[validate_model] modelo evaluado: {saved['selected_run']}")
    ok = metrics["accuracy"] >= MIN_TEST_ACCURACY
    print(f"[validate_model] accuracy {metrics['accuracy']} >= {MIN_TEST_ACCURACY}: {'OK' if ok else 'FALLA'}")
    failed |= not ok

    ok = metrics["overfitting_gap"] <= MAX_OVERFIT_GAP
    print(f"[validate_model] overfitting_gap {metrics['overfitting_gap']} <= {MAX_OVERFIT_GAP}: {'OK' if ok else 'FALLA'}")
    failed |= not ok

    pred = predict(contract["examples"]["valid"], artifact, contract)
    label = contract["model"]["class_labels"].get(str(pred), "?")
    ok = pred in contract["model"]["classes"]
    print(f"[validate_model] inferencia con ejemplo válido -> clase {pred} ({label}): {'OK' if ok else 'FALLA'}")
    failed |= not ok

    try:
        predict(contract["examples"]["invalid"], artifact, contract)
        print("[validate_model] ejemplo inválido NO fue rechazado: FALLA")
        failed = True
    except ValueError:
        print("[validate_model] inferencia con ejemplo inválido rechazada por el contrato: OK")

    print("[validate_model] RESULTADO:", "FALLÓ" if failed else "modelo validado")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
