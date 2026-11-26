"""Validación de datos de entrada sin librerías externas (solo biblioteca estándar).

Verifica, contra el contrato ``data_contract.json``:
  1. Tipos de datos correctos.
  2. Ausencia de valores nulos.
  3. Rangos lógicos por variable.
Además comprueba columnas faltantes o inesperadas.

Uso:
    python validate_data.py --csv data/iris.csv     # valida un CSV completo
    python validate_data.py --examples              # valida los ejemplos del contrato
    python validate_data.py --json entrada.json     # valida un registro JSON

Código de salida: 0 si todo es válido, 1 si hay errores.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CONTRACT = ROOT / "data_contract.json"
NULL_TOKENS = {"", "null", "none", "nan", "na", "n/a"}


def load_contract(path=DEFAULT_CONTRACT) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _is_null(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in NULL_TOKENS:
        return True
    return isinstance(value, float) and value != value  # NaN


def _convert(value, expected_type: str):
    """Devuelve (valor_convertido, error). Acepta cadenas numéricas (caso CSV),
    pero rechaza bool (en Python bool es subclase de int) y texto no numérico."""
    if isinstance(value, bool):
        return None, "booleano no es un valor numérico"
    if expected_type == "float":
        if isinstance(value, (int, float)):
            return float(value), None
        if isinstance(value, str):
            try:
                return float(value), None
            except ValueError:
                return None, f"no se puede convertir '{value}' a float"
    elif expected_type == "int":
        if isinstance(value, int):
            return value, None
        if isinstance(value, float) and value.is_integer():
            return int(value), None
        if isinstance(value, str):
            try:
                return int(value), None
            except ValueError:
                return None, f"no se puede convertir '{value}' a int"
    return None, f"tipo inválido (se esperaba {expected_type})"


def validate_record(record: dict, contract: dict) -> list:
    """Valida un registro (dict). Devuelve la lista de errores ([] si es válido)."""
    errors = []
    rules = contract.get("rules", {})
    names = [f["name"] for f in contract["features"]]

    if not rules.get("allow_missing_fields", False):
        for name in names:
            if name not in record:
                errors.append(f"{name}: campo requerido ausente")
    if not rules.get("allow_extra_fields", False):
        for name in record:
            if name not in names and name != contract["model"]["target"]:
                errors.append(f"{name}: campo no definido en el contrato")

    for feat in contract["features"]:
        name = feat["name"]
        if name not in record:
            continue
        value = record[name]
        if _is_null(value):
            if not rules.get("allow_null", False):
                errors.append(f"{name}: valor nulo no permitido")
            continue
        number, err = _convert(value, feat["type"])
        if err:
            errors.append(f"{name}: {err}")
            continue
        lo, hi = feat.get("min"), feat.get("max")
        if (lo is not None and number < lo) or (hi is not None and number > hi):
            errors.append(f"{name}: valor {number} fuera de rango [{lo}, {hi}]")
    return errors


def validate_csv(path, contract: dict):
    """Valida cada fila de un CSV. Devuelve (n_filas, {numero_fila: [errores]})."""
    problems = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        expected = [x["name"] for x in contract["features"]] + [contract["model"]["target"]]
        missing = [c for c in expected if c not in header]
        if missing:
            return 0, {0: [f"columnas ausentes en el CSV: {missing}"]}
        n = 0
        for n, row in enumerate(reader, start=1):
            errs = validate_record(row, contract)
            target = row.get(contract["model"]["target"])
            if _is_null(target):
                errs.append("target: valor nulo no permitido")
            elif target not in {str(c) for c in contract["model"]["classes"]}:
                errs.append(f"target: clase '{target}' no está en {contract['model']['classes']}")
            if errs:
                problems[n] = errs
    return n, problems


def _check_examples(contract: dict) -> int:
    ok_errors = validate_record(contract["examples"]["valid"], contract)
    bad_errors = validate_record(contract["examples"]["invalid"], contract)
    print("[contrato] ejemplo VÁLIDO  ->", "OK, sin errores" if not ok_errors else f"FALLA: {ok_errors}")
    print("[contrato] ejemplo INVÁLIDO ->", f"{len(bad_errors)} errores detectados")
    for e in bad_errors:
        print("   -", e)
    if ok_errors or not bad_errors:
        print("[contrato] ERROR: los ejemplos del contrato no se comportan como se espera.")
        return 1
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", help="ruta del CSV a validar")
    g.add_argument("--json", help="ruta de un registro JSON a validar")
    g.add_argument("--examples", action="store_true", help="probar ejemplos del contrato")
    args = p.parse_args(argv)
    contract = load_contract(args.contract)

    if args.examples:
        return _check_examples(contract)
    if args.json:
        with open(args.json, encoding="utf-8") as f:
            errs = validate_record(json.load(f), contract)
        print("[validate_data] registro válido" if not errs else "\n".join(f"- {e}" for e in errs))
        return 1 if errs else 0

    n, problems = validate_csv(args.csv, contract)
    if problems:
        print(f"[validate_data] {len(problems)} de {n} filas con errores:")
        for row, errs in list(problems.items())[:20]:
            for e in errs:
                print(f"   fila {row}: {e}")
        return 1
    print(f"[validate_data] OK: {n} filas validadas (tipos, nulos y rangos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
