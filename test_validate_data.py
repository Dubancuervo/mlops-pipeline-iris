"""Pruebas del validador de datos (solo biblioteca estándar: unittest).

Ejecutar con:  python -m unittest discover -s . -p "test_*.py" -v   (o `make test`)
"""
import copy
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_data  # noqa: E402

CONTRACT = validate_data.load_contract()
VALID = CONTRACT["examples"]["valid"]


def record(**changes):
    r = copy.deepcopy(VALID)
    r.update(changes)
    return r


class TiposDeDatos(unittest.TestCase):
    def test_registro_valido(self):
        self.assertEqual(validate_data.validate_record(VALID, CONTRACT), [])

    def test_texto_no_numerico(self):
        errs = validate_data.validate_record(record(sepal_length_cm="cinco"), CONTRACT)
        self.assertEqual(len(errs), 1)
        self.assertIn("sepal_length_cm", errs[0])

    def test_booleano_rechazado(self):
        self.assertTrue(validate_data.validate_record(record(petal_width_cm=True), CONTRACT))

    def test_cadena_numerica_aceptada(self):  # caso típico al leer un CSV
        self.assertEqual(validate_data.validate_record(record(sepal_length_cm="5.1"), CONTRACT), [])

    def test_entero_aceptado_como_float(self):
        self.assertEqual(validate_data.validate_record(record(sepal_length_cm=5), CONTRACT), [])


class Nulos(unittest.TestCase):
    def test_none(self):
        errs = validate_data.validate_record(record(sepal_width_cm=None), CONTRACT)
        self.assertIn("nulo", errs[0])

    def test_cadena_vacia_y_tokens_nulos(self):
        for token in ["", "  ", "NaN", "null", "NA"]:
            with self.subTest(token=token):
                self.assertTrue(validate_data.validate_record(record(sepal_width_cm=token), CONTRACT))

    def test_nan_float(self):
        self.assertTrue(validate_data.validate_record(record(sepal_width_cm=float("nan")), CONTRACT))


class Rangos(unittest.TestCase):
    def test_limites_inclusivos(self):
        for feat in CONTRACT["features"]:
            for edge in (feat["min"], feat["max"]):
                with self.subTest(feature=feat["name"], value=edge):
                    self.assertEqual(validate_data.validate_record(record(**{feat["name"]: edge}), CONTRACT), [])

    def test_fuera_de_rango(self):
        for feat in CONTRACT["features"]:
            for bad in (feat["min"] - 0.01, feat["max"] + 0.01):
                with self.subTest(feature=feat["name"], value=bad):
                    errs = validate_data.validate_record(record(**{feat["name"]: bad}), CONTRACT)
                    self.assertTrue(any("fuera de rango" in e for e in errs))

    def test_infinito_rechazado(self):
        self.assertTrue(validate_data.validate_record(record(petal_length_cm=float("inf")), CONTRACT))


class Estructura(unittest.TestCase):
    def test_campo_ausente(self):
        r = copy.deepcopy(VALID)
        del r["petal_width_cm"]
        errs = validate_data.validate_record(r, CONTRACT)
        self.assertIn("ausente", errs[0])

    def test_campo_extra(self):
        errs = validate_data.validate_record(record(color="rojo"), CONTRACT)
        self.assertIn("no definido", errs[0])


class ContratoYCsv(unittest.TestCase):
    def test_ejemplos_del_contrato(self):
        self.assertEqual(validate_data.validate_record(CONTRACT["examples"]["valid"], CONTRACT), [])
        errs = validate_data.validate_record(CONTRACT["examples"]["invalid"], CONTRACT)
        self.assertEqual(errs, CONTRACT["examples"]["invalid_expected_errors"])

    def _csv(self, rows):
        header = [f["name"] for f in CONTRACT["features"]] + [CONTRACT["model"]["target"]]
        tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="", encoding="utf-8")
        writer = csv.writer(tmp)
        writer.writerow(header)
        writer.writerows(rows)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return tmp.name

    def test_csv_valido(self):
        n, problems = validate_data.validate_csv(self._csv([[5.1, 3.5, 1.4, 0.2, 0], [6.3, 3.3, 6.0, 2.5, 2]]), CONTRACT)
        self.assertEqual((n, problems), (2, {}))

    def test_csv_con_errores_reporta_fila(self):
        n, problems = validate_data.validate_csv(
            self._csv([[5.1, 3.5, 1.4, 0.2, 0], ["x", "", 99, 0.2, 7]]), CONTRACT
        )
        self.assertEqual(n, 2)
        self.assertEqual(list(problems), [2])
        self.assertGreaterEqual(len(problems[2]), 4)

    def test_csv_con_columna_faltante(self):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
        tmp.write("sepal_length_cm,target\n5.1,0\n")
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        _, problems = validate_data.validate_csv(tmp.name, CONTRACT)
        self.assertIn(0, problems)


if __name__ == "__main__":
    unittest.main()
