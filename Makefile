# Automatización del pipeline: `make all` ejecuta todo con un solo comando.
# Flujo: entorno -> pruebas -> datos -> validación de datos -> entrenamiento -> validación del modelo

PYTHON ?= python3
VENV   := .venv
PY     := $(VENV)/bin/python

.PHONY: all setup test data validate-data validate-contract train validate-model clean help

help:  ## Muestra los comandos disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-18s %s\n", $$1, $$2}'

all: setup test data validate-contract validate-data train validate-model  ## Pipeline completo
	@echo "Pipeline completado correctamente."

setup: $(VENV)/.installed  ## Crea el entorno virtual e instala dependencias

$(VENV)/.installed: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -r requirements.txt
	@touch $@

test: setup  ## Ejecuta las pruebas unitarias del validador de datos
	$(PY) -m unittest discover -s . -p "test_*.py" -v

data: setup  ## Genera data/iris.csv
	$(PY) prepare_data.py

validate-contract: setup  ## Comprueba que los ejemplos del contrato se comportan como se espera
	$(PY) validate_data.py --examples

validate-data: data  ## Valida tipos, nulos y rangos del CSV
	$(PY) validate_data.py --csv data/iris.csv

train: validate-data  ## Entrena el modelo (solo si los datos son válidos)
	$(PY) train.py

validate-model: train  ## Verifica métricas, sobreajuste e inferencia con el contrato
	$(PY) validate_model.py

clean:  ## Elimina artefactos generados (conserva el entorno virtual)
	rm -rf data/iris.csv models metrics mlruns mlflow.db
