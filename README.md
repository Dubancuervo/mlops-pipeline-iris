# Pipeline MLOps reproducible con control de calidad de datos

Pipeline que entrena y valida el clasificador **RandomForest** de la línea base del Proyecto_Final_Grado (dataset Iris: 150 flores, 4 medidas en centímetros, 3 especies) con **un solo comando**. Incorpora validación de datos de entrada sin librerías externas y un contrato de datos en JSON para la futura inferencia.

Actividad 2.1 — *Estandarización de Pipelines y Control de Calidad de Datos* (MLOps, semana 2). Continúa la línea base reproducible de la semana 1.

## Tabla de contenido

1. [Objetivo](#objetivo)
2. [Requisitos previos](#requisitos-previos)
3. [Reproducción rápida (un comando)](#reproducción-rápida-un-comando)
4. [Estructura del repositorio](#estructura-del-repositorio)
5. [Etapas del pipeline](#etapas-del-pipeline)
6. [Validación de datos](#validación-de-datos)
7. [Contrato de datos](#contrato-de-datos)
8. [Resultados esperados](#resultados-esperados)
9. [Reproducibilidad](#reproducibilidad)
10. [Reproducción paso a paso (sin Make)](#reproducción-paso-a-paso-sin-make)
11. [Solución de problemas](#solución-de-problemas)
12. [Autoría y contexto](#autoría-y-contexto)
13. [Referencias](#referencias)

## Objetivo

Garantizar que cualquier persona pueda reproducir el entrenamiento desde cero y que ningún dato defectuoso llegue al modelo, ni en entrenamiento ni en inferencia.

## Requisitos previos

| Requisito | Versión probada |
|---|---|
| Python | 3.13 (se recomienda 3.10 o superior) |
| GNU Make | 4.x (Linux, macOS o WSL en Windows) |
| Acceso a internet | Solo para instalar dependencias con `pip` |

Las dependencias están fijadas en `requirements.txt` (`mlflow==3.18.0`, `scikit-learn==1.9.1`, `joblib==1.6.0`, `numpy==2.5.3`). La validación de datos y sus pruebas no usan ninguna librería externa.

## Reproducción rápida (un comando)

```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>
make all
```

`make all` crea el entorno virtual `.venv`, instala dependencias, ejecuta las pruebas del validador, genera los datos, valida contrato y datos, entrena las configuraciones, selecciona el modelo y lo valida. Si cualquier etapa falla, el proceso se detiene con código de salida distinto de cero. Para ver todos los comandos: `make help`.

## Estructura del repositorio

```text
.
├── Makefile                  # Orquestación del pipeline
├── README.md                 # Este documento
├── requirements.txt          # Dependencias fijadas
├── data_contract.json        # Contrato de datos oficial (esquema + ejemplos)
├── prepare_data.py           # Genera data/iris.csv (determinista)
├── validate_data.py          # Validación de tipos, nulos y rangos (solo biblioteca estándar)
├── train.py                  # Entrenamiento con MLflow; valida los datos antes de entrenar
├── validate_model.py         # Umbrales de métricas e inferencia protegida por contrato
├── test_validate_data.py     # 17 pruebas unitarias del validador (unittest)
├── .github/workflows/ci.yml  # Integración continua: ejecuta `make all` en cada push
├── data/                     # Generado: iris.csv
├── models/                   # Generado: model.joblib
├── metrics/                  # Generado: metrics.json
└── mlflow.db                 # Generado: experimentos de MLflow (SQLite)
```

## Etapas del pipeline

El orden de ejecución lo define el Makefile mediante dependencias entre objetivos:

| Objetivo | Qué hace |
|---|---|
| `make setup` | Crea `.venv` e instala `requirements.txt` |
| `make test` | Ejecuta las 17 pruebas unitarias del validador (tipos, nulos, rangos, límites, estructura y CSV) |
| `make data` | Genera `data/iris.csv` con `load_iris` de scikit-learn |
| `make validate-contract` | Verifica que el ejemplo válido pase y el inválido sea rechazado |
| `make validate-data` | Valida tipos, nulos y rangos de las 150 filas |
| `make train` | Entrena Baseline y 3 runs (experimento `Proyecto_Final_Grado` en MLflow), selecciona el de menor sobreajuste y guarda `models/model.joblib` y `metrics/metrics.json` |
| `make validate-model` | Comprueba umbrales de métricas, sobreajuste e inferencia con el contrato |
| `make clean` | Borra los artefactos generados |

Las configuraciones son las de la semana 1 (`n_estimators` / `max_depth`: 100/None, 50/3, 150/5, 200/10), con semilla fija (`random_state=42`) y división 80 % / 20 %, de modo que los resultados son idénticos entre ejecuciones. El modelo conservado es el de menor brecha de sobreajuste (entrenamiento menos prueba), con desempate por mayor exactitud en validación cruzada.

## Validación de datos

`validate_data.py` usa únicamente la biblioteca estándar de Python (`csv`, `json`, `argparse`). Verifica, para cada registro:

- **Tipos de datos:** cada variable debe poder interpretarse como el tipo declarado en el contrato (`float`); se rechazan textos no numéricos y booleanos.
- **Ausencia de nulos:** se rechazan `null`, cadenas vacías, `nan`, `none` y `na`.
- **Rangos lógicos:** cada variable debe estar entre `min` y `max` del contrato.
- **Estructura:** se rechazan campos faltantes o no definidos en el contrato, y clases de `target` fuera de `[0, 1, 2]`.

Uso manual:

```bash
python validate_data.py --csv data/iris.csv      # valida un CSV completo
python validate_data.py --json mi_entrada.json   # valida un registro individual
python validate_data.py --examples               # prueba los ejemplos del contrato
```

Devuelve código 0 si todo es válido y 1 si hay errores, con un mensaje por cada problema encontrado.

## Contrato de datos

`data_contract.json` es la fuente única de verdad para la inferencia. Contiene:

- `model`: nombre, tarea, variable objetivo, clases y etiquetas (setosa, versicolor, virginica).
- `rules`: política de nulos y de campos faltantes o extra.
- `features`: nombre, tipo, rango (`min`, `max`), unidad y descripción de las 4 variables.
- `examples.valid`: ejemplo de entrada válida (una flor setosa típica).
- `examples.invalid`: ejemplo de entrada inválida (texto en `sepal_length_cm`, nulo en `sepal_width_cm`, `petal_length_cm` = 15.0 fuera de rango) junto con los errores esperados.

Tanto `train.py` como `validate_model.py` leen este archivo, por lo que el esquema de entrenamiento y el de inferencia nunca se desincronizan.

## Resultados esperados

Con la configuración por defecto:

```text
[validate_data] OK: 150 filas validadas (tipos, nulos y rangos)
[train] modelo seleccionado: Run_1_n50_depth3 (menor sobreajuste)
[validate_model] inferencia con ejemplo válido -> clase 0 (setosa): OK
[validate_model] RESULTADO: modelo validado
```

Criterios de aceptación del modelo (`validate_model.py`): exactitud en prueba ≥ 0.90 y brecha entrenamiento-prueba ≤ 0.10.

Los runs quedan en `mlflow.db`; se pueden comparar con `.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Reproducibilidad

Siguiendo la lista de verificación del programa de reproducibilidad de NeurIPS (Pineau et al., 2021), el proyecto declara de forma explícita cada elemento que influye en el resultado. El código completo está en este repositorio y los datos se regeneran de forma determinista con `load_iris`, sin descargas externas. Las versiones de todas las dependencias están fijadas en `requirements.txt` y se probó con Python 3.13.18. Los hiperparámetros de las cuatro configuraciones están definidos en `train.py`, y la semilla (`random_state=42`) y la división 80 % / 20 % son fijas. Las métricas reportadas son la exactitud en prueba y en entrenamiento, la brecha de sobreajuste y la validación cruzada de cinco pliegues (media y desviación estándar). Todo el proceso se ejecuta en CPU en pocos segundos. Además, cada push al repositorio ejecuta `make all` mediante GitHub Actions (`.github/workflows/ci.yml`), lo que verifica de forma continua que el pipeline sigue reproduciéndose desde cero.

## Reproducción paso a paso (sin Make)

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s . -p "test_*.py" -v
python prepare_data.py
python validate_data.py --examples
python validate_data.py --csv data/iris.csv
python train.py
python validate_model.py
```

## Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `make: command not found` | Make no instalado | Instalar Make o usar la sección sin Make |
| `ModuleNotFoundError` | Entorno virtual no activo | Usar `make all` o activar `.venv` |
| `datos inválidos; entrenamiento cancelado` | El CSV incumple el contrato | Revisar las filas indicadas por `validate_data.py` |
| Error de MLflow con `mlruns` | Versiones recientes exigen backend de base de datos | Este proyecto ya usa SQLite (`mlflow.db`) |

## Autoría y contexto

Duban Fernando Cuervo Martínez. Especialización en Ciencia de Datos, Universidad Santo Tomás, Seccional Tunja. Seminario de Grado 2 (MLOps). Docente: Martha Susana Contreras Ortiz. Noviembre de 2026.

## Referencias

Camp BI. (2025, 19 de septiembre). *La mejor forma de crear pipelines de datos en Python* [Video]. YouTube. https://www.youtube.com/watch?v=zmSo35zs_6U

Mensh, B., & Kording, K. (2017). Ten simple rules for structuring papers. *PLoS Computational Biology, 13*(9), e1005619. https://www.biorxiv.org/content/10.1101/088278v5.full.pdf

Pineau, J., et al. (2021). Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program). *Journal of Machine Learning Research, 22*(164), 1-20. https://www.jmlr.org/papers/volume22/20-303/20-303.pdf

TechWorld with Nana. (2025, 13 de noviembre). *MLOps explained - what it is, why you need it and how it works* [Video]. YouTube. https://www.youtube.com/watch?v=biqYkVf-a7Y
