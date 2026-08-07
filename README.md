# Collection center location: evolutionary algorithm execution

This repository contains a version of the project for running optimization algorithms on an agricultural collection center location instance.

The purpose of this version is to allow another user to download the repository, use the included instance or replace it with their own data, run the algorithms, and generate experimental results.

## Repository structure

```text
.
├── data/
│   ├── raw/             # Original reference data
│   └── processed/       # Minimum processed instance required by the algorithms
├── src/                 # Scripts for data construction and algorithm execution
├── requirements.txt
├── .gitignore
└── README.md
```

## Included scripts

The `src/` directory contains the scripts required to prepare the instance and run the experiments:

- `build_municipios_productores.py`: builds the municipal production file from the original data.
- `build_distancias.py`: builds the Haversine distance matrix.
- `problem_definition.py`: loads and validates the problem instance.
- `baselines.py`: runs baseline methods used to compare against the mono-objective genetic algorithm.
- `ga_monoobjective.py`: runs the mono-objective genetic algorithm.
- `multiobjective_problem.py`: defines the bi-objective problem for PYMOO.
- `nsga2_multiobjective.py`: runs NSGA-II for the bi-objective formulation.
- `three_objective_problem.py`: defines the three-objective problem.
- `nsga2_three_objective.py`: runs NSGA-II for the three-objective formulation.

## Included data

The `data/raw/` directory keeps the original reference files:

- DGSIAP/SADER: municipal agricultural closing dataset and data dictionary.
- INEGI: AGEEML catalogue with official geographic coordinates.

The `data/processed/` directory contains the minimum ready-to-run instance:

- `municipios_productores.csv`
- `distancias.csv`

`municipios_productores.csv` must have the following structure:

```text
clave_inegi,municipio,latitud,longitud,produccion_ton
```

`distancias.csv` contains the origin-destination distance matrix between municipalities.

## Using another dataset

To run the algorithms with another instance, replace the files in `data/processed/` while preserving the same structure.

### `municipios_productores.csv`

Required columns:

- `clave_inegi`: unique identifier of the municipality, locality, or demand point.
- `municipio`: name of the point.
- `latitud`: decimal latitude.
- `longitud`: decimal longitude.
- `produccion_ton`: weight, demand, or production associated with the point.

### `distancias.csv`

This file must contain a distance matrix compatible with the same identifiers used in `municipios_productores.csv`.

Requirements:

- distances expressed in kilometers;
- zero diagonal;
- same municipalities or points as the production file;
- no missing values.

If using the original files included in this case study, the two processed files can be rebuilt by running the data construction scripts.

## Installation

Python 3.10 or later is recommended.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Recommended execution

All commands must be executed from the repository root.

### 1. Validate the instance

```bash
python src/problem_definition.py
```

This step checks that the processed files can be loaded correctly and that the distance matrix is compatible with the municipalities.

### 2. Run baselines

```bash
python src/baselines.py
```

Outputs:

```text
data/results/baselines_resultados.csv
data/results/baselines_centros.csv
data/results/baselines_cargas.csv
figures/comparacion_baselines.svg
```

This step should be executed before `ga_monoobjective.py` if `comparacion_ga_baselines.svg` is expected to include the baseline methods. If `data/results/baselines_resultados.csv` does not exist, that figure will only show the GA result.

### 3. Run the mono-objective genetic algorithm

```bash
python src/ga_monoobjective.py
```

This script solves the mono-objective problem for fixed values of `p`, minimizing the production-weighted average distance.

Outputs:

```text
data/results/ga_resultados.csv
data/results/ga_centros.csv
data/results/ga_cargas.csv
data/results/ga_convergencia.csv
figures/comparacion_ga_baselines.svg
figures/convergencia_ga.svg
```

### 4. Run bi-objective NSGA-II

```bash
python src/nsga2_multiobjective.py
```

Objectives:

1. minimize production-weighted average distance;
2. minimize maximum service distance.

Outputs:

```text
data/results/nsga2_resultados.csv
data/results/nsga2_frentes.csv
data/results/nsga2_centros.csv
data/results/nsga2_cargas.csv
figures/pareto_fronts_by_p.svg
```

### 5. Run three-objective NSGA-II

```bash
python src/nsga2_three_objective.py
```

Objectives:

1. minimize production-weighted average distance;
2. minimize maximum service distance;
3. minimize the number of selected centers.

Outputs:

```text
data/results/nsga2_3obj_resultados.csv
data/results/nsga2_3obj_frentes.csv
data/results/nsga2_3obj_centros.csv
data/results/nsga2_3obj_cargas.csv
figures/pareto_3obj_tradeoff.svg
figures/pareto_3obj_3d.svg
```

The figure `pareto_3obj_tradeoff.svg` shows the trade-off between `f1` and `f2`, with points colored by the number of centers. The figure `pareto_3obj_3d.svg` shows a three-dimensional projection of the Pareto front using `f1`, `f2`, and `f3`.

## CSV-only execution

To generate only CSV result files and avoid diagnostic figures in the scripts that support it:

```bash
python src/problem_definition.py
python src/baselines.py
python src/ga_monoobjective.py --skip-figures
python src/nsga2_multiobjective.py --skip-figures
python src/nsga2_three_objective.py --skip-figures
```

Note: `baselines.py` generates a simple baseline comparison figure. This figure is diagnostic and can be removed if it is not needed.

## Rebuilding the processed data

To rebuild the Tamaulipas instance from the original files included in `data/raw/`, run:

```bash
python src/build_municipios_productores.py
python src/build_distancias.py
```

This generates:

```text
data/processed/municipios_productores.csv
data/processed/distancias.csv
```

After rebuilding the data, run:

```bash
python src/problem_definition.py
python src/baselines.py
python src/ga_monoobjective.py
python src/nsga2_multiobjective.py
python src/nsga2_three_objective.py
```

## Full recommended order

To reproduce the experiments using the included instance:

```bash
pip install -r requirements.txt
python src/problem_definition.py
python src/baselines.py
python src/ga_monoobjective.py
python src/nsga2_multiobjective.py
python src/nsga2_three_objective.py
```

To rebuild the processed data first:

```bash
pip install -r requirements.txt
python src/build_municipios_productores.py
python src/build_distancias.py
python src/problem_definition.py
python src/baselines.py
python src/ga_monoobjective.py
python src/nsga2_multiobjective.py
python src/nsga2_three_objective.py
```

## Generated outputs

The scripts generate outputs in:

```text
data/results/
figures/
```

These folders are not included initially because they are derived products generated by running the code.
