"""
Definición del problema multiobjetivo con p fijo.

Problema:
- Seleccionar exactamente p cabeceras municipales como centros de acopio.
- Asignar cada municipio al centro seleccionado más cercano.

Objetivos:
- f1: minimizar distancia promedio ponderada por producción.
- f2: minimizar distancia máxima de atención.

Restricción:
- Exactamente p centros seleccionados.

Este módulo define evaluación y validación. No ejecuta todavía NSGA-II.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from problem_definition import (
    ProblemData,
    assign_to_nearest_center,
    evaluate_solution,
    load_problem_data,
    summarize_center_loads,
    validate_solution,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEST_MONOOBJECTIVE_PATH = PROJECT_ROOT / "data" / "results" / "monoobjective_best_solutions.csv"


@dataclass(frozen=True)
class MultiObjectiveEvaluation:
    """Resultado de evaluar una solución multiobjetivo."""

    p: int
    centros: list[str]
    f1_distancia_promedio_ponderada_km: float
    f2_distancia_maxima_km: float
    distancia_total_ponderada_ton_km: float
    produccion_total_ton: float
    municipios_atendidos: int


def evaluate_multiobjective_solution(
    problem_data: ProblemData,
    centros: Iterable[str],
    p: int,
) -> MultiObjectiveEvaluation:
    """
    Evalúa una solución con dos objetivos.

    f1 = distancia promedio ponderada por producción.
    f2 = distancia máxima de atención.
    """
    centros_validos = validate_solution(centros, problem_data, p)
    metricas = evaluate_solution(problem_data, centros_validos, p)

    return MultiObjectiveEvaluation(
        p=p,
        centros=centros_validos,
        f1_distancia_promedio_ponderada_km=float(metricas["distancia_promedio_ponderada_km"]),
        f2_distancia_maxima_km=float(metricas["distancia_maxima_km"]),
        distancia_total_ponderada_ton_km=float(metricas["distancia_total_ponderada_ton_km"]),
        produccion_total_ton=float(metricas["produccion_total_ton"]),
        municipios_atendidos=int(metricas["municipios_atendidos"]),
    )


def evaluate_binary_vector(
    problem_data: ProblemData,
    x: np.ndarray,
    p: int,
) -> MultiObjectiveEvaluation:
    """Evalúa un vector binario de longitud 43."""
    x = np.asarray(x, dtype=bool)
    claves = problem_data.municipios["clave_inegi"].tolist()

    if len(x) != len(claves):
        raise ValueError(f"El vector debe tener longitud {len(claves)}.")

    centros = [claves[i] for i in np.flatnonzero(x)]
    return evaluate_multiobjective_solution(problem_data, centros, p)


def centers_to_binary_vector(problem_data: ProblemData, centros: Iterable[str]) -> np.ndarray:
    """Convierte una lista de claves INEGI a vector binario."""
    centros_set = set(str(c) for c in centros)
    claves = problem_data.municipios["clave_inegi"].tolist()
    invalidos = sorted(centros_set - set(claves))
    if invalidos:
        raise ValueError(f"Centros inválidos: {invalidos}")
    return np.array([clave in centros_set for clave in claves], dtype=bool)


def multiobjective_assignment_table(
    problem_data: ProblemData,
    centros: Iterable[str],
    p: int,
) -> pd.DataFrame:
    """Devuelve asignación por municipio para una solución multiobjetivo."""
    return assign_to_nearest_center(problem_data, centros, p)


def multiobjective_center_loads(
    problem_data: ProblemData,
    centros: Iterable[str],
    p: int,
) -> pd.DataFrame:
    """Devuelve carga de producción y cobertura por centro."""
    return summarize_center_loads(problem_data, centros, p)


def objective_array_for_pymoo(
    problem_data: ProblemData,
    X: np.ndarray,
    p: int,
    penalty: float = 1e12,
) -> np.ndarray:
    """
    Evalúa una matriz binaria X para PYMOO.

    Devuelve un arreglo de forma (n_individuos, 2) con:
    - columna 0: f1 distancia promedio ponderada.
    - columna 1: f2 distancia máxima.

    Si un individuo no tiene exactamente p centros, recibe penalización alta.
    """
    X = np.asarray(X, dtype=bool)
    produccion = problem_data.municipios["produccion_ton"].to_numpy(dtype=float)
    produccion_total = float(produccion.sum())
    dist_matrix = problem_data.matriz_distancias.to_numpy(dtype=float)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.shape[1] != len(produccion):
        raise ValueError(f"Cada individuo debe tener longitud {len(produccion)}.")

    F = np.empty((X.shape[0], 2), dtype=float)

    for i, individuo in enumerate(X):
        if individuo.sum() != p:
            violation = abs(int(individuo.sum()) - p)
            F[i, :] = penalty + violation * 1e9
            continue

        centros_idx = np.flatnonzero(individuo)
        dist_min = dist_matrix[:, centros_idx].min(axis=1)
        f1 = float((produccion * dist_min).sum() / produccion_total)
        f2 = float(dist_min.max())
        F[i, :] = [f1, f2]

    return F


def load_best_ga_solutions() -> pd.DataFrame:
    """Carga mejores soluciones monoobjetivo para usarlas como validación."""
    if not BEST_MONOOBJECTIVE_PATH.exists():
        raise FileNotFoundError(BEST_MONOOBJECTIVE_PATH)

    best = pd.read_csv(BEST_MONOOBJECTIVE_PATH)
    return best[best["metodo"] == "ga_pymoo"].copy()


def validate_against_monoobjective_best(problem_data: ProblemData) -> pd.DataFrame:
    """
    Valida que f1 y f2 coincidan con las métricas monoobjetivo ya calculadas.
    """
    best_ga = load_best_ga_solutions()
    rows = []

    for row in best_ga.itertuples(index=False):
        centros = str(row.centros_clave_inegi).split(";")
        evaluation = evaluate_multiobjective_solution(problem_data, centros, int(row.p))
        x = centers_to_binary_vector(problem_data, centros)
        F = objective_array_for_pymoo(problem_data, x, int(row.p))[0]

        rows.append(
            {
                "p": int(row.p),
                "centros": ";".join(evaluation.centros),
                "f1_multiobjetivo": evaluation.f1_distancia_promedio_ponderada_km,
                "f1_monoobjetivo": float(row.distancia_promedio_ponderada_km),
                "diferencia_f1": abs(
                    evaluation.f1_distancia_promedio_ponderada_km
                    - float(row.distancia_promedio_ponderada_km)
                ),
                "f2_multiobjetivo": evaluation.f2_distancia_maxima_km,
                "f2_monoobjetivo": float(row.distancia_maxima_km),
                "diferencia_f2": abs(evaluation.f2_distancia_maxima_km - float(row.distancia_maxima_km)),
                "f1_pymoo_array": F[0],
                "f2_pymoo_array": F[1],
            }
        )

    validation = pd.DataFrame(rows).sort_values("p")

    if not (validation["diferencia_f1"] <= 1e-9).all():
        raise ValueError("f1 no coincide con los resultados monoobjetivo.")

    if not (validation["diferencia_f2"] <= 1e-9).all():
        raise ValueError("f2 no coincide con los resultados monoobjetivo.")

    return validation


def main() -> None:
    problem_data = load_problem_data()
    validation = validate_against_monoobjective_best(problem_data)

    print("Definición multiobjetivo validada con mejores soluciones monoobjetivo.")
    print("\nObjetivos:")
    print("f1 = distancia promedio ponderada por producción")
    print("f2 = distancia máxima de atención")
    print("\nValidación:")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
