"""
Definición del problema multiobjetivo de tres objetivos con número variable de centros.

Objetivos:
- f1: minimizar distancia promedio ponderada por producción.
- f2: minimizar distancia máxima de atención.
- f3: minimizar número de centros instalados.

Restricción:
- p_min <= número de centros <= p_max

Configuración recomendada:
- p_min = 3
- p_max = 10

Este módulo solo define evaluación y utilidades; la ejecución de NSGA-II se
implementa en nsga2_three_objective.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from problem_definition import ProblemData, assign_to_nearest_center, load_problem_data, summarize_center_loads


P_MIN = 3
P_MAX = 10


@dataclass(frozen=True)
class ThreeObjectiveEvaluation:
    centros: list[str]
    f1_distancia_promedio_ponderada_km: float
    f2_distancia_maxima_km: float
    f3_num_centros: int
    distancia_total_ponderada_ton_km: float
    produccion_total_ton: float
    municipios_atendidos: int
    factible: bool


def validate_center_count(n_centros: int, p_min: int = P_MIN, p_max: int = P_MAX) -> None:
    if n_centros < p_min or n_centros > p_max:
        raise ValueError(f"El número de centros debe estar entre {p_min} y {p_max}; recibido {n_centros}.")


def evaluate_three_objective_solution(
    problem_data: ProblemData,
    centros: Iterable[str],
    p_min: int = P_MIN,
    p_max: int = P_MAX,
) -> ThreeObjectiveEvaluation:
    """Evalúa una solución con número variable de centros."""
    centros_validos = sorted(set(str(c) for c in centros))
    if len(centros_validos) != len(list(centros)):
        raise ValueError("La solución contiene centros duplicados.")

    claves_validas = set(problem_data.municipios["clave_inegi"])
    invalidos = sorted(set(centros_validos) - claves_validas)
    if invalidos:
        raise ValueError(f"La solución contiene claves inexistentes: {invalidos}")

    validate_center_count(len(centros_validos), p_min, p_max)

    asignaciones = assign_to_nearest_center(problem_data, centros_validos, len(centros_validos))
    produccion_total = float(asignaciones["produccion_ton"].sum())
    distancia_total_ponderada = float((asignaciones["produccion_ton"] * asignaciones["distancia_km"]).sum())

    if produccion_total <= 0:
        raise ValueError("La producción total debe ser positiva.")

    return ThreeObjectiveEvaluation(
        centros=centros_validos,
        f1_distancia_promedio_ponderada_km=distancia_total_ponderada / produccion_total,
        f2_distancia_maxima_km=float(asignaciones["distancia_km"].max()),
        f3_num_centros=len(centros_validos),
        distancia_total_ponderada_ton_km=distancia_total_ponderada,
        produccion_total_ton=produccion_total,
        municipios_atendidos=int(len(asignaciones)),
        factible=True,
    )


def centers_to_binary_vector(problem_data: ProblemData, centros: Iterable[str]) -> np.ndarray:
    centros_set = set(str(c) for c in centros)
    claves = problem_data.municipios["clave_inegi"].tolist()
    invalidos = sorted(centros_set - set(claves))
    if invalidos:
        raise ValueError(f"Centros inválidos: {invalidos}")
    return np.array([clave in centros_set for clave in claves], dtype=bool)


def binary_vector_to_centers(problem_data: ProblemData, x: np.ndarray) -> list[str]:
    x = np.asarray(x, dtype=bool)
    claves = problem_data.municipios["clave_inegi"].tolist()
    if len(x) != len(claves):
        raise ValueError(f"El vector debe tener longitud {len(claves)}.")
    return [claves[i] for i in np.flatnonzero(x)]


def objective_array_for_pymoo_3obj(
    problem_data: ProblemData,
    X: np.ndarray,
    p_min: int = P_MIN,
    p_max: int = P_MAX,
    penalty: float = 1e12,
) -> np.ndarray:
    """
    Evalúa una matriz binaria X para PYMOO.

    Devuelve arreglo (n_individuos, 3):
    - f1 distancia promedio ponderada.
    - f2 distancia máxima.
    - f3 número de centros.

    Individuos fuera de [p_min, p_max] reciben penalización alta en f1/f2/f3.
    """
    X = np.asarray(X, dtype=bool)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    produccion = problem_data.municipios["produccion_ton"].to_numpy(dtype=float)
    produccion_total = float(produccion.sum())
    dist_matrix = problem_data.matriz_distancias.to_numpy(dtype=float)

    F = np.empty((X.shape[0], 3), dtype=float)

    for i, individuo in enumerate(X):
        n_centros = int(individuo.sum())
        if n_centros < p_min or n_centros > p_max:
            violation = min(abs(n_centros - p_min), abs(n_centros - p_max))
            F[i, :] = [penalty + violation * 1e9, penalty + violation * 1e9, penalty + violation * 1e9]
            continue

        centros_idx = np.flatnonzero(individuo)
        dist_min = dist_matrix[:, centros_idx].min(axis=1)
        f1 = float((produccion * dist_min).sum() / produccion_total)
        f2 = float(dist_min.max())
        f3 = float(n_centros)
        F[i, :] = [f1, f2, f3]

    return F


def three_objective_assignment_table(
    problem_data: ProblemData,
    centros: Iterable[str],
    p_min: int = P_MIN,
    p_max: int = P_MAX,
) -> pd.DataFrame:
    centros_validos = sorted(set(str(c) for c in centros))
    validate_center_count(len(centros_validos), p_min, p_max)
    return assign_to_nearest_center(problem_data, centros_validos, len(centros_validos))


def three_objective_center_loads(
    problem_data: ProblemData,
    centros: Iterable[str],
    p_min: int = P_MIN,
    p_max: int = P_MAX,
) -> pd.DataFrame:
    centros_validos = sorted(set(str(c) for c in centros))
    validate_center_count(len(centros_validos), p_min, p_max)
    return summarize_center_loads(problem_data, centros_validos, len(centros_validos))


def main() -> None:
    problem_data = load_problem_data()
    ejemplos = {
        "p3_mono": ["28012", "28033", "28035"],
        "p5_mono": ["28012", "28022", "28033", "28035", "28040"],
        "p10_mono": [
            "28003",
            "28008",
            "28012",
            "28022",
            "28023",
            "28032",
            "28033",
            "28035",
            "28037",
            "28040",
        ],
    }

    rows = []
    for nombre, centros in ejemplos.items():
        evaluation = evaluate_three_objective_solution(problem_data, centros)
        x = centers_to_binary_vector(problem_data, centros)
        F = objective_array_for_pymoo_3obj(problem_data, x)[0]
        rows.append(
            {
                "ejemplo": nombre,
                "f1": evaluation.f1_distancia_promedio_ponderada_km,
                "f2": evaluation.f2_distancia_maxima_km,
                "f3": evaluation.f3_num_centros,
                "f1_array": F[0],
                "f2_array": F[1],
                "f3_array": F[2],
                "centros": ";".join(evaluation.centros),
            }
        )

    print("Definición de tres objetivos validada con soluciones conocidas.")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
