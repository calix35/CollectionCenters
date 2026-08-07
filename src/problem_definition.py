"""
Definición reutilizable del problema monoobjetivo de centros de acopio.

Entrada:
- data/processed/municipios_productores.csv
- data/processed/distancias.csv

Problema:
- Seleccionar exactamente p cabeceras municipales como centros.
- Asignar cada municipio al centro seleccionado más cercano.
- Minimizar la distancia promedio ponderada por producción.

Este módulo no implementa todavía líneas base ni algoritmos evolutivos.
Solo define carga de datos, validación, asignación y evaluación de soluciones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUNICIPIOS_PATH = PROJECT_ROOT / "data" / "processed" / "municipios_productores.csv"
DISTANCIAS_PATH = PROJECT_ROOT / "data" / "processed" / "distancias.csv"


@dataclass(frozen=True)
class ProblemData:
    """Datos validados de la instancia."""

    municipios: pd.DataFrame
    distancias: pd.DataFrame
    matriz_distancias: pd.DataFrame


def load_problem_data(
    municipios_path: Path = MUNICIPIOS_PATH,
    distancias_path: Path = DISTANCIAS_PATH,
) -> ProblemData:
    """Carga datos procesados y valida consistencia básica."""
    municipios = pd.read_csv(municipios_path, dtype={"clave_inegi": str})
    distancias = pd.read_csv(distancias_path, dtype={"origen": str, "destino": str})

    validate_processed_data(municipios, distancias)

    matriz = distancias.pivot(index="origen", columns="destino", values="distancia_km")
    matriz = matriz.loc[municipios["clave_inegi"], municipios["clave_inegi"]]

    return ProblemData(
        municipios=municipios.sort_values("clave_inegi").reset_index(drop=True),
        distancias=distancias,
        matriz_distancias=matriz,
    )


def validate_processed_data(municipios: pd.DataFrame, distancias: pd.DataFrame) -> None:
    """Valida los dos CSV procesados que definen la instancia."""
    columnas_municipios = ["clave_inegi", "municipio", "latitud", "longitud", "produccion_ton"]
    columnas_distancias = ["origen", "destino", "distancia_km"]

    if list(municipios.columns) != columnas_municipios:
        raise ValueError(f"Columnas inesperadas en municipios: {list(municipios.columns)}")

    if list(distancias.columns) != columnas_distancias:
        raise ValueError(f"Columnas inesperadas en distancias: {list(distancias.columns)}")

    if len(municipios) != 43:
        raise ValueError(f"Se esperaban 43 municipios; se obtuvieron {len(municipios)}.")

    if municipios["clave_inegi"].nunique() != 43:
        raise ValueError("Hay claves municipales duplicadas.")

    if not municipios["clave_inegi"].str.fullmatch(r"\d{5}").all():
        raise ValueError("Todas las claves municipales deben ser texto de cinco dígitos.")

    if municipios[["latitud", "longitud", "produccion_ton"]].isna().any().any():
        raise ValueError("Hay valores faltantes en municipios_productores.csv.")

    if (municipios["produccion_ton"] < 0).any():
        raise ValueError("Hay producción negativa.")

    if not (municipios["longitud"] < 0).all():
        raise ValueError("Hay longitudes no negativas.")

    claves = set(municipios["clave_inegi"])

    if len(distancias) != 43 * 43:
        raise ValueError(f"Se esperaban 1849 distancias; se obtuvieron {len(distancias)}.")

    if set(distancias["origen"]) != claves:
        raise ValueError("Los orígenes de distancias no coinciden con las claves municipales.")

    if set(distancias["destino"]) != claves:
        raise ValueError("Los destinos de distancias no coinciden con las claves municipales.")

    if distancias[["origen", "destino"]].duplicated().any():
        raise ValueError("Hay pares origen-destino duplicados.")

    if distancias["distancia_km"].isna().any():
        raise ValueError("Hay distancias faltantes.")

    if (distancias["distancia_km"] < 0).any():
        raise ValueError("Hay distancias negativas.")

    diagonal = distancias[distancias["origen"] == distancias["destino"]]
    if not (diagonal["distancia_km"].abs() <= 1e-9).all():
        raise ValueError("La diagonal de la matriz de distancias debe ser cero.")

    matriz = distancias.pivot(index="origen", columns="destino", values="distancia_km")
    diferencia_maxima = (matriz - matriz.T).abs().to_numpy().max()
    if diferencia_maxima > 1e-6:
        raise ValueError(f"La matriz de distancias no es simétrica: diferencia {diferencia_maxima}.")


def validate_solution(centros: Iterable[str], problem_data: ProblemData, p: int) -> list[str]:
    """Valida una solución como lista de claves INEGI y devuelve centros ordenados."""
    centros_lista = [str(centro) for centro in centros]
    centros_unicos = sorted(set(centros_lista))

    if len(centros_lista) != len(centros_unicos):
        raise ValueError("La solución contiene centros duplicados.")

    if len(centros_unicos) != p:
        raise ValueError(f"La solución debe seleccionar exactamente p={p} centros.")

    claves_validas = set(problem_data.municipios["clave_inegi"])
    invalidos = sorted(set(centros_unicos) - claves_validas)
    if invalidos:
        raise ValueError(f"La solución contiene claves inexistentes: {invalidos}")

    return centros_unicos


def assign_to_nearest_center(problem_data: ProblemData, centros: Iterable[str], p: int) -> pd.DataFrame:
    """
    Asigna cada municipio al centro seleccionado más cercano.

    Devuelve una tabla con una fila por municipio:
    clave_inegi, municipio, produccion_ton, centro_asignado, distancia_km.
    """
    centros_validos = validate_solution(centros, problem_data, p)

    municipios = problem_data.municipios.copy()
    matriz = problem_data.matriz_distancias
    distancias_a_centros = matriz.loc[municipios["clave_inegi"], centros_validos]

    municipios["centro_asignado"] = distancias_a_centros.idxmin(axis=1).to_numpy()
    municipios["distancia_km"] = distancias_a_centros.min(axis=1).to_numpy()

    return municipios[
        ["clave_inegi", "municipio", "produccion_ton", "centro_asignado", "distancia_km"]
    ].copy()


def evaluate_solution(problem_data: ProblemData, centros: Iterable[str], p: int) -> dict[str, float | int | list[str]]:
    """
    Evalúa una solución monoobjetivo.

    Métrica principal:
    - distancia_promedio_ponderada_km

    Métricas auxiliares:
    - distancia_total_ponderada_ton_km
    - distancia_maxima_km
    - produccion_total_ton
    - municipios_atendidos
    - centros
    """
    centros_validos = validate_solution(centros, problem_data, p)
    asignaciones = assign_to_nearest_center(problem_data, centros_validos, p)

    produccion_total = float(asignaciones["produccion_ton"].sum())
    distancia_total_ponderada = float(
        (asignaciones["produccion_ton"] * asignaciones["distancia_km"]).sum()
    )

    if produccion_total <= 0:
        raise ValueError("La producción total debe ser positiva para calcular promedio ponderado.")

    distancia_promedio_ponderada = distancia_total_ponderada / produccion_total

    return {
        "p": p,
        "centros": centros_validos,
        "municipios_atendidos": int(len(asignaciones)),
        "produccion_total_ton": produccion_total,
        "distancia_total_ponderada_ton_km": distancia_total_ponderada,
        "distancia_promedio_ponderada_km": distancia_promedio_ponderada,
        "distancia_maxima_km": float(asignaciones["distancia_km"].max()),
    }


def summarize_center_loads(problem_data: ProblemData, centros: Iterable[str], p: int) -> pd.DataFrame:
    """Resume producción y número de municipios asignados por centro."""
    asignaciones = assign_to_nearest_center(problem_data, centros, p)
    nombres = problem_data.municipios[["clave_inegi", "municipio"]].rename(
        columns={"clave_inegi": "centro_asignado", "municipio": "centro_municipio"}
    )

    cargas = (
        asignaciones.groupby("centro_asignado", as_index=False)
        .agg(
            municipios_asignados=("clave_inegi", "size"),
            produccion_asignada_ton=("produccion_ton", "sum"),
            distancia_maxima_asignada_km=("distancia_km", "max"),
        )
        .merge(nombres, on="centro_asignado", how="left")
        [
            [
                "centro_asignado",
                "centro_municipio",
                "municipios_asignados",
                "produccion_asignada_ton",
                "distancia_maxima_asignada_km",
            ]
        ]
        .sort_values("produccion_asignada_ton", ascending=False)
        .reset_index(drop=True)
    )
    return cargas


def top_production_centers(problem_data: ProblemData, p: int) -> list[str]:
    """Selecciona los p municipios con mayor producción. Útil como prueba/heurística simple."""
    return (
        problem_data.municipios.sort_values(
            ["produccion_ton", "clave_inegi"],
            ascending=[False, True],
        )
        .head(p)["clave_inegi"]
        .tolist()
    )


def main() -> None:
    """Prueba mínima de validación: p=3 con los municipios de mayor producción."""
    problem_data = load_problem_data()
    p = 3
    centros = top_production_centers(problem_data, p)
    metricas = evaluate_solution(problem_data, centros, p)
    cargas = summarize_center_loads(problem_data, centros, p)

    nombres_centros = problem_data.municipios[
        problem_data.municipios["clave_inegi"].isin(centros)
    ][["clave_inegi", "municipio", "produccion_ton"]]

    print("Datos del problema cargados y validados.")
    print(f"Municipios: {len(problem_data.municipios)}")
    print(f"Distancias: {len(problem_data.distancias)}")
    print(f"Prueba: p={p}, centros con mayor producción")
    print("\nCentros seleccionados:")
    print(nombres_centros.to_string(index=False))
    print("\nMétricas:")
    for clave, valor in metricas.items():
        print(f"{clave}: {valor}")
    print("\nCargas por centro:")
    print(cargas.to_string(index=False))


if __name__ == "__main__":
    main()
