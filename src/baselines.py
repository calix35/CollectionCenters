"""
Líneas base para el problema monoobjetivo de centros de acopio.

Métodos implementados:
1. aleatoria: selecciona p centros al azar con semillas fijas.
2. top_produccion: selecciona los p municipios con mayor producción.
3. voraz: agrega centros iterativamente minimizando la distancia promedio ponderada.

Entradas:
- data/processed/municipios_productores.csv
- data/processed/distancias.csv

Salidas:
- data/results/baselines_resultados.csv
- data/results/baselines_centros.csv
- data/results/baselines_cargas.csv
- figures/comparacion_baselines.svg
"""

from __future__ import annotations

import random
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

from problem_definition import (
    ProblemData,
    evaluate_solution,
    load_problem_data,
    summarize_center_loads,
    top_production_centers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

P_VALUES = [3, 5, 7, 10]
RANDOM_SEEDS = list(range(1, 31))


def random_centers(problem_data: ProblemData, p: int, seed: int) -> list[str]:
    """Selecciona p centros al azar de forma reproducible."""
    rng = random.Random(seed)
    claves = problem_data.municipios["clave_inegi"].tolist()
    return sorted(rng.sample(claves, p))


def greedy_centers(problem_data: ProblemData, p: int) -> list[str]:
    """
    Heurística voraz.

    En cada paso agrega el municipio que produce el menor valor de distancia promedio ponderada.
    Usa clave INEGI como desempate determinista.
    """
    claves = problem_data.municipios["clave_inegi"].tolist()
    seleccionados: list[str] = []

    for _ in range(p):
        mejor_clave = None
        mejor_valor = None

        for candidato in claves:
            if candidato in seleccionados:
                continue

            prueba = sorted(seleccionados + [candidato])
            metricas = evaluate_solution(problem_data, prueba, len(prueba))
            valor = float(metricas["distancia_promedio_ponderada_km"])

            if mejor_valor is None or valor < mejor_valor or (
                valor == mejor_valor and candidato < str(mejor_clave)
            ):
                mejor_valor = valor
                mejor_clave = candidato

        if mejor_clave is None:
            raise RuntimeError("No se pudo seleccionar un nuevo centro en la heurística voraz.")

        seleccionados.append(mejor_clave)

    return sorted(seleccionados)


def add_result(
    resultados: list[dict],
    centros_rows: list[dict],
    cargas_rows: list[dict],
    problem_data: ProblemData,
    metodo: str,
    p: int,
    corrida: int,
    semilla: int | None,
    centros: list[str],
) -> None:
    """Evalúa y agrega resultados, centros y cargas a las listas acumuladoras."""
    metricas = evaluate_solution(problem_data, centros, p)
    cargas = summarize_center_loads(problem_data, centros, p)

    resultados.append(
        {
            "metodo": metodo,
            "p": p,
            "corrida": corrida,
            "semilla": semilla,
            "distancia_promedio_ponderada_km": metricas["distancia_promedio_ponderada_km"],
            "distancia_total_ponderada_ton_km": metricas["distancia_total_ponderada_ton_km"],
            "distancia_maxima_km": metricas["distancia_maxima_km"],
            "produccion_total_ton": metricas["produccion_total_ton"],
            "municipios_atendidos": metricas["municipios_atendidos"],
        }
    )

    nombres = problem_data.municipios[["clave_inegi", "municipio", "produccion_ton"]]
    centros_df = (
        pd.DataFrame({"clave_inegi": centros})
        .merge(nombres, on="clave_inegi", how="left")
        .sort_values("clave_inegi")
    )
    for orden, row in enumerate(centros_df.itertuples(index=False), start=1):
        centros_rows.append(
            {
                "metodo": metodo,
                "p": p,
                "corrida": corrida,
                "semilla": semilla,
                "orden": orden,
                "clave_inegi": row.clave_inegi,
                "municipio": row.municipio,
                "produccion_ton": row.produccion_ton,
            }
        )

    for row in cargas.itertuples(index=False):
        cargas_rows.append(
            {
                "metodo": metodo,
                "p": p,
                "corrida": corrida,
                "semilla": semilla,
                "centro_asignado": row.centro_asignado,
                "centro_municipio": row.centro_municipio,
                "municipios_asignados": row.municipios_asignados,
                "produccion_asignada_ton": row.produccion_asignada_ton,
                "distancia_maxima_asignada_km": row.distancia_maxima_asignada_km,
            }
        )


def run_baselines(problem_data: ProblemData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ejecuta líneas base para todos los valores de p."""
    resultados: list[dict] = []
    centros_rows: list[dict] = []
    cargas_rows: list[dict] = []

    for p in P_VALUES:
        centros_top = top_production_centers(problem_data, p)
        add_result(
            resultados,
            centros_rows,
            cargas_rows,
            problem_data,
            metodo="top_produccion",
            p=p,
            corrida=1,
            semilla=None,
            centros=centros_top,
        )

        centros_voraz = greedy_centers(problem_data, p)
        add_result(
            resultados,
            centros_rows,
            cargas_rows,
            problem_data,
            metodo="voraz",
            p=p,
            corrida=1,
            semilla=None,
            centros=centros_voraz,
        )

        for corrida, semilla in enumerate(RANDOM_SEEDS, start=1):
            centros_aleatorios = random_centers(problem_data, p, semilla)
            add_result(
                resultados,
                centros_rows,
                cargas_rows,
                problem_data,
                metodo="aleatoria",
                p=p,
                corrida=corrida,
                semilla=semilla,
                centros=centros_aleatorios,
            )

    resultados_df = pd.DataFrame(resultados)
    centros_df = pd.DataFrame(centros_rows)
    cargas_df = pd.DataFrame(cargas_rows)

    return resultados_df, centros_df, cargas_df


def build_comparison_figure(resultados: pd.DataFrame, salida: Path) -> None:
    """Genera figura SVG simple comparando distancia promedio ponderada por p y método."""
    resumen = (
        resultados.groupby(["p", "metodo"], as_index=False)
        .agg(
            media=("distancia_promedio_ponderada_km", "mean"),
            minimo=("distancia_promedio_ponderada_km", "min"),
            maximo=("distancia_promedio_ponderada_km", "max"),
        )
        .sort_values(["p", "metodo"])
    )

    ancho = 980
    alto = 540
    margen_izq = 80
    margen_der = 40
    margen_sup = 65
    margen_inf = 80
    ancho_graf = ancho - margen_izq - margen_der
    alto_graf = alto - margen_sup - margen_inf

    metodos = ["aleatoria", "top_produccion", "voraz"]
    colores = {
        "aleatoria": "#999999",
        "top_produccion": "#2f6f9f",
        "voraz": "#8a4f9e",
    }
    p_values = sorted(resultados["p"].unique().tolist())
    max_y = max(float(resumen["maximo"].max()), 1.0)
    min_y = 0.0

    def x_pos(p_idx: int, metodo_idx: int) -> float:
        grupo_ancho = ancho_graf / len(p_values)
        base = margen_izq + p_idx * grupo_ancho
        return base + grupo_ancho * (metodo_idx + 1) / (len(metodos) + 1)

    def y_pos(valor: float) -> float:
        return margen_sup + alto_graf - ((valor - min_y) / (max_y - min_y)) * alto_graf

    partes = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{ancho}" height="{alto}" viewBox="0 0 {ancho} {alto}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{ancho/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">Comparación de líneas base</text>',
        f'<text x="{ancho/2}" y="50" text-anchor="middle" font-family="Arial" font-size="12">Distancia promedio ponderada por producción (km). Menor es mejor.</text>',
        f'<line x1="{margen_izq}" y1="{margen_sup + alto_graf}" x2="{margen_izq + ancho_graf}" y2="{margen_sup + alto_graf}" stroke="#333"/>',
        f'<line x1="{margen_izq}" y1="{margen_sup}" x2="{margen_izq}" y2="{margen_sup + alto_graf}" stroke="#333"/>',
    ]

    for tick in range(6):
        valor = max_y * tick / 5
        y = y_pos(valor)
        partes.append(f'<line x1="{margen_izq - 5}" y1="{y:.2f}" x2="{margen_izq}" y2="{y:.2f}" stroke="#333"/>')
        partes.append(f'<line x1="{margen_izq}" y1="{y:.2f}" x2="{margen_izq + ancho_graf}" y2="{y:.2f}" stroke="#eee"/>')
        partes.append(f'<text x="{margen_izq - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="10">{valor:.0f}</text>')

    radio = 5
    for p_idx, p in enumerate(p_values):
        grupo_ancho = ancho_graf / len(p_values)
        centro_grupo = margen_izq + p_idx * grupo_ancho + grupo_ancho / 2
        partes.append(f'<text x="{centro_grupo:.2f}" y="{alto - 40}" text-anchor="middle" font-family="Arial" font-size="13">p={p}</text>')
        for metodo_idx, metodo in enumerate(metodos):
            row = resumen[(resumen["p"] == p) & (resumen["metodo"] == metodo)].iloc[0]
            x = x_pos(p_idx, metodo_idx)
            y_media = y_pos(float(row["media"]))
            y_min = y_pos(float(row["minimo"]))
            y_max = y_pos(float(row["maximo"]))
            partes.append(f'<line x1="{x:.2f}" y1="{y_min:.2f}" x2="{x:.2f}" y2="{y_max:.2f}" stroke="{colores[metodo]}" stroke-width="2"/>')
            partes.append(f'<circle cx="{x:.2f}" cy="{y_media:.2f}" r="{radio}" fill="{colores[metodo]}"/>')

    leyenda_x = margen_izq + 20
    leyenda_y = alto - 20
    for idx, metodo in enumerate(metodos):
        x = leyenda_x + idx * 190
        partes.append(f'<rect x="{x}" y="{leyenda_y - 10}" width="12" height="12" fill="{colores[metodo]}"/>')
        partes.append(f'<text x="{x + 18}" y="{leyenda_y}" font-family="Arial" font-size="12">{escape(metodo)}</text>')

    partes.append("</svg>")
    salida.write_text("\n".join(partes), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    problem_data = load_problem_data()
    resultados, centros, cargas = run_baselines(problem_data)

    resultados.to_csv(RESULTS_DIR / "baselines_resultados.csv", index=False, encoding="utf-8")
    centros.to_csv(RESULTS_DIR / "baselines_centros.csv", index=False, encoding="utf-8")
    cargas.to_csv(RESULTS_DIR / "baselines_cargas.csv", index=False, encoding="utf-8")
    build_comparison_figure(resultados, FIGURES_DIR / "comparacion_baselines.svg")

    resumen = (
        resultados.groupby(["p", "metodo"], as_index=False)
        .agg(
            n=("distancia_promedio_ponderada_km", "size"),
            media=("distancia_promedio_ponderada_km", "mean"),
            mejor=("distancia_promedio_ponderada_km", "min"),
            peor=("distancia_promedio_ponderada_km", "max"),
        )
        .sort_values(["p", "media"])
    )

    print("Líneas base generadas.")
    print(f"Resultados: {len(resultados)} filas")
    print(f"Centros: {len(centros)} filas")
    print(f"Cargas: {len(cargas)} filas")
    print("\nResumen por p y método:")
    print(resumen.to_string(index=False))


if __name__ == "__main__":
    main()
