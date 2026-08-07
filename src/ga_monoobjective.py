"""
Algoritmo genético monoobjetivo con PYMOO.

Problema:
- Seleccionar exactamente p cabeceras municipales como centros de acopio.
- Minimizar la distancia promedio ponderada por producción.

Representación:
- Vector binario de longitud 43.
- 1: municipio seleccionado como centro.
- 0: municipio no seleccionado.

Restricción:
- Exactamente p posiciones activas.

Estrategia:
- Muestreo inicial factible con exactamente p unos.
- Cruce de dos puntos.
- Mutación bit-flip.
- Reparación posterior para restaurar exactamente p centros.
- Selección y evolución gestionadas por GA de PYMOO.

Salidas:
- data/results/ga_resultados.csv
- data/results/ga_centros.csv
- data/results/ga_cargas.csv
- data/results/ga_convergencia.csv
- figures/comparacion_ga_baselines.svg
- figures/convergencia_ga.svg
"""

from __future__ import annotations

import time
import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

from problem_definition import (
    assign_to_nearest_center,
    evaluate_solution,
    load_problem_data,
    summarize_center_loads,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

P_VALUES = [3, 5, 7, 10]
SEEDS = list(range(1, 31))

POPULATION_SIZE = 80
GENERATIONS = 150
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.2


class FixedCardinalitySampling(Sampling):
    """Genera individuos binarios factibles con exactamente p unos."""

    def __init__(self, p: int):
        super().__init__()
        self.p = p

    def _do(self, problem, n_samples, **kwargs):
        rng = np.random.default_rng(kwargs.get("seed", None))
        x = np.zeros((n_samples, problem.n_var), dtype=bool)
        for i in range(n_samples):
            indices = rng.choice(problem.n_var, size=self.p, replace=False)
            x[i, indices] = True
        return x


class FixedCardinalityRepair(Repair):
    """Repara individuos para que tengan exactamente p centros seleccionados."""

    def __init__(self, p: int):
        super().__init__()
        self.p = p

    def _do(self, problem, X, **kwargs):
        X = np.asarray(X, dtype=bool).copy()

        for i in range(X.shape[0]):
            activos = np.flatnonzero(X[i])
            inactivos = np.flatnonzero(~X[i])

            if len(activos) > self.p:
                quitar = np.random.choice(activos, size=len(activos) - self.p, replace=False)
                X[i, quitar] = False
            elif len(activos) < self.p:
                agregar = np.random.choice(inactivos, size=self.p - len(activos), replace=False)
                X[i, agregar] = True

        return X


class CentrosAcopioProblem(Problem):
    """Problema monoobjetivo vectorizado para PYMOO."""

    def __init__(self, produccion: np.ndarray, dist_matrix: np.ndarray, p: int):
        super().__init__(
            n_var=len(produccion),
            n_obj=1,
            n_ieq_constr=0,
            xl=0,
            xu=1,
            vtype=bool,
        )
        self.produccion = produccion.astype(float)
        self.dist_matrix = dist_matrix.astype(float)
        self.p = p
        self.produccion_total = float(self.produccion.sum())
        if self.produccion_total <= 0:
            raise ValueError("La producción total debe ser positiva.")

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X, dtype=bool)
        f = np.empty(X.shape[0], dtype=float)

        for i, individuo in enumerate(X):
            if individuo.sum() != self.p:
                f[i] = 1e12 + abs(int(individuo.sum()) - self.p) * 1e9
                continue

            centros_idx = np.flatnonzero(individuo)
            dist_min = self.dist_matrix[:, centros_idx].min(axis=1)
            f[i] = float((self.produccion * dist_min).sum() / self.produccion_total)

        out["F"] = f.reshape(-1, 1)


class ConvergenceCallback(Callback):
    """Registra el mejor valor por generación."""

    def __init__(self):
        super().__init__()
        self.data["generation"] = []
        self.data["best_f"] = []

    def notify(self, algorithm):
        self.data["generation"].append(int(algorithm.n_gen))
        self.data["best_f"].append(float(algorithm.pop.get("F").min()))


def vector_to_centers(x: np.ndarray, claves: list[str]) -> list[str]:
    """Convierte vector binario en lista de claves INEGI seleccionadas."""
    return [claves[i] for i in np.flatnonzero(np.asarray(x, dtype=bool))]


def run_single_ga(problem_data, p: int, seed: int) -> dict:
    """Ejecuta una corrida del GA para un valor de p y semilla."""
    claves = problem_data.municipios["clave_inegi"].tolist()
    produccion = problem_data.municipios["produccion_ton"].to_numpy(dtype=float)
    dist_matrix = problem_data.matriz_distancias.to_numpy(dtype=float)

    problem = CentrosAcopioProblem(produccion=produccion, dist_matrix=dist_matrix, p=p)
    repair = FixedCardinalityRepair(p=p)
    callback = ConvergenceCallback()

    algorithm = GA(
        pop_size=POPULATION_SIZE,
        sampling=FixedCardinalitySampling(p=p),
        crossover=TwoPointCrossover(prob=CROSSOVER_PROBABILITY),
        mutation=BitflipMutation(prob=MUTATION_PROBABILITY),
        repair=repair,
        eliminate_duplicates=True,
    )

    start = time.perf_counter()
    result = minimize(
        problem,
        algorithm,
        ("n_gen", GENERATIONS),
        seed=seed,
        callback=callback,
        verbose=False,
        save_history=False,
    )
    elapsed = time.perf_counter() - start

    centros = sorted(vector_to_centers(result.X, claves))
    metricas = evaluate_solution(problem_data, centros, p)
    asignaciones = assign_to_nearest_center(problem_data, centros, p)
    cargas = summarize_center_loads(problem_data, centros, p)

    convergencia = pd.DataFrame(
        {
            "p": p,
            "semilla": seed,
            "generacion": callback.data["generation"],
            "mejor_distancia_promedio_ponderada_km": callback.data["best_f"],
        }
    )

    return {
        "p": p,
        "semilla": seed,
        "centros": centros,
        "metricas": metricas,
        "asignaciones": asignaciones,
        "cargas": cargas,
        "convergencia": convergencia,
        "tiempo_segundos": elapsed,
    }


def run_all_ga(p_values: list[int] | None = None, seeds: list[int] | None = None):
    """Ejecuta GA para todos los valores de p y semillas."""
    if p_values is None:
        p_values = P_VALUES
    if seeds is None:
        seeds = SEEDS

    problem_data = load_problem_data()
    resultados_rows = []
    centros_rows = []
    cargas_rows = []
    convergencia_frames = []

    nombres = problem_data.municipios[["clave_inegi", "municipio", "produccion_ton"]]

    for p in p_values:
        for corrida, seed in enumerate(seeds, start=1):
            run = run_single_ga(problem_data, p=p, seed=seed)
            metricas = run["metricas"]

            resultados_rows.append(
                {
                    "metodo": "ga_pymoo",
                    "p": p,
                    "corrida": corrida,
                    "semilla": seed,
                    "distancia_promedio_ponderada_km": metricas["distancia_promedio_ponderada_km"],
                    "distancia_total_ponderada_ton_km": metricas["distancia_total_ponderada_ton_km"],
                    "distancia_maxima_km": metricas["distancia_maxima_km"],
                    "produccion_total_ton": metricas["produccion_total_ton"],
                    "municipios_atendidos": metricas["municipios_atendidos"],
                    "tiempo_segundos": run["tiempo_segundos"],
                    "population_size": POPULATION_SIZE,
                    "generations": GENERATIONS,
                    "crossover_probability": CROSSOVER_PROBABILITY,
                    "mutation_probability": MUTATION_PROBABILITY,
                }
            )

            centros_df = (
                pd.DataFrame({"clave_inegi": run["centros"]})
                .merge(nombres, on="clave_inegi", how="left")
                .sort_values("clave_inegi")
            )
            for orden, row in enumerate(centros_df.itertuples(index=False), start=1):
                centros_rows.append(
                    {
                        "metodo": "ga_pymoo",
                        "p": p,
                        "corrida": corrida,
                        "semilla": seed,
                        "orden": orden,
                        "clave_inegi": row.clave_inegi,
                        "municipio": row.municipio,
                        "produccion_ton": row.produccion_ton,
                    }
                )

            cargas = run["cargas"].copy()
            cargas.insert(0, "semilla", seed)
            cargas.insert(0, "corrida", corrida)
            cargas.insert(0, "p", p)
            cargas.insert(0, "metodo", "ga_pymoo")
            cargas_rows.append(cargas)

            convergencia_frames.append(run["convergencia"])

    resultados = pd.DataFrame(resultados_rows)
    centros = pd.DataFrame(centros_rows)
    cargas = pd.concat(cargas_rows, ignore_index=True)
    convergencia = pd.concat(convergencia_frames, ignore_index=True)

    return resultados, centros, cargas, convergencia


def build_ga_baseline_figure(ga_resultados: pd.DataFrame, salida: Path) -> None:
    """Compara GA contra líneas base si están disponibles."""
    baseline_path = RESULTS_DIR / "baselines_resultados.csv"

    frames = [ga_resultados.copy()]
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        frames.append(baseline)

    data = pd.concat(frames, ignore_index=True)
    resumen = (
        data.groupby(["p", "metodo"], as_index=False)
        .agg(
            media=("distancia_promedio_ponderada_km", "mean"),
            mejor=("distancia_promedio_ponderada_km", "min"),
            peor=("distancia_promedio_ponderada_km", "max"),
        )
        .sort_values(["p", "media"])
    )

    metodos = [m for m in ["aleatoria", "top_produccion", "voraz", "ga_pymoo"] if m in set(data["metodo"])]
    colores = {
        "aleatoria": "#999999",
        "top_produccion": "#2f6f9f",
        "voraz": "#8a4f9e",
        "ga_pymoo": "#c43c39",
    }

    ancho = 1100
    alto = 560
    margen_izq = 80
    margen_der = 40
    margen_sup = 65
    margen_inf = 85
    ancho_graf = ancho - margen_izq - margen_der
    alto_graf = alto - margen_sup - margen_inf

    p_values = sorted(data["p"].unique().tolist())
    max_y = max(float(resumen["peor"].max()), 1.0)

    def y_pos(valor: float) -> float:
        return margen_sup + alto_graf - (valor / max_y) * alto_graf

    partes = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{ancho}" height="{alto}" viewBox="0 0 {ancho} {alto}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{ancho/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">GA PYMOO vs líneas base</text>',
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

    for p_idx, p in enumerate(p_values):
        grupo_ancho = ancho_graf / len(p_values)
        centro_grupo = margen_izq + p_idx * grupo_ancho + grupo_ancho / 2
        partes.append(f'<text x="{centro_grupo:.2f}" y="{alto - 45}" text-anchor="middle" font-family="Arial" font-size="13">p={p}</text>')

        for metodo_idx, metodo in enumerate(metodos):
            row = resumen[(resumen["p"] == p) & (resumen["metodo"] == metodo)].iloc[0]
            x = margen_izq + p_idx * grupo_ancho + grupo_ancho * (metodo_idx + 1) / (len(metodos) + 1)
            y_mejor = y_pos(float(row["mejor"]))
            y_peor = y_pos(float(row["peor"]))
            y_media = y_pos(float(row["media"]))
            color = colores[metodo]
            partes.append(f'<line x1="{x:.2f}" y1="{y_mejor:.2f}" x2="{x:.2f}" y2="{y_peor:.2f}" stroke="{color}" stroke-width="2"/>')
            partes.append(f'<circle cx="{x:.2f}" cy="{y_media:.2f}" r="5" fill="{color}"/>')

    leyenda_x = margen_izq + 20
    leyenda_y = alto - 20
    for idx, metodo in enumerate(metodos):
        x = leyenda_x + idx * 210
        partes.append(f'<rect x="{x}" y="{leyenda_y - 10}" width="12" height="12" fill="{colores[metodo]}"/>')
        partes.append(f'<text x="{x + 18}" y="{leyenda_y}" font-family="Arial" font-size="12">{escape(metodo)}</text>')

    partes.append("</svg>")
    salida.write_text("\n".join(partes), encoding="utf-8")


def build_convergence_figure(convergencia: pd.DataFrame, salida: Path) -> None:
    """Figura SVG con convergencia promedio del GA por p."""
    resumen = (
        convergencia.groupby(["p", "generacion"], as_index=False)
        .agg(media=("mejor_distancia_promedio_ponderada_km", "mean"))
        .sort_values(["p", "generacion"])
    )

    ancho = 1000
    alto = 560
    margen_izq = 80
    margen_der = 40
    margen_sup = 60
    margen_inf = 70
    ancho_graf = ancho - margen_izq - margen_der
    alto_graf = alto - margen_sup - margen_inf
    colores = {3: "#c43c39", 5: "#2f6f9f", 7: "#8a4f9e", 10: "#7a9e3a"}
    max_gen = int(resumen["generacion"].max())
    max_y = float(resumen["media"].max())
    min_y = float(resumen["media"].min())
    if max_y == min_y:
        max_y += 1

    def x_pos(gen: float) -> float:
        return margen_izq + (gen / max_gen) * ancho_graf

    def y_pos(valor: float) -> float:
        return margen_sup + alto_graf - ((valor - min_y) / (max_y - min_y)) * alto_graf

    partes = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{ancho}" height="{alto}" viewBox="0 0 {ancho} {alto}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{ancho/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">Convergencia promedio del GA PYMOO</text>',
        f'<line x1="{margen_izq}" y1="{margen_sup + alto_graf}" x2="{margen_izq + ancho_graf}" y2="{margen_sup + alto_graf}" stroke="#333"/>',
        f'<line x1="{margen_izq}" y1="{margen_sup}" x2="{margen_izq}" y2="{margen_sup + alto_graf}" stroke="#333"/>',
        f'<text x="{ancho/2}" y="{alto - 25}" text-anchor="middle" font-family="Arial" font-size="13">Generación</text>',
        f'<text x="18" y="{margen_sup + alto_graf/2}" text-anchor="middle" font-family="Arial" font-size="13" transform="rotate(-90 18 {margen_sup + alto_graf/2})">Mejor distancia promedio ponderada (km)</text>',
    ]

    for p in sorted(resumen["p"].unique()):
        sub = resumen[resumen["p"] == p]
        puntos = " ".join(
            f'{x_pos(row.generacion):.2f},{y_pos(row.media):.2f}'
            for row in sub.itertuples(index=False)
        )
        partes.append(f'<polyline points="{puntos}" fill="none" stroke="{colores[int(p)]}" stroke-width="2"/>')

    leyenda_x = margen_izq + 20
    leyenda_y = alto - 45
    for idx, p in enumerate(sorted(resumen["p"].unique())):
        x = leyenda_x + idx * 120
        color = colores[int(p)]
        partes.append(f'<rect x="{x}" y="{leyenda_y - 10}" width="12" height="12" fill="{color}"/>')
        partes.append(f'<text x="{x + 18}" y="{leyenda_y}" font-family="Arial" font-size="12">p={int(p)}</text>')

    partes.append("</svg>")
    salida.write_text("\n".join(partes), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta GA monoobjetivo con PYMOO.")
    parser.add_argument(
        "--p-values",
        nargs="+",
        type=int,
        default=P_VALUES,
        help="Valores de p a ejecutar.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Semillas a ejecutar.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Sufijo opcional para archivos de salida, por ejemplo _p3.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="No genera figuras SVG. Útil para ejecuciones por bloque.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    resultados, centros, cargas, convergencia = run_all_ga(
        p_values=args.p_values,
        seeds=args.seeds,
    )

    suffix = args.output_suffix
    resultados.to_csv(RESULTS_DIR / f"ga_resultados{suffix}.csv", index=False, encoding="utf-8")
    centros.to_csv(RESULTS_DIR / f"ga_centros{suffix}.csv", index=False, encoding="utf-8")
    cargas.to_csv(RESULTS_DIR / f"ga_cargas{suffix}.csv", index=False, encoding="utf-8")
    convergencia.to_csv(RESULTS_DIR / f"ga_convergencia{suffix}.csv", index=False, encoding="utf-8")

    if not args.skip_figures:
        build_ga_baseline_figure(resultados, FIGURES_DIR / "comparacion_ga_baselines.svg")
        build_convergence_figure(convergencia, FIGURES_DIR / "convergencia_ga.svg")

    resumen = (
        resultados.groupby("p", as_index=False)
        .agg(
            media=("distancia_promedio_ponderada_km", "mean"),
            mejor=("distancia_promedio_ponderada_km", "min"),
            peor=("distancia_promedio_ponderada_km", "max"),
            desviacion=("distancia_promedio_ponderada_km", "std"),
            tiempo_medio_segundos=("tiempo_segundos", "mean"),
        )
        .sort_values("p")
    )

    print("GA monoobjetivo con PYMOO generado.")
    print(f"Corridas: {len(resultados)}")
    print(f"Centros registrados: {len(centros)}")
    print(f"Cargas registradas: {len(cargas)}")
    print(f"Registros de convergencia: {len(convergencia)}")
    print("\nResumen GA por p:")
    print(resumen.to_string(index=False))


if __name__ == "__main__":
    main()
