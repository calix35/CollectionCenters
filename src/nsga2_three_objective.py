"""
NSGA-II para problema multiobjetivo de tres objetivos.

Objetivos:
- f1: minimizar distancia promedio ponderada por producción.
- f2: minimizar distancia máxima de atención.
- f3: minimizar número de centros instalados.

Restricción:
- 3 <= número de centros <= 10

Salidas:
- data/results/nsga2_3obj_resultados.csv
- data/results/nsga2_3obj_frentes.csv
- data/results/nsga2_3obj_centros.csv
- data/results/nsga2_3obj_cargas.csv
- figures/pareto_3obj_tradeoff.svg
- figures/pareto_3obj_3d.svg
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

from problem_definition import load_problem_data
from three_objective_problem import (
    P_MAX,
    P_MIN,
    binary_vector_to_centers,
    objective_array_for_pymoo_3obj,
    three_objective_center_loads,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

SEEDS = list(range(1, 31))
POPULATION_SIZE = 120
GENERATIONS = 250
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.2


class VariableCardinalitySampling(Sampling):
    """Muestreo factible con número de centros entre p_min y p_max."""

    def __init__(self, p_min: int = P_MIN, p_max: int = P_MAX):
        super().__init__()
        self.p_min = p_min
        self.p_max = p_max

    def _do(self, problem, n_samples, **kwargs):
        x = np.zeros((n_samples, problem.n_var), dtype=bool)
        for i in range(n_samples):
            p = np.random.randint(self.p_min, self.p_max + 1)
            idx = np.random.choice(problem.n_var, size=p, replace=False)
            x[i, idx] = True
        return x


class VariableCardinalityRepair(Repair):
    """Repara individuos para que cumplan p_min <= centros <= p_max."""

    def __init__(self, p_min: int = P_MIN, p_max: int = P_MAX):
        super().__init__()
        self.p_min = p_min
        self.p_max = p_max

    def _do(self, problem, X, **kwargs):
        X = np.asarray(X, dtype=bool).copy()
        for i in range(X.shape[0]):
            active = np.flatnonzero(X[i])
            inactive = np.flatnonzero(~X[i])
            n_active = len(active)

            if n_active > self.p_max:
                remove = np.random.choice(active, size=n_active - self.p_max, replace=False)
                X[i, remove] = False
            elif n_active < self.p_min:
                add = np.random.choice(inactive, size=self.p_min - n_active, replace=False)
                X[i, add] = True

        return X


class CentrosAcopioThreeObjectiveProblem(Problem):
    """Problema de tres objetivos para PYMOO."""

    def __init__(self, problem_data, p_min: int = P_MIN, p_max: int = P_MAX):
        super().__init__(
            n_var=len(problem_data.municipios),
            n_obj=3,
            n_ieq_constr=0,
            xl=0,
            xu=1,
            vtype=bool,
        )
        self.problem_data = problem_data
        self.p_min = p_min
        self.p_max = p_max

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = objective_array_for_pymoo_3obj(
            self.problem_data,
            X,
            p_min=self.p_min,
            p_max=self.p_max,
        )


def deduplicate_front(X: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seen = set()
    xs = []
    fs = []
    for x, f in zip(X, F):
        key = tuple(np.asarray(x, dtype=bool).tolist())
        if key in seen:
            continue
        seen.add(key)
        xs.append(np.asarray(x, dtype=bool))
        fs.append(np.asarray(f, dtype=float))
    return np.array(xs, dtype=bool), np.array(fs, dtype=float)


def run_single(problem_data, seed: int, p_min: int = P_MIN, p_max: int = P_MAX) -> dict:
    problem = CentrosAcopioThreeObjectiveProblem(problem_data, p_min=p_min, p_max=p_max)
    algorithm = NSGA2(
        pop_size=POPULATION_SIZE,
        sampling=VariableCardinalitySampling(p_min=p_min, p_max=p_max),
        crossover=TwoPointCrossover(prob=CROSSOVER_PROBABILITY),
        mutation=BitflipMutation(prob=MUTATION_PROBABILITY),
        repair=VariableCardinalityRepair(p_min=p_min, p_max=p_max),
        eliminate_duplicates=True,
    )

    start = time.perf_counter()
    result = minimize(
        problem,
        algorithm,
        ("n_gen", GENERATIONS),
        seed=seed,
        verbose=False,
        save_history=False,
    )
    elapsed = time.perf_counter() - start

    X = np.asarray(result.X, dtype=bool)
    F = np.asarray(result.F, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
        F = F.reshape(1, -1)
    X, F = deduplicate_front(X, F)
    return {"X": X, "F": F, "tiempo_segundos": elapsed}


def run_all(seeds: list[int], p_min: int = P_MIN, p_max: int = P_MAX):
    problem_data = load_problem_data()
    nombres = problem_data.municipios[["clave_inegi", "municipio", "produccion_ton"]]

    resultados_rows = []
    frentes_rows = []
    centros_rows = []
    cargas_frames = []

    for corrida, seed in enumerate(seeds, start=1):
        run = run_single(problem_data, seed=seed, p_min=p_min, p_max=p_max)
        X, F = run["X"], run["F"]

        resultados_rows.append(
            {
                "corrida": corrida,
                "semilla": seed,
                "n_solutions": len(F),
                "f1_min": float(F[:, 0].min()),
                "f1_mean": float(F[:, 0].mean()),
                "f1_max": float(F[:, 0].max()),
                "f2_min": float(F[:, 1].min()),
                "f2_mean": float(F[:, 1].mean()),
                "f2_max": float(F[:, 1].max()),
                "f3_min": int(F[:, 2].min()),
                "f3_mean": float(F[:, 2].mean()),
                "f3_max": int(F[:, 2].max()),
                "p_min": p_min,
                "p_max": p_max,
                "tiempo_segundos": run["tiempo_segundos"],
                "population_size": POPULATION_SIZE,
                "generations": GENERATIONS,
                "crossover_probability": CROSSOVER_PROBABILITY,
                "mutation_probability": MUTATION_PROBABILITY,
            }
        )

        order = np.lexsort((F[:, 2], F[:, 1], F[:, 0]))
        for local_id, idx in enumerate(order, start=1):
            solution_id = f"s{seed}_sol{local_id:03d}"
            x = X[idx]
            f = F[idx]
            centros = sorted(binary_vector_to_centers(problem_data, x))

            frentes_rows.append(
                {
                    "corrida": corrida,
                    "semilla": seed,
                    "solution_id": solution_id,
                    "f1_distancia_promedio_ponderada_km": float(f[0]),
                    "f2_distancia_maxima_km": float(f[1]),
                    "f3_num_centros": int(f[2]),
                    "centros_clave_inegi": ";".join(centros),
                }
            )

            centros_df = (
                pd.DataFrame({"clave_inegi": centros})
                .merge(nombres, on="clave_inegi", how="left")
                .sort_values("clave_inegi")
            )
            for centro_order, row in enumerate(centros_df.itertuples(index=False), start=1):
                centros_rows.append(
                    {
                        "corrida": corrida,
                        "semilla": seed,
                        "solution_id": solution_id,
                        "orden": centro_order,
                        "clave_inegi": row.clave_inegi,
                        "municipio": row.municipio,
                        "produccion_ton": row.produccion_ton,
                    }
                )

            cargas = three_objective_center_loads(problem_data, centros, p_min=p_min, p_max=p_max)
            cargas.insert(0, "solution_id", solution_id)
            cargas.insert(0, "semilla", seed)
            cargas.insert(0, "corrida", corrida)
            cargas_frames.append(cargas)

    resultados = pd.DataFrame(resultados_rows)
    frentes = pd.DataFrame(frentes_rows)
    centros = pd.DataFrame(centros_rows)
    cargas = pd.concat(cargas_frames, ignore_index=True) if cargas_frames else pd.DataFrame()
    return resultados, frentes, centros, cargas


def build_tradeoff_figure(frentes: pd.DataFrame, output: Path) -> None:
    width = 1050
    height = 620
    left = 80
    right = 160
    top = 65
    bottom = 75
    graph_w = width - left - right
    graph_h = height - top - bottom
    colors = {
        3: "#c43c39",
        4: "#e17c05",
        5: "#f1c40f",
        6: "#7a9e3a",
        7: "#2f9f73",
        8: "#2f6f9f",
        9: "#4b55a1",
        10: "#8a4f9e",
    }

    f1_min, f1_max = frentes["f1_distancia_promedio_ponderada_km"].min(), frentes["f1_distancia_promedio_ponderada_km"].max()
    f2_min, f2_max = frentes["f2_distancia_maxima_km"].min(), frentes["f2_distancia_maxima_km"].max()
    f1_pad = max((f1_max - f1_min) * 0.08, 1e-6)
    f2_pad = max((f2_max - f2_min) * 0.08, 1e-6)
    f1_min, f1_max = f1_min - f1_pad, f1_max + f1_pad
    f2_min, f2_max = f2_min - f2_pad, f2_max + f2_pad

    def x_pos(v: float) -> float:
        return left + ((v - f1_min) / (f1_max - f1_min)) * graph_w

    def y_pos(v: float) -> float:
        return top + graph_h - ((v - f2_min) / (f2_max - f2_min)) * graph_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">NSGA-II tres objetivos: trade-off f1, f2 y número de centros</text>',
        f'<text x="{width/2}" y="48" text-anchor="middle" font-family="Arial" font-size="12">Cada punto es no dominado; color indica f3 = número de centros.</text>',
        f'<line x1="{left}" y1="{top + graph_h}" x2="{left + graph_w}" y2="{top + graph_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + graph_h}" stroke="#333"/>',
        f'<text x="{left + graph_w/2}" y="{height - 28}" text-anchor="middle" font-family="Arial" font-size="13">f1 distancia promedio ponderada (km)</text>',
        f'<text x="22" y="{top + graph_h/2}" text-anchor="middle" font-family="Arial" font-size="13" transform="rotate(-90 22 {top + graph_h/2})">f2 distancia máxima (km)</text>',
    ]

    for row in frentes.itertuples(index=False):
        f3 = int(row.f3_num_centros)
        parts.append(
            f'<circle cx="{x_pos(row.f1_distancia_promedio_ponderada_km):.2f}" cy="{y_pos(row.f2_distancia_maxima_km):.2f}" r="3" fill="{colors.get(f3, "#555")}" opacity="0.55"/>'
        )

    legend_x = left + graph_w + 35
    legend_y = top + 20
    parts.append(f'<text x="{legend_x}" y="{legend_y - 10}" font-family="Arial" font-size="13" font-weight="700">f3 centros</text>')
    for idx, p in enumerate(range(P_MIN, P_MAX + 1)):
        y = legend_y + idx * 24
        parts.append(f'<rect x="{legend_x}" y="{y}" width="12" height="12" fill="{colors[p]}"/>')
        parts.append(f'<text x="{legend_x + 20}" y="{y + 11}" font-family="Arial" font-size="12">{p}</text>')

    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def build_3d_pareto_figure(frentes: pd.DataFrame, output: Path) -> None:
    """Genera una proyección 3D simple del frente f1-f2-f3 en SVG."""
    width = 1050
    height = 680
    origin_x = 240
    origin_y = 560
    x_len = 560
    y_len = 320
    z_len = 260

    colors = {
        3: "#c43c39",
        4: "#e17c05",
        5: "#f1c40f",
        6: "#7a9e3a",
        7: "#2f9f73",
        8: "#2f6f9f",
        9: "#4b55a1",
        10: "#8a4f9e",
    }

    f1_min = float(frentes["f1_distancia_promedio_ponderada_km"].min())
    f1_max = float(frentes["f1_distancia_promedio_ponderada_km"].max())
    f2_min = float(frentes["f2_distancia_maxima_km"].min())
    f2_max = float(frentes["f2_distancia_maxima_km"].max())
    f3_min = float(frentes["f3_num_centros"].min())
    f3_max = float(frentes["f3_num_centros"].max())

    def normalize(value: float, lower: float, upper: float) -> float:
        if upper == lower:
            return 0.5
        return (value - lower) / (upper - lower)

    def project(f1: float, f2: float, f3: float) -> tuple[float, float]:
        x_norm = normalize(f1, f1_min, f1_max)
        y_norm = normalize(f2, f2_min, f2_max)
        z_norm = normalize(f3, f3_min, f3_max)
        x = origin_x + x_norm * x_len + y_norm * 150
        y = origin_y - y_norm * y_len - z_norm * z_len
        return x, y

    x_axis_end = project(f1_max, f2_min, f3_min)
    y_axis_end = project(f1_min, f2_max, f3_min)
    z_axis_end = project(f1_min, f2_min, f3_max)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">NSGA-II tres objetivos: frente de Pareto 3D</text>',
        f'<text x="{width/2}" y="54" text-anchor="middle" font-family="Arial" font-size="12">Proyección de f1, f2 y f3. Cada punto representa una solución no dominada.</text>',
        f'<line x1="{origin_x}" y1="{origin_y}" x2="{x_axis_end[0]:.2f}" y2="{x_axis_end[1]:.2f}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{origin_x}" y1="{origin_y}" x2="{y_axis_end[0]:.2f}" y2="{y_axis_end[1]:.2f}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{origin_x}" y1="{origin_y}" x2="{z_axis_end[0]:.2f}" y2="{z_axis_end[1]:.2f}" stroke="#333" stroke-width="1.5"/>',
        f'<text x="{x_axis_end[0] + 18:.2f}" y="{x_axis_end[1] + 5:.2f}" font-family="Arial" font-size="13">f1 promedio ponderado (km)</text>',
        f'<text x="{y_axis_end[0] + 12:.2f}" y="{y_axis_end[1] - 10:.2f}" font-family="Arial" font-size="13">f2 máximo (km)</text>',
        f'<text x="{z_axis_end[0] - 55:.2f}" y="{z_axis_end[1] - 14:.2f}" font-family="Arial" font-size="13">f3 centros</text>',
    ]

    sorted_front = frentes.sort_values(
        ["f3_num_centros", "f2_distancia_maxima_km", "f1_distancia_promedio_ponderada_km"]
    )
    for row in sorted_front.itertuples(index=False):
        f1 = float(row.f1_distancia_promedio_ponderada_km)
        f2 = float(row.f2_distancia_maxima_km)
        f3 = int(row.f3_num_centros)
        x, y = project(f1, f2, f3)
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="{colors.get(f3, "#555")}" opacity="0.58"/>'
        )

    legend_x = 835
    legend_y = 120
    parts.append(f'<text x="{legend_x}" y="{legend_y - 12}" font-family="Arial" font-size="13" font-weight="700">f3 centros</text>')
    for idx, p in enumerate(range(P_MIN, P_MAX + 1)):
        y = legend_y + idx * 24
        parts.append(f'<rect x="{legend_x}" y="{y}" width="12" height="12" fill="{colors[p]}"/>')
        parts.append(f'<text x="{legend_x + 20}" y="{y + 11}" font-family="Arial" font-size="12">{p}</text>')

    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta NSGA-II de tres objetivos.")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    resultados, frentes, centros, cargas = run_all(args.seeds)
    suffix = args.output_suffix
    resultados.to_csv(RESULTS_DIR / f"nsga2_3obj_resultados{suffix}.csv", index=False, encoding="utf-8")
    frentes.to_csv(RESULTS_DIR / f"nsga2_3obj_frentes{suffix}.csv", index=False, encoding="utf-8")
    centros.to_csv(RESULTS_DIR / f"nsga2_3obj_centros{suffix}.csv", index=False, encoding="utf-8")
    cargas.to_csv(RESULTS_DIR / f"nsga2_3obj_cargas{suffix}.csv", index=False, encoding="utf-8")

    if not args.skip_figures:
        build_tradeoff_figure(frentes, FIGURES_DIR / "pareto_3obj_tradeoff.svg")
        build_3d_pareto_figure(frentes, FIGURES_DIR / "pareto_3obj_3d.svg")

    resumen = (
        frentes.groupby("f3_num_centros", as_index=False)
        .agg(
            soluciones=("solution_id", "nunique"),
            f1_min=("f1_distancia_promedio_ponderada_km", "min"),
            f1_median=("f1_distancia_promedio_ponderada_km", "median"),
            f2_min=("f2_distancia_maxima_km", "min"),
            f2_median=("f2_distancia_maxima_km", "median"),
        )
        .sort_values("f3_num_centros")
    )

    print("NSGA-II de tres objetivos generado.")
    print(f"Corridas: {len(resultados)}")
    print(f"Soluciones no dominadas: {len(frentes)}")
    print(f"Centros registrados: {len(centros)}")
    print(f"Cargas registradas: {len(cargas)}")
    print("\nResumen por número de centros:")
    print(resumen.to_string(index=False))


if __name__ == "__main__":
    main()
