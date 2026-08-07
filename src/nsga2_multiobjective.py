"""
NSGA-II multiobjetivo con p fijo usando PYMOO.

Objetivos:
- f1: minimizar distancia promedio ponderada por producción.
- f2: minimizar distancia máxima de atención.

Restricción:
- Seleccionar exactamente p centros.

Salidas:
- data/results/nsga2_resultados.csv
- data/results/nsga2_frentes.csv
- data/results/nsga2_centros.csv
- data/results/nsga2_cargas.csv
- figures/pareto_fronts_by_p.svg
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

from multiobjective_problem import objective_array_for_pymoo
from problem_definition import assign_to_nearest_center, load_problem_data, summarize_center_loads


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

P_VALUES = [3, 5, 7, 10]
SEEDS = list(range(1, 31))

POPULATION_SIZE = 100
GENERATIONS = 200
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.2


class FixedCardinalitySampling(Sampling):
    """Muestreo factible: exactamente p centros por individuo."""

    def __init__(self, p: int):
        super().__init__()
        self.p = p

    def _do(self, problem, n_samples, **kwargs):
        x = np.zeros((n_samples, problem.n_var), dtype=bool)
        for i in range(n_samples):
            idx = np.random.choice(problem.n_var, size=self.p, replace=False)
            x[i, idx] = True
        return x


class FixedCardinalityRepair(Repair):
    """Repara cada individuo para cumplir exactamente p centros."""

    def __init__(self, p: int):
        super().__init__()
        self.p = p

    def _do(self, problem, X, **kwargs):
        X = np.asarray(X, dtype=bool).copy()
        for i in range(X.shape[0]):
            active = np.flatnonzero(X[i])
            inactive = np.flatnonzero(~X[i])
            if len(active) > self.p:
                remove = np.random.choice(active, size=len(active) - self.p, replace=False)
                X[i, remove] = False
            elif len(active) < self.p:
                add = np.random.choice(inactive, size=self.p - len(active), replace=False)
                X[i, add] = True
        return X


class CentrosAcopioMultiObjectiveProblem(Problem):
    """Problema multiobjetivo para PYMOO."""

    def __init__(self, problem_data, p: int):
        super().__init__(
            n_var=len(problem_data.municipios),
            n_obj=2,
            n_ieq_constr=0,
            xl=0,
            xu=1,
            vtype=bool,
        )
        self.problem_data = problem_data
        self.p = p

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = objective_array_for_pymoo(self.problem_data, X, self.p)


def vector_to_centers(x: np.ndarray, claves: list[str]) -> list[str]:
    return [claves[i] for i in np.flatnonzero(np.asarray(x, dtype=bool))]


def deduplicate_front(X: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Elimina soluciones duplicadas por vector binario."""
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


def run_single_nsga2(problem_data, p: int, seed: int) -> dict:
    problem = CentrosAcopioMultiObjectiveProblem(problem_data, p)
    algorithm = NSGA2(
        pop_size=POPULATION_SIZE,
        sampling=FixedCardinalitySampling(p),
        crossover=TwoPointCrossover(prob=CROSSOVER_PROBABILITY),
        mutation=BitflipMutation(prob=MUTATION_PROBABILITY),
        repair=FixedCardinalityRepair(p),
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


def run_all_nsga2(p_values: list[int], seeds: list[int]):
    problem_data = load_problem_data()
    claves = problem_data.municipios["clave_inegi"].tolist()
    nombres = problem_data.municipios[["clave_inegi", "municipio", "produccion_ton"]]

    resultados_rows = []
    frentes_rows = []
    centros_rows = []
    cargas_frames = []

    for p in p_values:
        for corrida, seed in enumerate(seeds, start=1):
            run = run_single_nsga2(problem_data, p, seed)
            X, F = run["X"], run["F"]

            resultados_rows.append(
                {
                    "p": p,
                    "corrida": corrida,
                    "semilla": seed,
                    "n_solutions": len(F),
                    "f1_min": float(F[:, 0].min()),
                    "f1_mean": float(F[:, 0].mean()),
                    "f1_max": float(F[:, 0].max()),
                    "f2_min": float(F[:, 1].min()),
                    "f2_mean": float(F[:, 1].mean()),
                    "f2_max": float(F[:, 1].max()),
                    "tiempo_segundos": run["tiempo_segundos"],
                    "population_size": POPULATION_SIZE,
                    "generations": GENERATIONS,
                    "crossover_probability": CROSSOVER_PROBABILITY,
                    "mutation_probability": MUTATION_PROBABILITY,
                }
            )

            order = np.lexsort((F[:, 1], F[:, 0]))
            for local_id, idx in enumerate(order, start=1):
                solution_id = f"p{p}_s{seed}_sol{local_id:03d}"
                x = X[idx]
                f = F[idx]
                centros = sorted(vector_to_centers(x, claves))

                frentes_rows.append(
                    {
                        "p": p,
                        "corrida": corrida,
                        "semilla": seed,
                        "solution_id": solution_id,
                        "f1_distancia_promedio_ponderada_km": float(f[0]),
                        "f2_distancia_maxima_km": float(f[1]),
                        "n_centros": len(centros),
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
                            "p": p,
                            "corrida": corrida,
                            "semilla": seed,
                            "solution_id": solution_id,
                            "orden": centro_order,
                            "clave_inegi": row.clave_inegi,
                            "municipio": row.municipio,
                            "produccion_ton": row.produccion_ton,
                        }
                    )

                cargas = summarize_center_loads(problem_data, centros, p)
                cargas.insert(0, "solution_id", solution_id)
                cargas.insert(0, "semilla", seed)
                cargas.insert(0, "corrida", corrida)
                cargas.insert(0, "p", p)
                cargas_frames.append(cargas)

    resultados = pd.DataFrame(resultados_rows)
    frentes = pd.DataFrame(frentes_rows)
    centros = pd.DataFrame(centros_rows)
    cargas = pd.concat(cargas_frames, ignore_index=True) if cargas_frames else pd.DataFrame()
    return resultados, frentes, centros, cargas


def build_pareto_fronts_figure(frentes: pd.DataFrame, salida: Path) -> None:
    width = 1050
    height = 720
    margin = 65
    panel_gap = 45
    panel_w = (width - 2 * margin - panel_gap) / 2
    panel_h = (height - 2 * margin - panel_gap) / 2
    colors = {3: "#c43c39", 5: "#2f6f9f", 7: "#8a4f9e", 10: "#7a9e3a"}

    f1_min, f1_max = frentes["f1_distancia_promedio_ponderada_km"].min(), frentes["f1_distancia_promedio_ponderada_km"].max()
    f2_min, f2_max = frentes["f2_distancia_maxima_km"].min(), frentes["f2_distancia_maxima_km"].max()
    f1_pad = max((f1_max - f1_min) * 0.08, 1e-6)
    f2_pad = max((f2_max - f2_min) * 0.08, 1e-6)
    f1_min, f1_max = f1_min - f1_pad, f1_max + f1_pad
    f2_min, f2_max = f2_min - f2_pad, f2_max + f2_pad

    def panel_origin(p_idx: int) -> tuple[float, float]:
        row = p_idx // 2
        col = p_idx % 2
        return margin + col * (panel_w + panel_gap), margin + row * (panel_h + panel_gap)

    def x_pos(f1: float, x0: float) -> float:
        return x0 + ((f1 - f1_min) / (f1_max - f1_min)) * panel_w

    def y_pos(f2: float, y0: float) -> float:
        return y0 + panel_h - ((f2 - f2_min) / (f2_max - f2_min)) * panel_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">Frentes de Pareto NSGA-II por p</text>',
        f'<text x="{width/2}" y="48" text-anchor="middle" font-family="Arial" font-size="12">f1: distancia promedio ponderada; f2: distancia máxima. Menor es mejor en ambos.</text>',
    ]

    for idx, p in enumerate(sorted(frentes["p"].unique())):
        x0, y0 = panel_origin(idx)
        sub = frentes[frentes["p"] == p]
        parts.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="#fafafa" stroke="#ddd"/>')
        parts.append(f'<text x="{x0 + panel_w/2}" y="{y0 - 10}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700">p={int(p)}</text>')
        parts.append(f'<line x1="{x0}" y1="{y0 + panel_h}" x2="{x0 + panel_w}" y2="{y0 + panel_h}" stroke="#333"/>')
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + panel_h}" stroke="#333"/>')
        for row in sub.itertuples(index=False):
            cx = x_pos(row.f1_distancia_promedio_ponderada_km, x0)
            cy = y_pos(row.f2_distancia_maxima_km, y0)
            parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="3" fill="{colors[int(p)]}" opacity="0.65"/>')
        parts.append(f'<text x="{x0 + panel_w/2}" y="{y0 + panel_h + 28}" text-anchor="middle" font-family="Arial" font-size="11">f1 km</text>')
        parts.append(f'<text x="{x0 - 35}" y="{y0 + panel_h/2}" text-anchor="middle" font-family="Arial" font-size="11" transform="rotate(-90 {x0 - 35} {y0 + panel_h/2})">f2 km</text>')

    parts.append("</svg>")
    salida.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta NSGA-II multiobjetivo con p fijo.")
    parser.add_argument("--p-values", nargs="+", type=int, default=P_VALUES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    resultados, frentes, centros, cargas = run_all_nsga2(args.p_values, args.seeds)

    suffix = args.output_suffix
    resultados.to_csv(RESULTS_DIR / f"nsga2_resultados{suffix}.csv", index=False, encoding="utf-8")
    frentes.to_csv(RESULTS_DIR / f"nsga2_frentes{suffix}.csv", index=False, encoding="utf-8")
    centros.to_csv(RESULTS_DIR / f"nsga2_centros{suffix}.csv", index=False, encoding="utf-8")
    cargas.to_csv(RESULTS_DIR / f"nsga2_cargas{suffix}.csv", index=False, encoding="utf-8")

    if not args.skip_figures:
        build_pareto_fronts_figure(frentes, FIGURES_DIR / "pareto_fronts_by_p.svg")

    resumen = (
        frentes.groupby("p", as_index=False)
        .agg(
            soluciones=("solution_id", "nunique"),
            f1_min=("f1_distancia_promedio_ponderada_km", "min"),
            f1_max=("f1_distancia_promedio_ponderada_km", "max"),
            f2_min=("f2_distancia_maxima_km", "min"),
            f2_max=("f2_distancia_maxima_km", "max"),
        )
        .sort_values("p")
    )

    print("NSGA-II multiobjetivo generado.")
    print(f"Corridas: {len(resultados)}")
    print(f"Soluciones no dominadas registradas: {len(frentes)}")
    print(f"Centros registrados: {len(centros)}")
    print(f"Cargas registradas: {len(cargas)}")
    print("\nResumen por p:")
    print(resumen.to_string(index=False))


if __name__ == "__main__":
    main()
