"""
Construye data/processed/distancias.csv a partir de municipios_productores.csv.

Entrada:
- data/processed/municipios_productores.csv

Salida:
- data/processed/distancias.csv

Transformaciones:
1. Lee las 43 cabeceras municipales ya validadas.
2. Usa latitud y longitud decimal.
3. Calcula la distancia geográfica entre cada par origen-destino con Haversine.
4. Genera una matriz larga con 43 x 43 = 1,849 filas.
5. Valida diagonal cero, simetría, claves completas y distancias no negativas.
"""

from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "municipios_productores.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "distancias.csv"

RADIO_TIERRA_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula distancia Haversine en kilómetros entre dos puntos decimales."""
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)

    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    c = 2 * asin(sqrt(a))
    return RADIO_TIERRA_KM * c


def cargar_municipios() -> pd.DataFrame:
    """Carga y valida campos mínimos de municipios_productores.csv."""
    municipios = pd.read_csv(INPUT_PATH, dtype={"clave_inegi": str})

    columnas_requeridas = ["clave_inegi", "municipio", "latitud", "longitud", "produccion_ton"]
    faltantes = [col for col in columnas_requeridas if col not in municipios.columns]
    if faltantes:
        raise ValueError(f"Columnas faltantes en municipios_productores.csv: {faltantes}")

    if len(municipios) != 43:
        raise ValueError(f"Se esperaban 43 municipios; se obtuvieron {len(municipios)}.")

    if municipios["clave_inegi"].nunique() != 43:
        raise ValueError("Las claves INEGI no son únicas.")

    if not municipios["clave_inegi"].astype(str).str.fullmatch(r"\d{5}").all():
        raise ValueError("Hay claves INEGI que no tienen formato de cinco dígitos.")

    municipios["latitud"] = pd.to_numeric(municipios["latitud"], errors="raise")
    municipios["longitud"] = pd.to_numeric(municipios["longitud"], errors="raise")

    if municipios[["latitud", "longitud"]].isna().any().any():
        raise ValueError("Hay coordenadas faltantes.")

    if not (municipios["longitud"] < 0).all():
        raise ValueError("Hay longitudes sin signo negativo.")

    return municipios.sort_values("clave_inegi").reset_index(drop=True)


def construir_distancias(municipios: pd.DataFrame) -> pd.DataFrame:
    """Genera distancias origen-destino en formato largo."""
    filas = []

    for origen in municipios.itertuples(index=False):
        for destino in municipios.itertuples(index=False):
            distancia = haversine_km(
                float(origen.latitud),
                float(origen.longitud),
                float(destino.latitud),
                float(destino.longitud),
            )
            filas.append(
                {
                    "origen": origen.clave_inegi,
                    "destino": destino.clave_inegi,
                    "distancia_km": round(distancia, 6),
                }
            )

    distancias = pd.DataFrame(filas, columns=["origen", "destino", "distancia_km"])
    return distancias


def validar_distancias(distancias: pd.DataFrame, municipios: pd.DataFrame) -> None:
    """Valida estructura y propiedades básicas de la matriz de distancias."""
    claves = set(municipios["clave_inegi"])
    n = len(claves)

    if list(distancias.columns) != ["origen", "destino", "distancia_km"]:
        raise ValueError(f"Columnas inesperadas: {list(distancias.columns)}")

    if len(distancias) != n * n:
        raise ValueError(f"Se esperaban {n*n} filas; se obtuvieron {len(distancias)}.")

    if set(distancias["origen"]) != claves:
        raise ValueError("Las claves de origen no coinciden con municipios_productores.csv.")

    if set(distancias["destino"]) != claves:
        raise ValueError("Las claves de destino no coinciden con municipios_productores.csv.")

    if distancias[["origen", "destino"]].duplicated().any():
        raise ValueError("Hay pares origen-destino duplicados.")

    if distancias["distancia_km"].isna().any():
        raise ValueError("Hay distancias faltantes.")

    if not (distancias["distancia_km"] >= 0).all():
        raise ValueError("Hay distancias negativas.")

    diagonal = distancias[distancias["origen"] == distancias["destino"]]
    if not (diagonal["distancia_km"].abs() <= 1e-9).all():
        raise ValueError("La diagonal no es cero.")

    matriz = distancias.pivot(index="origen", columns="destino", values="distancia_km")
    diferencia_maxima = (matriz - matriz.T).abs().to_numpy().max()
    if diferencia_maxima > 1e-6:
        raise ValueError(f"La matriz no es simétrica. Diferencia máxima: {diferencia_maxima}")


def main() -> None:
    municipios = cargar_municipios()
    distancias = construir_distancias(municipios)
    validar_distancias(distancias, municipios)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    distancias.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    no_diagonal = distancias[distancias["origen"] != distancias["destino"]]
    print(f"Archivo generado: {OUTPUT_PATH}")
    print(f"Municipios: {len(municipios)}")
    print(f"Filas: {len(distancias)}")
    print(f"Distancia mínima no diagonal km: {no_diagonal['distancia_km'].min():.6f}")
    print(f"Distancia máxima km: {distancias['distancia_km'].max():.6f}")


if __name__ == "__main__":
    main()
