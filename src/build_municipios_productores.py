"""
Construye data/processed/municipios_productores.csv.

Fuentes de entrada intactas:
- data/raw/dgsiap/produccion_agricola_municipal_2024_original.csv
- data/raw/inegi/localidades_tamaulipas_ageeml_original.csv

Transformaciones:
1. Lee la producción agrícola municipal DGSIAP 2024.
2. Filtra Tamaulipas (Idestado = 28), año 2024 y cultivo Sorgo grano.
3. Verifica que la unidad de volumen sea Tonelada.
4. Construye clave INEGI municipal de cinco dígitos: Idestado(2) + Idmunicipio(3).
5. Suma Volumenproduccion por municipio.
6. Lee localidades INEGI AGEEML y toma la localidad 0001 de cada municipio.
7. Usa latitud/longitud decimal de INEGI como coordenadas de cabecera municipal.
8. Une coordenadas con producción agregada.
9. Conserva municipios sin producción con produccion_ton = 0.
10. Valida 43 municipios, claves únicas, coordenadas completas, longitudes negativas
    y producción no negativa.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRODUCCION_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dgsiap"
    / "produccion_agricola_municipal_2024_original.csv"
)
LOCALIDADES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "inegi"
    / "localidades_tamaulipas_ageeml_original.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "municipios_productores.csv"

ENTIDAD_TAMAULIPAS = "28"
ANIO = "2024"
CULTIVO = "Sorgo grano"
UNIDAD_ESPERADA = "Tonelada"


def construir_clave_municipal(entidad: pd.Series, municipio: pd.Series) -> pd.Series:
    """Construye clave INEGI municipal en formato texto de cinco dígitos."""
    return entidad.astype(str).str.strip().str.zfill(2) + municipio.astype(str).str.strip().str.zfill(3)


def cargar_produccion_agregada() -> pd.DataFrame:
    """Carga DGSIAP, filtra sorgo grano en Tamaulipas 2024 y agrega toneladas por municipio."""
    columnas = [
        "Anio",
        "Idestado",
        "Nomestado",
        "Idmunicipio",
        "Nommunicipio",
        "Nomcicloproductivo",
        "Nommodalidad",
        "Nomunidad",
        "Nomcultivo",
        "Volumenproduccion",
    ]
    produccion = pd.read_csv(
        PRODUCCION_PATH,
        encoding="latin-1",
        dtype=str,
        usecols=columnas,
    )

    produccion["Idestado_norm"] = produccion["Idestado"].astype(str).str.strip().str.zfill(2)
    produccion["Anio_norm"] = produccion["Anio"].astype(str).str.strip()
    produccion["Nomcultivo_norm"] = produccion["Nomcultivo"].astype(str).str.strip()

    filtrada = produccion[
        (produccion["Idestado_norm"] == ENTIDAD_TAMAULIPAS)
        & (produccion["Anio_norm"] == ANIO)
        & (produccion["Nomcultivo_norm"].str.casefold() == CULTIVO.casefold())
    ].copy()

    if filtrada.empty:
        raise ValueError(f"No se encontraron filas para {CULTIVO}, Tamaulipas, {ANIO}.")

    unidades = sorted(filtrada["Nomunidad"].dropna().astype(str).str.strip().unique().tolist())
    if unidades != [UNIDAD_ESPERADA]:
        raise ValueError(f"Unidad inesperada para {CULTIVO}: {unidades}")

    filtrada["clave_inegi"] = construir_clave_municipal(
        filtrada["Idestado"],
        filtrada["Idmunicipio"],
    )
    filtrada["produccion_ton"] = pd.to_numeric(
        filtrada["Volumenproduccion"],
        errors="raise",
    )

    agregada = (
        filtrada.groupby("clave_inegi", as_index=False)
        .agg(
            produccion_ton=("produccion_ton", "sum"),
            filas_fuente=("produccion_ton", "size"),
        )
        .sort_values("clave_inegi")
    )
    return agregada


def cargar_cabeceras_inegi() -> pd.DataFrame:
    """Carga AGEEML y toma localidad 0001 como cabecera municipal candidata."""
    localidades = pd.read_csv(
        LOCALIDADES_PATH,
        encoding="utf-8-sig",
        dtype=str,
    )

    columnas_requeridas = [
        "Clave de AGEE",
        "Clave de AGEM",
        "Nombre de AGEM",
        "Clave Localidad Geoestadística",
        "Latitud decimal",
        "Longitud decimal",
    ]
    faltantes = [col for col in columnas_requeridas if col not in localidades.columns]
    if faltantes:
        raise ValueError(f"Columnas faltantes en AGEEML: {faltantes}")

    localidades["clave_inegi"] = construir_clave_municipal(
        localidades["Clave de AGEE"],
        localidades["Clave de AGEM"],
    )

    cabeceras = localidades[
        (localidades["Clave de AGEE"].astype(str).str.strip().str.zfill(2) == ENTIDAD_TAMAULIPAS)
        & (localidades["Clave Localidad Geoestadística"].astype(str).str.strip().str.zfill(4) == "0001")
    ].copy()

    cabeceras = cabeceras[
        [
            "clave_inegi",
            "Nombre de AGEM",
            "Latitud decimal",
            "Longitud decimal",
        ]
    ].rename(
        columns={
            "Nombre de AGEM": "municipio",
            "Latitud decimal": "latitud",
            "Longitud decimal": "longitud",
        }
    )

    cabeceras["municipio"] = cabeceras["municipio"].astype(str).str.strip()
    cabeceras["latitud"] = pd.to_numeric(cabeceras["latitud"], errors="raise")
    cabeceras["longitud"] = pd.to_numeric(cabeceras["longitud"], errors="raise")

    return cabeceras.sort_values("clave_inegi").reset_index(drop=True)


def validar_municipios_productores(df: pd.DataFrame) -> None:
    """Valida reglas básicas del archivo final."""
    columnas_esperadas = ["clave_inegi", "municipio", "latitud", "longitud", "produccion_ton"]
    if list(df.columns) != columnas_esperadas:
        raise ValueError(f"Columnas inesperadas: {list(df.columns)}")

    if len(df) != 43:
        raise ValueError(f"Se esperaban 43 municipios; se obtuvieron {len(df)}.")

    if df["clave_inegi"].nunique() != 43:
        raise ValueError("Las claves INEGI no son únicas.")

    if not df["clave_inegi"].astype(str).str.fullmatch(r"\d{5}").all():
        raise ValueError("Hay claves INEGI que no tienen formato de cinco dígitos.")

    if df[["latitud", "longitud"]].isna().any().any():
        raise ValueError("Hay coordenadas faltantes.")

    if not (df["longitud"] < 0).all():
        raise ValueError("Hay longitudes sin signo negativo.")

    if df["produccion_ton"].isna().any():
        raise ValueError("Hay producción faltante.")

    if not (df["produccion_ton"] >= 0).all():
        raise ValueError("Hay producción negativa.")


def main() -> None:
    produccion = cargar_produccion_agregada()
    cabeceras = cargar_cabeceras_inegi()

    final = cabeceras.merge(
        produccion[["clave_inegi", "produccion_ton"]],
        on="clave_inegi",
        how="left",
        validate="one_to_one",
    )
    final["produccion_ton"] = final["produccion_ton"].fillna(0.0)

    final = final[
        ["clave_inegi", "municipio", "latitud", "longitud", "produccion_ton"]
    ].sort_values("clave_inegi")

    final["clave_inegi"] = final["clave_inegi"].astype(str)
    final["produccion_ton"] = final["produccion_ton"].round(2)

    validar_municipios_productores(final)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Archivo generado: {OUTPUT_PATH}")
    print(f"Municipios: {len(final)}")
    print(f"Producción total ton: {final['produccion_ton'].sum():,.2f}")
    print(f"Municipios con producción cero: {(final['produccion_ton'] == 0).sum()}")


if __name__ == "__main__":
    main()
