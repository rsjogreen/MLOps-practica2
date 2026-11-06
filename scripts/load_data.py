"""Carga el dataset de fuga de empleados, valida sus columnas y muestra su tamaño.

Por defecto lee `datos.ruta` de configs/config.yaml. Las rutas se resuelven desde el
directorio de trabajo, así que ejecuta el script desde la raíz de actividad-2.

Ejecución (desde la raíz de actividad-2):
    poetry run python scripts/load_data.py

Para usarla desde un flow:
    from scripts.load_data import cargar_datos
"""
import pandas as pd
import yaml

RUTA_CONFIG = "configs/config.yaml"

# Columnas del dataset de fuga: las variables predictivas temporales/demográficas y el objetivo `Target_Fuga`.
# Se excluyen identificadores granulares como `cod_empl` y `nro_cont`.
# Se incluye `ID_Jefe_Directo` ya que agrupa empleados y puede ser útil para capturar el efecto del liderazgo en la fuga.
COLUMNAS = [
    "Mes",
    "tip_cont",
    "Tenure_Meses",
    "Edad",
    "sex_empl",
    "est_civi",
    "cod_area",
    "cod_carg",
    "Meses_En_Cargo",
    "ID_Jefe_Directo",
    "Meses_Con_Jefe",
    "Salario_Mes",
    "Variacion_Absoluta_Salario",
    "Porcentaje_Variacion_Salario",
    "Ausentismo_Mes",
    "Ausentismo_Trimestre",
    "Dias_Vacaciones_Mes",
    "Target_Fuga",
]


def cargar_datos(ruta=None):
    """Devuelve el DataFrame de fuga con las columnas esperadas. Falla si falta alguna."""
    if ruta is None:
        with open(RUTA_CONFIG, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        ruta = config["datos"]["ruta_train"]

    df = pd.read_csv(ruta, low_memory=False)
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Al CSV le faltan columnas: {faltan}")
    
    return df[COLUMNAS].copy()


def main():
    ruta = None  # None = usa la ruta de configs/config.yaml

    df = cargar_datos(ruta)
    print(f"Filas: {len(df)} \ncolumnas: {df.shape[1]} \nnulos: {int(df.isna().sum().sum())}")
    print(f"Columnas: {list(df.columns)}")


if __name__ == "__main__":
    main()