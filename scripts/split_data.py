import pandas as pd
import yaml

# Por defecto lee `datos.ruta` de configs/config.yaml. Las rutas se resuelven desde el
# directorio de trabajo, así que ejecuta el script desde la raíz.
# se debe ejecutar unicament una vez para generar los archivos train_fuga.csv y test_fuga.csv en la carpeta data

RUTA_CONFIG = "configs/config.yaml"

def split_time_based_data(filepath=None, target_test_size=0.20, path=None):

    """Devuelve el DataFrame de churn con las columnas esperadas. Falla si falta alguna."""
    if filepath is None:
        with open(RUTA_CONFIG, encoding="utf-8") as f:
            config = yaml.safe_load(f)
            #print(config)

        # carga las rutas del archivo de configuración
        filepath = config["datos"]["ruta"]
        path = config["datos"]["path"]

    # 1. Cargar datos
    df = pd.read_csv(filepath, low_memory=False)
    
    # 2. Ordenar cronológicamente (YYYY-MM se ordena correctamente como string)
    df = df.sort_values('Mes')
    
    # 3. Definir los meses que irán a Test (los últimos 7 meses dan ~19%)
    test_months = ['2025-09', '2025-10', '2025-11', '2025-12', 
                   '2026-01', '2026-02', '2026-03']
    
    # 4. Separar los datasets
    test_df = df[df['Mes'].isin(test_months)]
    train_df = df[~df['Mes'].isin(test_months)]
    
    # 5. Guardar (opcional) o retornar
    train_df.to_csv(f"{path}/train_fuga.csv", index=False)
    test_df.to_csv(f"{path}/test_fuga.csv", index=False)
    
    return train_df, test_df

split_time_based_data()

