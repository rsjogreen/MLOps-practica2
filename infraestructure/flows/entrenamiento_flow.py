"""Actividad 2: pipeline de entrenamiento.

Pasos: cargar datos → entrenar y registrar en MLflow cada candidato de `candidatos` → promover a champion el mejor si mejora

El modelo recibe el alias `champion` solo si su AUC supera `registro.umbral_auc` y el AUC
del champion actual. Si no, el run queda en MLflow pero el alias no cambia.

La lógica del modelo está en domain/modelo.py. Este archivo solo orquesta y registra en MLflow.

Requisitos: servicios levantados (ver services/README.md) y las variables de entorno:
    PREFECT_API_URL=http://localhost:4200/api
    MLFLOW_TRACKING_URI=http://localhost:5000

Ejecución (desde la raíz de actividad-2):
    poetry run python infraestructure/flows/entrenamiento_flow.py
"""
import os

import mlflow
import yaml
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from prefect import flow, task
from sklearn.model_selection import train_test_split

from domain.modelo import OBJETIVO, calcular_metricas, construir_modelo, resolver_features
from scripts.load_data import cargar_datos as leer_datos

CONFIG = "configs/config.yaml"


@task
def cargar_datos(ruta):
    """Carga el CSV con scripts/load_data.py y valida sus columnas."""
    print(f"Cargando datos desde {ruta}...")
    return leer_datos(ruta)

@task
def entrenar_y_registrar(df, cfg, features):
    """Entrena el modelo, calcula sus métricas y registra el run en MLflow. Devuelve (run_id, model_uri, métricas)."""
    
    print(f"Entrenando {cfg['modelo']['tipo']} con features={features['nombre']}...")
    numericas, categoricas = resolver_features(features["columnas"])
    variables = numericas + categoricas
    
    # 1. Partir los datos de forma CRONOLÓGICA (Out-of-Time Validation)
    print("División cronológica train/test (último 20% como test)")
    
    # Ordenamos por la columna 'Mes' para garantizar el orden temporal
    df_sorted = df.sort_values('Mes').copy()
    
    # Calculamos el índice de corte
    test_size_ratio = cfg["division"]["test_size"]
    split_idx = int(len(df_sorted) * (1 - test_size_ratio))
    
    # Separamos secuencialmente
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    X_train, y_train = train_df[variables], train_df[OBJETIVO]
    X_test, y_test = test_df[variables], test_df[OBJETIVO]
    
    print(f"Conjunto de entrenamiento (Periodos antiguos): {len(X_train)} filas, {len(variables)} features")
    print(f"Conjunto de prueba (Periodos recientes): {len(X_test)} filas")
    print(f"Proporción de {OBJETIVO} en Test: {y_test.mean():.2%}")

    # 2. Crear y entrenar el modelo (preprocesado + estimador).
    print(f"Creando modelo {cfg['modelo']['tipo']} con parámetros {cfg['modelo'].get('parametros')}")
    tipo = cfg["modelo"]["tipo"]
    parametros = cfg["modelo"].get("parametros") or {}
    modelo = construir_modelo(tipo, parametros, numericas, categoricas)
    
    print("Entrenando modelo...")
    modelo.fit(X_train, y_train)
    print("Modelo entrenado")
    
    # 3. Calcular las métricas sobre el conjunto de prueba (Out-of-time).
    umbral = cfg["metricas"]["umbral"]
    proba = modelo.predict_proba(X_test)[:, 1]
    metricas = calcular_metricas(y_test, proba, umbral)
    print(f"Métricas sobre el conjunto de prueba: {metricas}")

    # 4. Registrar el run en MLflow. MLFLOW_TRACKING_URI, si existe, tiene prioridad sobre el YAML.
    print(f"Registrando run en MLflow (tracking_uri={os.environ.get('MLFLOW_TRACKING_URI', cfg['mlflow']['tracking_uri'])})...")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])
    
    with mlflow.start_run(run_name=cfg["mlflow"]["nombre_run"]) as run:
        mlflow.set_tag("fase", "entrenamiento")
        mlflow.log_params(
            {
                "modelo": tipo,
                **parametros,
                "split_type": "chronological",
                "test_size_ratio": test_size_ratio,
                "umbral": umbral,
                "features": features["nombre"],
                "n_features": len(variables),
            }
        )
        mlflow.log_metrics(metricas)
        info = mlflow.sklearn.log_model(
            modelo,
            name="model",
            signature=infer_signature(X_test, proba),
            input_example=X_test.head(3),
            # Gradient boosting guarda árboles de sklearn; skops los marca como no confiables por defecto.
            # Son nuestros propios modelos entrenados aquí, así que se confía solo en ese tipo.
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        run_id = run.info.run_id
        print(f"Run registrado en MLflow: {run_id} | URI del modelo: {info.model_uri}")

    return run_id, info.model_uri, metricas


@task
def promover_si_mejora(model_uri, metricas, cfg):
    """Asigna el alias champion solo si el AUC supera el mínimo y al champion actual."""
    print(f"Promoviendo a champion si mejora: AUC={metricas['auc']:.4f} (umbral={cfg['registro']['umbral_auc']})")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    client = MlflowClient()
    nombre = cfg["registro"]["nombre_modelo"]
    auc = metricas["auc"]
    print(f"Conectado a MLflow (tracking_uri={mlflow.get_tracking_uri()}) | Modelo: {nombre}")

    if auc < cfg["registro"]["umbral_auc"]:
        print(f"AUC {auc:.4f} por debajo del mínimo {cfg['registro']['umbral_auc']}: no se registra como champion")
        return None
    else:
        print(f"AUC {auc:.4f} supera el mínimo {cfg['registro']['umbral_auc']}: se intenta promover a champion")

    try:
        champion = client.get_model_version_by_alias(nombre, "champion")
    except MlflowException:
        champion = None  # todavía no existe ningún champion
    print(f"Champion actual: {champion.version if champion is not None else 'ninguno'}")

    if champion is not None:
        auc_champion = client.get_run(champion.run_id).data.metrics["auc"]
        if auc <= auc_champion:
            print(f"AUC {auc:.4f} no supera al champion ({auc_champion:.4f}): el alias no cambia")
            return None
        else:
            print(f"AUC {auc:.4f} supera al champion ({auc_champion:.4f}): se registra como champion")

    print(f"Registrando versión {model_uri} como champion...")
    version = mlflow.register_model(model_uri, nombre)
    client.set_registered_model_alias(nombre, "champion", version.version)
    client.set_model_version_tag(nombre, version.version, "estado", "aprobado")
    print(f"Versión {version.version} registrada como champion (AUC {auc:.4f})")
    return version.version


@flow(name="entrenamiento-churn-v2", log_prints=True)
def entrenamiento_flow():
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = cargar_datos(cfg["datos"]["ruta_train"])

    # Entrena cada candidato con cada conjunto de features. Cada combinación es un run propio.
    resultados = []
    print(f"Entrenando candidatos: {[c['nombre_run'] for c in cfg['candidatos']]} con features {[f['nombre'] for f in cfg['features']]}")
    for candidato in cfg["candidatos"]:
        for features in cfg["features"]:
            cfg_run = {
                **cfg,
                "modelo": {"tipo": candidato["tipo"], "parametros": candidato["parametros"]},
                "mlflow": {**cfg["mlflow"], "nombre_run": f"{candidato['nombre_run']}-{features['nombre']}"},
            }
            run_id, model_uri, metricas = entrenar_y_registrar(df, cfg_run, features)
            print(
                f"{candidato['nombre_run']} ({candidato['tipo']}) features={features['nombre']}: "
                f"AUC = {metricas['auc']:.4f}"
            )
            resultados.append((run_id, model_uri, metricas))

    # Solo el mejor candidato (mayor AUC) puede pasar a champion.
    mejor_run_id, mejor_uri, mejor_metricas = max(resultados, key=lambda r: r[2]["auc"])
    promover_si_mejora(mejor_uri, mejor_metricas, cfg)
    print(f"Mejor run de MLflow: {mejor_run_id} | AUC = {mejor_metricas['auc']:.4f}")
    return mejor_run_id


if __name__ == "__main__":
    entrenamiento_flow()
