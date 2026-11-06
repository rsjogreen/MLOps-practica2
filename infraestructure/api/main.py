"""Actividad 2: API de inferencia. Sirve el modelo con el alias `champion` de MLflow.

Ejecución (desde la raíz de actividad-2, con MLFLOW_TRACKING_URI definida):
    poetry run uvicorn infraestructure.api.main:app --port 8000
Documentación interactiva: http://localhost:8000/docs

El modelo se carga al arrancar la API. Si el pipeline de entrenamiento asigna un nuevo
champion, hay que reiniciar la API para que lo use.
"""
import os

import mlflow
import pandas as pd
import yaml
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from pydantic import BaseModel


# 1. Cargar la configuración
with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

NOMBRE_MODELO = cfg["registro"]["nombre_modelo"]
UMBRAL = cfg["metricas"]["umbral"]

# 2. Conectar a MLflow y cargar el Champion
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}@champion")

# Obtener la versión exacta que tiene el alias champion actualmente
VERSION = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, "champion").version

# 3. Inicializar la app FastAPI
app = FastAPI(
    title="API de Retención de Talento (Fuga de Empleados)", 
    description="Predicción de probabilidad de fuga de un empleado usando el modelo champion."
)

class Empleado(BaseModel):
    """Datos de un empleado. Los nombres coinciden exactamente con las columnas del CSV."""
    tip_cont: str
    Tenure_Meses: int
    Edad: int
    sex_empl: int
    est_civi: str
    cod_area: int
    cod_carg: str
    Meses_En_Cargo: int
    ID_Jefe_Directo: str
    Meses_Con_Jefe: int
    Salario_Mes: float
    Variacion_Absoluta_Salario: float
    Porcentaje_Variacion_Salario: float
    Ausentismo_Mes: int
    Ausentismo_Trimestre: int
    Dias_Vacaciones_Mes: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "tip_cont": "I",
                "Tenure_Meses": 24,
                "Edad": 35,
                "sex_empl": 0,
                "est_civi": "C",
                "cod_area": 332,
                "cod_carg": "130",
                "Meses_En_Cargo": 12,
                "ID_Jefe_Directo": "EMP_01369",
                "Meses_Con_Jefe": 10,
                "Salario_Mes": 3500000.0,
                "Variacion_Absoluta_Salario": 150000.0,
                "Porcentaje_Variacion_Salario": 0.04,
                "Ausentismo_Mes": 0,
                "Ausentismo_Trimestre": 2,
                "Dias_Vacaciones_Mes": 0
            }
        }
    }


@app.get("/health")
def health():
    """Endpoint de estado para comprobar qué modelo está sirviendo la API."""
    return {"estado": "ok", "modelo": NOMBRE_MODELO, "version": VERSION}


@app.post("/predict")
def predict(empleado: Empleado):
    """Calcula la probabilidad de fuga para un único empleado."""
    # Convertimos el JSON de entrada a DataFrame y filtramos solo las columnas que espera el modelo
    datos = pd.DataFrame([empleado.model_dump()])[list(modelo.feature_names_in_)]
    
    # Predecimos la probabilidad (clase 1 = Fuga)
    proba = float(modelo.predict_proba(datos)[0, 1])
    
    return {
        "probabilidad_fuga": round(proba, 4),
        "prediccion_fuga": int(proba >= UMBRAL),
        "version_modelo": VERSION,
    }