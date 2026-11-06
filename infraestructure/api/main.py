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
from pydantic import BaseModel, Field


with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
NOMBRE_MODELO = cfg["registro"]["nombre_modelo"]
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}@champion")
VERSION = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, "champion").version

app = FastAPI(title="API de churn", description="Predicción de abandono de clientes con el modelo champion.")


class Cliente(BaseModel):
    """Datos de un cliente. Los nombres son los mismos que las columnas del CSV."""

    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Complain: int
    satisfaction_score: int = Field(alias="Satisfaction Score")
    card_type: str = Field(alias="Card Type")
    point_earned: int = Field(alias="Point Earned")


@app.get("/health")
def health():
    return {"estado": "ok", "modelo": NOMBRE_MODELO, "version": VERSION}


@app.post("/predict")
def predict(cliente: Cliente):
    # Usa las mismas columnas con las que se entrenó el modelo.
    datos = pd.DataFrame([cliente.model_dump(by_alias=True)])[list(modelo.feature_names_in_)]
    proba = float(modelo.predict_proba(datos)[0, 1])
    return {
        "probabilidad_churn": round(proba, 4),
        "prediccion": int(proba >= 0.5),
        "version_modelo": VERSION,
    }
