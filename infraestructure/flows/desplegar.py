"""Actividad 2: despliegue de las pipelines en Prefect.

- entrenamiento-churn: sin horario. Se lanza a mano desde la UI de Prefect (http://localhost:4200).
- inferencia-churn: se ejecuta todos los días a las 06:00 (cron "0 6 * * *", en hora UTC).

Ejecución (desde la raíz de actividad-2):
    poetry run python infraestructure/flows/desplegar.py

Deja esta terminal abierta: mientras esté abierta, Prefect puede lanzar las ejecuciones.
"""
from prefect import serve

from entrenamiento_flow import entrenamiento_flow
from inferencia_flow import inferencia_flow

if __name__ == "__main__":
    despliegue_entrenamiento = entrenamiento_flow.to_deployment(name="entrenamiento-churn")
    despliegue_inferencia = inferencia_flow.to_deployment(name="inferencia-diaria", cron="0 6 * * *")
    serve(despliegue_entrenamiento, despliegue_inferencia)
