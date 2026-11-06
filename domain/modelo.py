"""Lógica de negocio del modelo de fuga de empleados: variables, pipeline y métricas.

No usa Prefect ni MLflow: se puede probar sin servicios levantados.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Variables del modelo adaptadas al dataset de Fuga de Empleados. 
# La columna 'Mes' se excluye aquí porque se usa para separar los datos, no para entrenar.
NUMERICAS = [
    "Tenure_Meses",
    "Edad",
    "Meses_En_Cargo",
    "Meses_Con_Jefe",
    "Salario_Mes",
    "Variacion_Absoluta_Salario",
    "Porcentaje_Variacion_Salario",
    "Ausentismo_Mes",
    "Ausentismo_Trimestre",
    "Dias_Vacaciones_Mes"
]

CATEGORICAS = [
    "tip_cont", 
    "sex_empl", 
    "est_civi", 
    "cod_area", 
    "cod_carg", 
    "ID_Jefe_Directo"
]

VARIABLES = NUMERICAS + CATEGORICAS
OBJETIVO = "Target_Fuga"


def resolver_features(columnas):
    """Convierte `columnas` de config.yaml ('ALL' o una lista de nombres) en (numericas, categoricas)."""
    if columnas == "ALL":
        elegidas = VARIABLES
    else:
        desconocidas = [v for v in columnas if v not in VARIABLES]
        if desconocidas:
            raise ValueError(f"variables desconocidas en 'columnas': {desconocidas}")
        elegidas = columnas

    if not elegidas:
        raise ValueError("la selección de features no tiene ninguna variable")
    numericas = [v for v in NUMERICAS if v in elegidas]
    categoricas = [v for v in CATEGORICAS if v in elegidas]
    return numericas, categoricas


def construir_modelo(tipo, parametros, numericas, categoricas):
    """Preprocesado (escalar y one-hot) más el estimador elegido en la configuración."""
    if tipo == "logistic_regression":
        estimador = LogisticRegression(**parametros)
    elif tipo == "gradient_boosting":
        estimador = GradientBoostingClassifier(**parametros)
    else:
        raise ValueError(f"modelo '{tipo}' no soportado")

    transformadores = []
    if numericas:
        transformadores.append(("num", StandardScaler(), numericas))
    if categoricas:
        # handle_unknown="ignore" es clave porque en Test pueden aparecer jefes o cargos nuevos
        transformadores.append(("cat", OneHotEncoder(handle_unknown="ignore"), categoricas))
    
    preprocesado = ColumnTransformer(transformadores)
    return Pipeline([("preprocesado", preprocesado), ("modelo", estimador)])


def calcular_metricas(y_true, proba, umbral):
    """Métricas de clasificación binaria a partir de la probabilidad de fuga."""
    pred = (proba >= umbral).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }