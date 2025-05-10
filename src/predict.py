# predict.py

import pandas as pd


def predict_new(model, features, input_data, threshold=0.3):
    """
    Predice si lloverá o no para un nuevo registro horario.

    Args:
        model: Modelo entrenado (con predict_proba).
        features (list): Lista de variables usadas en el entrenamiento.
        input_data (dict): Ejemplo con las variables requeridas.
        threshold (float): Umbral de clasificación para decidir si lloverá.

    Returns:
        pred (int): 0 o 1 (no lluvia / lluvia) según el threshold.
        proba (float): Probabilidad de lluvia predicha por el modelo.
    """
    # Convertir el diccionario a DataFrame
    df_input = pd.DataFrame([input_data])

    # Ordenar columnas como las del modelo
    df_input = df_input[features]

    # Obtener probabilidad de lluvia
    proba = model.predict_proba(df_input)[0][1]

    # Aplicar umbral personalizado
    pred = int(proba >= threshold)

    return pred, proba
