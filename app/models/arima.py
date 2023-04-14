from app.database import db
from app.login import get_current_user
from flask import Blueprint
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

arima_blueprint = Blueprint('arima', __name__)


@arima_blueprint.route("/arima", methods=["GET"])
def model_arima():
    df = get_hist_data()
    result = arima_predictions(df)
    print(result)
    result.to_csv('result_arima.csv', index=True)
    return "arima done"


def get_hist_data():
    user = get_current_user().json["id"]
    table_name = "historical data_" + user
    df = pd.read_sql_table(table_name, db.engine.connect())
    # Obtener la lista de columnas a partir de la posición 14
    columnas = df.columns[13:]
    df.rename(columns={col: pd.to_datetime(col, format='%Y-%m-%d %H:%M:%S') for col in columnas}, inplace=True)
    return df


def arima_predictions(dataframe):
    # Define el número de periodos a predecir
    forecast_periods = 8
    df = dataframe.copy()

    # Define el orden del modelo ARIMA
    arima_order = (1, 1, 0)

    # Crea un dataframe vacío para almacenar los resultados de predicción
    df_pred = pd.DataFrame(columns=['product', 'date', 'predicted_value'])

    # Itera sobre cada fila del dataframe de ventas
    for product, row in df.iterrows():
        # Crea una serie de tiempo a partir de los valores de ventas
        time_series = pd.Series(row.iloc[13:]).astype(float)

        # Divide la serie de tiempo en un conjunto de entrenamiento y validación
        train_data = time_series[:-forecast_periods]
        test_data = time_series

        # Realiza la predicción con el modelo ARIMA
        model = ARIMA(train_data, order=arima_order)
        model_fit = model.fit()
        predictions = model_fit.forecast(len(test_data))

        # Almacena los resultados de predicción en el dataframe
        for i, pred in enumerate(predictions):
            pred_date = test_data.index[i]
            df_pred = df_pred.append({'product': product, 'date': pred_date, 'predicted_value': pred}, ignore_index=True)

    # Convierte la columna de fechas a formato fecha
    df_pred['date'] = pd.to_datetime(df_pred['date'])

    # Pivota el dataframe para tener las fechas como columnas y los productos como filas
    df_pred_pivot = df_pred.pivot(index='product', columns='date', values='predicted_value')

    # Combina los dataframes de ventas y predicciones en uno solo
    df_combined = pd.concat([df, df_pred_pivot], axis=1)

    # Devuelve el dataframe combinado
    return df_combined



