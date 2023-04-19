from app.database import db

from app.login import get_current_user
from flask import Blueprint, render_template_string
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import datetime as dt
from flask_cors import CORS, cross_origin
import plotly.graph_objs as go
import json
import plotly


arima_blueprint = Blueprint('arima', __name__)


@arima_blueprint.route("/arima", methods=["GET"])
@cross_origin(supports_credentials=True)
def model_arima():
    df = get_hist_data()
    result = arima_predictions(df)
    mape = mape_calc(result)

    # Crear un objeto ExcelWriter
    writer = pd.ExcelWriter('result_arima.xlsx', engine='xlsxwriter')

    df.to_excel(writer, sheet_name='data', index=False)

    # Escribir la hoja "result"
    result.to_excel(writer, sheet_name='result', index=True)

    # Escribir la hoja "mape"
    mape.to_excel(writer, sheet_name='mape', index=True)

    # Guardar el archivo Excel
    writer.save()

    # Crear el gráfico de líneas con Plotly
    fig = go.Figure()
    for product, row in result.iterrows():
        fig.add_trace(go.Scatter(x=row.index, y=row.values, name="trace" + str(product)))


    fig.update_layout(title='ARIMA Predictions',
                      xaxis_title='Date',
                      yaxis_title='Value')

    # Incrustar el gráfico en la página web
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    with open('arima.html', 'r') as f:
        template_string = f.read()
    return render_template_string(template_string, graphJSON=graphJSON)


def get_hist_data():
    user = get_current_user().json["id"]
    table_name = "historical data_" + user
    df = pd.read_sql_table(table_name, db.engine.connect())
    # Reemplazar los NaN de las columnas a partir de la posición 14 por ceros
    df.iloc[:, 13:] = df.iloc[:, 13:].replace(to_replace=["NaN", "null", "nan"], value=np.nan)
    df.iloc[:, 13:] = df.iloc[:, 13:].fillna(0).apply(pd.to_numeric, errors='coerce').values

    # Obtener la lista de columnas a partir de la posición 14
    columnas = df.columns[13:]

    # Convertir las columnas a formato datetime
    df.rename(columns={col: dt.datetime.strptime(col, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d') for col in columnas}, inplace=True)
    # Reemplazar los valores NaN con 0 en las columnas a partir de la posición 14
    df.iloc[:, 13:] = df.iloc[:, 13:].fillna(0)

    return df


def arima_predictions(dataframe):
    # Define el número de periodos a predecir
    forecast_periods = len(dataframe.columns) - 13
    df = dataframe.copy()

    # Define el orden del modelo ARIMA
    arima_order = (1, 1, 0)

    # Crea un dataframe vacío para almacenar los resultados de predicción
    df_pred = pd.DataFrame(columns=['product', 'date', 'value', 'Model'])

    # Itera sobre cada fila del dataframe de ventas
    for product, row in df.iterrows():
        # Crea una serie de tiempo a partir de los valores de ventas
        time_series = pd.Series(row.iloc[13:]).astype(float)

        # Usa toda la serie de tiempo como datos de entrenamiento
        train_data = time_series
        test_data = time_series[-forecast_periods:]

        # Realiza la predicción con el modelo ARIMA
        model = ARIMA(train_data, order=arima_order)
        model_fit = model.fit()
        predictions = model_fit.forecast(len(test_data))

        # Almacena los resultados de predicción y el MAPE en el dataframe
        for i, pred in enumerate(predictions):
            pred_date = test_data.index[i]
            df_pred = df_pred.append({'product': product, 'date': pred_date, 'value': pred, 'Model': 'arima'},
                                     ignore_index=True)
            df_pred = df_pred.append(
                {'product': product, 'date': pred_date, 'value': row[pred_date], 'Model': 'original'},
                ignore_index=True)


    df_pred['date'] = pd.to_datetime(df_pred['date'], errors="ignore")
    # Pivota el dataframe para tener las fechas como columnas y los productos como filas
    df_pred_pivot = df_pred.pivot(index=['product', 'Model'], columns='date', values='value')
    # Combina los dataframes de ventas y predicciones en uno solo
    df_combined = pd.concat([df_pred_pivot], axis=1)
    # Devuelve el dataframe combinado
    return df_combined


# Filtrar las filas que corresponden a las predicciones ARIMA
def mape_calc(df):
    arima_pred = df.xs('arima', level='Model')
    original_data = df.xs('original', level='Model')

    # Calcular el error absoluto y relativo para cada producto
    error_list = []
    for product in arima_pred.index:
        arima_vals = arima_pred.loc[product].fillna(0).values
        original_vals = original_data.loc[product].fillna(0).values
        if 0 in original_vals:
            error = np.mean(np.abs(original_vals - arima_vals))
            tipo = "Absoluto"
        else:
            error = np.mean(np.abs((original_vals - arima_vals) / original_vals)) * 100
            tipo = "MAPE"
        error_list.append(error)

    # Crear un dataframe con los resultados del error
    error_df = pd.DataFrame({'product': arima_pred.index.get_level_values(0), 'error': error_list, 'tipo': tipo})

    return error_df




