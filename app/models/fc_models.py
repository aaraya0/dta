from app.database import db
from app.login import get_current_user
from flask import Blueprint
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import datetime as dt
from flask_cors import cross_origin
from flask import jsonify, request
import pmdarima as pm
from sklearn.linear_model import LinearRegression

models_blueprint = Blueprint('models', __name__)


@models_blueprint.route("/model", methods=["POST"])
@cross_origin(supports_credentials=True)
def testing_model():
    json_data = request.json
    prediction_p = json_data.get("prediction_p")
    if not prediction_p:
        return jsonify({'message': 'Missing parameters'}), 400
    df = get_hist_data()
    result_1 = linear_regression_predictions(df, prediction_p)
    result_2 = arima_predictions(df, prediction_p)
    result_3 = exp_smoothing_predictions(df, prediction_p)
    writer = pd.ExcelWriter('prediction_results.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='data', index=False)
    result_1[0].to_excel(writer, sheet_name='result_reg', index=True)
    result_1[1].to_excel(writer, sheet_name='error_reg', index=True)
    result_2[0].to_excel(writer, sheet_name = 'result_arima', index = True)
    result_2[1].to_excel(writer, sheet_name = 'error_arima', index = True)
    result_3[0].to_excel(writer, sheet_name = 'result_exp', index = True)
    result_3[1].to_excel(writer, sheet_name = 'error_exp', index = True)
    writer.save()
    return jsonify({'message': 'Model results saved to prediction_results.xlsx'}), 200


def get_hist_data():
    user = get_current_user().json["id"]
    table_name = "historical data_" + user
    df = pd.read_sql_table(table_name, db.engine.connect())

    df.iloc[:, 13:] = df.iloc[:, 13:].replace(to_replace=["NaN", "null", "nan"], value=np.nan)
    df.iloc[:, 13:] = df.iloc[:, 13:].fillna(0).apply(pd.to_numeric, errors='coerce').values

    columnas = df.columns[13:]

    df.rename(columns={col: dt.datetime.strptime(col, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d') for col in columnas},
              inplace=True)

    df.iloc[:, 13:] = df.iloc[:, 13:].fillna(0)

    return df


def mape_calc(df, model_name):
    model_pred = df.xs(model_name, level='model')
    original_data = df.xs('original', level='model')
    error_list = []
    for product in model_pred.index:
        model_vals = model_pred.loc[product].fillna(0).values
        original_vals = original_data.loc[product].fillna(0).values
        mask = original_vals != 0
        error = np.mean(np.abs((original_vals[mask] - model_vals[mask]) / original_vals[mask])) * 100
        if np.isnan(error):
            error = 100
        error_list.append(error)

    error_df = pd.DataFrame({'product': model_pred.index.get_level_values(0), 'error': error_list})
    return error_df


def arima_predictions(dataframe, prediction_periods):

    df = dataframe.copy()

    # Crea un dataframe vacío para almacenar los resultados de predicción
    df_pred = pd.DataFrame(columns=['product', 'model',  'date', 'value'])

    # Itera sobre cada fila del df original
    for product, row in df.iterrows():
        # se hace una serie de tiempo de cada producto solo con las filas de fecha
        time_series = pd.Series(row.iloc[13:]).astype(float)

        # se seleccionan las columnas con periodos para entrenamiento y para prediccion
        train_data = time_series[:-prediction_periods]
        test_data = time_series.iloc[-prediction_periods:]

        # se seleccionan automaticamente los parametros para modelar el conjunto de entrenamiento
        model = pm.auto_arima(train_data)
        arima_order = model.order
        # se calcula el modelo arima para los datos de entrenamiento
        model = ARIMA(train_data, order=arima_order)
        # se ajusta el modelo
        model_fit = model.fit()

        # en base al modelo ajustado, se hace el forecast sobre los datos a predecir,
        # cada prediccion es un valor en el df
        predictions = model_fit.forecast(len(test_data))
        originals = model_fit.forecast(len(train_data))

        # se itera sobre los valores de entrenamiento y se completan los valores de prediccion y originales del df nuevo
        for i, og in enumerate(originals):
            # se toma la fecha de prediccion como el valor del indice de este valor
            og_date = train_data.index[i]
            df_pred = df_pred.append(
                {'product': product, 'model': 'original', 'date': og_date, 'value': row[og_date]},
                ignore_index = True)
            df_pred = df_pred.append({'product': product, 'model': 'arima', 'date': og_date, 'value': og},
                                     ignore_index = True)

        # se enumeran las preddiciones y se pasa por cada una en una misma fila
        for i, pred in enumerate(predictions):
            # se toma la fecha de prediccion como el valor del indice de este valor
            pred_date = test_data.index[i]
            # al df resultado se le agrega product (que es un contador de la fila), date (q es el indice de este valor)
            # value (que es el valor que tiene la fila del indice que se tomo como fecha en el df original) y model (que
            # indica que es el original) se ignora el indice porque ya esta el de producto
            df_pred = df_pred.append(
                {'product': product, 'model': 'original', 'date': pred_date, 'value': row[pred_date]},
                ignore_index=True)
            # tambien, por cada valor, se agrega una fila con el mismo valor de product (contador de filas), la misma
            # fecha, el valor de la prediccion y la aclaracion de que el modelo es arima.
            df_pred = df_pred.append({'product': product, 'model': 'arima', 'date': pred_date, 'value': pred},
                                     ignore_index=True)

    # a cada valor del nuevo df en la columna date se lo convierte a datetime (antes era string, creo)
    df_pred['date'] = pd.to_datetime(df_pred['date'], errors="ignore")
    # genera una tabla mas "ancha" resumiendo y agrupando los datos por producto
    df_pred_pivot = df_pred.pivot(index=['product', 'model'], columns='date', values='value')

    mape_error = mape_calc(df_pred_pivot, 'arima')

    return df_pred_pivot, mape_error


def linear_regression_predictions(dataframe, prediction_periods):
    df = dataframe.copy()

    # Create an empty dataframe to store the prediction results
    df_pred = pd.DataFrame(columns=['product', 'model', 'date', 'value'])

    # Iterate over each row of the original df
    for product, row in df.iterrows():
        # Create a time series for each product using only the date columns
        time_series = pd.Series(row.iloc[13:]).astype(float)

        # Split the time series into training and testing data
        train_data = time_series[:-prediction_periods:]
        test_data = time_series.iloc[-prediction_periods:]

        # Fit a linear regression model to the training data
        model = LinearRegression()
        X_train = pd.DataFrame(pd.to_numeric(pd.to_datetime(train_data.index))).astype(int).values.reshape(-1, 1)
        y_train = train_data.values.reshape(-1, 1)
        model.fit(X_train, y_train)

        # Use the model to make predictions on the testing data
        X_test = pd.DataFrame(pd.to_numeric(pd.to_datetime(test_data.index))).astype(int).values.reshape(-1, 1)

        # predictions = model.predict(X_test)
        predictions = np.squeeze(model.predict(X_test))
        # originals = model.predict(X_train)
        originals = np.squeeze(model.predict(X_train))

        for i, og in enumerate(originals):
            # Take the prediction date as the index value for this row
            og_date = train_data.index[i]
            # Add a row to the result df with product, date, value, and model = 'original'
            df_pred = df_pred.append(
                {'product': product, 'model': 'original', 'date': og_date, 'value': row[og_date]},
                ignore_index=True)
            # Add a row to the result df with product, date, prediction value, and model = 'linear'
            df_pred = df_pred.append({'product': product, 'model': 'linear', 'date': og_date, 'value': og},
                                     ignore_index=True)

        for i, pred in enumerate(predictions):
            # Take the prediction date as the index value for this row
            pred_date = test_data.index[i]
            # Add a row to the result df with product, date, value, and model = 'original'
            df_pred = df_pred.append(
                {'product': product, 'model': 'original', 'date': pred_date, 'value': row[pred_date]},
                ignore_index=True)
            # Add a row to the result df with product, date, prediction value, and model = 'linear'
            df_pred = df_pred.append({'product': product, 'model': 'linear', 'date': pred_date, 'value': pred},
                                     ignore_index=True)

    # Convert the date column of the new df to datetime format
    df_pred['date'] = pd.to_datetime(df_pred['date'], errors="ignore")
    # Pivot the data to create a wider table summarizing the results by product
    df_pred_pivot = df_pred.pivot(index=['product', 'model'], columns='date', values='value')

    mape_error = mape_calc(df_pred_pivot, 'linear')

    return df_pred_pivot, mape_error


def exp_smoothing_predictions(dataframe, prediction_periods):
    df = dataframe.copy()

    # Create an empty dataframe to store the prediction results
    df_pred = pd.DataFrame(columns=['product', 'model', 'date', 'value'])

    # Iterate over each row of the original df
    for product, row in df.iterrows():
        # Create a time series for each product using only the date columns
        time_series = pd.Series(row.iloc[13:]).astype(float)

        # Split the time series into training and testing data
        train_data = time_series[:-prediction_periods:]
        test_data = time_series.iloc[-prediction_periods:]

        # Use Pandas' exponential smoothing function to create the model and make predictions
        model = pd.Series(train_data).ewm(span=10).mean()
        predictions = model[-prediction_periods:]
        originals = model[:-prediction_periods]

        for i, og in enumerate(originals):
            og_date = train_data.index[i]
            df_pred = df_pred.append(
                {'product': product, 'model': 'original', 'date': og_date, 'value': row[og_date]},
                ignore_index=True)
            df_pred = df_pred.append({'product': product, 'model': 'exp_smoothing', 'date': og_date, 'value': og},
                                     ignore_index=True)

        for i, pred in enumerate(predictions):
            pred_date = test_data.index[i]
            df_pred = df_pred.append(
                {'product': product, 'model': 'original', 'date': pred_date, 'value': row[pred_date]},
                ignore_index=True)
            df_pred = df_pred.append({'product': product, 'model': 'exp_smoothing', 'date': pred_date, 'value': pred},
                                     ignore_index=True)

    df_pred['date'] = pd.to_datetime(df_pred['date'], errors="ignore")
    df_pred_pivot = df_pred.pivot(index=['product', 'model'], columns='date', values='value')

    mape_error = mape_calc(df_pred_pivot, 'exp_smoothing')

    return df_pred_pivot, mape_error