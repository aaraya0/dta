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
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objs as go
import plotly.offline as pyo
from dateutil.relativedelta import relativedelta

models_blueprint = Blueprint('models', __name__)


@models_blueprint.route("/model", methods=["POST"])
@cross_origin(supports_credentials=True)
def testing_model():
    json_data = request.json
    test_p = json_data.get("test_p")
    if not test_p:
        return jsonify({'message': 'Missing parameters'}), 400
    pred_p = json_data.get("prediction_p")
    if not pred_p:
        return jsonify({'message': 'Missing parameters'}), 400
    df = get_hist_data()
    result = best_model(df, test_p, pred_p)
    writer = pd.ExcelWriter('prediction_results.xlsx', engine='xlsxwriter')
    result.to_excel(writer, sheet_name='result', index=True)
    writer.save()
    return jsonify({'message': 'Model results saved to prediction_results.xlsx'}), 200

@models_blueprint.route("/graph", methods=["POST"])
@cross_origin(supports_credentials=True)
def testing_graph():
    json_data = request.json
    prediction_p = json_data.get("prediction_p")
    if not prediction_p:
        return jsonify({'message': 'Missing parameters'}), 400
    df = get_hist_data()
    plot_predictions(df, prediction_p)
    return jsonify({'message': 'Model results plotted'}), 200

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
    predicted_df = df.xs(model_name, level='model').iloc[:, -12:]
    actual_df = df.xs('actual', level='model').iloc[:, -12:]
    absolute_errors = []
    for col in predicted_df.columns:
        predicted_col = predicted_df[col]
        actual_col = actual_df[col]
        n = len(actual_col)
        col_errors = []
        for i in range(n):
            if actual_col[i] == 0 and predicted_col[i] != 0:
                col_errors.append(1)  # 100% de error relativo
            else:
                col_errors.append(abs(predicted_col[i] - actual_col[i]) / actual_col[i])
        col_mape = sum(col_errors) / n * 100
        absolute_errors.append(col_mape)
    mape = sum(absolute_errors) / len(absolute_errors)

    return mape


def best_model (dataframe, test_p, pred_p):
    df = dataframe.copy()
    df_pred = pd.DataFrame()
    for product, row in df.iterrows():
        arima_df, arima_mape, future_pred_arima = arima_predictions(row, test_p, pred_p)
        linear_df, linear_mape, future_pred_linear = linear_regression_predictions(row, test_p, pred_p)
        exp_df, exp_mape, future_pred_exp = exp_smoothing_predictions(row, test_p, pred_p)
        holt_df, holt_mape, future_pred_holt = holt_winters_predictions(row, test_p, pred_p)

        mape_list = [arima_mape, linear_mape, exp_mape, holt_mape]
        best_model_idx = mape_list.index(min(mape_list))

        if best_model_idx == 0:
            best_df = arima_df
            best_df['MAPE'] = arima_mape

        elif best_model_idx == 1:
            best_df = linear_df
            best_df['MAPE'] = linear_mape

        elif best_model_idx == 2:
            best_df = exp_df
            best_df['MAPE'] = exp_mape

        else:
            best_df = holt_df
            best_df['MAPE'] = holt_mape

        df_pred = df_pred.append(best_df,  ignore_index=False)

    return df_pred



def arima_predictions(fila, test_periods, prediction_periods):
    df_pred = pd.DataFrame(columns = ['family', 'region', 'salesman', 'client', 'category', 'subcategory',
                                      'sku', 'description', 'model', 'date', 'value'])

    time_series = pd.Series(fila.iloc[13:]).astype(float)
    train_data = time_series[:-test_periods]

    test_data = time_series.iloc[-test_periods:]
    n_train = len(train_data)
# --------------------------------------------------------
    model = pm.auto_arima(train_data)
    arima_order = model.order
    model = ARIMA(train_data, order=arima_order)
    model_fit = model.fit()

    train_predictions = model_fit.predict(start = 0, end = n_train - 1)
    test_predictions = model_fit.predict(start = n_train, end = len(time_series) - 1)

# predictions = model_fit.predict(start=len(time_series), end = len(time_series)+prediction_periods-1)
    model = ARIMA(test_data, order = arima_order)
    model_fit = model.fit()
    predictions = model_fit.forecast(prediction_periods)


# --------------------------------------------------------

    for i, og in enumerate(train_predictions):

        og_date = train_data.index[i]

        df_pred = df_pred.append(
            {'family': fila.iloc[1], 'region':fila.iloc[2] , 'salesman':fila.iloc[3], 'client':fila.iloc[4],
             'category': fila.iloc[5], 'subcategory':fila.iloc[6],
                                  'sku':fila.iloc[7], 'description':fila.iloc[8], 'model':'actual',
             'date': og_date, 'value':fila[og_date]}, ignore_index = True)

        df_pred = df_pred.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3],
             'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'arima',
             'date': og_date, 'value': og}, ignore_index = True)

    for i, test in enumerate(test_predictions):

        test_date = test_data.index[i]
        df_pred = df_pred.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3],
             'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'actual',
             'date': test_date, 'value': fila[test_date]}, ignore_index = True)

        df_pred = df_pred.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3],
             'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'arima',
             'date':test_date, 'value':test},ignore_index = True)


    df_pred_pivot = df_pred.pivot( values= 'value', index = ['family', 'region', 'salesman', 'client', 'category',
                                                     'subcategory', 'sku', 'description', 'model'], columns = 'date')
    mape = mape_calc(df_pred_pivot, 'arima')


    return df_pred_pivot, mape, predictions


def linear_regression_predictions(fila, test_periods, prediction_periods):
    df_pred = pd.DataFrame(columns = ['family', 'region', 'salesman', 'client', 'category', 'subcategory',
                                      'sku', 'description', 'model', 'date', 'value'])

    time_series = pd.Series(fila.iloc[13:]).astype(float)
    train_data = time_series[:-test_periods]
    test_data = time_series.iloc[-test_periods:]
    n_train = len(train_data)

    model = LinearRegression()
    X_train = pd.DataFrame(pd.to_numeric(pd.to_datetime(train_data.index))).astype(int).values.reshape(-1, 1)
    y_train = train_data.values.reshape(-1, 1)
    model.fit(X_train, y_train)

    # Use the model to make predictions on the testing data
    X_test = pd.DataFrame(pd.to_numeric(pd.to_datetime(test_data.index))).astype(int).values.reshape(-1, 1)

    test_predictions = np.squeeze(model.predict(X_test))

    train_predictions = np.squeeze(model.predict(X_train))
    # Obtener la última fecha conocida en el conjunto de datos
    last_date = test_data.index[-1]

    future_dates = pd.date_range(last_date, periods = prediction_periods, freq = 'M')
    X_future = pd.DataFrame(pd.to_numeric(pd.to_datetime(future_dates))).astype(int).values.reshape(-1, 1)
    predictions = np.squeeze(model.predict(X_future))
    print("LINEAR PREDICTIONS", predictions)

    for i, og in enumerate(train_predictions):
        og_date = train_data.index[i]
        #
        df_pred = df_pred.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3], 'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'actual',
             'date': og_date, 'value': fila[og_date]}, ignore_index = True)

        df_pred = df_pred.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3],
             'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'linear',
             'date': og_date, 'value': og}, ignore_index = True)

    for i, test in enumerate(test_predictions):
        test_date = test_data.index[i]
        df_pred = df_pred.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3],
             'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'actual',
             'date': test_date, 'value': fila[test_date]}, ignore_index = True)

        df_pred = df_pred.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3],
             'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'linear',
             'date': test_date, 'value': test}, ignore_index = True)

    df_pred_pivot = df_pred.pivot(values = 'value', index = ['family', 'region', 'salesman', 'client', 'category',
                                                             'subcategory', 'sku', 'description', 'model'],
                                  columns = 'date')

    mape = mape_calc(df_pred_pivot, 'linear')
    return df_pred_pivot, mape, predictions


def exp_smoothing_predictions(fila, test_periods, prediction_periods):
    df_pred = pd.DataFrame(columns = ['family', 'region', 'salesman', 'client', 'category', 'subcategory',
                                      'sku', 'description', 'model', 'date', 'value'])

    time_series = pd.Series(fila.iloc[13:]).astype(float)
    train_data = time_series[:-test_periods]
    test_data = time_series.iloc[-test_periods:]
    n_train = len(train_data)

    # Use Pandas' exponential smoothing function to create the model and make predictions
    model = pd.Series(train_data).ewm(span=10).mean()
    test_predictions = model[-test_periods:]
    train_predictions = model[:-test_periods]
    # Extend the time series with future dates
    future_dates = pd.date_range(start = time_series.index[-1], periods = prediction_periods, freq = 'MS')
    extended_series = time_series.append(pd.Series(index = future_dates))
    # Make a prediction for the extended series
    extended_model = pd.Series(extended_series[:n_train]).ewm(span = len(test_data)).mean()
    predictions = extended_model[-prediction_periods:]

    for i, og in enumerate(train_predictions):
        og_date = train_data.index[i]

        df_pred = df_pred.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3], 'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'actual',
             'date': og_date, 'value': fila[og_date]}, ignore_index = True)

        df_pred = df_pred.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3],
             'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'exp_smooth',
             'date': og_date, 'value': og}, ignore_index = True)

    for i, test in enumerate(test_predictions):
        test_date = test_data.index[i]
        df_pred = df_pred.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3],
             'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'actual',
             'date': test_date, 'value': fila[test_date]}, ignore_index = True)

        df_pred = df_pred.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3],
             'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'exp_smooth',
             'date': test_date, 'value': test}, ignore_index = True)

    df_pred_pivot = df_pred.pivot(values = 'value', index = ['family', 'region', 'salesman', 'client', 'category',
                                                             'subcategory', 'sku', 'description', 'model'],
                                  columns = 'date')
    print(df_pred_pivot)
    mape = mape_calc(df_pred_pivot, 'exp_smooth')

    return df_pred_pivot, mape, predictions


def holt_winters_predictions(fila, test_periods, prediction_periods):
    df_pred = pd.DataFrame(columns = ['family', 'region', 'salesman', 'client', 'category', 'subcategory',
                                      'sku', 'description', 'model', 'date', 'value'])

    time_series = pd.Series(fila.iloc[13:]).astype(float)
    train_data = time_series[:-test_periods]

    test_data = time_series.iloc[-test_periods:]
    n_train = len(train_data)

    # Create the Holt-Winters model using the training data
    model = ExponentialSmoothing(train_data, seasonal_periods=12, trend='add', seasonal='add')

    # Fit the model
    model_fit = model.fit()

    # Make predictions on the testing data
    test_predictions = model_fit.forecast(test_periods)

    # Get the original values
    train_predictions = model_fit.fittedvalues
    predictions = model_fit.forecast(prediction_periods)

    for i, og in enumerate(train_predictions):
        og_date = train_data.index[i]

        df_pred = df_pred.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3], 'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'actual',
             'date': og_date, 'value': fila[og_date]}, ignore_index = True)

        df_pred = df_pred.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3],
             'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'holt_winters',
             'date': og_date, 'value': og}, ignore_index = True)

    for i, test in enumerate(test_predictions):
        test_date = test_data.index[i]
        df_pred = df_pred.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3],
             'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'actual',
             'date': test_date, 'value': fila[test_date]}, ignore_index = True)

        df_pred = df_pred.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3],
             'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'holt_winters',
             'date': test_date, 'value': test}, ignore_index = True)

    df_pred_pivot = df_pred.pivot(values = 'value', index = ['family', 'region', 'salesman', 'client', 'category',
                                                             'subcategory', 'sku', 'description', 'model'],
                                  columns = 'date')
    print(df_pred_pivot)
    mape = mape_calc(df_pred_pivot, 'holt_winters')

    return df_pred_pivot, mape, predictions


def plot_predictions(dataframe, prediction_periods):
    # Obtiene las predicciones y el error MAPE utilizando la función arima_predictions
    df_arima = arima_predictions(dataframe, prediction_periods)
    df_linear = linear_regression_predictions(dataframe, prediction_periods)
    df_smooth = exp_smoothing_predictions(dataframe, prediction_periods)
    df_holt = holt_winters_predictions(dataframe, prediction_periods)
    df_arima = df_arima.reset_index()
    df_linear = df_linear.reset_index()
    df_smooth = df_smooth.reset_index()
    df_holt = df_holt.reset_index()

    df_original = df_arima[df_arima['model'] == 'original']
    original_sales = df_original.iloc[:, 2:].sum()

    df_linear = df_linear[df_linear['model'] == 'linear']
    linear_sales = df_linear.iloc[:, 2:].sum()
    df_arima= df_arima[df_arima['model'] == 'arima']
    arima_sales = df_arima.iloc[:, 2:].sum()
    df_smooth = df_smooth[df_smooth['model'] == 'exp_smoothing']
    smooth_sales = df_smooth.iloc[:, 2:].sum()
    df_holt = df_holt[df_holt['model'] == 'holt_winters']
    holt_sales = df_holt.iloc[:, 2:].sum()

    trace1 = go.Scatter(
        x = original_sales.index,
        y = original_sales.values,
        name = 'Original'
    )

    trace2 = go.Scatter(
        x = linear_sales.index,
        y = linear_sales.values,
        name = 'Linear Regression'
    )
    trace3 = go.Scatter(
        x = arima_sales.index,
        y = arima_sales.values,
        name = 'Arima'
    )
    trace4 = go.Scatter(
        x = smooth_sales.index,
        y = smooth_sales.values,
        name = 'Exponential Smoothing'
    )
    trace5 = go.Scatter(
        x = holt_sales.index,
        y = holt_sales.values,
        name = 'Holt Winters'
    )

    data = [trace1, trace2, trace3, trace4, trace5]

    layout = go.Layout(
        title = 'Ventas por fecha',
        xaxis = dict(title = 'Fecha'),
        yaxis = dict(title = 'Ventas',  tickformat='.if')
    )

    fig = go.Figure(data = data, layout = layout)

    pyo.plot(fig, filename='ventas.html')


