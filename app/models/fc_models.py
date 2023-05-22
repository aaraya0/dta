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
    result.to_excel(writer, sheet_name='result', index=True, merge_cells=False)
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
    # plot_predictions(df, prediction_p)
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
        arima_df, arima_mape = arima_predictions(row, test_p, pred_p)
        linear_df, linear_mape = linear_regression_predictions(row, test_p, pred_p)
        exp_df, exp_mape = exp_smoothing_predictions(row, test_p, pred_p)
        holt_df, holt_mape = holt_winters_predictions(row, test_p, pred_p)

    # -------------------------------------------------------------------

    # -------------------------------------------------------------------


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
    plot_predictions(df_pred)
    return df_pred


def arima_predictions(fila, test_periods, prediction_periods):
    df_pred = pd.DataFrame(columns = ['family', 'region', 'salesman', 'client', 'category', 'subcategory',
                                      'sku', 'description', 'model', 'date', 'value'])
    df_pred_fc = df_pred.copy()
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

# -------------------------------------------------------------
    start_date = pd.to_datetime(test_data.index[-1])
    next_month = start_date + pd.DateOffset(months = 1)
    future_dates = pd.date_range(start = next_month, periods = prediction_periods, freq = 'MS')
    future_dates = future_dates.strftime('%Y-%m-%d')
    future_predictions = model_fit.forecast(prediction_periods)



# -------------------------------------------------------------

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

    for i, future in enumerate(future_dates):

        fut_date = future_dates[i]
        df_pred_fc = df_pred_fc.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3], 'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'actual',
             'date': fut_date, 'value': None}, ignore_index = True)

        df_pred_fc = df_pred_fc.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3],
             'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'arima',
             'date': fut_date, 'value': future_predictions[i]}, ignore_index = True)

    df_pred_fc_pivot = df_pred_fc.pivot(values = 'value', index = ['family', 'region', 'salesman', 'client', 'category',
                                                             'subcategory', 'sku', 'description', 'model'],
                                  columns = 'date')
    result = pd.concat([df_pred_pivot, df_pred_fc_pivot], axis = 1)

    return result, mape


def linear_regression_predictions(fila, test_periods, prediction_periods):
    df_pred = pd.DataFrame(columns = ['family', 'region', 'salesman', 'client', 'category', 'subcategory',
                                      'sku', 'description', 'model', 'date', 'value'])

    df_pred_fc = df_pred.copy()
    time_series = pd.Series(fila.iloc[13:]).astype(float)
    train_data = time_series[:-test_periods]
    test_data = time_series.iloc[-test_periods:]


    model = LinearRegression()
    X_train = pd.DataFrame(pd.to_numeric(pd.to_datetime(train_data.index))).astype(int).values.reshape(-1, 1)
    y_train = train_data.values.reshape(-1, 1)
    model.fit(X_train, y_train)

    # Use the model to make predictions on the testing data
    X_test = pd.DataFrame(pd.to_numeric(pd.to_datetime(test_data.index))).astype(int).values.reshape(-1, 1)

    test_predictions = np.squeeze(model.predict(X_test))
    train_predictions = np.squeeze(model.predict(X_train))

# --------------------------------------------------------------------
    start_date = pd.to_datetime(test_data.index[-1])
    next_month = start_date + pd.DateOffset(months = 1)
    future_dates = pd.date_range(start = next_month, periods = prediction_periods, freq = 'MS')
    future_dates = future_dates.strftime('%Y-%m-%d')


    X_future = pd.DataFrame(pd.to_numeric(pd.to_datetime(future_dates))).astype(int).values.reshape(-1, 1)
    future_predictions = np.squeeze(model.predict(X_future))

    # ----------------------------------------------------------------
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

    for i, future in  enumerate(future_dates):
        fut_date = future_dates[i]
        df_pred_fc = df_pred_fc.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3], 'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'actual',
             'date': fut_date, 'value': None}, ignore_index = True)

        df_pred_fc = df_pred_fc.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3],
             'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'linear',
             'date': fut_date, 'value': future_predictions[i]}, ignore_index = True)

    df_pred_fc_pivot = df_pred_fc.pivot(values = 'value', index = ['family', 'region', 'salesman', 'client', 'category',
                                                                   'subcategory', 'sku', 'description', 'model'],
                                        columns = 'date')
    result = pd.concat([df_pred_pivot, df_pred_fc_pivot], axis = 1)

    return result, mape


def exp_smoothing_predictions(fila, test_periods, prediction_periods):
    df_pred = pd.DataFrame(columns = ['family', 'region', 'salesman', 'client', 'category', 'subcategory',
                                      'sku', 'description', 'model', 'date', 'value'])
    df_pred_fc = df_pred.copy()
    time_series = pd.Series(fila.iloc[13:]).astype(float)
    train_data = time_series[:-test_periods]
    test_data = time_series.iloc[-test_periods:]


    # Use Pandas' exponential smoothing function to create the model and make predictions
    model = pd.Series(train_data).ewm(span=10).mean()
    test_predictions = model[-test_periods:]
    train_predictions = model[:-test_periods]
    model = model.fillna(0)  # Reemplazar los valores no válidos con ceros

    # ------------------------------------------------------------------------------------
    start_date = pd.to_datetime(test_data.index[-1])
    next_month = start_date + pd.DateOffset(months = 1)
    future_dates = pd.date_range(start = next_month, periods = prediction_periods, freq = 'MS')
    future_dates = future_dates.strftime('%Y-%m-%d')

    future_predictions = model.ewm(span=10, min_periods=0).mean().iloc[-1:].repeat(len(future_dates))

    # -------------------------------------------------------
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
    mape = mape_calc(df_pred_pivot, 'exp_smooth')

    for i, future in enumerate(future_dates):
        fut_date = future_dates[i]
        df_pred_fc = df_pred_fc.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3], 'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'actual',
             'date': fut_date, 'value': None}, ignore_index = True)

        df_pred_fc = df_pred_fc.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3],
             'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'exp_smooth',
             'date': fut_date, 'value': future_predictions[i]}, ignore_index = True)

    df_pred_fc_pivot = df_pred_fc.pivot(values = 'value', index = ['family', 'region', 'salesman', 'client', 'category',
                                                                   'subcategory', 'sku', 'description', 'model'],
                                        columns = 'date')
    result = pd.concat([df_pred_pivot, df_pred_fc_pivot], axis = 1)

    return result, mape


def holt_winters_predictions(fila, test_periods, prediction_periods):
    df_pred = pd.DataFrame(columns = ['family', 'region', 'salesman', 'client', 'category', 'subcategory',
                                      'sku', 'description', 'model', 'date', 'value'])
    df_pred_fc = df_pred.copy()
    time_series = pd.Series(fila.iloc[13:]).astype(float)
    train_data = time_series[:-test_periods]

    test_data = time_series.iloc[-test_periods:]


    # Create the Holt-Winters model using the training data
    model = ExponentialSmoothing(train_data, seasonal_periods=12, trend='add', seasonal='add')

    # Fit the model
    model_fit = model.fit()

    # Make predictions on the testing data
    test_predictions = model_fit.forecast(test_periods)
    # Get the original values
    train_predictions = model_fit.fittedvalues
# -----------------------------------------------------------------------

    start_date = pd.to_datetime(test_data.index[-1])
    next_month = start_date + pd.DateOffset(months = 1)
    future_dates = pd.date_range(start = next_month, periods = prediction_periods, freq = 'MS')
    future_dates = future_dates.strftime('%Y-%m-%d')
    future_predictions = model_fit.forecast(prediction_periods)
    # -----------------------------------------------------------------------

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

    mape = mape_calc(df_pred_pivot, 'holt_winters')

    for i, future in enumerate(future_dates):
        fut_date = future_dates[i]
        df_pred_fc = df_pred_fc.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3], 'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'actual',
             'date': fut_date, 'value': None}, ignore_index = True)

        df_pred_fc = df_pred_fc.append(
            {'family': fila.iloc[1], 'region': fila.iloc[2], 'salesman': fila.iloc[3],
             'client': fila.iloc[4],
             'category': fila.iloc[5], 'subcategory': fila.iloc[6],
             'sku': fila.iloc[7], 'description': fila.iloc[8], 'model': 'holt_winters',
             'date': fut_date, 'value': future_predictions[i]}, ignore_index = True)

    df_pred_fc_pivot = df_pred_fc.pivot(values = 'value', index = ['family', 'region', 'salesman', 'client', 'category',
                                                                   'subcategory', 'sku', 'description', 'model'],
                                        columns = 'date')
    result = pd.concat([df_pred_pivot, df_pred_fc_pivot], axis = 1)

    return result, mape



def plot_predictions(df_pred):

    df_pred = df_pred.reset_index()


    df_actual = df_pred[df_pred['model'] == 'actual'].iloc[:,9:]
    df_models = df_pred[~(df_pred['model'] == 'actual')].iloc[:,9:]



    df_total_actual = df_actual.sum(axis = 1)
    df_total_models = df_models.sum(axis = 1)

    trace1 = go.Scatter(
        x = df_total_actual.index,
        y = df_total_actual.values,
        name = 'Actual'
    )

    trace2 = go.Scatter(
        x = df_total_models.index,
        y = df_total_models.values,
        name = 'Prediccion'
    )

    data = [trace1, trace2]

    layout = go.Layout(
        title = 'Ventas por fecha',
        xaxis = dict(title = 'Fecha'),
        yaxis = dict(title = 'Ventas',  tickformat='.if')
    )

    fig = go.Figure(data = data, layout = layout)

    pyo.plot(fig, filename='ventas.html')
