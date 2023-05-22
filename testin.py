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

dates = pd.date_range(start='2021-01-01', end='2021-12-01', freq='MS')
data = pd.Series(data=np.random.rand(len(dates)), index=dates)
last_date = data.index[-1]

# Obtener el día siguiente a la última fecha
next_date = last_date + pd.DateOffset(months = 1)
# Generar fechas futuras a partir del día siguiente
future_dates = pd.date_range(start=next_date, periods=10, freq='MS')
print( future_dates)