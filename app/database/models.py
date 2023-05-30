from uuid import uuid4
from .db import db


def get_uuid():
    return uuid4().hex


class User(db.Model):
    id = db.Column(db.String(32), primary_key=True, unique=True, default=get_uuid())
    email = db.Column(db.String(345), unique=True)
    password = db.Column(db.Text, nullable=False)
    name = db.Column(db.String(80), unique=True, nullable=False)


# tabla de referencia de archivos, incluye el usuario y la fecha de subida del archivo
class FileRef(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(32))
    file_name = db.Column(db.String(64), nullable=False)
    upload_date = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp(), nullable=False)


# selectores para preparacion de datos
class DataSelectors(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    run_mode = db.Column(db.String(32))
    top_down_dims = db.Column(db.String(32))
    forecast_dims = db.Column(db.String(32))
    replace_outliers = db.Column(db.String(32))
    interpolate_negatives = db.Column(db.String(345))
    interpolate_zeros = db.Column(db.String(345))
    interpolation_methods = db.Column(db.String(345))
    outliers_detection = db.Column(db.String(345))
    missing_values_t = db.Column(db.String(345))
    pex_variables = db.Column(db.String(32))
    mapes100 = db.Column(db.String(32))
