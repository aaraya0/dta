import pandas as pd
from flask import Blueprint
from flask import request, jsonify

from app.database import db
from app.database.models import FileRef, DataSelectors
from app.login import get_current_user
from flask_cors import cross_origin

# crea las rutas para exportar
excel_blueprint = Blueprint('excel', __name__)


# subir archivos excel
@excel_blueprint.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    user_id = get_current_user().json["id"]
    new_file = FileRef(file_name=file.filename, user_id=user_id)
    db.session.add(new_file)
    db.session.commit()
    save_df(file)
    response = jsonify({'message': 'Upload successful!'})
    return response


# obtener todos los nombres y fecha de los archivos subidos por un determinado usuario
@excel_blueprint.route('/files', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_files(user_id):
    files_list = FileRef.query.filter_by(user_id=user_id)
    for file in files_list:
        return jsonify({
            "id": file.id,
            "file_name": file.file_name,
            "upload_date": file.upload_date,
        })


# funcion que guarda el archivo en la base de datos, creando una nueva tabla con las mismas col que el archivo
# el nombre es nombre_del_archivo + _ + id_del_usuario para asociar a cada tabla con el usuario que la sube
def save_df(file):
    user_id = get_current_user().json["id"]
    excel_file = pd.ExcelFile(file)
    dataframe = pd.read_excel(excel_file, sheet_name=0)
    dataframe = dataframe.astype(str)  # Convert all columns to string type
    table_name = file.filename[:-5] + '_' + user_id
    if len(table_name) > 64:
        table_name = "launches_discontinued" + '_' + user_id
    dataframe.to_sql(table_name, db.engine, if_exists="replace")


# subir selectores para la preparacion de datos
@excel_blueprint.route("/data_sel", methods=["POST"])
@cross_origin(supports_credentials=True)
def data_selection():

    run_mode = request.json["run_mode"]

    top_down_dims = request.json["top_down_dims"]
    top_down_dims = ','.join(map(str, top_down_dims))

    forecast_dims = request.json["forecast_dims"]
    forecast_dims = ','.join(map(str, forecast_dims))

    replace_outliers = request.json["replace_outliers"]
    interpolate_negatives = request.json["interpolate_negatives"]
    interpolate_zeros = request.json["interpolate_zeros"]

    interpolation_methods = request.json["interpolation_methods"]
    interpolation_methods = ','.join(map(str, interpolation_methods))

    outliers_detection = request.json["outliers_detection"]
    missing_values_t = request.json["missing_values_t"]
    pex_variables = request.json["pex_variables"]
    mapes100 = request.json["mapes100"]

    new_selection = DataSelectors(run_mode=run_mode, top_down_dims=top_down_dims,
                                  forecast_dims=forecast_dims, replace_outliers=replace_outliers,
                                  interpolate_negatives=interpolate_negatives, interpolate_zeros=interpolate_zeros,
                                  interpolation_methods=interpolation_methods, outliers_detection=outliers_detection,
                                  missing_values_t=missing_values_t,  pex_variables=pex_variables, mapes100=mapes100)

    db.session.add(new_selection)
    db.session.commit()

    return jsonify({
        "id": new_selection.id,
        "run_mode": new_selection.run_mode,
        "top_down_dims": new_selection.top_down_dims,
        "forecast_dims": new_selection.forecast_dims,
        "replace_outliers": new_selection.replace_outliers,
        "interpolate_negatives": new_selection.interpolate_negatives,
        "interpolate_zeros": new_selection.interpolate_zeros,
        "interpolation_methods": new_selection.interpolation_methods,
        "outliers_detection": new_selection.outliers_detection,
        "missing_values_t": new_selection.missing_values_t,
        "pex_variables": new_selection.pex_variables,
        "mapes100": new_selection.mapes100
    })
