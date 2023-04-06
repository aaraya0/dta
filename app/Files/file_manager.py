from flask import request, jsonify
import sys
sys.path.append('C:\\Users\\Usuario\\OneDrive\\Escritorio\\2023\\Forecasting\\dta')
from app.Login.models import FileRef
import pandas as pd
from app.main import get_current_user, db as bd, engine


def upload_file():
    file = request.files['file']
    user = get_current_user().json["id"]
    new_file = FileRef(file_name=file.filename, user_id=user)
    bd.session.add(new_file)
    bd.session.commit()
    save_df(file)
    return 'File uploaded successfully'


def save_df(file):
    user_id = get_current_user().json["id"]
    excel_file = pd.ExcelFile(file)
    dataframe = pd.read_excel(excel_file, sheet_name=0)
    dataframe = dataframe.astype(str)  # Convert all columns to string type
    table_name = file.filename + '_' + user_id
    dataframe.to_sql(table_name, engine, if_exists="replace")


def get_files(user_id):
    files_list = FileRef.query.filter_by(user_id=user_id)
    for file in files_list:
        return jsonify({
            "id": file.id,
            "file_name": file.file_name,
            "upload_date": file.upload_date,
        })

