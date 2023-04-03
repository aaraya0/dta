from flask import request
from models import db, FileRef
from ..Login.login import app, get_current_user
from data_preparation import read_data

with app.app_context():
    db.create_all()


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    user = get_current_user().json["id"]
    new_file = FileRef(file_name=file.filename, user_id=user)
    db.session.add(new_file)
    db.session.commit()
    save_df(file)
    return 'File uploaded successfully'


def file_convert(excel_file):
    if excel_file.filename == "Historical Data.xlsx":
        df = read_data.read_historical_data(excel_file)
    elif excel_file.filename == "Launches and Discontinued Data.xlsx":
        df = read_data.read_absolute_launches_data(excel_file)
    elif excel_file.filename == "Allocation Matrix.xlsx":
        df = read_data.read_sku_allocation_matrix(excel_file)
    else:
        df = None
    return df


def save_df(file):
    user_id = get_current_user().json["id"]
    dataframe = file_convert(file)
    if dataframe is not None:
        dataframe.to_sql(file.filename+' '+user_id, con=db, if_exists='replace', chunksize=1000)
    return "File saved to database"

@app.route('/files/<string:user_id>', methods=['GET'])
def get_files (user_id):
