from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:aaraya0@localhost/dta_files'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def __repr__(self):
        return '<User %r>' % self.username


class ExcelFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(255), nullable=False)
    file_data = db.Column(db.LargeBinary, nullable=False)
    upload_date = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp(), nullable=False)

    def __repr__(self):
        return f"<ExcelFile {self.file_name}>"


if __name__ == '__main__':
    with app.app_context():
        db.create_all()


@app.route('/')
def index():
    return 'Hello'


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    new_file = ExcelFile(file_name=file.filename, file_data=file.read())
    db.session.add(new_file)
    db.session.commit()
    return 'File uploaded successfully'


@app.route('/download/<int:file_id>', methods=['GET'])
def download_file(file_id):
    excel_file = ExcelFile.query.get_or_404(file_id)
    file_name = excel_file.file_name
    file_data = excel_file.file_data
    with open(file_name, 'wb') as f:
        f.write(file_data)
    return f"File {file_name} downloaded successfully to {os.getcwd()}"


app.run()
