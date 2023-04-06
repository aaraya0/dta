from flask_sqlalchemy import SQLAlchemy
from uuid import uuid4
from app.Login.config import ApplicationConfig
from flask import Flask
from flask_session import Session
db = SQLAlchemy()


def get_uuid():
    return uuid4().hex


class User(db.Model):
    id = db.Column(db.String(32), primary_key=True, unique=True, default=get_uuid())
    email = db.Column(db.String(345), unique=True)
    password = db.Column(db.Text, nullable=False)
    name = db.Column(db.String(80), unique=True, nullable=False)
class FileRef(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(32))
    file_name = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp(), nullable=False)


def create_app():
    app = Flask(__name__)
    app.config.from_object(ApplicationConfig)
    server_session = Session(app)
    db.init_app(app)


    return app
