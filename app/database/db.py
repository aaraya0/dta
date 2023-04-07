from flask_sqlalchemy import SQLAlchemy
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ECHO = True
SQLALCHEMY_DATABASE_URI = 'mysql://root:aaraya0@localhost/dta_files'
db = SQLAlchemy()

# definicion de la base de datos, configuracion de motor y schema
