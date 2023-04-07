from flask import Flask
from flask_cors import CORS
from flask_session import Session
import sys
sys.path.append('C:\\Users\\Usuario\\OneDrive\\Escritorio\\2023\\Forecasting\\dta')
from app.excel import excel_blueprint
from app.login import login_blueprint, bcrypt
from app.login.config import ApplicationConfig
from app.database import db

# creacion de app, configuracion de sesion, importa rutas de excel y login, inicializa db
app = Flask(__name__)
app.config.from_object(ApplicationConfig)
server_session = Session(app)
CORS(app, supports_credentials=True)
bcrypt.init_app(app)
app.register_blueprint(login_blueprint)
app.register_blueprint(excel_blueprint)
db.init_app(app)

# crea tablas de la bd

with app.app_context():
    engine = db.engine
    db.create_all()

# inicializa app
if __name__ == "__main__":

    app.run(debug=True)
