from flask import jsonify, session
from app.Login.models import db, User





def logout_user():
    session.pop("user_id")
    return "200"

