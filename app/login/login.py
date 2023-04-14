from flask import Blueprint
from flask import jsonify, session, request
from flask_bcrypt import Bcrypt

from app.database import User, db

login_blueprint = Blueprint('login', __name__)

bcrypt = Bcrypt()


@login_blueprint.route("/@me", methods=["GET"])
def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    user = db.session.query(User).filter_by(id=user_id).first()
    response = jsonify({
        "id": user.id,
        "email": user.email,
        "name": user.name
    })
    response.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin'))
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


@login_blueprint.route("/register", methods=["POST"])
def register_user():
    email = request.json["email"]
    password = request.json["password"]
    name = request.json["name"]
    user_exists = User.query.filter_by(email=email).first() is not None
    if user_exists:
        return jsonify({"error": "User already exists"}), 409
    hashed_password = bcrypt.generate_password_hash(password)
    new_user = User(email=email, password=hashed_password, name=name)
    db.session.add(new_user)
    db.session.commit()

    session["user_id"] = new_user.id

    return jsonify({
        "id": new_user.id,
        "email": new_user.email,
        "name": new_user.name
    }), 200


@login_blueprint.route("/login", methods=["POST"])
def login_user():
    email = request.json["email"]
    password = request.json["password"]
    user = User.query.filter_by(email=email).first()
    if user is None:
        return jsonify({"error": "Unauthorized"}), 401
    if not bcrypt.check_password_hash(user.password, password):
        return jsonify({"error": "Unauthorized"}), 401

    session["user_id"] = user.id

    return jsonify({
        "id": user.id,
        "email": user.email,
        "name": user.name
    })


@login_blueprint.route("/logout", methods=["POST"])
def logout_user():
    session.pop("user_id")
    return "200"

