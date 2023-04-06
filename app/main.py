from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask import request, jsonify, session
import Files.file_manager as file
from Login.models import db, User, create_app
import Login.login as login

app = create_app()

with app.app_context():
    engine = db.engine
    db.create_all()
bcrypt = Bcrypt(app)
cors = CORS(app, supports_credentials=True)


@app.route("/@me", methods=["GET"])
def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    user = db.session.query(User).filter_by(id=user_id).first()
    return jsonify({
        "id": user.id,
        "email": user.email,
        "name": user.name
    })


@app.route("/register", methods=["POST"])
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
    })


@app.route("/login", methods=["POST"])
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


@app.route("/logout", methods=["POST"])
def logout():
    login.logout_user()


@app.route('/upload', methods=['POST'])
def upload():
    file.upload_file()
    return 'File Uploaded Successfully'


@app.route('/files', methods=['GET'])
def get_files():
    user_id = get_current_user()
    file.get_files(user_id)


if __name__ == "__main__":

    app.run(debug=True)
