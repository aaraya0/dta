from ..Login.models import db


class FileRef(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(32))
    file_name = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp(), nullable=False)


