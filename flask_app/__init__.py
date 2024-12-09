from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'flask_app/static/uploads'
    app.secret_key = "b';\x13\xda\xe81-\xc0K\xfd<R\xe7nb@\xe3\xea\x8f}bX\x1d\x8dO'"

    from flask_app.routes import main
    app.register_blueprint(main)

    return app
