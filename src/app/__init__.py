 # Inicializa a aplicação Flask
 
from flask import Flask

def create_app():
    app = Flask(__name__)
    from .routes.routes import bp
    app.register_blueprint(bp)
    return app
