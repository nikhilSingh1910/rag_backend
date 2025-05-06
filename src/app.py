from flask import Flask
from flask_async import AsyncFlask
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Flask app with async support
app = AsyncFlask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-jwt-secret-key')

# Initialize JWT
jwt = JWTManager(app)

# Import and register blueprints
from api.auth_routes import auth_bp
from api.document_routes import document_bp
from api.qa_routes import qa_bp

app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(document_bp, url_prefix='/api/documents')
app.register_blueprint(qa_bp, url_prefix='/api/qa')

if __name__ == '__main__':
    app.run(debug=True) 