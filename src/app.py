from flask import Flask
from flask_async import AsyncFlask
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv
from utils.logging_config import setup_logging
import os
import logging

# Load environment variables
load_dotenv()

# Setup logging
loggers = setup_logging()
logger = logging.getLogger('api')

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
from api.health_routes import health_bp

app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(document_bp, url_prefix='/api/documents')
app.register_blueprint(qa_bp, url_prefix='/api/qa')
app.register_blueprint(health_bp, url_prefix='/api')

# Error handlers
@app.errorhandler(404)
async def not_found_error(error):
    logger.warning(f"404 error: {str(error)}")
    return {'error': 'Not found'}, 404

@app.errorhandler(500)
async def internal_error(error):
    logger.error(f"500 error: {str(error)}")
    return {'error': 'Internal server error'}, 500

@app.before_request
async def log_request_info():
    logger.info(f"Request: {request.method} {request.url}")

@app.after_request
async def log_response_info(response):
    logger.info(f"Response: {response.status}")
    return response

if __name__ == '__main__':
    logger.info("Starting RAG application")
    app.run(debug=True) 