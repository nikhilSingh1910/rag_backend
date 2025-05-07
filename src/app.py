from quart import Quart, jsonify, request
from quart_jwt_extended import JWTManager
from config.settings import settings
from utils.logging_config import setup_logging
from api.auth_routes import auth_bp
from api.document_routes import document_bp
from api.qa_routes import qa_bp
from api.health_routes import health_bp

# Setup logging
loggers = setup_logging()
logger = loggers['api']

# Initialize Quart app
app = Quart(__name__)

# JWT configuration
app.config['JWT_SECRET_KEY'] = settings.jwt_secret_key
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = settings.jwt_access_token_expires
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = settings.jwt_refresh_token_expires
jwt = JWTManager(app)

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(document_bp, url_prefix='/api/documents')
app.register_blueprint(qa_bp, url_prefix='/api/qa')
app.register_blueprint(health_bp, url_prefix='/api')

# Error handlers
@app.errorhandler(404)
async def not_found_error(error):
    logger.warning(f"404 error: {error}")
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
async def internal_error(error):
    logger.error(f"500 error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Middleware for logging
@app.before_request
async def log_request_info():
    logger.info(f"Request: {request.method} {request.url}")

@app.after_request
async def log_response_info(response):
    logger.info(f"Response: {response.status}")
    return response

if __name__ == '__main__':
    debug = settings.flask_debug
    logger.info(f"Starting application in {'debug' if debug else 'production'} mode")
    app.run(debug=debug) 