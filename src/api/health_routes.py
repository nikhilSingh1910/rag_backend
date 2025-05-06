from quart import Blueprint, jsonify
from models.database import engine
from sqlalchemy import text
import psutil
import os
from datetime import datetime
import logging

health_bp = Blueprint('health', __name__)
logger = logging.getLogger('api')

@health_bp.route('/health', methods=['GET'])
async def health_check():
    """Basic health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@health_bp.route('/health/detailed', methods=['GET'])
async def detailed_health_check():
    """Detailed health check with system metrics"""
    try:
        # Check database connection
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        db_status = "unhealthy"

    # Get system metrics
    system_metrics = {
        'cpu_usage': psutil.cpu_percent(),
        'memory_usage': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent
    }

    # Get process metrics
    process = psutil.Process(os.getpid())
    process_metrics = {
        'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
        'cpu_percent': process.cpu_percent(),
        'threads': process.num_threads()
    }

    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'components': {
            'database': db_status,
            'system': system_metrics,
            'process': process_metrics
        }
    }), 200 if db_status == "healthy" else 500

@health_bp.route('/health/ready', methods=['GET'])
async def readiness_check():
    """Readiness check for Kubernetes"""
    try:
        # Check database connection
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            return jsonify({
                'status': 'ready',
                'timestamp': datetime.utcnow().isoformat()
            }), 200
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return jsonify({
            'status': 'not_ready',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 503 