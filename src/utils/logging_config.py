import logging
import logging.handlers
import os
from datetime import datetime

def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # File handler for all logs
    all_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, 'app.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    all_handler.setFormatter(file_formatter)
    all_handler.setLevel(logging.INFO)

    # File handler for errors
    error_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, 'error.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setFormatter(file_formatter)
    error_handler.setLevel(logging.ERROR)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    # Add handlers to root logger
    root_logger.addHandler(all_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    # Create loggers for different components
    loggers = {
        'api': logging.getLogger('api'),
        'rag': logging.getLogger('rag'),
        'db': logging.getLogger('db'),
        'auth': logging.getLogger('auth')
    }

    # Set levels for component loggers
    for logger in loggers.values():
        logger.setLevel(logging.INFO)

    return loggers 