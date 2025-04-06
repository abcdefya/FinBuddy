import logging
import os
from datetime import datetime
from pathlib import Path
import sys


def setup_logger(
    log_dir="logs",
    log_level=logging.INFO,
    console_output=True,
    log_format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
):
    """
    Configure application logging with both file and optional console output.
    
    Args:
        log_dir (str): Directory where logs will be stored
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG)
        console_output (bool): Whether to also output logs to console
        log_format (str): Format string for log messages
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create timestamp for unique log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"finbuddy_{timestamp}.log"
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True, parents=True)
    
    # Full path to log file
    log_file_path = log_path / log_filename
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates on reloads
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file_path}")
    return logger


# Create a default logger on module import
default_logger = setup_logger()

if __name__ == "__main__":
    # Test the logger by logging messages at different levels
    logger = default_logger
    
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
