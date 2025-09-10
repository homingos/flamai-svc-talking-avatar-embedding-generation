import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from ..config.settings import settings


class Logger:
    """
    Centralized logging configuration using settings manager
    """
    _instance: Optional['Logger'] = None
    _logger: Optional[logging.Logger] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'Logger':
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize logger"""
        if not self._initialized:
            self._setup_logger()
            self._initialized = True
    
    def _setup_logger(self) -> None:
        """Setup logger configuration from settings"""
        try:
            # Get logging configuration from settings
            log_config = settings.get_logging_config()
            
            # Create logger
            self._logger = logging.getLogger(settings.get("app.name", "fastapi-app"))
            
            # Set log level
            log_level = log_config.get("level", "info").upper()
            self._logger.setLevel(getattr(logging, log_level, logging.INFO))
            
            # Clear existing handlers
            self._logger.handlers.clear()
            
            # Setup formatters
            file_formatter = logging.Formatter(
                log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            console_formatter = logging.Formatter(
                log_config.get("console_format", "%(levelname)s - %(message)s")
            )
            
            # Setup handlers based on configuration
            handlers = log_config.get("handlers", ["console"])
            
            if "console" in handlers:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(console_formatter)
                self._logger.addHandler(console_handler)
            
            if "file" in handlers:
                file_path = log_config.get("file_path", "logs/app.log")
                
                # Create logs directory if it doesn't exist
                log_dir = Path(file_path).parent
                log_dir.mkdir(parents=True, exist_ok=True)
                
                # Setup rotating file handler
                max_size = self._parse_size(log_config.get("file_max_size", "50MB"))
                backup_count = log_config.get("file_backup_count", 5)
                
                file_handler = logging.handlers.RotatingFileHandler(
                    file_path,
                    maxBytes=max_size,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                file_handler.setFormatter(file_formatter)
                self._logger.addHandler(file_handler)
            
            # Prevent duplicate logs
            self._logger.propagate = False
            
            print(f"✅ Logger initialized with level: {log_level}")
            
        except Exception as e:
            print(f"❌ Failed to setup logger: {e}")
            # Fallback to basic logging
            self._logger = logging.getLogger(settings.get("app.name", "fastapi-app"))
            self._logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
            self._logger.addHandler(handler)
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '50MB' to bytes"""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-3]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance"""
        if not self._initialized:
            self._setup_logger()
        return self._logger
    
    def reload_config(self) -> None:
        """Reload logger configuration from settings"""
        self._initialized = False
        self._setup_logger()
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message"""
        self._logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message"""
        self._logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message"""
        self._logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message"""
        self._logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message"""
        self._logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs) -> None:
        """Log exception with traceback"""
        self._logger.exception(message, *args, **kwargs)


# Create global logger instance
logger = Logger()
