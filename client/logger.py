"""
Centralized logging system for services to log to the Rich console UI.
"""

from typing import Optional, Callable
import threading


class ServiceLogger:
    """Logger that services can use to log to the console UI"""
    
    def __init__(self, service_name: str, log_callback: Optional[Callable[[str, str, str], None]] = None):
        """
        Initialize logger for a service.
        
        Args:
            service_name: Name of the service (e.g., "Audio (TTS)", "Face Detection")
            log_callback: Optional callback function (service, level, message) -> None
        """
        self.service_name = service_name
        self.log_callback = log_callback
        self.lock = threading.Lock()
    
    def _log(self, level: str, message: str):
        """Internal logging method"""
        if self.log_callback:
            try:
                self.log_callback(self.service_name, level, message)
            except Exception:
                # Don't let logging errors break services
                pass
    
    def info(self, message: str):
        """Log an info message"""
        self._log("INFO", message)
    
    def warning(self, message: str):
        """Log a warning message"""
        self._log("WARNING", message)
    
    def error(self, message: str):
        """Log an error message"""
        self._log("ERROR", message)
    
    def debug(self, message: str):
        """Log a debug message"""
        self._log("DEBUG", message)


# Global logger registry - services can get loggers from here
_logger_registry: dict[str, ServiceLogger] = {}
_registry_lock = threading.Lock()


def get_logger(service_name: str, log_callback: Optional[Callable[[str, str, str], None]] = None) -> ServiceLogger:
    """
    Get or create a logger for a service.
    
    Args:
        service_name: Name of the service
        log_callback: Optional callback function for logging
    
    Returns:
        ServiceLogger instance
    """
    with _registry_lock:
        if service_name not in _logger_registry:
            _logger_registry[service_name] = ServiceLogger(service_name, log_callback)
        elif log_callback:
            # Update callback if provided
            _logger_registry[service_name].log_callback = log_callback
        return _logger_registry[service_name]


def set_log_callback(log_callback: Callable[[str, str, str], None]):
    """Set the global log callback for all loggers"""
    with _registry_lock:
        for logger in _logger_registry.values():
            logger.log_callback = log_callback

