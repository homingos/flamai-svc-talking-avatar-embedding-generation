from typing import Any, Dict
from src.utils.config.modules.providers import BaseConfigProvider


class AppConfigProvider(BaseConfigProvider):
    """Application configuration provider"""
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration"""
        return self.get("app", {})
    
    def get_name(self) -> str:
        """Get application name"""
        return self.get("app.name", "unknown")
    
    def get_version(self) -> str:
        """Get application version"""
        return self.get("app.version", "unknown")
    
    def get_debug(self) -> bool:
        """Get debug mode"""
        return self.get("app.debug", False)
    
    def get_temp_dir(self) -> str:
        """Get temp directory"""
        return self.get("app.temp_dir", "temp")
    
    def get_assets_dir(self) -> str:
        """Get assets directory"""
        return self.get("app.assets_dir", "assets")


class ServerConfigProvider(BaseConfigProvider):
    """Server configuration provider"""
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration"""
        return self.get("server", {})
    
    def get_host(self) -> str:
        """Get server host"""
        return self.get("server.host", "0.0.0.0")
    
    def get_port(self) -> int:
        """Get server port"""
        return self.get("server.port", 8000)
    
    def get_reload(self) -> bool:
        """Get reload setting"""
        return self.get("server.reload", False)


class LoggingConfigProvider(BaseConfigProvider):
    """Logging configuration provider"""
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get("logging", {})
    
    def get_level(self) -> str:
        """Get log level"""
        return self.get("logging.level", "INFO")
    
    def get_file_path(self) -> str:
        """Get log file path"""
        return self.get("logging.file_path", "logs/app.log")


class SecurityConfigProvider(BaseConfigProvider):
    """Security configuration provider"""
    
    def get_jwt_config(self) -> Dict[str, Any]:
        """Get JWT configuration"""
        return self.get("jwt", {})
    
    def get_secret_key(self) -> str:
        """Get JWT secret key"""
        return self.get("jwt.secret_key", "")
    
    def get_algorithm(self) -> str:
        """Get JWT algorithm"""
        return self.get("jwt.algorithm", "HS256")
    
    def get_api_keys(self) -> Dict[str, Any]:
        """Get API keys configuration"""
        return self.get("api_keys", {})


class ServerManagerConfigProvider(BaseConfigProvider):
    """Server manager configuration provider"""
    
    def get_server_manager_config(self) -> Dict[str, Any]:
        """Get server manager configuration"""
        return self.get("server_manager", {})
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get server manager app configuration"""
        return self.get("server_manager.app", {})
    
    def get_directories_config(self) -> Dict[str, Any]:
        """Get server manager directories configuration"""
        return self.get("server_manager.directories", {})
    
    def get_housekeeping_config(self) -> Dict[str, Any]:
        """Get server manager housekeeping configuration"""
        return self.get("server_manager.housekeeping", {})
    
    def get_cleanup_config(self) -> Dict[str, Any]:
        """Get server manager cleanup configuration"""
        return self.get("server_manager.cleanup", {})
    
    def get_services_config(self) -> Dict[str, Any]:
        """Get server manager services configuration"""
        return self.get("server_manager.services", {})
    
    def get_device_config(self) -> Dict[str, Any]:
        """Get server manager device configuration"""
        return self.get("server_manager.device", {})
    
    def get_signal_config(self) -> Dict[str, Any]:
        """Get server manager signal handling configuration"""
        return self.get("server_manager.signals", {})


class ProcessManagerConfigProvider(BaseConfigProvider):
    """Process manager configuration provider (legacy support)"""
    
    def get_process_manager_config(self) -> Dict[str, Any]:
        """Get process manager configuration"""
        return self.get("process_manager", {})
    
    def get_type(self) -> str:
        """Get process manager type (stateful or stateless)"""
        return self.get("process_manager.type", "stateful")
    
    def get_stateful_config(self) -> Dict[str, Any]:
        """Get stateful process manager configuration"""
        return self.get("process_manager.stateful", {})
    
    def get_stateless_config(self) -> Dict[str, Any]:
        """Get stateless process manager configuration"""
        return self.get("process_manager.stateless", {})
    
    def get_cleanup_config(self) -> Dict[str, Any]:
        """Get process manager cleanup configuration"""
        return self.get("process_manager.cleanup", {})
    
    def get_file_tracking_config(self) -> Dict[str, Any]:
        """Get process manager file tracking configuration"""
        return self.get("process_manager.file_tracking", {})
    
    def get_directories_config(self) -> Dict[str, Any]:
        """Get process manager directories configuration"""
        return self.get("process_manager.directories", {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get process manager performance configuration"""
        return self.get("process_manager.performance", {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get process manager monitoring configuration"""
        return self.get("process_manager.monitoring", {})