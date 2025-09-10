import os
from typing import Any, Dict, List
from .interfaces import ConfigValidator


class BaseConfigValidator(ConfigValidator):
    """Base configuration validator"""
    
    def __init__(self):
        self.errors: List[str] = []
    
    def validate(self, config: Dict[str, Any]) -> bool:
        """Validate configuration"""
        self.errors.clear()
        return self._validate_impl(config)
    
    def _validate_impl(self, config: Dict[str, Any]) -> bool:
        """Implementation of validation logic"""
        raise NotImplementedError
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors"""
        return self.errors.copy()
    
    def _add_error(self, error: str) -> None:
        """Add validation error"""
        self.errors.append(error)


class DirectoryValidator(BaseConfigValidator):
    """Validates and creates required directories"""
    
    def _validate_impl(self, config: Dict[str, Any]) -> bool:
        """Validate and create directories"""
        try:
            # Check required directories
            temp_dir = self._get_nested_value(config, "app.temp_dir", "temp")
            assets_dir = self._get_nested_value(config, "app.assets_dir", "assets")
            
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(assets_dir, exist_ok=True)
            
            # Check logs directory
            log_path = self._get_nested_value(config, "logging.file_path", "logs/app.log")
            logs_dir = os.path.dirname(log_path)
            if logs_dir:
                os.makedirs(logs_dir, exist_ok=True)
            
            # Validate server manager directories
            server_manager_dirs = self._get_nested_value(config, "server_manager.directories", {})
            for dir_key, dir_path in server_manager_dirs.items():
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"✅ Created server manager directory: {dir_path}")
            
            # Validate process manager directories (legacy support)
            pm_dirs = self._get_nested_value(config, "process_manager.directories", {})
            for dir_key, dir_path in pm_dirs.items():
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"✅ Created process manager directory: {dir_path}")
            
            return True
            
        except Exception as e:
            self._add_error(f"Directory validation failed: {e}")
            return False
    
    def _get_nested_value(self, config: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get nested value using dot notation"""
        keys = key.split('.')
        value = config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default


class ProductionValidator(BaseConfigValidator):
    """Validates production-specific requirements"""
    
    def _validate_impl(self, config: Dict[str, Any]) -> bool:
        """Validate production requirements"""
        try:
            # Check JWT secret key
            jwt_secret = self._get_nested_value(config, "jwt.secret_key")
            if not jwt_secret or jwt_secret == "your-secret-key-here":
                self._add_error("JWT secret key not configured for production")
                return False
            
            return True
            
        except Exception as e:
            self._add_error(f"Production validation failed: {e}")
            return False
    
    def _get_nested_value(self, config: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get nested value using dot notation"""
        keys = key.split('.')
        value = config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default


class ConfigValidatorChain:
    """Chain of configuration validators"""
    
    def __init__(self):
        self.validators: List[ConfigValidator] = []
    
    def add_validator(self, validator: ConfigValidator) -> 'ConfigValidatorChain':
        """Add a validator to the chain"""
        self.validators.append(validator)
        return self
    
    def validate(self, config: Dict[str, Any]) -> bool:
        """Validate configuration through all validators"""
        all_valid = True
        all_errors = []
        
        for validator in self.validators:
            if not validator.validate(config):
                all_valid = False
                all_errors.extend(validator.get_validation_errors())
        
        if not all_valid:
            print(f"❌ Configuration validation failed: {'; '.join(all_errors)}")
        else:
            print("✅ Configuration validation passed")
        
        return all_valid

