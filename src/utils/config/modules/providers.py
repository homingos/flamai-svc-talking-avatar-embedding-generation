from typing import Any, Dict, Optional
from .interfaces import ConfigProvider


class BaseConfigProvider(ConfigProvider):
    """Base configuration provider implementation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._loaded = config is not None
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        if not self._loaded or not self._config:
            print(f"⚠️ Configuration not loaded, returning default for key: {key}")
            return default
        
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation"""
        if not self._loaded or not self._config:
            print(f"⚠️ Configuration not loaded, cannot set key: {key}")
            return
        
        keys = key.split('.')
        target = self._config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        # Set the final value
        target[keys[-1]] = value
    
    def is_loaded(self) -> bool:
        """Check if configuration is loaded"""
        return self._loaded
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary"""
        if not self._loaded or not self._config:
            print("⚠️ Configuration not loaded")
            return {}
        return self._config.copy()
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update the entire configuration"""
        self._config = config
        self._loaded = True