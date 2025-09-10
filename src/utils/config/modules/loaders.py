import yaml
import json
import os
from typing import Any, Dict
from pathlib import Path
from .interfaces import ConfigLoader


class YAMLConfigLoader(ConfigLoader):
    """YAML configuration loader"""
    
    def load(self, path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        if not config:
            raise ValueError(f"Empty or invalid configuration file: {path}")
        
        return config
    
    def supports_format(self, path: str) -> bool:
        """Check if file is YAML format"""
        return path.lower().endswith(('.yaml', '.yml'))


class JSONConfigLoader(ConfigLoader):
    """JSON configuration loader"""
    
    def load(self, path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        
        if not config:
            raise ValueError(f"Empty or invalid configuration file: {path}")
        
        return config
    
    def supports_format(self, path: str) -> bool:
        """Check if file is JSON format"""
        return path.lower().endswith('.json')


class ConfigLoaderFactory:
    """Factory for creating appropriate config loaders"""
    
    _loaders = [
        YAMLConfigLoader(),
        JSONConfigLoader(),
    ]
    
    @classmethod
    def get_loader(cls, path: str) -> ConfigLoader:
        """Get appropriate loader for the given path"""
        for loader in cls._loaders:
            if loader.supports_format(path):
                return loader
        
        raise ValueError(f"No loader found for file: {path}")
