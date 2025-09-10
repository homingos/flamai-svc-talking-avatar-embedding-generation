from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pathlib import Path
import yaml
import json
import os


class ConfigProvider(ABC):
    """Abstract base class for configuration providers"""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation"""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if configuration is loaded"""
        pass


class ConfigValidator(ABC):
    """Abstract base class for configuration validators"""
    
    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> bool:
        """Validate configuration"""
        pass
    
    @abstractmethod
    def get_validation_errors(self) -> list[str]:
        """Get list of validation errors"""
        pass


class ConfigLoader(ABC):
    """Abstract base class for configuration loaders"""
    
    @abstractmethod
    def load(self, path: str) -> Dict[str, Any]:
        """Load configuration from source"""
        pass
    
    @abstractmethod
    def supports_format(self, path: str) -> bool:
        """Check if loader supports the given file format"""
        pass


class ConfigProcessor(ABC):
    """Abstract base class for configuration processors"""
    
    @abstractmethod
    def process(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process configuration data"""
        pass
