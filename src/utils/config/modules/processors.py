import os
from typing import Any, Dict
from .interfaces import ConfigProcessor


class EnvironmentVariableProcessor(ConfigProcessor):
    """Process environment variable substitutions in config"""
    
    def process(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Replace ${VAR_NAME} patterns with environment variables"""
        return self._replace_env_variables(config)
    
    def _replace_env_variables(self, obj: Any) -> Any:
        """Recursively replace environment variables"""
        if isinstance(obj, dict):
            return {key: self._replace_env_variables(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_env_variables(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]  # Remove ${ and }
            return os.getenv(env_var, obj)  # Keep original if env var not found
        return obj


class SecretsProcessor(ConfigProcessor):
    """Process API keys and secrets with environment variable precedence"""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.secrets_mapping = {
            "api_keys.openai": "OPENAI_API_KEY",
            "api_keys.anthropic": "ANTHROPIC_API_KEY",
            "api_keys.google": "GOOGLE_API_KEY",
            "api_keys.custom": "CUSTOM_API_KEY",
            "jwt.secret_key": "JWT_SECRET_KEY",
            "jwt.algorithm": "JWT_ALGORITHM",
        }
    
    def process(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process secrets in configuration"""
        for config_key, env_var in self.secrets_mapping.items():
            env_value = os.getenv(env_var)
            
            if env_value:
                # Environment variable takes precedence in development
                if self.environment == "development" or not self._get_nested_value(config, config_key):
                    self._set_nested_value(config, config_key, env_value)
                    print(f"🔑 Using {env_var} from environment variable")
            elif self.environment == "development":
                # In development, warn if required secrets are missing
                if config_key in ["jwt.secret_key"]:
                    print(f"⚠️ Missing environment variable {env_var} for {config_key}")
        
        return config
    
    def _get_nested_value(self, config: Dict[str, Any], key: str) -> Any:
        """Get nested value using dot notation"""
        keys = key.split('.')
        value = config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return None
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any) -> None:
        """Set nested value using dot notation"""
        keys = key.split('.')
        target = config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        # Set the final value
        target[keys[-1]] = value


class ConfigProcessorChain:
    """Chain of configuration processors"""
    
    def __init__(self):
        self.processors: list[ConfigProcessor] = []
    
    def add_processor(self, processor: ConfigProcessor) -> 'ConfigProcessorChain':
        """Add a processor to the chain"""
        self.processors.append(processor)
        return self
    
    def process(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process configuration through all processors"""
        result = config
        for processor in self.processors:
            result = processor.process(result)
        return result

