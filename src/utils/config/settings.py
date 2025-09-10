import os
from typing import Any, Dict, Optional
from pathlib import Path

from src.utils.config.modules.loaders import ConfigLoaderFactory
from src.utils.config.modules.processors import ConfigProcessorChain, EnvironmentVariableProcessor, SecretsProcessor
from src.utils.config.modules.validators import ConfigValidatorChain, DirectoryValidator, ProductionValidator
from src.utils.config.modules.providers import BaseConfigProvider
from src.utils.config.modules.embedding_generation import EmbeddingGenerationConfigProvider
from src.utils.config.modules.section_providers import *

class SettingsManager:
    """
    Refactored settings manager with separation of concerns.
    Uses composition and dependency injection for better maintainability.
    """
    _instance: Optional['SettingsManager'] = None
    _config_path: Optional[str] = None
    _environment: str = "production"
    
    def __new__(cls) -> 'SettingsManager':
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize settings manager"""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._config_provider: Optional[BaseConfigProvider] = None
            self._section_providers: Dict[str, BaseConfigProvider] = {}
            
            try:
                # Determine environment from env var or default to production
                self._environment = os.getenv("ENVIRONMENT", "production").lower()
                self._setup_providers()
                self.load_config()
            except Exception as e:
                print(f"⚠️ Failed to load configuration during initialization: {e}")
    
    def _setup_providers(self) -> None:
        """Setup configuration providers"""
        # Main config provider
        self._config_provider = BaseConfigProvider()
        
        # Section-specific providers
        self._section_providers = {
            'app': AppConfigProvider(),
            'server': ServerConfigProvider(),
            'logging': LoggingConfigProvider(),
            'security': SecurityConfigProvider(),
            'server_manager': ServerManagerConfigProvider(),
            'process_manager': ProcessManagerConfigProvider(),
            'embedding_generation': EmbeddingGenerationConfigProvider(),
        }
    
    def load_config(self, config_path: Optional[str] = None) -> None:
        """
        Load configuration from file
        
        Args:
            config_path: Path to the configuration file. If None, uses default path.
        """
        if config_path is None:
            # Default config path relative to this file
            current_dir = Path(__file__).parent
            config_path = current_dir / "config.yaml"
        
        self._config_path = str(config_path)
        
        try:
            # Load configuration using appropriate loader
            loader = ConfigLoaderFactory.get_loader(str(config_path))
            config = loader.load(str(config_path))
            
            # Process configuration through processor chain
            processor_chain = ConfigProcessorChain()
            processor_chain.add_processor(EnvironmentVariableProcessor())
            processor_chain.add_processor(SecretsProcessor(self._environment))
            
            processed_config = processor_chain.process(config)
            
            # Update all providers with the new configuration
            self._config_provider.update_config(processed_config)
            for provider in self._section_providers.values():
                provider.update_config(processed_config)
            
            # Validate configuration
            self._validate_config(processed_config)
            
            print(f"✅ Configuration loaded from {config_path}")
            
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            raise
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration using validator chain"""
        validator_chain = ConfigValidatorChain()
        validator_chain.add_validator(DirectoryValidator())
        
        # Add production-specific validations
        if self._environment == "production":
            validator_chain.add_validator(ProductionValidator())
        
        return validator_chain.validate(config)
    
    # Core configuration methods
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        return self._config_provider.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation"""
        self._config_provider.set(key, value)
        # Update all section providers
        for provider in self._section_providers.values():
            provider.set(key, value)
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary"""
        return self._config_provider.get_all_config()
    
    def is_loaded(self) -> bool:
        """Check if configuration is loaded"""
        return self._config_provider.is_loaded()
    
    # Environment methods
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self._environment == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self._environment == "production"
    
    def get_environment(self) -> str:
        """Get current environment"""
        return self._environment
    
    # Section-specific getters (delegated to section providers)
    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration"""
        return self._section_providers['app'].get_app_config()
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration"""
        return self._section_providers['server'].get_server_config()
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self._section_providers['logging'].get_logging_config()
    
    def get_jwt_config(self) -> Dict[str, Any]:
        """Get JWT configuration"""
        return self._section_providers['security'].get_jwt_config()
    
    def get_api_keys(self) -> Dict[str, Any]:
        """Get API keys configuration"""
        return self._section_providers['security'].get_api_keys()
    
    def get_server_manager_config(self) -> Dict[str, Any]:
        """Get server manager configuration"""
        return self._section_providers['server_manager'].get_server_manager_config()
    
    def get_process_manager_config(self) -> Dict[str, Any]:
        """Get process manager configuration (legacy)"""
        return self._section_providers['process_manager'].get_process_manager_config()
    
    # Embedding Generation specific methods
    def get_embedding_generation_config(self) -> Dict[str, Any]:
        """Get embedding generation configuration"""
        return self._section_providers['embedding_generation'].get_embedding_generation_config()
    
    def get_embedding_model_config(self) -> Dict[str, Any]:
        """Get embedding model configuration"""
        return self._section_providers['embedding_generation'].get_model_config()
    
    def get_embedding_device_config(self) -> Dict[str, Any]:
        """Get embedding device configuration"""
        return self._section_providers['embedding_generation'].get_device_config()
    
    def get_embedding_processing_config(self) -> Dict[str, Any]:
        """Get embedding processing configuration"""
        return self._section_providers['embedding_generation'].get_processing_config()
    
    def get_embedding_optimization_config(self) -> Dict[str, Any]:
        """Get embedding optimization configuration"""
        return self._section_providers['embedding_generation'].get_optimization_config()
    
    def get_embedding_validation_config(self) -> Dict[str, Any]:
        """Get embedding validation configuration"""
        return self._section_providers['embedding_generation'].get_validation_config()
    
    def get_embedding_timeouts_config(self) -> Dict[str, Any]:
        """Get embedding timeouts configuration"""
        return self._section_providers['embedding_generation'].get_timeouts_config()
    
    def get_embedding_memory_config(self) -> Dict[str, Any]:
        """Get embedding memory configuration"""
        return self._section_providers['embedding_generation'].get_memory_config()
    
    # Convenience methods for embedding generation
    def get_embedding_model_name(self) -> str:
        """Get embedding model name"""
        return self.get("embedding_generation.model.name", "all-MiniLM-L6-v2")
    
    def get_embedding_device(self) -> str:
        """Get embedding device preference"""
        return self.get("embedding_generation.device.preferred_device", "auto")
    
    def get_embedding_batch_size(self) -> int:
        """Get default embedding batch size"""
        return self.get("embedding_generation.processing.default_batch_size", 32)
    
    def get_embedding_max_seq_length(self) -> int:
        """Get embedding max sequence length"""
        return self.get("embedding_generation.model.max_seq_length", 512)
    
    def get_embedding_normalize(self) -> bool:
        """Get embedding normalization setting"""
        return self.get("embedding_generation.model.normalize_embeddings", True)
    
    def get_embedding_max_text_length(self) -> int:
        """Get maximum text length for embeddings"""
        return self.get("embedding_generation.validation.max_text_length", 10000)
    
    def get_embedding_max_batch_size(self) -> int:
        """Get maximum batch size for embeddings"""
        return self.get("embedding_generation.validation.max_batch_size", 128)
    
    def get_embedding_max_parallel_workers(self) -> int:
        """Get maximum parallel workers for embeddings"""
        return self.get("embedding_generation.validation.max_parallel_workers", 16)
    
    def get_embedding_max_texts_per_parallel(self) -> int:
        """Get maximum texts per parallel request"""
        return self.get("embedding_generation.validation.max_texts_per_parallel", 10000)
    
    def get_embedding_max_texts_per_batch(self) -> int:
        """Get maximum texts per batch request"""
        return self.get("embedding_generation.validation.max_texts_per_batch", 1000)
    
    def get_embedding_enable_half_precision(self) -> bool:
        """Get half precision setting"""
        return self.get("embedding_generation.optimization.enable_half_precision", True)
    
    def get_embedding_enable_cudnn_benchmark(self) -> bool:
        """Get cuDNN benchmark setting"""
        return self.get("embedding_generation.optimization.enable_cudnn_benchmark", True)
    
    def get_embedding_enable_mixed_precision(self) -> bool:
        """Get mixed precision setting"""
        return self.get("embedding_generation.optimization.enable_mixed_precision", True)
    
    def get_embedding_lazy_load_optional(self) -> bool:
        """Get lazy load optional setting"""
        return self.get("embedding_generation.optimization.lazy_load_optional", True)
    
    def get_embedding_model_initialization_timeout(self) -> float:
        """Get model initialization timeout"""
        return self.get("embedding_generation.timeouts.model_initialization", 120.0)
    
    def get_embedding_single_timeout(self) -> float:
        """Get single embedding timeout"""
        return self.get("embedding_generation.timeouts.single_embedding", 30.0)
    
    def get_embedding_batch_timeout(self) -> float:
        """Get batch embedding timeout"""
        return self.get("embedding_generation.timeouts.batch_embedding", 300.0)
    
    def get_embedding_parallel_timeout(self) -> float:
        """Get parallel embedding timeout"""
        return self.get("embedding_generation.timeouts.parallel_embedding", 600.0)
    
    def get_embedding_health_check_timeout(self) -> float:
        """Get health check timeout"""
        return self.get("embedding_generation.timeouts.health_check", 10.0)
    
    def get_embedding_enable_gpu_cache_clearing(self) -> bool:
        """Get GPU cache clearing setting"""
        return self.get("embedding_generation.memory.enable_gpu_cache_clearing", True)
    
    def get_embedding_cache_clear_interval(self) -> int:
        """Get cache clear interval"""
        return self.get("embedding_generation.memory.cache_clear_interval", 300)
    
    def get_embedding_max_memory_usage(self) -> float:
        """Get maximum memory usage fraction"""
        return self.get("embedding_generation.memory.max_memory_usage", 0.9)
    
    def get_embedding_enable_garbage_collection(self) -> bool:
        """Get garbage collection setting"""
        return self.get("embedding_generation.memory.enable_garbage_collection", True)
    
    # Convenience methods for common configurations
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration"""
        return self.get("cors", {})
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration"""
        return self.get("storage", {})
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration"""
        return self.get("rate_limit", {})
    
    # Server Manager specific methods
    def get_server_manager_app_config(self) -> Dict[str, Any]:
        """Get server manager app configuration"""
        return self._section_providers['server_manager'].get_app_config()
    
    def get_server_manager_directories_config(self) -> Dict[str, Any]:
        """Get server manager directories configuration"""
        return self._section_providers['server_manager'].get_directories_config()
    
    def get_server_manager_housekeeping_config(self) -> Dict[str, Any]:
        """Get server manager housekeeping configuration"""
        return self._section_providers['server_manager'].get_housekeeping_config()
    
    def get_server_manager_cleanup_config(self) -> Dict[str, Any]:
        """Get server manager cleanup configuration"""
        return self._section_providers['server_manager'].get_cleanup_config()
    
    def get_server_manager_services_config(self) -> Dict[str, Any]:
        """Get server manager services configuration"""
        return self._section_providers['server_manager'].get_services_config()
    
    def get_server_manager_device_config(self) -> Dict[str, Any]:
        """Get server manager device configuration"""
        return self._section_providers['server_manager'].get_device_config()
    
    def get_server_manager_signal_config(self) -> Dict[str, Any]:
        """Get server manager signal handling configuration"""
        return self._section_providers['server_manager'].get_signal_config()
    
    # Process Manager specific methods (legacy support)
    def get_process_manager_type(self) -> str:
        """Get process manager type (stateful or stateless)"""
        return self._section_providers['process_manager'].get_type()
    
    def get_process_manager_stateful_config(self) -> Dict[str, Any]:
        """Get stateful process manager configuration"""
        return self._section_providers['process_manager'].get_stateful_config()
    
    def get_process_manager_stateless_config(self) -> Dict[str, Any]:
        """Get stateless process manager configuration"""
        return self._section_providers['process_manager'].get_stateless_config()
    
    def get_process_manager_cleanup_config(self) -> Dict[str, Any]:
        """Get process manager cleanup configuration"""
        return self._section_providers['process_manager'].get_cleanup_config()
    
    def get_process_manager_file_tracking_config(self) -> Dict[str, Any]:
        """Get process manager file tracking configuration"""
        return self._section_providers['process_manager'].get_file_tracking_config()
    
    def get_process_manager_directories_config(self) -> Dict[str, Any]:
        """Get process manager directories configuration"""
        return self._section_providers['process_manager'].get_directories_config()
    
    def get_process_manager_performance_config(self) -> Dict[str, Any]:
        """Get process manager performance configuration"""
        return self._section_providers['process_manager'].get_performance_config()
    
    def get_process_manager_monitoring_config(self) -> Dict[str, Any]:
        """Get process manager monitoring configuration"""
        return self._section_providers['process_manager'].get_monitoring_config()
    
    # Utility methods
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        if not self.is_loaded():
            print("❌ Configuration not loaded")
            return False
        
        return self._validate_config(self.get_all_config())
    
    def reload_config(self, config_path: Optional[str] = None) -> None:
        """Reload configuration from file"""
        try:
            self.load_config(config_path)
        except Exception as e:
            print(f"❌ Failed to reload configuration: {e}")
    
    def print_config_summary(self) -> None:
        """Print a summary of the current configuration for debugging"""
        if not self.is_loaded():
            print("\n❌ Configuration Summary: Not loaded")
            return
        
        print(f"\n📋 Configuration Summary (Environment: {self._environment.upper()}):")
        print(f"   App: {self.get('app.name', 'unknown')} v{self.get('app.version', 'unknown')}")
        print(f"   Server: {self.get('server.host', '0.0.0.0')}:{self.get('server.port', 8000)}")
        print(f"   Debug: {self.get('app.debug', False)}")
        
        # Embedding Generation info
        model_name = self.get_embedding_model_name()
        device = self.get_embedding_device()
        batch_size = self.get_embedding_batch_size()
        max_seq_length = self.get_embedding_max_seq_length()
        normalize = self.get_embedding_normalize()
        print(f"   Embedding Model: {model_name}")
        print(f"   Device: {device}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Max Seq Length: {max_seq_length}")
        print(f"   Normalize: {normalize}")
        
        # Server Manager info
        server_manager_config = self.get_server_manager_config()
        if server_manager_config:
            app_name = server_manager_config.get('app', {}).get('name', 'unknown')
            housekeeping_interval = server_manager_config.get('housekeeping', {}).get('interval', 'N/A')
            print(f"   Server Manager: {app_name}")
            print(f"   Housekeeping Interval: {housekeeping_interval}s")
        
        # Process Manager info (legacy)
        pm_type = self.get_process_manager_type()
        pm_config = self.get_process_manager_config()
        print(f"   Process Manager: {pm_type}")
        if pm_config:
            cleanup_enabled = pm_config.get('cleanup', {}).get('enabled', True)
            max_processes = pm_config.get('stateful', {}).get('max_processes', 'N/A')
            print(f"   PM Cleanup: {'enabled' if cleanup_enabled else 'disabled'}")
            if pm_type == 'stateful':
                print(f"   PM Max Processes: {max_processes}")
        
        # API Keys info (without exposing values)
        api_keys = self.get_api_keys()
        if api_keys:
            configured_keys = [key for key, value in api_keys.items() if value and value != f"${{{key.upper()}}}" and not value.startswith("${")]
            print(f"   API Keys: {len(configured_keys)} configured")
        
        # Directories
        print(f"   Temp Dir: {self.get('app.temp_dir', 'temp')}")
        print(f"   Assets Dir: {self.get('app.assets_dir', 'assets')}")
        print(f"   Log Level: {self.get('logging.level', 'INFO')}")
        print()
    
    def __str__(self) -> str:
        """String representation of the settings manager"""
        return f"SettingsManager(env={self._environment}, loaded={self.is_loaded()})"


# Create global instance
settings = SettingsManager()
