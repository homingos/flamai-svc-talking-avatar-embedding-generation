from typing import Any, Dict, Optional
from src.utils.config.modules.providers import BaseConfigProvider


class EmbeddingGenerationConfigProvider(BaseConfigProvider):
    """Configuration provider for embedding generation settings"""
    
    def get_embedding_generation_config(self) -> Dict[str, Any]:
        """Get complete embedding generation configuration"""
        return self.get("embedding_generation", {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get embedding model configuration"""
        return self.get("embedding_generation.model", {})
    
    def get_device_config(self) -> Dict[str, Any]:
        """Get embedding device configuration"""
        return self.get("embedding_generation.device", {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get embedding processing configuration"""
        return self.get("embedding_generation.processing", {})
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """Get embedding optimization configuration"""
        return self.get("embedding_generation.optimization", {})
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get embedding validation configuration"""
        return self.get("embedding_generation.validation", {})
    
    def get_timeouts_config(self) -> Dict[str, Any]:
        """Get embedding timeouts configuration"""
        return self.get("embedding_generation.timeouts", {})
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Get embedding memory configuration"""
        return self.get("embedding_generation.memory", {})
