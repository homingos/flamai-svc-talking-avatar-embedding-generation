from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import time
import numpy as np


class ProcessingMode(str, Enum):
    """Enum for processing modes"""
    SINGLE = "single"
    BATCH = "batch"
    PARALLEL = "parallel"


class HealthStatus(str, Enum):
    """Enum for health check status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


# Request Models
class SingleEmbeddingRequest(BaseModel):
    """Request model for single text embedding generation"""
    text: str = Field(
        ..., 
        description="The text to generate embedding for",
        min_length=1,
        max_length=10000,
        example="This is a sample text for embedding generation"
    )
    normalize: Optional[bool] = Field(
        default=True,
        description="Whether to normalize the embedding to unit vector"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is a sample text for embedding generation",
                "normalize": True
            }
        }


class BatchEmbeddingRequest(BaseModel):
    """Request model for batch text embedding generation"""
    texts: List[str] = Field(
        ..., 
        description="List of texts to generate embeddings for",
        min_items=1,
        max_items=1000,
        example=[
            "First sample text",
            "Second sample text", 
            "Third sample text"
        ]
    )
    batch_size: Optional[int] = Field(
        default=32,
        description="Batch size for processing",
        ge=1,
        le=128
    )
    normalize: Optional[bool] = Field(
        default=True,
        description="Whether to normalize embeddings to unit vectors"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "First sample text",
                    "Second sample text", 
                    "Third sample text",
                    "Fourth sample text"
                ],
                "batch_size": 16,
                "normalize": True
            }
        }


class ParallelEmbeddingRequest(BaseModel):
    """Request model for parallel text embedding generation"""
    texts: List[str] = Field(
        ..., 
        description="List of texts to generate embeddings for",
        min_items=1,
        max_items=10000,
        example=[f"Sample text number {i}" for i in range(1, 101)]
    )
    num_workers: Optional[int] = Field(
        default=4,
        description="Number of parallel workers",
        ge=1,
        le=16
    )
    batch_size: Optional[int] = Field(
        default=32,
        description="Batch size for each worker",
        ge=1,
        le=128
    )
    normalize: Optional[bool] = Field(
        default=True,
        description="Whether to normalize embeddings to unit vectors"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [f"Sample text number {i}" for i in range(1, 21)],
                "num_workers": 4,
                "batch_size": 16,
                "normalize": True
            }
        }


# Response Models
class SingleEmbeddingResponse(BaseModel):
    """Response model for single text embedding generation"""
    success: bool = Field(..., description="Whether the operation was successful")
    text: str = Field(..., description="The original input text")
    embedding: List[float] = Field(..., description="The generated embedding vector")
    embedding_dimension: int = Field(..., description="Dimension of the embedding vector")
    processing_time: float = Field(..., description="Processing time in seconds")
    normalized: bool = Field(..., description="Whether the embedding was normalized")
    message: Optional[str] = Field(default=None, description="Additional information")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "text": "This is a sample text for embedding generation",
                "embedding": [0.1, -0.2, 0.3, 0.4, -0.5],
                "embedding_dimension": 384,
                "processing_time": 0.045,
                "normalized": True,
                "message": "Embedding generated successfully"
            }
        }


class BatchEmbeddingResponse(BaseModel):
    """Response model for batch text embedding generation"""
    success: bool = Field(..., description="Whether the operation was successful")
    total_texts: int = Field(..., description="Total number of input texts")
    embeddings_generated: int = Field(..., description="Number of embeddings generated")
    embedding_dimension: int = Field(..., description="Dimension of each embedding vector")
    processing_time: float = Field(..., description="Total processing time in seconds")
    batch_size: int = Field(..., description="Batch size used for processing")
    normalized: bool = Field(..., description="Whether embeddings were normalized")
    embeddings: List[List[float]] = Field(..., description="List of generated embedding vectors")
    message: Optional[str] = Field(default=None, description="Additional information")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "total_texts": 4,
                "embeddings_generated": 4,
                "embedding_dimension": 384,
                "processing_time": 0.156,
                "batch_size": 16,
                "normalized": True,
                "embeddings": [
                    [0.1, -0.2, 0.3, 0.4, -0.5],
                    [0.2, -0.1, 0.4, 0.3, -0.6],
                    [0.3, -0.3, 0.2, 0.5, -0.4],
                    [0.4, -0.4, 0.1, 0.6, -0.3]
                ],
                "message": "Batch embeddings generated successfully"
            }
        }


class ParallelEmbeddingResponse(BaseModel):
    """Response model for parallel text embedding generation"""
    success: bool = Field(..., description="Whether the operation was successful")
    total_texts: int = Field(..., description="Total number of input texts")
    embeddings_generated: int = Field(..., description="Number of embeddings generated")
    embedding_dimension: int = Field(..., description="Dimension of each embedding vector")
    processing_time: float = Field(..., description="Total processing time in seconds")
    num_workers: int = Field(..., description="Number of parallel workers used")
    batch_size: int = Field(..., description="Batch size used for each worker")
    normalized: bool = Field(..., description="Whether embeddings were normalized")
    embeddings: List[List[float]] = Field(..., description="List of generated embedding vectors")
    message: Optional[str] = Field(default=None, description="Additional information")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "total_texts": 100,
                "embeddings_generated": 100,
                "embedding_dimension": 384,
                "processing_time": 2.345,
                "num_workers": 4,
                "batch_size": 16,
                "normalized": True,
                "embeddings": [[0.1, -0.2, 0.3], [0.2, -0.1, 0.4]],
                "message": "Parallel embeddings generated successfully"
            }
        }


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    success: bool = Field(..., description="Whether the operation was successful")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    message: Optional[str] = Field(default=None, description="Additional information")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "model_info": {
                    "model_name": "all-MiniLM-L6-v2",
                    "device": "cuda",
                    "batch_size": 32,
                    "max_seq_length": 512,
                    "normalize_embeddings": True,
                    "embedding_dimension": 384,
                    "cuda_available": True,
                    "cuda_device_count": 1
                },
                "message": "Model information retrieved successfully"
            }
        }


class HealthCheckResponse(BaseModel):
    """Response model for health check API"""
    status: HealthStatus = Field(..., description="Overall health status")
    timestamp: str = Field(..., description="Timestamp of the health check")
    service_name: str = Field(..., description="Name of the service")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Service uptime in seconds")
    embedding_service_status: Dict[str, Any] = Field(..., description="Embedding service status")
    system_info: Dict[str, Any] = Field(..., description="System information")
    processing_metrics: Dict[str, Any] = Field(..., description="Processing metrics")
    message: Optional[str] = Field(default=None, description="Additional status information")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "service_name": "Embedding Generation Service",
                "version": "1.0.0",
                "uptime": 3600.5,
                "embedding_service_status": {
                    "model_name": "all-MiniLM-L6-v2",
                    "device": "cuda",
                    "initialized": True,
                    "embedding_dimension": 384
                },
                "system_info": {
                    "python_version": "3.11.0",
                    "torch_version": "2.8.0",
                    "cuda_available": True,
                    "gpu_name": "NVIDIA GeForce RTX 4090",
                    "gpu_memory": 24.0
                },
                "processing_metrics": {
                    "total_requests": 150,
                    "successful_requests": 148,
                    "failed_requests": 2,
                    "average_processing_time": 0.245
                },
                "message": "All systems operational"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(default=False, description="Always false for error responses")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: str = Field(..., description="Timestamp when error occurred")

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "ValidationError",
                "message": "Invalid input parameters",
                "details": {
                    "field": "text",
                    "issue": "Text cannot be empty"
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


# Data Models (for internal use)
@dataclass
class EmbeddingRequest:
    """Data model for embedding request"""
    text: str
    normalize: bool = True
    request_id: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.request_id is None:
            self.request_id = f"req_{int(time.time() * 1000)}"


@dataclass
class BatchEmbeddingRequest:
    """Data model for batch embedding request"""
    texts: List[str]
    normalize: bool = True
    batch_size: int = 32
    request_id: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.request_id is None:
            self.request_id = f"batch_req_{int(time.time() * 1000)}"


@dataclass
class ParallelEmbeddingRequest:
    """Data model for parallel embedding request"""
    texts: List[str]
    normalize: bool = True
    num_workers: int = 4
    batch_size: int = 32
    request_id: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.request_id is None:
            self.request_id = f"parallel_req_{int(time.time() * 1000)}"


@dataclass
class EmbeddingResult:
    """Data model for embedding result"""
    text: str
    embedding: np.ndarray
    processing_time: float
    normalized: bool
    request_id: str
    timestamp: datetime


@dataclass
class BatchEmbeddingResult:
    """Data model for batch embedding result"""
    texts: List[str]
    embeddings: List[np.ndarray]
    processing_time: float
    batch_size: int
    normalized: bool
    request_id: str
    timestamp: datetime


@dataclass
class ParallelEmbeddingResult:
    """Data model for parallel embedding result"""
    texts: List[str]
    embeddings: List[np.ndarray]
    processing_time: float
    num_workers: int
    batch_size: int
    normalized: bool
    request_id: str
    timestamp: datetime


@dataclass
class ServiceStatus:
    """Data model for service status"""
    name: str
    initialized: bool
    error: Optional[str] = None


@dataclass
class SystemInfo:
    """Data model for system information"""
    python_version: str
    torch_version: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    gpu_memory: Optional[float] = None
    cpu_count: Optional[int] = None
    memory_total: Optional[float] = None


@dataclass
class ProcessingMetrics:
    """Data model for processing metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_processing_time: float = 0.0
    last_request_time: Optional[datetime] = None
    start_time: datetime = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()


@dataclass
class ErrorInfo:
    """Data model for error information"""
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()