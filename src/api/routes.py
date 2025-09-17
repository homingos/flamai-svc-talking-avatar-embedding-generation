from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import time
from datetime import datetime

from src.api.handlers import (
    embedding_handler, 
    health_check_handler
)
from src.api.schema import (
    SingleEmbeddingRequest, SingleEmbeddingResponse,
    BatchEmbeddingRequest, BatchEmbeddingResponse,
    ParallelEmbeddingRequest, ParallelEmbeddingResponse,
    CSVEmbeddingRequest, CSVEmbeddingResponse,
    ModelInfoResponse, HealthCheckResponse, ErrorResponse
)
from src.utils.resources.logger import logger

# Create router
router = APIRouter(
    prefix="/api/v1",
    tags=["Embedding Generation"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)


@router.post(
    "/embedding/single",
    response_model=SingleEmbeddingResponse,
    summary="Generate Single Embedding",
    description="Generate embedding for a single text string",
    responses={
        200: {"description": "Embedding generated successfully"},
        400: {"description": "Invalid request parameters"},
        503: {"description": "Service not available"},
        500: {"description": "Internal server error"}
    }
)
async def generate_single_embedding(
    request_data: SingleEmbeddingRequest,
    request: Request
) -> SingleEmbeddingResponse:
    """
    Generate embedding for a single text string.
    
    This endpoint is optimized for processing individual text inputs
    and provides the most efficient path for single embedding generation.
    
    Args:
        request_data: SingleEmbeddingRequest containing:
            - text: The text to generate embedding for
            - normalize: Whether to normalize the embedding (default: True)
        request: FastAPI request object
    
    Returns:
        SingleEmbeddingResponse: Single embedding result with metadata
        
    Raises:
        HTTPException: For various error conditions
    """
    try:
        logger.info(f"Single embedding request received: {request_data.text[:50]}...")
        return await embedding_handler.generate_single_embedding(request_data, request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in single embedding: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Single embedding failed: {str(e)}"
        )


@router.post(
    "/embedding/batch",
    response_model=BatchEmbeddingResponse,
    summary="Generate Batch Embeddings",
    description="Generate embeddings for multiple text strings using batch processing",
    responses={
        200: {"description": "Batch embeddings generated successfully"},
        400: {"description": "Invalid request parameters"},
        503: {"description": "Service not available"},
        500: {"description": "Internal server error"}
    }
)
async def generate_batch_embeddings(
    request_data: BatchEmbeddingRequest,
    request: Request
) -> BatchEmbeddingResponse:
    """
    Generate embeddings for multiple text strings using optimized batch processing.
    
    This endpoint is designed for processing multiple texts efficiently
    by leveraging batch processing capabilities of the underlying model.
    
    Args:
        request_data: BatchEmbeddingRequest containing:
            - texts: List of texts to generate embeddings for
            - batch_size: Batch size for processing (default: 32)
            - normalize: Whether to normalize embeddings (default: True)
        request: FastAPI request object
    
    Returns:
        BatchEmbeddingResponse: Batch embedding results with metadata
        
    Raises:
        HTTPException: For various error conditions
    """
    try:
        logger.info(f"Batch embedding request received: {len(request_data.texts)} texts")
        return await embedding_handler.generate_batch_embeddings(request_data, request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in batch embedding: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch embedding failed: {str(e)}"
        )


@router.post(
    "/embedding/parallel",
    response_model=ParallelEmbeddingResponse,
    summary="Generate Parallel Embeddings",
    description="Generate embeddings for large datasets using parallel processing",
    responses={
        200: {"description": "Parallel embeddings generated successfully"},
        400: {"description": "Invalid request parameters"},
        503: {"description": "Service not available"},
        500: {"description": "Internal server error"}
    }
)
async def generate_parallel_embeddings(
    request_data: ParallelEmbeddingRequest,
    request: Request
) -> ParallelEmbeddingResponse:
    """
    Generate embeddings for large datasets using parallel processing.
    
    This endpoint is optimized for processing large numbers of texts
    by distributing the workload across multiple workers.
    
    Args:
        request_data: ParallelEmbeddingRequest containing:
            - texts: List of texts to generate embeddings for
            - num_workers: Number of parallel workers (default: 4)
            - batch_size: Batch size for each worker (default: 32)
            - normalize: Whether to normalize embeddings (default: True)
        request: FastAPI request object
    
    Returns:
        ParallelEmbeddingResponse: Parallel embedding results with metadata
        
    Raises:
        HTTPException: For various error conditions
    """
    try:
        logger.info(f"Parallel embedding request received: {len(request_data.texts)} texts, {request_data.num_workers} workers")
        return await embedding_handler.generate_parallel_embeddings(request_data, request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in parallel embedding: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Parallel embedding failed: {str(e)}"
        )
        
@router.post(
    "/embedding/csv",
    response_model=CSVEmbeddingResponse,
    summary="Generate Embeddings from CSV",
    description="Download a CSV file from URL and generate embeddings for all text content",
    responses={
        200: {"description": "CSV embeddings generated successfully"},
        400: {"description": "Invalid request parameters or CSV processing failed"},
        503: {"description": "Service not available"},
        500: {"description": "Internal server error"}
    }
)
async def generate_embeddings_from_csv(
    request_data: CSVEmbeddingRequest,
    request: Request
) -> CSVEmbeddingResponse:
    """
    Download a CSV file from the provided URL and generate embeddings for all text content.
    
    This endpoint combines CSV file processing with embedding generation:
    1. Downloads the CSV file from the provided URL
    2. Extracts text content from specified or all columns
    3. Combines or separates column data based on configuration
    4. Generates embeddings using optimized batch processing
    5. Returns embeddings with detailed metadata about the CSV processing
    
    Args:
        request_data: CSVEmbeddingRequest containing:
            - url: URL to the CSV file
            - text_columns: Specific columns to process (None = all columns)
            - combine_columns: Whether to combine multiple columns per row
            - separator: Separator when combining columns
            - batch_size: Batch size for embedding generation
            - normalize: Whether to normalize embeddings
            - skip_empty: Whether to skip empty cells/rows
            - max_texts: Maximum number of texts to process
        request: FastAPI request object
    
    Returns:
        CSVEmbeddingResponse: CSV embedding results with metadata including:
        - Generated embeddings
        - Processing statistics
        - CSV file information
        - Performance metrics
        
    Raises:
        HTTPException: For various error conditions including:
        - Invalid or inaccessible CSV URL
        - CSV parsing errors
        - Empty or invalid CSV content
        - Service unavailability
        - Processing failures
    """
    try:
        logger.info(f"CSV embedding request received for URL: {request_data.url}")
        return await embedding_handler.generate_embeddings_from_csv(request_data, request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in CSV embedding: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CSV embedding failed: {str(e)}"
        )

@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Get Model Information",
    description="Get detailed information about the loaded embedding model",
    responses={
        200: {"description": "Model information retrieved successfully"},
        503: {"description": "Service not available"},
        500: {"description": "Internal server error"}
    }
)
async def get_model_info(request: Request) -> ModelInfoResponse:
    """
    Get detailed information about the loaded embedding model.
    
    This endpoint provides comprehensive information about the model
    including configuration, device information, and capabilities.
    
    Args:
        request: FastAPI request object
    
    Returns:
        ModelInfoResponse: Model information including:
        - Model name and configuration
        - Device information
        - Embedding dimensions
        - CUDA availability
        - Performance settings
    """
    try:
        logger.info("Model info request received")
        return await embedding_handler.get_model_info(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in model info: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model info failed: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Get comprehensive health status of the embedding generation service",
    responses={
        200: {"description": "Health status retrieved successfully"},
        500: {"description": "Health check failed"}
    }
)
async def get_health_status(request: Request) -> HealthCheckResponse:
    """
    Get comprehensive health status of the embedding generation service.
    
    This endpoint provides detailed health information including:
    - Service status and availability
    - System information (GPU, CPU, memory)
    - Processing metrics and statistics
    - Model status and configuration
    
    Args:
        request: FastAPI request object
    
    Returns:
        HealthCheckResponse: Comprehensive health status information
    """
    try:
        logger.info("Health check requested")
        return await health_check_handler.get_health_status(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in health check: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


# Additional utility endpoints
@router.get(
    "/status",
    summary="Service Status",
    description="Get basic service status information",
    responses={
        200: {"description": "Status retrieved successfully"}
    }
)
async def get_service_status(request: Request) -> Dict[str, Any]:
    """
    Get basic service status information.
    This is a lightweight alternative to the health check endpoint.
    """
    try:
        # Get server manager and embedding service
        from src.core.managers import get_server_manager
        server_manager = get_server_manager(request)
        embedding_service = server_manager.get_service("embedding_generation")
        
        if embedding_service and hasattr(embedding_service, 'embedding_generator') and embedding_service.embedding_generator:
            model_info = embedding_service.embedding_generator.get_model_info()
            return {
                "service": "Embedding Generation Service",
                "status": "running",
                "model_name": model_info.get("model_name", "unknown"),
                "device": model_info.get("device", "unknown"),
                "embedding_dimension": model_info.get("embedding_dimension", 0),
                "initialized": embedding_service.is_initialized,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "service": "Embedding Generation Service",
                "status": "not_initialized",
                "model_name": "unknown",
                "device": "unknown",
                "embedding_dimension": 0,
                "initialized": False,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        return {
            "service": "Embedding Generation Service",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get(
    "/metrics",
    summary="Processing Metrics",
    description="Get processing metrics and statistics",
    responses={
        200: {"description": "Metrics retrieved successfully"}
    }
)
async def get_metrics(request: Request) -> Dict[str, Any]:
    """
    Get processing metrics and statistics for the service.
    """
    try:
        metrics = embedding_handler.processing_metrics
        return {
            "total_requests": metrics.total_requests,
            "successful_requests": metrics.successful_requests,
            "failed_requests": metrics.failed_requests,
            "success_rate": (
                metrics.successful_requests / metrics.total_requests 
                if metrics.total_requests > 0 else 0
            ),
            "average_processing_time": metrics.average_processing_time,
            "last_request_time": (
                metrics.last_request_time.isoformat() 
                if metrics.last_request_time else None
            ),
            "uptime": time.time() - metrics.start_time.timestamp(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


@router.post(
    "/cache/clear",
    summary="Clear Cache",
    description="Clear GPU cache and run garbage collection",
    responses={
        200: {"description": "Cache cleared successfully"},
        503: {"description": "Service not available"},
        500: {"description": "Internal server error"}
    }
)
async def clear_cache(request: Request) -> Dict[str, Any]:
    """
    Clear GPU cache and run garbage collection.
    This can help free up memory and improve performance.
    """
    try:
        # Get the embedding generation service
        from src.core.managers import get_server_manager
        server_manager = get_server_manager(request)
        embedding_service = server_manager.get_service("embedding_generation")
        
        if not embedding_service or not embedding_service.embedding_generator:
            raise HTTPException(
                status_code=503,
                detail="Embedding generation service not available"
            )
        
        # Clear cache
        embedding_service.embedding_generator.clear_cache()
        
        return {
            "success": True,
            "message": "Cache cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )
