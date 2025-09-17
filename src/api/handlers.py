import time
import torch
import asyncio
import concurrent.futures
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import HTTPException, Request

from src.services.embedding_generator import EmbeddingGenerator
from src.core.server_manager import AIService, ServiceConfig
from src.core.managers import get_server_manager
from src.api.models import (
    EmbeddingRequest, BatchEmbeddingRequest, ParallelEmbeddingRequest,
    EmbeddingResult, BatchEmbeddingResult, ParallelEmbeddingResult,
    HealthStatus, ServiceStatus, SystemInfo, ProcessingMetrics,
    ErrorInfo, HealthCheckResponse
)
from src.api.schema import (
    SingleEmbeddingRequest as SingleEmbeddingRequestSchema,
    BatchEmbeddingRequest as BatchEmbeddingRequestSchema,
    ParallelEmbeddingRequest as ParallelEmbeddingRequestSchema,
    SingleEmbeddingResponse, BatchEmbeddingResponse, ParallelEmbeddingResponse,
    CSVEmbeddingRequest, CSVEmbeddingResponse,
    ModelInfoResponse, HealthCheckResponse as HealthCheckResponseSchema,
    ErrorResponse
)
from src.utils.resources.logger import logger
from src.utils.config.settings import settings
from src.utils.io.reader import CSVTextReader


class EmbeddingService(AIService):
    """Embedding generation service that integrates with the core system"""
    
    def __init__(self, config: ServiceConfig, device: Optional[str] = None):
        super().__init__(config, device)
        self.embedding_generator: Optional[EmbeddingGenerator] = None
        self.processing_metrics = ProcessingMetrics()
        self.start_time = time.time()
    
    async def initialize(self) -> bool:
        """Initialize the embedding generation service"""
        try:
            logger.info("Initializing embedding generation service...")
            
            # Get configuration from service config
            model_name = self.config.config.get("model_name", "all-MiniLM-L6-v2")
            batch_size = self.config.config.get("batch_size", 32)
            max_seq_length = self.config.config.get("max_seq_length", 512)
            normalize_embeddings = self.config.config.get("normalize_embeddings", True)
            
            # Log configuration being used
            logger.info(f"Embedding Generation Configuration:")
            logger.info(f"  - Model Name: {model_name}")
            logger.info(f"  - Device: {self.device}")
            logger.info(f"  - Batch Size: {batch_size}")
            logger.info(f"  - Max Sequence Length: {max_seq_length}")
            logger.info(f"  - Normalize Embeddings: {normalize_embeddings}")
            
            self.embedding_generator = EmbeddingGenerator(
                model_name=model_name,
                device=self.device,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                normalize_embeddings=normalize_embeddings
            )
            
            # Apply optimizations
            self.embedding_generator.optimize_for_inference()
            
            self.is_initialized = True
            logger.info("Embedding generation service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding generation service: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the embedding generation service"""
        logger.info("Shutting down embedding generation service...")
        self.is_shutting_down = True
        
        if self.embedding_generator:
            try:
                self.embedding_generator.clear_cache()
                logger.info("Embedding generation service shutdown complete")
            except Exception as e:
                logger.error(f"Error during embedding generation service shutdown: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get embedding generation service status"""
        if not self.embedding_generator:
            return {
                "name": self.config.name,
                "initialized": self.is_initialized,
                "error": "Service not initialized"
            }
        
        try:
            model_info = self.embedding_generator.get_model_info()
            model_info.update({
                "name": self.config.name,
                "initialized": self.is_initialized,
                "processing_metrics": {
                    "total_requests": self.processing_metrics.total_requests,
                    "successful_requests": self.processing_metrics.successful_requests,
                    "failed_requests": self.processing_metrics.failed_requests,
                    "average_processing_time": self.processing_metrics.average_processing_time,
                    "uptime": time.time() - self.start_time
                }
            })
            return model_info
        except Exception as e:
            return {
                "name": self.config.name,
                "initialized": self.is_initialized,
                "error": str(e)
            }


class EmbeddingHandler:
    """Handler for embedding generation operations using the core system"""
    
    def __init__(self):
        self.processing_metrics = ProcessingMetrics()
        self.start_time = time.time()
    
    def _get_embedding_service(self, request: Request) -> EmbeddingService:
        """Get the embedding generation service from the server manager"""
        try:
            server_manager = get_server_manager(request)
            service = server_manager.get_service("embedding_generation")
            
            if not service:
                raise HTTPException(
                    status_code=503,
                    detail="Embedding generation service not registered"
                )
            
            if not isinstance(service, EmbeddingService):
                raise HTTPException(
                    status_code=503,
                    detail="Invalid embedding generation service type"
                )
            
            if not service.is_initialized:
                raise HTTPException(
                    status_code=503,
                    detail="Embedding generation service not initialized"
                )
            
            return service
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting embedding generation service: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Failed to access embedding generation service: {str(e)}"
            )
    
    def _validate_single_request(self, request_data: SingleEmbeddingRequestSchema) -> None:
        """Validate single embedding request parameters"""
        if not request_data.text or not request_data.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty"
            )
        
        if len(request_data.text) > 10000:
            raise HTTPException(
                status_code=400,
                detail="Text cannot exceed 10000 characters"
            )
    
    def _validate_batch_request(self, request_data: BatchEmbeddingRequestSchema) -> None:
        """Validate batch embedding request parameters"""
        if not request_data.texts:
            raise HTTPException(
                status_code=400,
                detail="Texts list cannot be empty"
            )
        
        if len(request_data.texts) > 1000:
            raise HTTPException(
                status_code=400,
                detail="Number of texts cannot exceed 1000"
            )
        
        # Check for empty texts
        valid_texts = [text for text in request_data.texts if text and text.strip()]
        if not valid_texts:
            raise HTTPException(
                status_code=400,
                detail="No valid texts found in input list"
            )
    
    def _validate_parallel_request(self, request_data: ParallelEmbeddingRequestSchema) -> None:
        """Validate parallel embedding request parameters"""
        if not request_data.texts:
            raise HTTPException(
                status_code=400,
                detail="Texts list cannot be empty"
            )
        
        if len(request_data.texts) > 10000:
            raise HTTPException(
                status_code=400,
                detail="Number of texts cannot exceed 10000"
            )
        
        # Check for empty texts
        valid_texts = [text for text in request_data.texts if text and text.strip()]
        if not valid_texts:
            raise HTTPException(
                status_code=400,
                detail="No valid texts found in input list"
            )
    def _validate_csv_request(self, request_data: CSVEmbeddingRequest) -> None:
        """Validate CSV embedding request parameters"""
        if not request_data.url or not request_data.url.strip():
            raise HTTPException(
                status_code=400,
                detail="CSV URL cannot be empty"
            )
        
        # Basic URL validation
        if not request_data.url.startswith(('http://', 'https://')):
            raise HTTPException(
                status_code=400,
                detail="URL must start with http:// or https://"
            )
        
        if request_data.max_texts and request_data.max_texts > 10000:
            raise HTTPException(
                status_code=400,
                detail="max_texts cannot exceed 10000"
            )
    
    async def generate_single_embedding(
        self, 
        request_data: SingleEmbeddingRequestSchema,
        request: Request
    ) -> SingleEmbeddingResponse:
        """
        Handle single text embedding generation request
        
        Args:
            request_data: The embedding request data
            request: FastAPI request object
            
        Returns:
            SingleEmbeddingResponse with embedding result
        """
        start_time = time.time()
        request_id = f"req_{int(time.time() * 1000)}"
        
        try:
            # Validate request parameters
            self._validate_single_request(request_data)
            
            # Get the embedding generation service from core system
            embedding_service = self._get_embedding_service(request)
            
            logger.info(f"Processing single embedding request {request_id}")
            logger.info(f"Text: '{request_data.text[:100]}...'")
            logger.info(f"Normalize: {request_data.normalize}")
            
            # Generate embedding
            embedding = embedding_service.embedding_generator.generate_embedding(
                request_data.text
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update metrics
            self.processing_metrics.total_requests += 1
            self.processing_metrics.successful_requests += 1
            self.processing_metrics.last_request_time = datetime.now()
            
            # Calculate average processing time
            if self.processing_metrics.total_requests > 0:
                total_time = (self.processing_metrics.average_processing_time * 
                            (self.processing_metrics.total_requests - 1) + processing_time)
                self.processing_metrics.average_processing_time = total_time / self.processing_metrics.total_requests
            
            # Create response
            response = SingleEmbeddingResponse(
                success=True,
                text=request_data.text,
                embedding=embedding.tolist(),
                embedding_dimension=len(embedding),
                processing_time=processing_time,
                normalized=request_data.normalize,
                message="Embedding generated successfully"
            )
            
            logger.info(f"Single embedding request {request_id} completed successfully in {processing_time:.3f}s")
            return response
            
        except HTTPException:
            # Re-raise HTTP exceptions
            self.processing_metrics.failed_requests += 1
            raise
        except Exception as e:
            # Handle unexpected errors
            self.processing_metrics.failed_requests += 1
            logger.error(f"Unexpected error in single embedding request {request_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )
    
    async def generate_batch_embeddings(
        self, 
        request_data: BatchEmbeddingRequestSchema,
        request: Request
    ) -> BatchEmbeddingResponse:
        """
        Handle batch text embedding generation request
        
        Args:
            request_data: The batch embedding request data
            request: FastAPI request object
            
        Returns:
            BatchEmbeddingResponse with embedding results
        """
        start_time = time.time()
        request_id = f"req_{int(time.time() * 1000)}"
        
        try:
            # Validate request parameters
            self._validate_batch_request(request_data)
            
            # Get the embedding generation service from core system
            embedding_service = self._get_embedding_service(request)
            
            logger.info(f"Processing batch embedding request {request_id}")
            logger.info(f"Texts count: {len(request_data.texts)}")
            logger.info(f"Batch size: {request_data.batch_size}")
            logger.info(f"Normalize: {request_data.normalize}")
            
            # Generate embeddings
            embeddings = embedding_service.embedding_generator.generate_embeddings_batch(
                request_data.texts,
                batch_size=request_data.batch_size
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update metrics
            self.processing_metrics.total_requests += 1
            self.processing_metrics.successful_requests += 1
            self.processing_metrics.last_request_time = datetime.now()
            
            # Calculate average processing time
            if self.processing_metrics.total_requests > 0:
                total_time = (self.processing_metrics.average_processing_time * 
                            (self.processing_metrics.total_requests - 1) + processing_time)
                self.processing_metrics.average_processing_time = total_time / self.processing_metrics.total_requests
            
            # Convert embeddings to list format
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            
            # Create response
            response = BatchEmbeddingResponse(
                success=True,
                total_texts=len(request_data.texts),
                embeddings_generated=len(embeddings),
                embedding_dimension=len(embeddings[0]) if embeddings else 0,
                processing_time=processing_time,
                batch_size=request_data.batch_size,
                normalized=request_data.normalize,
                embeddings=embeddings_list,
                message="Batch embeddings generated successfully"
            )
            
            logger.info(f"Batch embedding request {request_id} completed successfully in {processing_time:.3f}s")
            return response
            
        except HTTPException:
            # Re-raise HTTP exceptions
            self.processing_metrics.failed_requests += 1
            raise
        except Exception as e:
            # Handle unexpected errors
            self.processing_metrics.failed_requests += 1
            logger.error(f"Unexpected error in batch embedding request {request_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )
    
    async def generate_parallel_embeddings(
        self, 
        request_data: ParallelEmbeddingRequestSchema,
        request: Request
    ) -> ParallelEmbeddingResponse:
        """
        Handle parallel text embedding generation request
        
        Args:
            request_data: The parallel embedding request data
            request: FastAPI request object
            
        Returns:
            ParallelEmbeddingResponse with embedding results
        """
        start_time = time.time()
        request_id = f"req_{int(time.time() * 1000)}"
        
        try:
            # Validate request parameters
            self._validate_parallel_request(request_data)
            
            # Get the embedding generation service from core system
            embedding_service = self._get_embedding_service(request)
            
            logger.info(f"Processing parallel embedding request {request_id}")
            logger.info(f"Texts count: {len(request_data.texts)}")
            logger.info(f"Num workers: {request_data.num_workers}")
            logger.info(f"Batch size: {request_data.batch_size}")
            logger.info(f"Normalize: {request_data.normalize}")
            
            # Generate embeddings
            embeddings = embedding_service.embedding_generator.generate_embeddings_parallel(
                request_data.texts,
                num_workers=request_data.num_workers,
                batch_size=request_data.batch_size
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update metrics
            self.processing_metrics.total_requests += 1
            self.processing_metrics.successful_requests += 1
            self.processing_metrics.last_request_time = datetime.now()
            
            # Calculate average processing time
            if self.processing_metrics.total_requests > 0:
                total_time = (self.processing_metrics.average_processing_time * 
                            (self.processing_metrics.total_requests - 1) + processing_time)
                self.processing_metrics.average_processing_time = total_time / self.processing_metrics.total_requests
            
            # Convert embeddings to list format
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            
            # Create response
            response = ParallelEmbeddingResponse(
                success=True,
                total_texts=len(request_data.texts),
                embeddings_generated=len(embeddings),
                embedding_dimension=len(embeddings[0]) if embeddings else 0,
                processing_time=processing_time,
                num_workers=request_data.num_workers,
                batch_size=request_data.batch_size,
                normalized=request_data.normalize,
                embeddings=embeddings_list,
                message="Parallel embeddings generated successfully"
            )
            
            logger.info(f"Parallel embedding request {request_id} completed successfully in {processing_time:.3f}s")
            return response
            
        except HTTPException:
            # Re-raise HTTP exceptions
            self.processing_metrics.failed_requests += 1
            raise
        except Exception as e:
            # Handle unexpected errors
            self.processing_metrics.failed_requests += 1
            logger.error(f"Unexpected error in parallel embedding request {request_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )
    async def generate_embeddings_from_csv(
        self, 
        request_data: CSVEmbeddingRequest,
        request: Request
    ) -> CSVEmbeddingResponse:
        """
        Handle CSV file embedding generation request
        
        Args:
            request_data: The CSV embedding request data
            request: FastAPI request object
            
        Returns:
            CSVEmbeddingResponse with embedding results
        """
        start_time = time.time()
        request_id = f"csv_req_{int(time.time() * 1000)}"
        
        try:
            # Validate request parameters
            self._validate_csv_request(request_data)
            
            # Get the embedding generation service from core system
            embedding_service = self._get_embedding_service(request)
            
            logger.info(f"Processing CSV embedding request {request_id}")
            logger.info(f"URL: {request_data.url}")
            logger.info(f"Text columns: {request_data.text_columns}")
            logger.info(f"Combine columns: {request_data.combine_columns}")
            logger.info(f"Batch size: {request_data.batch_size}")
            
            # Step 1: Download and read CSV file
            csv_reader = CSVTextReader()
            csv_read_result = csv_reader.read_csv_from_url(
                url=request_data.url,
                text_columns=request_data.text_columns,
                combine_columns=request_data.combine_columns,
                separator=request_data.separator,
                skip_empty=request_data.skip_empty
            )
            
            # Check if CSV reading was successful
            if not csv_read_result.success:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to read CSV file: {csv_read_result.message}"
                )
            
            # Get texts from CSV
            texts = csv_read_result.texts
            
            # Apply max_texts limit if specified
            if request_data.max_texts and len(texts) > request_data.max_texts:
                texts = texts[:request_data.max_texts]
                logger.info(f"Limited texts to {request_data.max_texts} items")
            
            # Check if we have any texts to process
            if not texts:
                raise HTTPException(
                    status_code=400,
                    detail="No texts found in CSV file after processing"
                )
            
            logger.info(f"Successfully extracted {len(texts)} texts from CSV")
            
            # Step 2: Generate embeddings using existing batch processing
            embeddings = embedding_service.embedding_generator.generate_embeddings_batch(
                texts,
                batch_size=request_data.batch_size
            )
            
            # Calculate total processing time
            processing_time = time.time() - start_time
            
            # Update metrics
            self.processing_metrics.total_requests += 1
            self.processing_metrics.successful_requests += 1
            self.processing_metrics.last_request_time = datetime.now()
            
            # Calculate average processing time
            if self.processing_metrics.total_requests > 0:
                total_time = (self.processing_metrics.average_processing_time * 
                            (self.processing_metrics.total_requests - 1) + processing_time)
                self.processing_metrics.average_processing_time = total_time / self.processing_metrics.total_requests
            
            # Convert embeddings to list format
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            
            # Prepare CSV information
            csv_info = {
                "source_url": request_data.url,
                "total_rows": csv_read_result.total_rows,
                "columns_processed": csv_read_result.columns_processed,
                "download_time": csv_read_result.processing_time,
                "texts_extracted": len(csv_read_result.texts),
                "texts_processed": len(texts)
            }
            
            # Create response
            response = CSVEmbeddingResponse(
                success=True,
                total_texts=len(texts),
                embeddings_generated=len(embeddings),
                embedding_dimension=len(embeddings[0]) if embeddings else 0,
                processing_time=processing_time,
                batch_size=request_data.batch_size,
                normalized=request_data.normalize,
                embeddings=embeddings_list,
                csv_info=csv_info,
                message=f"Successfully generated embeddings for {len(embeddings)} texts from CSV"
            )
            
            logger.info(f"CSV embedding request {request_id} completed successfully in {processing_time:.3f}s")
            return response
            
        except HTTPException:
            # Re-raise HTTP exceptions
            self.processing_metrics.failed_requests += 1
            raise
        except Exception as e:
            # Handle unexpected errors
            self.processing_metrics.failed_requests += 1
            logger.error(f"Unexpected error in CSV embedding request {request_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error during CSV processing: {str(e)}"
            )
    
    async def get_model_info(self, request: Request) -> ModelInfoResponse:
        """
        Get model information
        
        Args:
            request: FastAPI request object
            
        Returns:
            ModelInfoResponse with model information
        """
        try:
            # Get the embedding generation service from core system
            embedding_service = self._get_embedding_service(request)
            
            # Get model information
            model_info = embedding_service.embedding_generator.get_model_info()
            
            response = ModelInfoResponse(
                success=True,
                model_info=model_info,
                message="Model information retrieved successfully"
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting model info: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get model info: {str(e)}"
            )


class HealthCheckHandler:
    """Handler for health check operations using the core system with async optimization"""
    
    def __init__(self, embedding_handler: EmbeddingHandler):
        self.embedding_handler = embedding_handler
        self.start_time = time.time()
        # Create a thread pool executor for CPU-bound operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    def _get_basic_system_info(self) -> Dict[str, Any]:
        """Get basic system information (fast operations)"""
        import sys
        
        return {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        }
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information in a separate thread"""
        gpu_info = {"gpu_name": None, "gpu_memory": None}
        
        if torch.cuda.is_available():
            try:
                gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
                gpu_info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            except Exception as e:
                logger.warning(f"Could not get GPU info: {e}")
        
        return gpu_info
    
    def _get_cpu_memory_info(self) -> Dict[str, Any]:
        """Get CPU and memory information in a separate thread"""
        cpu_memory_info = {"cpu_count": None, "memory_total": None}
        
        try:
            import psutil
            cpu_memory_info["cpu_count"] = psutil.cpu_count()
            cpu_memory_info["memory_total"] = psutil.virtual_memory().total / 1024**3
        except Exception as e:
            logger.warning(f"Could not get system info: {e}")
        
        return cpu_memory_info
    
    def _get_embedding_service_status(self, embedding_service) -> Dict[str, Any]:
        """Get embedding service status in a separate thread"""
        if not embedding_service or not isinstance(embedding_service, EmbeddingService):
            return {"error": "Embedding generation service not available"}
        
        try:
            return embedding_service.get_status()
        except Exception as e:
            logger.warning(f"Could not get embedding generation status: {e}")
            return {"error": str(e)}
    
    async def _get_system_info_async(self) -> SystemInfo:
        """Get system information asynchronously using thread pool"""
        # Get basic info immediately (fast)
        basic_info = self._get_basic_system_info()
        
        # Run potentially slow operations in parallel
        loop = asyncio.get_event_loop()
        
        # Submit tasks to thread pool
        gpu_future = loop.run_in_executor(self.executor, self._get_gpu_info)
        cpu_memory_future = loop.run_in_executor(self.executor, self._get_cpu_memory_info)
        
        # Wait for all tasks to complete
        gpu_info, cpu_memory_info = await asyncio.gather(gpu_future, cpu_memory_future)
        
        # Combine all information
        system_info = SystemInfo(
            python_version=basic_info["python_version"],
            torch_version=basic_info["torch_version"],
            cuda_available=basic_info["cuda_available"],
            gpu_name=gpu_info["gpu_name"],
            gpu_memory=gpu_info["gpu_memory"],
            cpu_count=cpu_memory_info["cpu_count"],
            memory_total=cpu_memory_info["memory_total"]
        )
        
        return system_info
    
    async def get_health_status(self, request: Request) -> HealthCheckResponseSchema:
        """
        Get comprehensive health status using the core system with async optimization
        
        Args:
            request: FastAPI request object
            
        Returns:
            HealthCheckResponseSchema with detailed status information
        """
        try:
            # Calculate uptime (fast operation)
            uptime = time.time() - self.start_time
            
            # Get server manager and embedding service (fast operation)
            server_manager = get_server_manager(request)
            embedding_service = server_manager.get_service("embedding_generation")
            
            # Run potentially slow operations in parallel
            loop = asyncio.get_event_loop()
            
            # Submit tasks to thread pool
            system_info_future = self._get_system_info_async()
            embedding_status_future = loop.run_in_executor(
                self.executor, 
                self._get_embedding_service_status, 
                embedding_service
            )
            
            # Wait for all tasks to complete
            system_info, embedding_status = await asyncio.gather(
                system_info_future, 
                embedding_status_future
            )
            
            # Determine overall health status (fast operation)
            status = "healthy"
            message = "All systems operational"
            
            if not embedding_status or not embedding_status.get('initialized', False):
                status = "unhealthy"
                message = "Embedding generation service not ready"
            elif embedding_status.get('error'):
                status = "degraded"
                message = f"Embedding generation service has issues: {embedding_status['error']}"
            
            # Create response
            response = HealthCheckResponseSchema(
                status=status,
                timestamp=datetime.now().isoformat(),
                service_name="Embedding Generation Service",
                version="1.0.0",
                uptime=uptime,
                embedding_service_status=embedding_status or {},
                system_info={
                    "python_version": system_info.python_version,
                    "torch_version": system_info.torch_version,
                    "cuda_available": system_info.cuda_available,
                    "gpu_name": system_info.gpu_name,
                    "gpu_memory": system_info.gpu_memory,
                    "cpu_count": system_info.cpu_count,
                    "memory_total": system_info.memory_total
                },
                processing_metrics={
                    "total_requests": self.embedding_handler.processing_metrics.total_requests,
                    "successful_requests": self.embedding_handler.processing_metrics.successful_requests,
                    "failed_requests": self.embedding_handler.processing_metrics.failed_requests,
                    "average_processing_time": self.embedding_handler.processing_metrics.average_processing_time,
                    "uptime": uptime
                },
                message=message
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in health check: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Health check failed: {str(e)}"
            )
    
    def __del__(self):
        """Cleanup thread pool executor"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Global handler instances
embedding_handler = EmbeddingHandler()
health_check_handler = HealthCheckHandler(embedding_handler)