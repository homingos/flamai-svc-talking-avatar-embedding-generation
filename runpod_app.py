import asyncio
import time
import uuid
import json
from typing import Optional, Dict, Any, List
import runpod
from src.services.embedding_generator import EmbeddingGenerator
from src.utils.resources.logger import logger
from src.utils.config.settings import settings


class EmbeddingGenerationServerlessSystem:
    """
    RunPod Serverless Embedding Generation System
    Single unified endpoint structure for all operations
    """
    
    def __init__(self):
        """Initialize the Embedding Generation system for RunPod"""
        self.settings = settings
        
        # Initialize Embedding Generation service
        logger.info(" Initializing Embedding Generation Service...")
        
        # Get configuration from settings
        model_name = self.settings.get_embedding_model_name()
        device = self.settings.get_embedding_device()
        batch_size = self.settings.get_embedding_batch_size()
        max_seq_length = self.settings.get_embedding_max_seq_length()
        normalize_embeddings = self.settings.get_embedding_normalize()
        
        logger.info(f"📋 Embedding Generation Configuration:")
        logger.info(f"  - Model Name: {model_name}")
        logger.info(f"  - Device: {device}")
        logger.info(f"  - Batch Size: {batch_size}")
        logger.info(f"  - Max Sequence Length: {max_seq_length}")
        logger.info(f"  - Normalize Embeddings: {normalize_embeddings}")
        
        self.embedding_generator = EmbeddingGenerator(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            normalize_embeddings=normalize_embeddings
        )
        
        # Apply optimizations
        self.embedding_generator.optimize_for_inference()
        
        # Add performance optimizations for serverless
        self._processing_cache = {}
        self._max_cache_size = 100
        
        # Processing metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
        
        logger.info("✅ Embedding Generation Serverless System initialized!")
    
    async def generate_single_embedding(
        self, 
        text: str, 
        normalize: bool = True,
        client_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate embedding for a single text"""
        start_time = time.time()
        request_id = client_id or f"req_{int(time.time() * 1000)}"
        
        try:
            logger.info(f"🎯 Processing single embedding request (Request ID: {request_id})")
            logger.info(f"📝 Text: '{text[:100]}...'")
            logger.info(f" Normalize: {normalize}")
            
            # Validate input parameters
            if not text or not text.strip():
                return {
                    "success": False,
                    "error_message": "Text cannot be empty",
                    "processing_time": time.time() - start_time
                }
            
            # Check text length
            max_text_length = self.settings.get_embedding_max_text_length()
            if len(text) > max_text_length:
                return {
                    "success": False,
                    "error_message": f"Text cannot exceed {max_text_length} characters",
                    "processing_time": time.time() - start_time
                }
            
            # Generate embedding
            logger.info("🔄 Generating embedding...")
            embedding = self.embedding_generator.generate_embedding(text)
            
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Single embedding request {request_id} completed successfully in {processing_time:.3f}s")
            
            return {
                "success": True,
                "text": text,
                "embedding": embedding.tolist(),
                "embedding_dimension": len(embedding),
                "processing_time": processing_time,
                "normalized": normalize,
                "message": "Embedding generated successfully"
            }
                
        except Exception as e:
            logger.error(f"❌ Error processing single embedding request: {str(e)}")
            return {
                "success": False,
                "error_message": f"Processing error: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    async def generate_batch_embeddings(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None,
        normalize: bool = True,
        client_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate embeddings for multiple texts using batch processing"""
        start_time = time.time()
        request_id = client_id or f"batch_req_{int(time.time() * 1000)}"
        
        try:
            logger.info(f"🎯 Processing batch embedding request (Request ID: {request_id})")
            logger.info(f"📝 Texts count: {len(texts)}")
            logger.info(f" Batch size: {batch_size}")
            logger.info(f" Normalize: {normalize}")
            
            # Validate input parameters
            if not texts:
                return {
                    "success": False,
                    "error_message": "Texts list cannot be empty",
                    "processing_time": time.time() - start_time
                }
            
            # Check texts count
            max_batch_size = self.settings.get_embedding_max_texts_per_batch()
            if len(texts) > max_batch_size:
                return {
                    "success": False,
                    "error_message": f"Number of texts cannot exceed {max_batch_size}",
                    "processing_time": time.time() - start_time
                }
            
            # Check for empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                return {
                    "success": False,
                    "error_message": "No valid texts found in input list",
                    "processing_time": time.time() - start_time
                }
            
            # Use default batch size if not provided
            if batch_size is None:
                batch_size = self.settings.get_embedding_batch_size()
            
            # Generate embeddings
            logger.info("🔄 Generating batch embeddings...")
            embeddings = self.embedding_generator.generate_embeddings_batch(
                valid_texts, 
                batch_size=batch_size
            )
            
            processing_time = time.time() - start_time
            
            # Convert embeddings to list format
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            
            logger.info(f"✅ Batch embedding request {request_id} completed successfully in {processing_time:.3f}s")
            
            return {
                "success": True,
                "total_texts": len(texts),
                "embeddings_generated": len(embeddings),
                "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                "processing_time": processing_time,
                "batch_size": batch_size,
                "normalized": normalize,
                "embeddings": embeddings_list,
                "message": "Batch embeddings generated successfully"
            }
                
        except Exception as e:
            logger.error(f"❌ Error processing batch embedding request: {str(e)}")
            return {
                "success": False,
                "error_message": f"Processing error: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    async def generate_parallel_embeddings(
        self, 
        texts: List[str], 
        num_workers: int = 4,
        batch_size: Optional[int] = None,
        normalize: bool = True,
        client_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate embeddings for large datasets using parallel processing"""
        start_time = time.time()
        request_id = client_id or f"parallel_req_{int(time.time() * 1000)}"
        
        try:
            logger.info(f"🎯 Processing parallel embedding request (Request ID: {request_id})")
            logger.info(f"📝 Texts count: {len(texts)}")
            logger.info(f" Num workers: {num_workers}")
            logger.info(f" Batch size: {batch_size}")
            logger.info(f" Normalize: {normalize}")
            
            # Validate input parameters
            if not texts:
                return {
                    "success": False,
                    "error_message": "Texts list cannot be empty",
                    "processing_time": time.time() - start_time
                }
            
            # Check texts count
            max_parallel_size = self.settings.get_embedding_max_texts_per_parallel()
            if len(texts) > max_parallel_size:
                return {
                    "success": False,
                    "error_message": f"Number of texts cannot exceed {max_parallel_size}",
                    "processing_time": time.time() - start_time
                }
            
            # Check num_workers
            max_workers = self.settings.get_embedding_max_parallel_workers()
            if num_workers > max_workers:
                return {
                    "success": False,
                    "error_message": f"Number of workers cannot exceed {max_workers}",
                    "processing_time": time.time() - start_time
                }
            
            # Check for empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                return {
                    "success": False,
                    "error_message": "No valid texts found in input list",
                    "processing_time": time.time() - start_time
                }
            
            # Use default batch size if not provided
            if batch_size is None:
                batch_size = self.settings.get_embedding_batch_size()
            
            # Generate embeddings
            logger.info("🔄 Generating parallel embeddings...")
            embeddings = self.embedding_generator.generate_embeddings_parallel(
                valid_texts,
                num_workers=num_workers,
                batch_size=batch_size
            )
            
            processing_time = time.time() - start_time
            
            # Convert embeddings to list format
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            
            logger.info(f"✅ Parallel embedding request {request_id} completed successfully in {processing_time:.3f}s")
            
            return {
                "success": True,
                "total_texts": len(texts),
                "embeddings_generated": len(embeddings),
                "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                "processing_time": processing_time,
                "num_workers": num_workers,
                "batch_size": batch_size,
                "normalized": normalize,
                "embeddings": embeddings_list,
                "message": "Parallel embeddings generated successfully"
            }
                
        except Exception as e:
            logger.error(f"❌ Error processing parallel embedding request: {str(e)}")
            return {
                "success": False,
                "error_message": f"Processing error: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check with comprehensive system information"""
        try:
            logger.info("🩺 Performing system health check...")
            
            # Check Embedding Generation service
            model_info = self.embedding_generator.get_model_info()
            embedding_healthy = model_info.get('model_loaded', False)
            logger.info(f"🤖 Embedding Generation health: {'✅ OK' if embedding_healthy else '❌ FAILED'}")
            
            # Calculate uptime
            uptime = time.time() - self.start_time
            
            # Get system information
            import sys
            import torch
            
            system_info = {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": model_info.get('device', 'unknown'),
                "model_loaded": model_info.get('model_loaded', False),
                "model_name": model_info.get('model_name', 'unknown'),
                "embedding_dimension": model_info.get('embedding_dimension', 0),
                "batch_size": model_info.get('batch_size', 0),
                "max_seq_length": model_info.get('max_seq_length', 0)
            }
            
            # Get GPU info if available
            if torch.cuda.is_available():
                try:
                    system_info["gpu_name"] = torch.cuda.get_device_name(0)
                    system_info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
                except Exception as e:
                    logger.warning(f"Could not get GPU info: {e}")
            
            # Calculate processing metrics
            success_rate = (
                self.successful_requests / self.total_requests 
                if self.total_requests > 0 else 0
            )
            
            processing_metrics = {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": success_rate,
                "uptime": uptime
            }
            
            overall_health = embedding_healthy
            status = "healthy" if overall_health else "unhealthy"
            
            logger.info(f"🏥 Overall system health: {'✅ HEALTHY' if overall_health else '❌ UNHEALTHY'}")
            
            return {
                "success": True,
                "status": status,
                "timestamp": time.time(),
                "service_name": "Embedding Generation Service",
                "version": "1.0.0",
                "uptime": uptime,
                "components": {
                    "embedding_generation": embedding_healthy
                },
                "system_info": system_info,
                "processing_metrics": processing_metrics,
                "model_info": model_info
            }
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {str(e)}")
            return {
                "success": False,
                "status": "unhealthy",
                "error_message": f"Health check failed: {str(e)}"
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get detailed system information"""
        try:
            logger.info("📋 Getting system information...")
            
            # Get model information
            model_info = self.embedding_generator.get_model_info()
            
            # Get system information
            import sys
            import torch
            
            system_info = {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": model_info.get('device', 'unknown'),
                "model_loaded": model_info.get('model_loaded', False),
                "model_name": model_info.get('model_name', 'unknown'),
                "embedding_dimension": model_info.get('embedding_dimension', 0),
                "batch_size": model_info.get('batch_size', 0),
                "max_seq_length": model_info.get('max_seq_length', 0),
                "normalize_embeddings": model_info.get('normalize_embeddings', True)
            }
            
            # Get GPU info if available
            if torch.cuda.is_available():
                try:
                    system_info["gpu_name"] = torch.cuda.get_device_name(0)
                    system_info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
                except Exception as e:
                    logger.warning(f"Could not get GPU info: {e}")
            
            # Get CPU and memory info
            try:
                import psutil
                system_info["cpu_count"] = psutil.cpu_count()
                system_info["memory_total"] = psutil.virtual_memory().total / 1024**3
            except Exception as e:
                logger.warning(f"Could not get system info: {e}")
            
            # Calculate processing metrics
            uptime = time.time() - self.start_time
            success_rate = (
                self.successful_requests / self.total_requests 
                if self.total_requests > 0 else 0
            )
            
            processing_metrics = {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": success_rate,
                "uptime": uptime
            }
            
            logger.info(f"✅ System information retrieved successfully")
            
            return {
                "success": True,
                "status": "healthy" if model_info.get('model_loaded', False) else "unhealthy",
                "timestamp": time.time(),
                "service_name": "Embedding Generation Service",
                "version": "1.0.0",
                "uptime": uptime,
                "model_info": model_info,
                "system_info": system_info,
                "processing_metrics": processing_metrics
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting system info: {str(e)}")
            return {
                "success": False,
                "error_message": f"Failed to get system info: {str(e)}"
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics and statistics"""
        try:
            logger.info(" Getting processing metrics...")
            
            uptime = time.time() - self.start_time
            success_rate = (
                self.successful_requests / self.total_requests 
                if self.total_requests > 0 else 0
            )
            
            metrics = {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": success_rate,
                "uptime": uptime,
                "timestamp": time.time()
            }
            
            logger.info(f"✅ Processing metrics retrieved successfully")
            
            return {
                "success": True,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting metrics: {str(e)}")
            return {
                "success": False,
                "error_message": f"Failed to get metrics: {str(e)}"
            }
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        try:
            if hasattr(self.embedding_generator, 'clear_cache'):
                self.embedding_generator.clear_cache()
            logger.info(" GPU memory cleaned up")
        except Exception as e:
            logger.warning(f"Could not clean up GPU memory: {e}")


# Global system instance
_system_instance: Optional[EmbeddingGenerationServerlessSystem] = None

def get_system_instance() -> EmbeddingGenerationServerlessSystem:
    """Get or create the system instance"""
    global _system_instance
    if _system_instance is None:
        _system_instance = EmbeddingGenerationServerlessSystem()
    return _system_instance

# UNIFIED RUNPOD HANDLER - Single Template Structure
async def handler(job):
    """
    🎯 UNIFIED RUNPOD HANDLER
    Single template structure for ALL endpoints
    
    Expected Input Format (ALL endpoints use this same structure):
    {
        "input": {
            "endpoint": "single" | "batch" | "parallel" | "health_check" | "system_info" | "metrics",
            "data": {
                // Endpoint-specific parameters go here
            }
        }
    }
    
    Output Format (ALL endpoints return this same structure):
    {
        "success": true/false,
        "data": {
            // Endpoint-specific response data
        },
        "error_message": "string" | null,
        "processing_time": float,
        "endpoint": "string"
    }
    """
    
    start_time = time.time()
    
    try:
        # Extract input from RunPod job
        input_data = job.get("input", {})
        endpoint = input_data.get("endpoint")
        data = input_data.get("data", {})
        
        if not endpoint:
            return {
                "success": False,
                "data": {},
                "error_message": "Missing required parameter: endpoint",
                "processing_time": time.time() - start_time,
                "endpoint": "unknown"
            }
        
        logger.info(f"🚀 Processing RunPod request - Endpoint: {endpoint}")
        
        # Get system instance
        system = get_system_instance()
        
        # Route to appropriate endpoint handler
        if endpoint == "single":
            # Single Embedding Endpoint
            text = data.get("text")
            normalize = data.get("normalize", True)
            client_id = data.get("client_id")
            
            if not text:
                return {
                    "success": False,
                    "data": {},
                    "error_message": "Missing required parameter: text",
                    "processing_time": time.time() - start_time,
                    "endpoint": endpoint
                }
            
            result = await system.generate_single_embedding(
                text=text,
                normalize=normalize,
                client_id=client_id
            )
            
            # Update metrics
            system.total_requests += 1
            if result["success"]:
                system.successful_requests += 1
            else:
                system.failed_requests += 1
            
            return {
                "success": result["success"],
                "data": {
                    "text": result.get("text"),
                    "embedding": result.get("embedding"),
                    "embedding_dimension": result.get("embedding_dimension"),
                    "normalized": result.get("normalized"),
                    "message": result.get("message")
                },
                "error_message": result.get("error_message"),
                "processing_time": time.time() - start_time,
                "endpoint": endpoint
            }
        
        elif endpoint == "batch":
            # Batch Embedding Endpoint
            texts = data.get("texts")
            batch_size = data.get("batch_size")
            normalize = data.get("normalize", True)
            client_id = data.get("client_id")
            
            if not texts:
                return {
                    "success": False,
                    "data": {},
                    "error_message": "Missing required parameter: texts",
                    "processing_time": time.time() - start_time,
                    "endpoint": endpoint
                }
            
            result = await system.generate_batch_embeddings(
                texts=texts,
                batch_size=batch_size,
                normalize=normalize,
                client_id=client_id
            )
            
            # Update metrics
            system.total_requests += 1
            if result["success"]:
                system.successful_requests += 1
            else:
                system.failed_requests += 1
            
            return {
                "success": result["success"],
                "data": {
                    "total_texts": result.get("total_texts"),
                    "embeddings_generated": result.get("embeddings_generated"),
                    "embedding_dimension": result.get("embedding_dimension"),
                    "batch_size": result.get("batch_size"),
                    "normalized": result.get("normalized"),
                    "embeddings": result.get("embeddings"),
                    "message": result.get("message")
                },
                "error_message": result.get("error_message"),
                "processing_time": time.time() - start_time,
                "endpoint": endpoint
            }
        
        elif endpoint == "parallel":
            # Parallel Embedding Endpoint
            texts = data.get("texts")
            num_workers = data.get("num_workers", 4)
            batch_size = data.get("batch_size")
            normalize = data.get("normalize", True)
            client_id = data.get("client_id")
            
            if not texts:
                return {
                    "success": False,
                    "data": {},
                    "error_message": "Missing required parameter: texts",
                    "processing_time": time.time() - start_time,
                    "endpoint": endpoint
                }
            
            result = await system.generate_parallel_embeddings(
                texts=texts,
                num_workers=num_workers,
                batch_size=batch_size,
                normalize=normalize,
                client_id=client_id
            )
            
            # Update metrics
            system.total_requests += 1
            if result["success"]:
                system.successful_requests += 1
            else:
                system.failed_requests += 1
            
            return {
                "success": result["success"],
                "data": {
                    "total_texts": result.get("total_texts"),
                    "embeddings_generated": result.get("embeddings_generated"),
                    "embedding_dimension": result.get("embedding_dimension"),
                    "num_workers": result.get("num_workers"),
                    "batch_size": result.get("batch_size"),
                    "normalized": result.get("normalized"),
                    "embeddings": result.get("embeddings"),
                    "message": result.get("message")
                },
                "error_message": result.get("error_message"),
                "processing_time": time.time() - start_time,
                "endpoint": endpoint
            }
        
        elif endpoint == "health_check":
            # Health Check Endpoint
            result = await system.health_check()
            
            return {
                "success": result["success"],
                "data": {
                    "status": result.get("status"),
                    "timestamp": result.get("timestamp"),
                    "service_name": result.get("service_name"),
                    "version": result.get("version"),
                    "uptime": result.get("uptime"),
                    "components": result.get("components"),
                    "system_info": result.get("system_info"),
                    "processing_metrics": result.get("processing_metrics"),
                    "model_info": result.get("model_info")
                },
                "error_message": result.get("error_message"),
                "processing_time": time.time() - start_time,
                "endpoint": endpoint
            }
        
        elif endpoint == "system_info":
            # System Info Endpoint
            result = system.get_system_info()
            
            return {
                "success": result["success"],
                "data": {
                    "status": result.get("status"),
                    "timestamp": result.get("timestamp"),
                    "service_name": result.get("service_name"),
                    "version": result.get("version"),
                    "uptime": result.get("uptime"),
                    "model_info": result.get("model_info"),
                    "system_info": result.get("system_info"),
                    "processing_metrics": result.get("processing_metrics")
                },
                "error_message": result.get("error_message"),
                "processing_time": time.time() - start_time,
                "endpoint": endpoint
            }
        
        elif endpoint == "metrics":
            # Metrics Endpoint
            result = system.get_metrics()
            
            return {
                "success": result["success"],
                "data": {
                    "metrics": result.get("metrics")
                },
                "error_message": result.get("error_message"),
                "processing_time": time.time() - start_time,
                "endpoint": endpoint
            }
        
        else:
            return {
                "success": False,
                "data": {},
                "error_message": f"Unknown endpoint: {endpoint}. Available: single, batch, parallel, health_check, system_info, metrics",
                "processing_time": time.time() - start_time,
                "endpoint": endpoint
            }
            
    except Exception as e:
        logger.error(f"❌ RunPod handler error: {str(e)}")
        return {
            "success": False,
            "data": {},
            "error_message": f"Handler error: {str(e)}",
            "processing_time": time.time() - start_time,
            "endpoint": input_data.get("endpoint", "unknown")
        }

# RunPod serverless setup
if __name__ == "__main__":
    logger.info("🚀 Starting RunPod Embedding Generation System...")
    
    # Initialize the system at startup
    system = get_system_instance()
    logger.info("✅ System initialized successfully!")
    
    logger.info("📋 Available Endpoints:")
    logger.info("  - single: Generate embedding for a single text")
    logger.info("  - batch: Generate embeddings for multiple texts using batch processing")
    logger.info("  - parallel: Generate embeddings for large datasets using parallel processing")
    logger.info("  - health_check: Get system health status")
    logger.info("  - system_info: Get detailed system information")
    logger.info("  - metrics: Get processing metrics and statistics")
    
    # Start the RunPod serverless worker
    runpod.serverless.start({"handler": handler})
