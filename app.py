# main.py
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Import the new embedding generation API routes
from src.api.routes import router
from src.core.server_manager import create_server_manager, ServerManager
from src.core.process_manager import create_process_manager, ProcessManager
from src.core.managers import get_server_manager, get_process_manager
from src.utils.resources.logger import logger
from src.utils.config.settings import settings


# Global instances for server and process management
server_manager: ServerManager = None
process_manager: ProcessManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle application lifespan events (startup and shutdown)
    with integrated server and process management
    """
    global server_manager, process_manager
    
    # Startup logic
    logger.info(f"Starting {settings.get('app.name', 'Embedding Generation Service')}")
    
    try:
        # Validate configuration
        if not settings.validate_config():
            logger.error("Configuration validation failed")
            raise RuntimeError("Configuration validation failed")
        
        # Print configuration summary in development
        if settings.is_development():
            settings.print_config_summary()
        
        # Initialize process manager
        logger.info("Initializing process manager...")
        process_manager = create_process_manager()
        logger.info("Process manager initialized successfully")
        
        # Initialize server manager
        logger.info("Initializing server manager...")
        server_manager = create_server_manager()
        
        # Register services
        logger.info("Registering services...")
        await _register_services(server_manager)
        
        # Setup signal handlers for graceful shutdown
        server_manager.setup_signal_handlers()
        
        # Initialize server components and services
        logger.info("Initializing server components...")
        if not await server_manager.initialize():
            logger.error("Server manager initialization failed")
            raise RuntimeError("Server manager initialization failed")
        
        # Store managers in app state for access in routes
        app.state.server_manager = server_manager
        app.state.process_manager = process_manager
        
        logger.info(f"{settings.get('app.name', 'Embedding Generation Service')} started successfully")
        logger.info("All services and components are ready")
        
    except Exception as e:
        logger.error(f"Error during application startup: {str(e)}", exc_info=True)
        raise
    
    # Yield control to FastAPI
    yield
    
    # Shutdown logic - this runs when FastAPI is shutting down
    logger.info(f"Shutting down {settings.get('app.name', 'Embedding Generation Service')}")
    
    try:
        # Shutdown server manager (includes all services and cleanup)
        if server_manager:
            logger.info("Shutting down server manager...")
            await server_manager.shutdown()
            logger.info("Server manager shutdown complete")
        
        # Process manager shutdown is handled by server manager
        # but we can add additional cleanup here if needed
        if process_manager:
            logger.info("Process manager shutdown handled by server manager")
        
        logger.info("Application shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during application shutdown: {str(e)}", exc_info=True)
        # Don't re-raise the exception to allow FastAPI to complete shutdown


async def _register_services(server_manager: ServerManager) -> None:
    """Register all configured services with the server manager"""
    from src.api.handlers import EmbeddingService
    from src.core.server_manager import ModelDownloaderService, ServiceConfig
    
    # Get services configuration
    services_config = settings.get("server_manager.services", {})
    
    # Register embedding generation service
    if "embedding_generation" in services_config:
        embedding_config = services_config["embedding_generation"]
        if embedding_config.get("enabled", True):
            logger.info("Registering embedding generation service...")
            
            service_config = ServiceConfig(
                name="embedding_generation",
                enabled=embedding_config.get("enabled", True),
                initialization_timeout=embedding_config.get("initialization_timeout", 120.0),
                shutdown_timeout=embedding_config.get("shutdown_timeout", 10.0),
                dependencies=embedding_config.get("dependencies", []),
                config=embedding_config.get("config", {})
            )
            
            # Get device from server manager
            device = server_manager.device_manager.get_device()
            
            # Create and register the service
            embedding_service = EmbeddingService(service_config, device)
            server_manager.register_service(embedding_service)
            logger.info("Embedding generation service registered successfully")
    
    # Register model downloader service
    if "model_downloader" in services_config:
        downloader_config = services_config["model_downloader"]
        if downloader_config.get("enabled", True):
            logger.info("Registering model downloader service...")
            
            service_config = ServiceConfig(
                name="model_downloader",
                enabled=downloader_config.get("enabled", True),
                initialization_timeout=downloader_config.get("initialization_timeout", 60.0),
                shutdown_timeout=downloader_config.get("shutdown_timeout", 10.0),
                dependencies=downloader_config.get("dependencies", []),
                config=downloader_config.get("config", {})
            )
            
            # Create and register the service
            model_downloader_service = ModelDownloaderService(service_config)
            server_manager.register_service(model_downloader_service)
            logger.info("Model downloader service registered successfully")
    
    logger.info(f"Registered {len(server_manager.services)} services")


# Create FastAPI application with lifespan manager
app = FastAPI(
    title=settings.get("app.name", "Embedding Generation Service"),
    description=settings.get("app.description", "FastAPI service for generating text embeddings using SentenceTransformers"),
    version=settings.get("app.version", "1.0.0"),
    lifespan=lifespan,
    debug=settings.get("app.debug", False),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Setup CORS middleware
cors_config = settings.get_cors_config()
if cors_config:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.get("allow_origins", ["*"]),
        allow_credentials=cors_config.get("allow_credentials", True),
        allow_methods=cors_config.get("allow_methods", ["*"]),
        allow_headers=cors_config.get("allow_headers", ["*"]),
        expose_headers=cors_config.get("expose_headers", ["*"])
    )

# Include API routes
app.include_router(router)

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint providing basic service information
    """
    return {
        "service": "Embedding Generation Service",
        "version": settings.get("app.version", "1.0.0"),
        "description": "FastAPI service for generating text embeddings using SentenceTransformers",
        "docs": "/docs",
        "health": "/api/v1/health",
        "status": "/status"
    }

# Basic status endpoint (legacy compatibility)
@app.get("/status")
async def get_status(request: Request):
    """
    Basic status endpoint for compatibility
    """
    server_mgr = get_server_manager(request)
    process_mgr = get_process_manager(request)
    
    return {
        "service": "Embedding Generation Service",
        "version": settings.get("app.version", "1.0.0"),
        "server_status": server_mgr.get_server_status() if server_mgr else None,
        "process_metrics": process_mgr.get_metrics() if process_mgr else None,
        "timestamp": "2024-01-15T10:30:00Z"
    }

# Health endpoint (redirect to API health check)
@app.get("/health")
async def health_redirect():
    """
    Redirect to the comprehensive health check endpoint
    """
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/api/v1/health", status_code=302)

# Custom shutdown handler for uvicorn
def handle_exit(sig, frame):
    """
    Custom signal handler for uvicorn to ensure proper shutdown
    """
    logger.info(f"Received exit signal in uvicorn process")
    # Don't call sys.exit() here as it will be handled by the lifespan


# Run the application if executed directly
if __name__ == "__main__":
    logger.info("Starting Embedding Generation Service...")
    
    # Get server configuration from settings
    server_config = settings.get_server_config()
    
    # Configure uvicorn
    config = uvicorn.Config(
        app=app,
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 8000),
        reload=server_config.get("reload", False),
        workers=server_config.get("workers", 1),
        timeout_keep_alive=server_config.get("keep_alive", 2),
        log_level=settings.get("logging.level", "info").lower()
    )
    
    # Create and run server
    server = uvicorn.Server(config)
    
    try:
        # Run the server in the main thread
        server.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
        # Let the lifespan handle the shutdown gracefully
    except Exception as e:
        logger.error(f"Unexpected error during server execution: {e}")
        raise
