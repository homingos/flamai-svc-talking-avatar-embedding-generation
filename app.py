# main.py
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# from src.api.routes import router
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
    logger.info(f"Starting {settings.get('app.name', 'FastAPI App')}")
    
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
        
        logger.info(f"{settings.get('app.name', 'FastAPI App')} started successfully")
        logger.info("All services and components are ready")
        
    except Exception as e:
        logger.error(f"Error during application startup: {str(e)}", exc_info=True)
        raise
    
    # Yield control to FastAPI
    yield
    
    # Shutdown logic - this runs when FastAPI is shutting down
    logger.info(f"Shutting down {settings.get('app.name', 'FastAPI App')}")
    
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


# Create FastAPI application with lifespan manager
app = FastAPI(
    title=settings.get("app.name", "FastAPI REST Template"),
    description=settings.get("app.description", "FastAPI REST API Template"),
    version=settings.get("app.version", "1.0.0"),
    lifespan=lifespan,
    debug=settings.get("app.debug", False)
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
# app.include_router(router)

@app.get("/status")
async def get_status(request: Request):
    server_mgr = get_server_manager(request)
    process_mgr = get_process_manager(request)
    
    return {
        "server_status": server_mgr.get_server_status() if server_mgr else None,
        "process_metrics": process_mgr.get_metrics() if process_mgr else None
    }

# Custom shutdown handler for uvicorn
def handle_exit(sig, frame):
    """
    Custom signal handler for uvicorn to ensure proper shutdown
    """
    logger.info(f"Received exit signal in uvicorn process")
    # Don't call sys.exit() here as it will be handled by the lifespan


# Run the application if executed directly
if __name__ == "__main__":
    logger.info("Starting FastAPI Server...")
    
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
