import signal
import sys
import asyncio
import os
import glob
import time
import shutil
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.resources.logger import logger
from src.utils.config.settings import settings
from src.core.process_manager import ProcessManager, create_process_manager


@dataclass
class ServiceConfig:
    """Configuration for AI services"""
    name: str
    enabled: bool = True
    initialization_timeout: float = 30.0
    shutdown_timeout: float = 10.0
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerConfig:
    """Configuration for the server manager"""
    app_name: str = "FastAPI AI Server"
    temp_dir: str = "./runtime/temp"
    outputs_dir: str = "./runtime/outputs"
    logs_dir: str = "./runtime/logs"
    housekeeping_interval: int = 600  # 10 minutes
    force_cleanup_on_startup: bool = True
    force_cleanup_on_shutdown: bool = True
    graceful_shutdown_timeout: float = 5.0
    services: Dict[str, ServiceConfig] = field(default_factory=dict)


class AIService(ABC):
    """Abstract base class for AI services"""
    
    def __init__(self, config: ServiceConfig, device: Optional[str] = None):
        self.config = config
        self.device = device
        self.is_initialized = False
        self.is_shutting_down = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the AI service"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the AI service"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get service status information"""
        pass


class ModelDownloaderService(AIService):
    """Service for downloading AI models"""
    
    def __init__(self, config: ServiceConfig, device: Optional[str] = None):
        super().__init__(config, device)
        self.downloaded_models: List[str] = []
    
    async def initialize(self) -> bool:
        """Initialize model downloader"""
        try:
            logger.info(f"Initializing {self.config.name}")
            
            # Import model downloader if available
            try:
                from src.utils.resources.downloader import ModelDownloader
                
                # Get model list from config
                model_list = self.config.config.get("model_list", [])
                if model_list:
                    model_downloader = ModelDownloader()
                    model_downloader.download_multiple_objects(model_list)
                    self.downloaded_models = model_list
                    logger.info(f"Downloaded {len(model_list)} models")
                
                self.is_initialized = True
                return True
            except ImportError:
                logger.warning("ModelDownloader not available, skipping model downloads")
                self.is_initialized = True
                return True
                
        except Exception as e:
            logger.error(f"Error initializing {self.config.name}: {str(e)}", exc_info=True)
            return False
    
    async def shutdown(self) -> None:
        """Shutdown model downloader"""
        logger.info(f"Shutting down {self.config.name}")
        self.is_shutting_down = True
    
    def get_status(self) -> Dict[str, Any]:
        """Get model downloader status"""
        return {
            "name": self.config.name,
            "initialized": self.is_initialized,
            "downloaded_models": len(self.downloaded_models),
            "models": self.downloaded_models
        }


class DeviceManager:
    """Manages AI device configuration (CPU/GPU)"""
    
    def __init__(self):
        self.device = None
        self.device_type = "cpu"
        self._setup_device()
    
    def _setup_device(self) -> None:
        """Setup AI device (CPU/GPU)"""
        try:
            # Try to import torch for GPU detection
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = torch.device('cuda')
                    self.device_type = "cuda"
                    torch.backends.cudnn.enabled = True
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                    logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
                else:
                    self.device = torch.device('cpu')
                    self.device_type = "cpu"
                    logger.info("Using CPU device")
            except ImportError:
                # PyTorch not available, use CPU
                self.device = "cpu"
                self.device_type = "cpu"
                logger.info("PyTorch not available, using CPU device")
                
        except Exception as e:
            logger.error(f"Error setting up device: {str(e)}")
            self.device = "cpu"
            self.device_type = "cpu"
    
    def get_device(self) -> Union[str, Any]:
        """Get the configured device"""
        return self.device
    
    def get_device_type(self) -> str:
        """Get the device type"""
        return self.device_type
    
    def clear_gpu_cache(self) -> None:
        """Clear GPU cache if available"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
        except ImportError:
            pass


class ServerManager:
    """
    Generalized server manager for AI projects using FastAPI.
    Manages server lifecycle, AI services, resources, and handles signals.
    """
    
    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or self._load_config_from_settings()
        self.is_shutting_down = False
        
        # Core components
        self.device_manager = DeviceManager()
        self.process_manager = create_process_manager()
        
        # Services registry
        self.services: Dict[str, AIService] = {}
        self.service_initialization_order: List[str] = []
        
        # Background tasks
        self.housekeeping_task: Optional[asyncio.Task] = None
        self._exit_event = asyncio.Event()
        
        # Signal handling
        self._original_sigint_handler = None
        
        # Setup directories
        self._setup_directories()
        
        logger.info(f"Server manager initialized for {self.config.app_name}")
    
    def _load_config_from_settings(self) -> ServerConfig:
        """Load server configuration from settings"""
        app_config = settings.get_app_config()
        
        # Create service configs from settings
        services_config = {}
        services_data = settings.get("services", {})
        
        for service_name, service_data in services_data.items():
            services_config[service_name] = ServiceConfig(
                name=service_name,
                enabled=service_data.get("enabled", True),
                initialization_timeout=service_data.get("initialization_timeout", 30.0),
                shutdown_timeout=service_data.get("shutdown_timeout", 10.0),
                dependencies=service_data.get("dependencies", []),
                config=service_data.get("config", {})
            )
        
        return ServerConfig(
            app_name=app_config.get("name", "FastAPI AI Server"),
            temp_dir=app_config.get("temp_dir", "./runtime/temp"),
            outputs_dir=app_config.get("outputs_dir", "./runtime/outputs"),
            logs_dir=app_config.get("logs_dir", "./runtime/logs"),
            housekeeping_interval=app_config.get("housekeeping_interval", 600),
            force_cleanup_on_startup=app_config.get("force_cleanup_on_startup", True),
            force_cleanup_on_shutdown=app_config.get("force_cleanup_on_shutdown", True),
            graceful_shutdown_timeout=app_config.get("graceful_shutdown_timeout", 5.0),
            services=services_config
        )
    
    def _setup_directories(self) -> None:
        """Setup required directories"""
        directories = [
            self.config.temp_dir,
            self.config.outputs_dir,
            self.config.logs_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    def register_service(self, service: AIService) -> None:
        """Register an AI service"""
        self.services[service.config.name] = service
        logger.info(f"Registered service: {service.config.name}")
    
    def get_service(self, name: str) -> Optional[AIService]:
        """Get a registered service by name"""
        return self.services.get(name)
    
    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        # Only setup signal handlers if we're in the main process
        # and not already handled by uvicorn
        try:
            # Check if we're in a child process by comparing with parent PID
            # If getppid() fails, we're likely in the main process
            parent_pid = os.getppid()
            current_pid = os.getpid()
            
            # Only setup signal handlers if we haven't already set them up
            # and we're not in a worker process (when workers > 1)
            if not hasattr(self, '_original_sigint_handler') and parent_pid != current_pid:
                self._original_sigint_handler = signal.getsignal(signal.SIGINT)
                
                signal.signal(signal.SIGINT, self.handle_shutdown_signal)
                signal.signal(signal.SIGTERM, self.handle_shutdown_signal)
                logger.info("Signal handlers installed")
            else:
                logger.info("Skipping signal handler setup - already handled or in worker process")
        except (OSError, ProcessLookupError) as e:
            # If we can't get parent PID, assume we're in main process
            logger.debug(f"Could not determine parent process: {e}")
            if not hasattr(self, '_original_sigint_handler'):
                self._original_sigint_handler = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, self.handle_shutdown_signal)
                signal.signal(signal.SIGTERM, self.handle_shutdown_signal)
                logger.info("Signal handlers installed (fallback)")
            else:
                logger.info("Signal handlers already installed")

    def handle_shutdown_signal(self, signum: int, frame) -> None:
        """Handle shutdown signals with proper exit behavior - robust against multiple interrupts"""
        sig_name = signal.Signals(signum).name
        
        # Use a lock to prevent race conditions from multiple interrupts
        if not hasattr(self, '_shutdown_lock'):
            import threading
            self._shutdown_lock = threading.Lock()
        
        with self._shutdown_lock:
            if self.is_shutting_down:
                logger.warning(f"Received {sig_name} again while shutting down. Ignoring additional signals.")
                return
            
            logger.info(f"Received {sig_name} signal, initiating graceful shutdown")
            self.is_shutting_down = True
            
            # Disable signal handlers to prevent further interrupts during shutdown
            try:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                logger.debug("Signal handlers disabled during shutdown")
            except Exception as e:
                logger.debug(f"Could not disable signal handlers: {e}")
            
            # Shutdown process manager
            try:
                self.process_manager.shutdown()
            except Exception as e:
                logger.error(f"Error during process manager shutdown: {e}")
            
            # Set the exit event to trigger asyncio tasks to complete
            if hasattr(self, '_exit_event'):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.call_soon_threadsafe(self._exit_event.set)
                except RuntimeError:
                    # Event loop might be closed
                    pass
    
    async def wait_for_exit(self, timeout: Optional[float] = None) -> None:
        """Wait for exit signal with timeout"""
        timeout = timeout or self.config.graceful_shutdown_timeout
        try:
            await asyncio.wait_for(self._exit_event.wait(), timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Shutdown timeout of {timeout}s reached")
    
    async def initialize(self) -> bool:
        """Initialize server components and services"""
        try:
            logger.info("Initializing server components")
            
            # Force initial cleanup if configured
            if self.config.force_cleanup_on_startup:
                await self._force_cleanup_folders()
            
            # Initialize services in dependency order
            await self._initialize_services()
            
            # Start process manager cleanup task
            logger.info("Starting process manager cleanup task")
            await self.process_manager.start_cleanup_task()
            
            # Start housekeeping task
            self.housekeeping_task = asyncio.create_task(self._housekeeping_loop())
            
            # Log current state
            self._log_server_state()
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during server initialization: {str(e)}", exc_info=True)
            return False
    
    async def _initialize_services(self) -> None:
        """Initialize all registered services in dependency order"""
        # Sort services by dependencies
        self.service_initialization_order = self._resolve_service_dependencies()
        
        for service_name in self.service_initialization_order:
            service = self.services.get(service_name)
            if not service or not service.config.enabled:
                continue
            
            logger.info(f"Initializing service: {service_name}")
            
            try:
                # Set timeout for service initialization
                success = await asyncio.wait_for(
                    service.initialize(),
                    timeout=service.config.initialization_timeout
                )
                
                if success:
                    logger.info(f"Service {service_name} initialized successfully")
                else:
                    logger.error(f"Service {service_name} failed to initialize")
                    
            except asyncio.TimeoutError:
                logger.error(f"Service {service_name} initialization timed out")
            except Exception as e:
                logger.error(f"Error initializing service {service_name}: {str(e)}", exc_info=True)
    
    def _resolve_service_dependencies(self) -> List[str]:
        """Resolve service initialization order based on dependencies"""
        # Simple topological sort for dependencies
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(service_name: str) -> None:
            if service_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {service_name}")
            if service_name in visited:
                return
            
            temp_visited.add(service_name)
            
            # Visit dependencies first
            service = self.services.get(service_name)
            if service:
                for dep in service.config.dependencies:
                    if dep in self.services:
                        visit(dep)
            
            temp_visited.remove(service_name)
            visited.add(service_name)
            result.append(service_name)
        
        # Visit all services
        for service_name in self.services.keys():
            if service_name not in visited:
                visit(service_name)
        
        return result
    
    async def _force_cleanup_folders(self) -> None:
        """Force cleanup of temp and outputs folders"""
        logger.info("Forcing initial cleanup of directories")
        
        directories_to_clean = [
            self.config.temp_dir,
            self.config.outputs_dir
        ]
        
        for directory in directories_to_clean:
            await self._cleanup_directory(directory)
    
    async def _cleanup_directory(self, directory: str) -> None:
        """Clean up a directory"""
        if not os.path.exists(directory):
            return
        
        try:
            files_removed = 0
            dirs_removed = 0
            
            # Clean files and subdirectories
            for item_path in glob.glob(os.path.join(directory, "*")):
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                        files_removed += 1
                        logger.debug(f"Removed file: {item_path}")
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                        dirs_removed += 1
                        logger.debug(f"Removed directory: {item_path}")
                except Exception as e:
                    logger.error(f"Error removing {item_path}: {str(e)}")
            
            if files_removed > 0 or dirs_removed > 0:
                logger.info(f"Cleaned {directory}: {files_removed} files, {dirs_removed} directories")
                
        except Exception as e:
            logger.error(f"Error cleaning directory {directory}: {str(e)}", exc_info=True)
    
    async def _housekeeping_loop(self) -> None:
        """Additional housekeeping task that runs periodically"""
        while not self.is_shutting_down:
            try:
                await asyncio.sleep(self.config.housekeeping_interval)
                
                logger.info("Running housekeeping task")
                
                # Log process summary
                self.process_manager.log_process_summary()
                
                # Log server state
                self._log_server_state()
                
                # Trigger orphaned file cleanup
                await self.process_manager.cleanup_orphaned_files()
                
                # Log service statuses
                self._log_service_statuses()
                
            except asyncio.CancelledError:
                logger.info("Housekeeping task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in housekeeping task: {str(e)}", exc_info=True)
    
    def _log_server_state(self) -> None:
        """Log the current state of server directories"""
        try:
            # Count files in directories
            temp_files = len([f for f in glob.glob(os.path.join(self.config.temp_dir, "**", "*"), recursive=True) 
                            if os.path.isfile(f)])
            output_files = len([f for f in glob.glob(os.path.join(self.config.outputs_dir, "**", "*"), recursive=True) 
                              if os.path.isfile(f)])
            
            # Calculate sizes
            temp_size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(self.config.temp_dir, "**", "*"), recursive=True) 
                          if os.path.isfile(f)) / (1024 * 1024)  # MB
            output_size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(self.config.outputs_dir, "**", "*"), recursive=True) 
                            if os.path.isfile(f)) / (1024 * 1024)  # MB
            
            logger.info(f"Server state: {temp_files} temp files ({temp_size:.2f} MB), "
                       f"{output_files} output files ({output_size:.2f} MB)")
            
        except Exception as e:
            logger.error(f"Error logging server state: {str(e)}", exc_info=True)
    
    def _log_service_statuses(self) -> None:
        """Log status of all services"""
        for service_name, service in self.services.items():
            try:
                status = service.get_status()
                logger.info(f"Service {service_name}: {status}")
            except Exception as e:
                logger.error(f"Error getting status for service {service_name}: {str(e)}")
    
    async def shutdown(self) -> None:
        """Shutdown server components and services"""
        logger.info("Shutting down server")
        
        # Cancel housekeeping task
        if self.housekeeping_task:
            logger.info("Cancelling housekeeping task")
            self.housekeeping_task.cancel()
            try:
                await self.housekeeping_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error cancelling housekeeping task: {str(e)}")
        
        # Stop process manager cleanup task
        logger.info("Stopping process manager cleanup task")
        await self.process_manager.stop_cleanup_task()
        
        # Shutdown services in reverse order
        await self._shutdown_services()
        
        # Clean up process resources
        logger.info("Cleaning up all process resources")
        await self.process_manager.cleanup_all_resources()
        
        # Clear GPU cache
        self.device_manager.clear_gpu_cache()
        
        # Force final cleanup if configured
        if self.config.force_cleanup_on_shutdown:
            logger.info("Forcing final cleanup of all files")
            await self._force_cleanup_folders()
        
        # Verify cleanup
        self._verify_cleanup()
        
        logger.info("Server shutdown complete")
        # Don't call sys.exit() here - let FastAPI handle the process exit
    
    async def _shutdown_services(self) -> None:
        """Shutdown all services in reverse initialization order"""
        for service_name in reversed(self.service_initialization_order):
            service = self.services.get(service_name)
            if not service:
                continue
            
            logger.info(f"Shutting down service: {service_name}")
            
            try:
                await asyncio.wait_for(
                    service.shutdown(),
                    timeout=service.config.shutdown_timeout
                )
                logger.info(f"Service {service_name} shut down successfully")
                
            except asyncio.TimeoutError:
                logger.error(f"Service {service_name} shutdown timed out")
            except Exception as e:
                logger.error(f"Error shutting down service {service_name}: {str(e)}", exc_info=True)
    
    def _verify_cleanup(self) -> None:
        """Verify that cleanup was successful"""
        try:
            temp_files = len([f for f in glob.glob(os.path.join(self.config.temp_dir, "**", "*"), recursive=True) 
                            if os.path.isfile(f)])
            output_files = len([f for f in glob.glob(os.path.join(self.config.outputs_dir, "**", "*"), recursive=True) 
                              if os.path.isfile(f)])
            
            logger.info(f"Cleanup verification: {temp_files} temp files, {output_files} output files remaining")
            
            # If files remain, try aggressive cleanup
            if temp_files > 0 or output_files > 0:
                logger.warning("Files still remain after cleanup, attempting aggressive cleanup")
                self._aggressive_cleanup()
                
        except Exception as e:
            logger.error(f"Error during cleanup verification: {str(e)}", exc_info=True)
    
    def _aggressive_cleanup(self) -> None:
        """Perform aggressive cleanup by recreating directories"""
        try:
            # Recreate temp directory
            if os.path.exists(self.config.temp_dir):
                logger.info(f"Removing entire temp directory: {self.config.temp_dir}")
                shutil.rmtree(self.config.temp_dir, ignore_errors=True)
            os.makedirs(self.config.temp_dir, exist_ok=True)
            
            # Recreate outputs directory
            if os.path.exists(self.config.outputs_dir):
                logger.info(f"Removing entire outputs directory: {self.config.outputs_dir}")
                shutil.rmtree(self.config.outputs_dir, ignore_errors=True)
            os.makedirs(self.config.outputs_dir, exist_ok=True)
            
            logger.info("Aggressive cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during aggressive cleanup: {str(e)}", exc_info=True)
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get comprehensive server status"""
        return {
            "app_name": self.config.app_name,
            "is_shutting_down": self.is_shutting_down,
            "device": {
                "type": self.device_manager.get_device_type(),
                "device": str(self.device_manager.get_device())
            },
            "services": {
                name: service.get_status() 
                for name, service in self.services.items()
            },
            "process_manager": self.process_manager.get_metrics(),
            "directories": {
                "temp": self.config.temp_dir,
                "outputs": self.config.outputs_dir,
                "logs": self.config.logs_dir
            }
        }


# Factory function to create server manager with default configuration
def create_server_manager(config: Optional[ServerConfig] = None) -> ServerManager:
    """
    Create a ServerManager instance with optional custom configuration.
    
    Args:
        config: Optional ServerConfig. If None, loads from settings.
    
    Returns:
        ServerManager instance
    """
    return ServerManager(config)


# Convenience function for creating the default server manager
def get_default_server_manager() -> ServerManager:
    """Get the default server manager instance based on settings"""
    return create_server_manager()


# Global server manager instance (can be used across the application)
server_manager = get_default_server_manager()
