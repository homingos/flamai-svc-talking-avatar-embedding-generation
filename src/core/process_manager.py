import os
import time
import uuid
import asyncio
import threading
import shutil
import glob
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict
from enum import Enum

from src.utils.resources.logger import logger
from src.utils.config.settings import settings


class ProcessState(Enum):
    """Process status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessData:
    """Data structure for process information"""
    process_id: str
    process_type: str
    status: ProcessState = ProcessState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)
    
    def update(self, **kwargs) -> None:
        """Update process data"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Auto-set completion time if status is completed or failed
        if self.status in [ProcessState.COMPLETED, ProcessState.FAILED, ProcessState.CANCELLED] and self.completed_at is None:
            self.completed_at = time.time()


class ProcessManagerConfig:
    """Configuration for ProcessManager loaded from settings"""
    def __init__(self):
        # Load configuration from settings
        self._load_from_settings()
    
    def _load_from_settings(self) -> None:
        """Load configuration from settings"""
        # Get process manager type
        self.stateful = settings.get_process_manager_type() == "stateful"
        
        # Get directories configuration
        dirs_config = settings.get_process_manager_directories_config()
        self.temp_dir = dirs_config.get("temp", "./runtime/temp")
        self.output_dir = dirs_config.get("outputs", "./runtime/outputs")
        self.logs_dir = dirs_config.get("logs", "./runtime/logs/process_manager")
        
        # Get cleanup configuration
        cleanup_config = settings.get_process_manager_cleanup_config()
        self.auto_cleanup = cleanup_config.get("enabled", True)
        self.cleanup_interval = cleanup_config.get("interval", 300)
        self.process_ttl = cleanup_config.get("process_ttl", 1800)
        self.orphaned_files_ttl = cleanup_config.get("orphaned_files_ttl", 3600)
        
        # Get file tracking configuration
        file_config = settings.get_process_manager_file_tracking_config()
        self.enable_file_tracking = file_config.get("enabled", True)
        self.track_temp_files = file_config.get("track_temp_files", True)
        self.track_output_files = file_config.get("track_output_files", True)
        
        # Get performance configuration
        perf_config = settings.get_process_manager_performance_config()
        self.thread_pool_size = perf_config.get("thread_pool_size", 4)
        self.async_cleanup = perf_config.get("async_cleanup", True)
        self.batch_cleanup_size = perf_config.get("batch_cleanup_size", 100)
        
        # Get monitoring configuration
        monitor_config = settings.get_process_manager_monitoring_config()
        self.enable_metrics = monitor_config.get("enable_metrics", True)
        self.log_process_events = monitor_config.get("log_process_events", True)
        
        # Get stateful/stateless specific config
        if self.stateful:
            stateful_config = settings.get_process_manager_stateful_config()
            self.max_processes = stateful_config.get("max_processes", 1000)
            self.enable_memory_optimization = stateful_config.get("enable_memory_optimization", True)
        else:
            stateless_config = settings.get_process_manager_stateless_config()
            self.cache_ttl = stateless_config.get("cache_ttl", 300)
            self.enable_persistence = stateless_config.get("enable_persistence", False)
            self.max_processes = 0  # No limit for stateless


class ProcessManager(ABC):
    """
    Abstract base class for process management.
    Provides common functionality for both stateful and stateless implementations.
    """
    
    def __init__(self, config: Optional[ProcessManagerConfig] = None):
        self.config = config or ProcessManagerConfig()
        self._lock = threading.RLock()
        self._is_shutting_down = False
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Initialize directories
        self._setup_directories()
        
        # Initialize metrics if enabled
        if self.config.enable_metrics:
            self._init_metrics()
        
        logger.info(f"ProcessManager initialized: stateful={self.config.stateful}, "
                   f"max_processes={self.config.max_processes}, cleanup_interval={self.config.cleanup_interval}s")
    
    def _setup_directories(self) -> None:
        """Setup required directories"""
        os.makedirs(self.config.temp_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.logs_dir, exist_ok=True)
    
    def _init_metrics(self) -> None:
        """Initialize metrics tracking"""
        self._metrics = {
            "processes_created": 0,
            "processes_completed": 0,
            "processes_failed": 0,
            "files_cleaned": 0,
            "cleanup_runs": 0,
            "last_cleanup": None
        }
    
    def generate_process_id(self) -> str:
        """Generate a unique process ID"""
        return str(uuid.uuid4())
    
    @abstractmethod
    def create_process(self, process_type: str, **kwargs) -> str:
        """Create a new process"""
        pass
    
    @abstractmethod
    def get_process(self, process_id: str) -> Optional[ProcessData]:
        """Get process data by ID"""
        pass
    
    @abstractmethod
    def update_process(self, process_id: str, **kwargs) -> bool:
        """Update process data"""
        pass
    
    @abstractmethod
    def remove_process(self, process_id: str) -> bool:
        """Remove a process"""
        pass
    
    @abstractmethod
    def list_processes(self, process_type: Optional[str] = None) -> Dict[str, ProcessData]:
        """List processes, optionally filtered by type"""
        pass
    
    @abstractmethod
    def cleanup_old_processes(self) -> int:
        """Clean up old processes and return count of cleaned processes"""
        pass
    
    def should_track_file(self, file_path: str) -> bool:
        """Check if a file should be tracked based on configuration"""
        if not self.config.enable_file_tracking:
            return False
        
        abs_path = os.path.abspath(file_path)
        temp_dir = os.path.abspath(self.config.temp_dir)
        output_dir = os.path.abspath(self.config.output_dir)
        
        # Check if file is in temp directory
        if abs_path.startswith(temp_dir):
            return self.config.track_temp_files
        
        # Check if file is in output directory
        if abs_path.startswith(output_dir):
            return self.config.track_output_files
        
        return False
    
    def track_file(self, process_id: str, file_path: str) -> None:
        """Track a file for a process"""
        if not self.should_track_file(file_path):
            return
        
        process = self.get_process(process_id)
        if process and file_path not in process.files:
            process.files.append(file_path)
            if self.config.log_process_events:
                logger.debug(f"Tracking file {file_path} for process {process_id}")
    
    def untrack_file(self, process_id: str, file_path: str) -> None:
        """Stop tracking a file for a process"""
        process = self.get_process(process_id)
        if process and file_path in process.files:
            process.files.remove(file_path)
            if self.config.log_process_events:
                logger.debug(f"Untracking file {file_path} for process {process_id}")
    
    async def start_cleanup_task(self) -> None:
        """Start the background cleanup task"""
        if self._cleanup_task is not None:
            logger.warning("Cleanup task already running")
            return
        
        if not self.config.auto_cleanup:
            logger.info("Auto cleanup disabled")
            return
        
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info(f"Started cleanup task with {self.config.cleanup_interval}s interval")
    
    async def stop_cleanup_task(self) -> None:
        """Stop the background cleanup task"""
        if self._cleanup_task is None:
            return
        
        logger.info("Stopping cleanup task")
        self._cleanup_task.cancel()
        
        try:
            await self._cleanup_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error stopping cleanup task: {str(e)}")
        finally:
            self._cleanup_task = None
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while not self._is_shutting_down:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                
                if self._is_shutting_down:
                    break
                
                logger.debug("Running scheduled cleanup")
                
                # Clean up old processes
                processes_cleaned = self.cleanup_old_processes()
                
                # Clean up orphaned files
                files_cleaned = await self.cleanup_orphaned_files()
                
                if self.config.enable_metrics:
                    self._metrics["cleanup_runs"] += 1
                    self._metrics["last_cleanup"] = time.time()
                
                if processes_cleaned > 0 or files_cleaned > 0:
                    logger.info(f"Cleanup completed: {processes_cleaned} processes, {files_cleaned} files")
                
            except asyncio.CancelledError:
                logger.info("Cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}", exc_info=True)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get process manager metrics"""
        if not self.config.enable_metrics:
            return {}
        
        return dict(self._metrics)
    
    def get_process_count(self) -> int:
        """Get the current number of processes"""
        return len(self.list_processes())
    
    def log_process_summary(self) -> None:
        """Log a summary of current processes"""
        processes = self.list_processes()
        if not processes:
            logger.info("No active processes")
            return
        
        # Count by status
        status_counts = {}
        for process in processes.values():
            status = process.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        logger.info(f"Process summary: {len(processes)} total processes - {status_counts}")
    
    async def cleanup_expired_processes(self) -> int:
        """Clean up expired processes based on TTL"""
        cutoff_time = time.time() - self.config.process_ttl
        processes_to_remove = []
        
        for process_id, process in self.list_processes().items():
            # Remove completed/failed processes that are old enough
            if (process.status in [ProcessState.COMPLETED, ProcessState.FAILED, ProcessState.CANCELLED] 
                and process.completed_at and process.completed_at < cutoff_time):
                processes_to_remove.append(process_id)
            # Remove very old processes regardless of status
            elif process.created_at < cutoff_time - 3600:  # 1 hour buffer
                processes_to_remove.append(process_id)
        
        # Remove processes in batches
        for i in range(0, len(processes_to_remove), self.config.batch_cleanup_size):
            batch = processes_to_remove[i:i + self.config.batch_cleanup_size]
            for process_id in batch:
                self.remove_process(process_id)
                if self.config.log_process_events:
                    logger.info(f"Cleaned up expired process: {process_id}")
    
    async def cleanup_orphaned_files(self) -> int:
        """Clean up orphaned files"""
        if not self.config.enable_file_tracking:
            return 0
            
        # Get all tracked files
        tracked_files = set()
        for process in self.list_processes().values():
            tracked_files.update(process.files)
        
        # Clean temp directory
        temp_cleaned = await self._cleanup_directory(self.config.temp_dir, tracked_files)
        
        # Clean output directory
        output_cleaned = await self._cleanup_directory(self.config.output_dir, tracked_files)
        
        if self.config.enable_metrics:
            self._metrics["files_cleaned"] += temp_cleaned + output_cleaned
        
        return temp_cleaned + output_cleaned
    
    async def _cleanup_directory(self, directory: str, tracked_files: set) -> int:
        """Clean up orphaned files in a directory"""
        if not os.path.exists(directory):
            return 0
            
        cutoff_time = time.time() - self.config.orphaned_files_ttl
        files_cleaned = 0
        
        for file_path in glob.glob(os.path.join(directory, "**", "*"), recursive=True):
            if os.path.isfile(file_path):
                abs_path = os.path.abspath(file_path)
                
                # Skip if file is tracked
                if abs_path in tracked_files:
                    continue
                
                # Remove if file is old enough
                try:
                    if os.path.getmtime(file_path) < cutoff_time:
                        os.remove(file_path)
                        files_cleaned += 1
                        if self.config.log_process_events:
                            logger.debug(f"Removed orphaned file: {file_path}")
                except Exception as e:
                    logger.error(f"Error removing file {file_path}: {e}")
        
        if files_cleaned > 0 and self.config.log_process_events:
            logger.info(f"Cleaned {files_cleaned} orphaned files from {directory}")
        
        return files_cleaned
    
    def shutdown(self) -> None:
        """Shutdown the process manager"""
        self._is_shutting_down = True
        logger.info("Process manager shutting down")
    
    async def cleanup_all_resources(self) -> None:
        """Clean up all resources - comprehensive cleanup method"""
        logger.info("Starting comprehensive resource cleanup")
        
        try:
            # Stop the cleanup task if running
            await self.stop_cleanup_task()
            
            # Clean up expired processes
            await self.cleanup_expired_processes()
            
            # Clean up orphaned files
            await self.cleanup_orphaned_files()
            
            # Clean up all remaining processes
            processes = self.list_processes()
            for process_id in list(processes.keys()):
                self.remove_process(process_id)
                if self.config.log_process_events:
                    logger.debug(f"Removed process during cleanup: {process_id}")
            
            logger.info("Comprehensive resource cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during comprehensive cleanup: {e}")
            # Don't re-raise to allow graceful shutdown


class StatefulProcessManager(ProcessManager):
    """
    Stateful process manager that maintains process state in memory.
    Fast and efficient for single-instance applications.
    """
    
    def __init__(self, config: Optional[ProcessManagerConfig] = None):
        super().__init__(config)
        self._processes: OrderedDict[str, ProcessData] = OrderedDict()
    
    def create_process(self, process_type: str, **kwargs) -> str:
        """Create a new process"""
        process_id = self.generate_process_id()
        
        with self._lock:
            # Create process data
            process_data = ProcessData(
                process_id=process_id,
                process_type=process_type,
                metadata=kwargs
            )
            
            self._processes[process_id] = process_data
            
            # Create process-specific temp directory
            process_temp_dir = os.path.join(self.config.temp_dir, process_id)
            os.makedirs(process_temp_dir, exist_ok=True)
            
            # Manage size
            self._manage_size()
            
            if self.config.enable_metrics:
                self._metrics["processes_created"] += 1
            
            if self.config.log_process_events:
                logger.debug(f"Created stateful process: {process_id}")
            return process_id
    
    def get_process(self, process_id: str) -> Optional[ProcessData]:
        """Get process data by ID"""
        with self._lock:
            return self._processes.get(process_id)
    
    def update_process(self, process_id: str, **kwargs) -> bool:
        """Update process data"""
        with self._lock:
            if process_id in self._processes:
                old_status = self._processes[process_id].status
                self._processes[process_id].update(**kwargs)
                new_status = self._processes[process_id].status
                
                # Update metrics
                if self.config.enable_metrics:
                    if new_status == ProcessState.COMPLETED and old_status != ProcessState.COMPLETED:
                        self._metrics["processes_completed"] += 1
                    elif new_status == ProcessState.FAILED and old_status != ProcessState.FAILED:
                        self._metrics["processes_failed"] += 1
                
                if self.config.log_process_events:
                    logger.debug(f"Updated process: {process_id} -> {new_status.value}")
                return True
            return False
    
    def remove_process(self, process_id: str) -> bool:
        """Remove a process"""
        with self._lock:
            if process_id in self._processes:
                process = self._processes[process_id]
                
                # Clean up files
                self._cleanup_process_files(process)
                
                # Clean up temp directory
                self._cleanup_process_temp_dir(process_id)
                
                del self._processes[process_id]
                if self.config.log_process_events:
                    logger.debug(f"Removed process: {process_id}")
                return True
            return False
    
    def list_processes(self, process_type: Optional[str] = None) -> Dict[str, ProcessData]:
        """List all processes"""
        with self._lock:
            if process_type:
                return {pid: data for pid, data in self._processes.items() 
                       if data.process_type == process_type}
            return dict(self._processes)
    
    def cleanup_old_processes(self) -> int:
        """Clean up old processes"""
        cutoff_time = time.time() - self.config.process_ttl
        processes_to_remove = []
        
        with self._lock:
            for process_id, process in self._processes.items():
                # Remove completed/failed processes that are old enough
                if (process.status in [ProcessState.COMPLETED, ProcessState.FAILED, ProcessState.CANCELLED] 
                    and process.completed_at and process.completed_at < cutoff_time):
                    processes_to_remove.append(process_id)
                # Remove very old processes regardless of status
                elif process.created_at < cutoff_time - 3600:  # 1 hour buffer
                    processes_to_remove.append(process_id)
        
        # Remove processes in batches
        for i in range(0, len(processes_to_remove), self.config.batch_cleanup_size):
            batch = processes_to_remove[i:i + self.config.batch_cleanup_size]
            for process_id in batch:
                self.remove_process(process_id)
                if self.config.log_process_events:
                    logger.info(f"Cleaned up expired process: {process_id}")
        
        return len(processes_to_remove)
    
    def _manage_size(self) -> None:
        """Manage the size of the process tracker"""
        if not self.config.enable_memory_optimization:
            return
            
        with self._lock:
            while len(self._processes) > self.config.max_processes:
                oldest_id, oldest_process = self._processes.popitem(last=False)
                self._cleanup_process_files(oldest_process)
                self._cleanup_process_temp_dir(oldest_id)
                if self.config.log_process_events:
                    logger.info(f"Removed oldest process due to size limit: {oldest_id}")
    
    def _cleanup_process_files(self, process: ProcessData) -> None:
        """Clean up files associated with a process"""
        for file_path in process.files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    if self.config.log_process_events:
                        logger.debug(f"Removed file: {file_path}")
            except Exception as e:
                logger.error(f"Error removing file {file_path}: {e}")
    
    def _cleanup_process_temp_dir(self, process_id: str) -> None:
        """Clean up process temp directory"""
        process_temp_dir = os.path.join(self.config.temp_dir, process_id)
        if os.path.exists(process_temp_dir):
            try:
                shutil.rmtree(process_temp_dir)
                if self.config.log_process_events:
                    logger.debug(f"Removed temp directory: {process_temp_dir}")
            except Exception as e:
                logger.error(f"Error removing temp directory {process_temp_dir}: {e}")


class StatelessProcessManager(ProcessManager):
    """
    Stateless process manager that doesn't maintain process state in memory.
    Suitable for distributed systems or when memory usage needs to be minimized.
    """
    
    def __init__(self, config: Optional[ProcessManagerConfig] = None):
        super().__init__(config)
        self._process_cache: Dict[str, ProcessData] = {}
        self._cache_ttl = self.config.cache_ttl
    
    def create_process(self, process_type: str, **kwargs) -> str:
        """Create a new process"""
        process_id = self.generate_process_id()
        
        # Create process data
        process_data = ProcessData(
            process_id=process_id,
            process_type=process_type,
            metadata=kwargs
        )
        
        # Create process-specific temp directory
        process_temp_dir = os.path.join(self.config.temp_dir, process_id)
        os.makedirs(process_temp_dir, exist_ok=True)
        
        # Store in cache
        self._process_cache[process_id] = process_data
        
        if self.config.enable_metrics:
            self._metrics["processes_created"] += 1
        
        if self.config.log_process_events:
            logger.debug(f"Created stateless process: {process_id}")
        return process_id
    
    def get_process(self, process_id: str) -> Optional[ProcessData]:
        """Get process data by ID"""
        # Check cache first
        if process_id in self._process_cache:
            return self._process_cache[process_id]
        
        # In a real stateless implementation, you would load from persistent storage
        # For this example, we'll return None if not in cache
        return None
    
    def update_process(self, process_id: str, **kwargs) -> bool:
        """Update process data"""
        if process_id in self._process_cache:
            old_status = self._process_cache[process_id].status
            self._process_cache[process_id].update(**kwargs)
            new_status = self._process_cache[process_id].status
            
            # Update metrics
            if self.config.enable_metrics:
                if new_status == ProcessState.COMPLETED and old_status != ProcessState.COMPLETED:
                    self._metrics["processes_completed"] += 1
                elif new_status == ProcessState.FAILED and old_status != ProcessState.FAILED:
                    self._metrics["processes_failed"] += 1
            
            if self.config.log_process_events:
                logger.debug(f"Updated process: {process_id} -> {new_status.value}")
            return True
        return False
    
    def remove_process(self, process_id: str) -> bool:
        """Remove a process"""
        if process_id in self._process_cache:
            process = self._process_cache[process_id]
            
            # Clean up files
            self._cleanup_process_files(process)
            
            # Clean up temp directory
            self._cleanup_process_temp_dir(process_id)
            
            del self._process_cache[process_id]
            if self.config.log_process_events:
                logger.debug(f"Removed process: {process_id}")
            return True
        return False
    
    def list_processes(self, process_type: Optional[str] = None) -> Dict[str, ProcessData]:
        """List all processes"""
        if process_type:
            return {pid: data for pid, data in self._process_cache.items() 
                   if data.process_type == process_type}
        return dict(self._process_cache)
    
    def cleanup_old_processes(self) -> int:
        """Clean up old processes from cache"""
        cutoff_time = time.time() - self.config.cache_ttl
        processes_to_remove = []
        
        for process_id, process in self._process_cache.items():
            if process.created_at < cutoff_time:
                processes_to_remove.append(process_id)
        
        for process_id in processes_to_remove:
            self.remove_process(process_id)
            if self.config.log_process_events:
                logger.debug(f"Cleaned up cached process: {process_id}")
        
        return len(processes_to_remove)
    
    def _cleanup_process_files(self, process: ProcessData) -> None:
        """Clean up files associated with a process"""
        for file_path in process.files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    if self.config.log_process_events:
                        logger.debug(f"Removed file: {file_path}")
            except Exception as e:
                logger.error(f"Error removing file {file_path}: {e}")
    
    def _cleanup_process_temp_dir(self, process_id: str) -> None:
        """Clean up process temp directory"""
        process_temp_dir = os.path.join(self.config.temp_dir, process_id)
        if os.path.exists(process_temp_dir):
            try:
                shutil.rmtree(process_temp_dir)
                if self.config.log_process_events:
                    logger.debug(f"Removed temp directory: {process_temp_dir}")
            except Exception as e:
                logger.error(f"Error removing temp directory {process_temp_dir}: {e}")


def create_process_manager(config: Optional[ProcessManagerConfig] = None) -> ProcessManager:
    """
    Factory function to create a ProcessManager instance based on settings configuration.
    
    Args:
        config: Optional ProcessManagerConfig. If None, loads from settings.
    
    Returns:
        ProcessManager instance
    """
    if config is None:
        config = ProcessManagerConfig()
    
    if config.stateful:
        return StatefulProcessManager(config)
    else:
        return StatelessProcessManager(config)


# Convenience function for creating the default process manager
def get_default_process_manager() -> ProcessManager:
    """Get the default process manager instance based on settings"""
    return create_process_manager()


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_process_manager():
        # Test with settings-based configuration
        print("Testing Settings-Based Process Manager:")
        
        # Create manager (will use settings configuration)
        manager = create_process_manager()
        
        # Create some processes
        process1 = manager.create_process("image_processing", input_file="test1.jpg")
        process2 = manager.create_process("data_analysis", dataset="data.csv")
        
        print(f"Created processes: {process1}, {process2}")
        print(f"Process count: {manager.get_process_count()}")
        print(f"Manager type: {'Stateful' if manager.config.stateful else 'Stateless'}")
        
        # Update processes
        manager.update_process(process1, status=ProcessState.RUNNING)
        manager.update_process(process2, status=ProcessState.COMPLETED)
        
        # List processes
        processes = manager.list_processes()
        for pid, process in processes.items():
            print(f"Process {pid}: {process.process_type} - {process.status.value}")
        
        # Get metrics
        metrics = manager.get_metrics()
        if metrics:
            print(f"Metrics: {metrics}")
        
        # Cleanup
        manager.shutdown()
    
    # Run the test
    asyncio.run(test_process_manager())
