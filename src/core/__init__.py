"""
Core module for the FastAPI REST API Template
"""

from .process_manager import ProcessManager, ProcessManagerConfig

__all__ = ["ProcessManager", "ProcessManagerConfig"]