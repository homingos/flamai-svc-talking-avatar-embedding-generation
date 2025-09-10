"""
Manager utility functions for accessing server and process managers from FastAPI routes.

This module provides convenient functions to access the server manager and process manager
instances that are stored in the FastAPI application state during the lifespan events.
"""

from typing import Optional
from fastapi import Request, HTTPException

from src.core.server_manager import ServerManager
from src.core.process_manager import ProcessManager
from src.utils.resources.logger import logger


def get_server_manager(request: Request) -> Optional[ServerManager]:
    """
    Get the server manager instance from the FastAPI request state.
    
    Args:
        request: FastAPI Request object containing the application state
        
    Returns:
        ServerManager instance if available, None otherwise
        
    Raises:
        HTTPException: If server manager is not available
    """
    try:
        server_manager = getattr(request.app.state, 'server_manager', None)
        if server_manager is None:
            logger.warning("Server manager not found in application state")
            raise HTTPException(
                status_code=503, 
                detail="Server manager not available"
            )
        return server_manager
    except AttributeError as e:
        logger.error(f"Error accessing server manager: {e}")
        raise HTTPException(
            status_code=503, 
            detail="Server manager not available"
        )


def get_process_manager(request: Request) -> Optional[ProcessManager]:
    """
    Get the process manager instance from the FastAPI request state.
    
    Args:
        request: FastAPI Request object containing the application state
        
    Returns:
        ProcessManager instance if available, None otherwise
        
    Raises:
        HTTPException: If process manager is not available
    """
    try:
        process_manager = getattr(request.app.state, 'process_manager', None)
        if process_manager is None:
            logger.warning("Process manager not found in application state")
            raise HTTPException(
                status_code=503, 
                detail="Process manager not available"
            )
        return process_manager
    except AttributeError as e:
        logger.error(f"Error accessing process manager: {e}")
        raise HTTPException(
            status_code=503, 
            detail="Process manager not available"
        )
