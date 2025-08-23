#!/usr/bin/env python3
"""
Startup script for Enhanced Matching Engine.

This script provides a simple way to start the matching service
for development and testing purposes.
"""

import sys
import os
import asyncio
import uvicorn
import structlog

# Add the backend/core directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

from app import app

logger = structlog.get_logger(__name__)

def main():
    """Main startup function"""
    try:
        logger.info("Starting Enhanced Matching Engine",
                   service="enhanced-matching",
                   version="1.0.0",
                   port=8008)
        
        # Run the FastAPI application
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8008,
            log_level="info",
            access_log=True,
            reload=False  # Set to True for development
        )
        
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error("Service startup failed", error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()