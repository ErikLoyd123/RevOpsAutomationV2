#!/usr/bin/env python3
"""
RevOps Automation - Data Ingestion Service Runner
=================================================

Docker container service wrapper for data ingestion operations.
Provides REST API endpoints for triggering Odoo and APN data extraction.

This script is called by the ingestion-service Docker container and provides
a lightweight FastAPI service for orchestrating data ingestion tasks.
"""

import os
import sys
import asyncio
import logging
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional

# Add project paths
sys.path.append('/app')
sys.path.append('/app/core')
sys.path.append('/app/scripts')

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import our core modules
try:
    from backend.core.config import get_config
    from backend.core.database import DatabaseManager
except ImportError as e:
    print(f"Warning: Could not import core modules: {e}")
    print("Running in minimal mode without database connectivity")

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.environ.get('APP_LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/ingestion-service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# FastAPI application
app = FastAPI(
    title="RevOps Data Ingestion Service",
    description="REST API for orchestrating data ingestion from Odoo and APN systems",
    version="1.0.0"
)

# Global state for tracking jobs
active_jobs: Dict[str, Dict[str, Any]] = {}


class JobRequest(BaseModel):
    """Request model for starting ingestion jobs"""
    source: str  # 'odoo' or 'apn'
    full_sync: bool = True
    tables: Optional[list] = None


class JobStatus(BaseModel):
    """Response model for job status"""
    job_id: str
    source: str
    status: str  # 'running', 'completed', 'failed'
    started_at: str
    completed_at: Optional[str] = None
    records_processed: int = 0
    error_message: Optional[str] = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Basic health check - can be enhanced with database connectivity
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "ingestion",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/status")
async def get_service_status():
    """Get overall service status and active jobs"""
    return {
        "service": "ingestion",
        "status": "running",
        "active_jobs": len(active_jobs),
        "jobs": list(active_jobs.keys())
    }


@app.post("/ingestion/odoo/sync")
async def start_odoo_sync(request: JobRequest, background_tasks: BackgroundTasks):
    """Start Odoo data synchronization"""
    if request.source != 'odoo':
        raise HTTPException(status_code=400, detail="Source must be 'odoo' for this endpoint")
    
    job_id = f"odoo_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # Register job
    active_jobs[job_id] = {
        "job_id": job_id,
        "source": "odoo",
        "status": "running",
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "records_processed": 0,
        "error_message": None
    }
    
    # Start background task
    background_tasks.add_task(run_odoo_extraction, job_id, request.full_sync, request.tables)
    
    return {"job_id": job_id, "status": "started", "message": "Odoo sync job queued"}


@app.post("/ingestion/apn/sync")
async def start_apn_sync(request: JobRequest, background_tasks: BackgroundTasks):
    """Start APN data synchronization"""
    if request.source != 'apn':
        raise HTTPException(status_code=400, detail="Source must be 'apn' for this endpoint")
    
    job_id = f"apn_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # Register job
    active_jobs[job_id] = {
        "job_id": job_id,
        "source": "apn",
        "status": "running",
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "records_processed": 0,
        "error_message": None
    }
    
    # Start background task
    background_tasks.add_task(run_apn_extraction, job_id, request.full_sync, request.tables)
    
    return {"job_id": job_id, "status": "started", "message": "APN sync job queued"}


@app.get("/ingestion/status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific ingestion job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return active_jobs[job_id]


async def run_odoo_extraction(job_id: str, full_sync: bool, tables: Optional[list]):
    """Background task to run Odoo data extraction"""
    try:
        logger.info(f"Starting Odoo extraction job {job_id}")
        
        # Build command
        cmd = ["python", "/app/scripts/03-data/08_extract_odoo_data.py"]
        if not full_sync:
            cmd.append("--incremental")
        if tables:
            cmd.extend(["--tables"] + tables)
        
        # Run extraction script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            # Parse output for record count if available
            records_processed = 0
            for line in result.stdout.split('\n'):
                if 'records processed' in line.lower():
                    try:
                        records_processed = int(line.split()[-1])
                        break
                    except (ValueError, IndexError):
                        pass
            
            # Mark job completed
            active_jobs[job_id].update({
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "records_processed": records_processed
            })
            logger.info(f"Odoo extraction job {job_id} completed successfully")
        else:
            # Mark job failed
            active_jobs[job_id].update({
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "error_message": result.stderr[:500] if result.stderr else "Unknown error"
            })
            logger.error(f"Odoo extraction job {job_id} failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        active_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.utcnow().isoformat(),
            "error_message": "Job timed out after 1 hour"
        })
        logger.error(f"Odoo extraction job {job_id} timed out")
    except Exception as e:
        active_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.utcnow().isoformat(),
            "error_message": str(e)[:500]
        })
        logger.error(f"Odoo extraction job {job_id} failed with exception: {e}")


async def run_apn_extraction(job_id: str, full_sync: bool, tables: Optional[list]):
    """Background task to run APN data extraction"""
    try:
        logger.info(f"Starting APN extraction job {job_id}")
        
        # Build command
        cmd = ["python", "/app/scripts/03-data/09_extract_apn_data.py"]
        if not full_sync:
            cmd.append("--incremental")
        if tables:
            cmd.extend(["--tables"] + tables)
        
        # Run extraction script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            # Parse output for record count if available
            records_processed = 0
            for line in result.stdout.split('\n'):
                if 'records processed' in line.lower():
                    try:
                        records_processed = int(line.split()[-1])
                        break
                    except (ValueError, IndexError):
                        pass
            
            # Mark job completed
            active_jobs[job_id].update({
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "records_processed": records_processed
            })
            logger.info(f"APN extraction job {job_id} completed successfully")
        else:
            # Mark job failed
            active_jobs[job_id].update({
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "error_message": result.stderr[:500] if result.stderr else "Unknown error"
            })
            logger.error(f"APN extraction job {job_id} failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        active_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.utcnow().isoformat(),
            "error_message": "Job timed out after 1 hour"
        })
        logger.error(f"APN extraction job {job_id} timed out")
    except Exception as e:
        active_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.utcnow().isoformat(),
            "error_message": str(e)[:500]
        })
        logger.error(f"APN extraction job {job_id} failed with exception: {e}")


if __name__ == "__main__":
    # Get service configuration
    port = int(os.environ.get("SERVICE_PORT", 8001))
    host = os.environ.get("SERVICE_HOST", "0.0.0.0")
    log_level = os.environ.get("APP_LOG_LEVEL", "info").lower()
    
    logger.info(f"Starting RevOps Data Ingestion Service on {host}:{port}")
    
    # Run the service
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=True
    )