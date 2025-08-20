#!/usr/bin/env python3
"""
RevOps Automation - Data Transformation Service Runner
======================================================

Docker container service wrapper for data transformation operations.
Provides REST API endpoints for triggering opportunity normalization and AWS account processing.

This script is called by the transformation-service Docker container and provides
a lightweight FastAPI service for orchestrating data transformation tasks.
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

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.environ.get('APP_LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/transformation-service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# FastAPI application
app = FastAPI(
    title="RevOps Data Transformation Service",
    description="REST API for orchestrating data transformation from RAW to CORE schema",
    version="1.0.0"
)

# Global state for tracking jobs
active_jobs: Dict[str, Dict[str, Any]] = {}


class TransformRequest(BaseModel):
    """Request model for starting transformation jobs"""
    transform_type: str  # 'opportunities', 'aws_accounts', 'all'
    force_rebuild: bool = False


class JobStatus(BaseModel):
    """Response model for job status"""
    job_id: str
    transform_type: str
    status: str  # 'running', 'completed', 'failed'
    started_at: str
    completed_at: Optional[str] = None
    records_processed: int = 0
    error_message: Optional[str] = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "transformation",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/status")
async def get_service_status():
    """Get overall service status and active jobs"""
    return {
        "service": "transformation",
        "status": "running",
        "active_jobs": len(active_jobs),
        "jobs": list(active_jobs.keys())
    }


@app.post("/transform/opportunities")
async def transform_opportunities(request: TransformRequest, background_tasks: BackgroundTasks):
    """Transform opportunity data from RAW to CORE schema"""
    if request.transform_type not in ['opportunities', 'all']:
        raise HTTPException(status_code=400, detail="Invalid transform_type for opportunities endpoint")
    
    job_id = f"opportunities_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # Register job
    active_jobs[job_id] = {
        "job_id": job_id,
        "transform_type": "opportunities",
        "status": "running",
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "records_processed": 0,
        "error_message": None
    }
    
    # Start background task
    background_tasks.add_task(run_opportunity_transformation, job_id, request.force_rebuild)
    
    return {"job_id": job_id, "status": "started", "message": "Opportunity transformation job queued"}


@app.post("/transform/aws_accounts")
async def transform_aws_accounts(request: TransformRequest, background_tasks: BackgroundTasks):
    """Transform AWS account data from RAW to CORE schema"""
    if request.transform_type not in ['aws_accounts', 'all']:
        raise HTTPException(status_code=400, detail="Invalid transform_type for AWS accounts endpoint")
    
    job_id = f"aws_accounts_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # Register job
    active_jobs[job_id] = {
        "job_id": job_id,
        "transform_type": "aws_accounts",
        "status": "running",
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "records_processed": 0,
        "error_message": None
    }
    
    # Start background task
    background_tasks.add_task(run_aws_account_transformation, job_id, request.force_rebuild)
    
    return {"job_id": job_id, "status": "started", "message": "AWS account transformation job queued"}


@app.post("/transform/all")
async def transform_all(request: TransformRequest, background_tasks: BackgroundTasks):
    """Transform all data types from RAW to CORE schema"""
    job_id = f"all_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # Register job
    active_jobs[job_id] = {
        "job_id": job_id,
        "transform_type": "all",
        "status": "running",
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "records_processed": 0,
        "error_message": None
    }
    
    # Start background task
    background_tasks.add_task(run_full_transformation, job_id, request.force_rebuild)
    
    return {"job_id": job_id, "status": "started", "message": "Full transformation job queued"}


@app.get("/transform/status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific transformation job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return active_jobs[job_id]


async def run_opportunity_transformation(job_id: str, force_rebuild: bool):
    """Background task to run opportunity transformation"""
    try:
        logger.info(f"Starting opportunity transformation job {job_id}")
        
        # Build command
        cmd = ["python", "/app/scripts/03-data/10_normalize_opportunities.py"]
        if force_rebuild:
            cmd.append("--force-rebuild")
        
        # Run transformation script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        if result.returncode == 0:
            # Parse output for record count if available
            records_processed = 0
            for line in result.stdout.split('\n'):
                if 'opportunities processed' in line.lower() or 'records processed' in line.lower():
                    try:
                        # Look for numbers in the line
                        words = line.split()
                        for word in words:
                            if word.isdigit():
                                records_processed = int(word)
                                break
                    except (ValueError, IndexError):
                        pass
            
            # Mark job completed
            active_jobs[job_id].update({
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "records_processed": records_processed
            })
            logger.info(f"Opportunity transformation job {job_id} completed successfully")
        else:
            # Mark job failed
            active_jobs[job_id].update({
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "error_message": result.stderr[:500] if result.stderr else "Unknown error"
            })
            logger.error(f"Opportunity transformation job {job_id} failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        active_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.utcnow().isoformat(),
            "error_message": "Job timed out after 30 minutes"
        })
        logger.error(f"Opportunity transformation job {job_id} timed out")
    except Exception as e:
        active_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.utcnow().isoformat(),
            "error_message": str(e)[:500]
        })
        logger.error(f"Opportunity transformation job {job_id} failed with exception: {e}")


async def run_aws_account_transformation(job_id: str, force_rebuild: bool):
    """Background task to run AWS account transformation"""
    try:
        logger.info(f"Starting AWS account transformation job {job_id}")
        
        # Build command
        cmd = ["python", "/app/scripts/03-data/11_normalize_aws_accounts.py"]
        if force_rebuild:
            cmd.append("--force-rebuild")
        
        # Run transformation script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        if result.returncode == 0:
            # Parse output for record count if available
            records_processed = 0
            for line in result.stdout.split('\n'):
                if 'accounts processed' in line.lower() or 'records processed' in line.lower():
                    try:
                        # Look for numbers in the line
                        words = line.split()
                        for word in words:
                            if word.isdigit():
                                records_processed = int(word)
                                break
                    except (ValueError, IndexError):
                        pass
            
            # Mark job completed
            active_jobs[job_id].update({
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "records_processed": records_processed
            })
            logger.info(f"AWS account transformation job {job_id} completed successfully")
        else:
            # Mark job failed
            active_jobs[job_id].update({
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "error_message": result.stderr[:500] if result.stderr else "Unknown error"
            })
            logger.error(f"AWS account transformation job {job_id} failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        active_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.utcnow().isoformat(),
            "error_message": "Job timed out after 30 minutes"
        })
        logger.error(f"AWS account transformation job {job_id} timed out")
    except Exception as e:
        active_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.utcnow().isoformat(),
            "error_message": str(e)[:500]
        })
        logger.error(f"AWS account transformation job {job_id} failed with exception: {e}")


async def run_full_transformation(job_id: str, force_rebuild: bool):
    """Background task to run full transformation (opportunities + AWS accounts)"""
    try:
        logger.info(f"Starting full transformation job {job_id}")
        
        total_records = 0
        
        # Run opportunity transformation first
        cmd = ["python", "/app/scripts/03-data/10_normalize_opportunities.py"]
        if force_rebuild:
            cmd.append("--force-rebuild")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            raise Exception(f"Opportunity transformation failed: {result.stderr}")
        
        # Parse opportunity records
        for line in result.stdout.split('\n'):
            if 'opportunities processed' in line.lower():
                try:
                    words = line.split()
                    for word in words:
                        if word.isdigit():
                            total_records += int(word)
                            break
                except (ValueError, IndexError):
                    pass
        
        # Run AWS account transformation
        cmd = ["python", "/app/scripts/03-data/11_normalize_aws_accounts.py"]
        if force_rebuild:
            cmd.append("--force-rebuild")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            raise Exception(f"AWS account transformation failed: {result.stderr}")
        
        # Parse AWS account records
        for line in result.stdout.split('\n'):
            if 'accounts processed' in line.lower():
                try:
                    words = line.split()
                    for word in words:
                        if word.isdigit():
                            total_records += int(word)
                            break
                except (ValueError, IndexError):
                    pass
        
        # Mark job completed
        active_jobs[job_id].update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "records_processed": total_records
        })
        logger.info(f"Full transformation job {job_id} completed successfully")
        
    except subprocess.TimeoutExpired:
        active_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.utcnow().isoformat(),
            "error_message": "Job timed out"
        })
        logger.error(f"Full transformation job {job_id} timed out")
    except Exception as e:
        active_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.utcnow().isoformat(),
            "error_message": str(e)[:500]
        })
        logger.error(f"Full transformation job {job_id} failed with exception: {e}")


if __name__ == "__main__":
    # Get service configuration
    port = int(os.environ.get("SERVICE_PORT", 8002))
    host = os.environ.get("SERVICE_HOST", "0.0.0.0")
    log_level = os.environ.get("APP_LOG_LEVEL", "info").lower()
    
    logger.info(f"Starting RevOps Data Transformation Service on {host}:{port}")
    
    # Run the service
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=True
    )