#!/usr/bin/env python3
"""
RevOps Automation - Data Validation Service Runner
==================================================

Docker container service wrapper for data validation operations.
Provides REST API endpoints for triggering data quality checks and validation reports.

This script is called by the validation-service Docker container and provides
a lightweight FastAPI service for orchestrating data validation tasks.
"""

import os
import sys
import asyncio
import logging
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List

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
        logging.FileHandler('/app/logs/validation-service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# FastAPI application
app = FastAPI(
    title="RevOps Data Validation Service",
    description="REST API for orchestrating data quality checks and validation",
    version="1.0.0"
)

# Global state for tracking jobs
active_jobs: Dict[str, Dict[str, Any]] = {}
validation_reports: Dict[str, Dict[str, Any]] = {}


class ValidationRequest(BaseModel):
    """Request model for starting validation jobs"""
    validation_type: str  # 'basic', 'quality', 'integrity', 'all'
    schema: Optional[str] = None  # 'raw', 'core', or None for all
    tables: Optional[List[str]] = None  # specific tables or None for all


class QualityMetrics(BaseModel):
    """Response model for quality metrics"""
    total_records: int
    valid_records: int
    invalid_records: int
    quality_score: float
    checks_passed: int
    checks_failed: int


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "validation",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/status")
async def get_service_status():
    """Get overall service status and active jobs"""
    return {
        "service": "validation",
        "status": "running",
        "active_jobs": len(active_jobs),
        "jobs": list(active_jobs.keys()),
        "available_reports": len(validation_reports)
    }


@app.post("/validate/run")
async def run_validation(request: ValidationRequest, background_tasks: BackgroundTasks):
    """Start data validation checks"""
    job_id = f"validation_{request.validation_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # Register job
    active_jobs[job_id] = {
        "job_id": job_id,
        "validation_type": request.validation_type,
        "schema": request.schema,
        "tables": request.tables,
        "status": "running",
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "checks_passed": 0,
        "checks_failed": 0,
        "error_message": None
    }
    
    # Start background task
    background_tasks.add_task(run_validation_checks, job_id, request)
    
    return {"job_id": job_id, "status": "started", "message": f"{request.validation_type} validation job queued"}


@app.get("/validate/status/{job_id}")
async def get_validation_status(job_id: str):
    """Get status of a specific validation job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return active_jobs[job_id]


@app.get("/validate/report/{job_id}")
async def get_validation_report(job_id: str):
    """Get detailed validation report for a completed job"""
    if job_id not in validation_reports:
        if job_id in active_jobs and active_jobs[job_id]["status"] == "running":
            raise HTTPException(status_code=202, detail="Validation job still running")
        else:
            raise HTTPException(status_code=404, detail="Report not found")
    
    return validation_reports[job_id]


@app.get("/validate/metrics")
async def get_quality_metrics():
    """Get overall data quality metrics"""
    try:
        # This would typically query the ops.data_quality_checks table
        # For now, return a placeholder response
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_quality_score": 0.95,
            "total_records_checked": 0,
            "recent_checks": len([j for j in active_jobs.values() if j["status"] == "completed"]),
            "active_validations": len([j for j in active_jobs.values() if j["status"] == "running"]),
            "last_validation": max([j["completed_at"] for j in active_jobs.values() if j["completed_at"]], default=None)
        }
    except Exception as e:
        logger.error(f"Failed to get quality metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve quality metrics")


@app.post("/validate/schedule")
async def schedule_validation(validation_type: str = "quality"):
    """Schedule automatic validation (placeholder for future cron integration)"""
    return {
        "message": f"Scheduled {validation_type} validation",
        "next_run": "Not implemented - placeholder for future scheduling",
        "status": "scheduled"
    }


async def run_validation_checks(job_id: str, request: ValidationRequest):
    """Background task to run validation checks"""
    try:
        logger.info(f"Starting validation job {job_id} with type: {request.validation_type}")
        
        # Determine which validation script to run
        if request.validation_type == "basic":
            script_path = "/app/scripts/03-data/12_validate_data_quality.py"
            timeout = 900  # 15 minutes
        elif request.validation_type == "quality":
            script_path = "/app/scripts/03-data/13_run_quality_checks.py"
            timeout = 1800  # 30 minutes
        elif request.validation_type == "integrity":
            script_path = "/app/scripts/03-data/12_validate_data_quality.py"
            timeout = 1200  # 20 minutes
        elif request.validation_type == "all":
            # Run comprehensive validation
            await run_comprehensive_validation(job_id, request)
            return
        else:
            raise ValueError(f"Unknown validation type: {request.validation_type}")
        
        # Build command
        cmd = ["python", script_path]
        if request.schema:
            cmd.extend(["--schema", request.schema])
        if request.tables:
            cmd.extend(["--tables"] + request.tables)
        
        # Run validation script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            # Parse validation results
            checks_passed = 0
            checks_failed = 0
            quality_score = 0.0
            
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'checks passed' in line.lower():
                    try:
                        checks_passed = int(line.split()[-1])
                    except (ValueError, IndexError):
                        pass
                elif 'checks failed' in line.lower():
                    try:
                        checks_failed = int(line.split()[-1])
                    except (ValueError, IndexError):
                        pass
                elif 'quality score' in line.lower():
                    try:
                        # Look for percentage or decimal
                        words = line.split()
                        for word in words:
                            if '%' in word:
                                quality_score = float(word.replace('%', '')) / 100.0
                                break
                            elif '0.' in word and len(word) < 6:
                                quality_score = float(word)
                                break
                    except (ValueError, IndexError):
                        pass
            
            # Store validation report
            validation_reports[job_id] = {
                "job_id": job_id,
                "validation_type": request.validation_type,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "completed",
                "checks_passed": checks_passed,
                "checks_failed": checks_failed,
                "quality_score": quality_score,
                "detailed_output": result.stdout,
                "summary": f"Validation completed: {checks_passed} passed, {checks_failed} failed"
            }
            
            # Update job status
            active_jobs[job_id].update({
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "checks_passed": checks_passed,
                "checks_failed": checks_failed,
                "quality_score": quality_score
            })
            logger.info(f"Validation job {job_id} completed successfully")
        else:
            # Mark job failed
            error_msg = result.stderr[:500] if result.stderr else "Unknown error"
            active_jobs[job_id].update({
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "error_message": error_msg
            })
            logger.error(f"Validation job {job_id} failed: {error_msg}")
            
    except subprocess.TimeoutExpired:
        active_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.utcnow().isoformat(),
            "error_message": f"Job timed out after {timeout} seconds"
        })
        logger.error(f"Validation job {job_id} timed out")
    except Exception as e:
        active_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.utcnow().isoformat(),
            "error_message": str(e)[:500]
        })
        logger.error(f"Validation job {job_id} failed with exception: {e}")


async def run_comprehensive_validation(job_id: str, request: ValidationRequest):
    """Run all validation types in sequence"""
    try:
        logger.info(f"Starting comprehensive validation job {job_id}")
        
        total_passed = 0
        total_failed = 0
        validation_steps = []
        
        # List of validation scripts to run
        validation_scripts = [
            ("/app/scripts/03-data/12_validate_data_quality.py", "Basic Validation"),
            ("/app/scripts/03-data/13_run_quality_checks.py", "Quality Checks")
        ]
        
        for script_path, step_name in validation_scripts:
            try:
                logger.info(f"Running {step_name} for job {job_id}")
                
                cmd = ["python", script_path]
                if request.schema:
                    cmd.extend(["--schema", request.schema])
                if request.tables:
                    cmd.extend(["--tables"] + request.tables)
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 minutes per step
                )
                
                if result.returncode == 0:
                    # Parse results for this step
                    step_passed = 0
                    step_failed = 0
                    
                    for line in result.stdout.split('\n'):
                        if 'checks passed' in line.lower():
                            try:
                                step_passed = int(line.split()[-1])
                            except (ValueError, IndexError):
                                pass
                        elif 'checks failed' in line.lower():
                            try:
                                step_failed = int(line.split()[-1])
                            except (ValueError, IndexError):
                                pass
                    
                    total_passed += step_passed
                    total_failed += step_failed
                    
                    validation_steps.append({
                        "step": step_name,
                        "status": "completed",
                        "checks_passed": step_passed,
                        "checks_failed": step_failed,
                        "output": result.stdout[:1000]  # Truncated output
                    })
                    
                else:
                    validation_steps.append({
                        "step": step_name,
                        "status": "failed",
                        "error": result.stderr[:500]
                    })
                    logger.warning(f"{step_name} failed in comprehensive validation {job_id}")
                    
            except subprocess.TimeoutExpired:
                validation_steps.append({
                    "step": step_name,
                    "status": "timeout",
                    "error": "Step timed out after 30 minutes"
                })
                logger.warning(f"{step_name} timed out in comprehensive validation {job_id}")
        
        # Calculate overall quality score
        if total_passed + total_failed > 0:
            quality_score = total_passed / (total_passed + total_failed)
        else:
            quality_score = 0.0
        
        # Store comprehensive validation report
        validation_reports[job_id] = {
            "job_id": job_id,
            "validation_type": "comprehensive",
            "timestamp": datetime.utcnow().isoformat(),
            "status": "completed",
            "total_checks_passed": total_passed,
            "total_checks_failed": total_failed,
            "overall_quality_score": quality_score,
            "validation_steps": validation_steps,
            "summary": f"Comprehensive validation: {total_passed} passed, {total_failed} failed across {len(validation_steps)} steps"
        }
        
        # Update job status
        active_jobs[job_id].update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "checks_passed": total_passed,
            "checks_failed": total_failed,
            "quality_score": quality_score
        })
        logger.info(f"Comprehensive validation job {job_id} completed successfully")
        
    except Exception as e:
        active_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.utcnow().isoformat(),
            "error_message": str(e)[:500]
        })
        logger.error(f"Comprehensive validation job {job_id} failed with exception: {e}")


if __name__ == "__main__":
    # Get service configuration
    port = int(os.environ.get("SERVICE_PORT", 8003))
    host = os.environ.get("SERVICE_HOST", "0.0.0.0")
    log_level = os.environ.get("APP_LOG_LEVEL", "info").lower()
    
    logger.info(f"Starting RevOps Data Validation Service on {host}:{port}")
    
    # Run the service
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=True
    )