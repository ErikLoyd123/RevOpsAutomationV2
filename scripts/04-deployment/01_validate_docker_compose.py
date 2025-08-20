#!/usr/bin/env python3
"""
Docker Compose Configuration Validator for RevOps Automation Platform

This script validates the docker-compose.yml configuration and ensures all
required dependencies and configurations are properly set up.

Usage:
    python scripts/04-deployment/01_validate_docker_compose.py
    
Features:
- Validates docker-compose.yml syntax
- Checks for required environment variables
- Validates network configuration
- Verifies volume mounts and permissions
- Tests GPU runtime configuration
- Validates service dependencies
"""

import os
import sys
import json
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


class DockerComposeValidator:
    """Validates Docker Compose configuration for RevOps platform."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.compose_file = project_root / "docker-compose.yml"
        self.env_example = project_root / ".env.example"
        self.env_file = project_root / ".env"
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate_all(self) -> bool:
        """Run all validation checks."""
        print("🔍 Validating Docker Compose configuration...")
        
        # Core file validation
        self._validate_compose_file_exists()
        self._validate_compose_syntax()
        self._validate_env_files()
        
        # Configuration validation
        self._validate_services()
        self._validate_networks()
        self._validate_volumes()
        self._validate_gpu_configuration()
        self._validate_dependencies()
        
        # Runtime validation
        self._validate_docker_requirements()
        
        # Report results
        self._report_results()
        
        return len(self.errors) == 0
    
    def _validate_compose_file_exists(self):
        """Check if docker-compose.yml exists."""
        if not self.compose_file.exists():
            self.errors.append(f"❌ docker-compose.yml not found at {self.compose_file}")
            return
        print(f"✅ Found docker-compose.yml at {self.compose_file}")
    
    def _validate_compose_syntax(self):
        """Validate YAML syntax of docker-compose.yml."""
        try:
            with open(self.compose_file, 'r') as f:
                self.compose_config = yaml.safe_load(f)
            print("✅ docker-compose.yml has valid YAML syntax")
        except yaml.YAMLError as e:
            self.errors.append(f"❌ Invalid YAML syntax in docker-compose.yml: {e}")
            return
        except Exception as e:
            self.errors.append(f"❌ Error reading docker-compose.yml: {e}")
            return
    
    def _validate_env_files(self):
        """Validate environment file configuration."""
        if not self.env_example.exists():
            self.warnings.append(f"⚠️  .env.example not found at {self.env_example}")
        else:
            print("✅ Found .env.example file")
        
        if not self.env_file.exists():
            self.warnings.append(f"⚠️  .env file not found. Copy from .env.example")
            print("💡 Run: cp .env.example .env")
        else:
            print("✅ Found .env file")
    
    def _validate_services(self):
        """Validate service configurations."""
        if not hasattr(self, 'compose_config'):
            return
        
        services = self.compose_config.get('services', {})
        required_services = ['postgres', 'redis']
        data_processing_services = ['ingestion-service', 'transformation-service', 'validation-service']
        
        for service in required_services:
            if service not in services:
                self.errors.append(f"❌ Required service '{service}' not found")
            else:
                print(f"✅ Found required service: {service}")
        
        # Validate data processing services for database infrastructure phase
        for service in data_processing_services:
            if service not in services:
                self.errors.append(f"❌ Required data processing service '{service}' not found")
            else:
                print(f"✅ Found data processing service: {service}")
        
        # Validate BGE service configurations (optional for future)
        bge_services = ['bge-service', 'bge-service-cpu']
        bge_found = any(svc in services for svc in bge_services)
        
        if not bge_found:
            self.warnings.append("⚠️  No BGE service configuration found (not required for database infrastructure phase)")
        else:
            print("✅ Found BGE service configurations")
        
        # Check for proper profiles
        profiles_found = set()
        for service_name, service_config in services.items():
            profiles = service_config.get('profiles', [])
            profiles_found.update(profiles)
        
        expected_profiles = {'dev', 'gpu', 'prod', 'full', 'cpu', 'monitoring'}
        missing_profiles = expected_profiles - profiles_found
        
        if missing_profiles:
            self.warnings.append(f"⚠️  Missing profiles: {missing_profiles}")
        else:
            print("✅ All expected profiles are configured")
    
    def _validate_networks(self):
        """Validate network configuration."""
        if not hasattr(self, 'compose_config'):
            return
        
        networks = self.compose_config.get('networks', {})
        
        if 'revops-network' not in networks:
            self.errors.append("❌ Required network 'revops-network' not found")
        else:
            network_config = networks['revops-network']
            if network_config.get('driver') != 'bridge':
                self.warnings.append("⚠️  Network driver should be 'bridge'")
            
            # Check subnet configuration
            ipam = network_config.get('ipam', {})
            config = ipam.get('config', [])
            if not config or not any('subnet' in c for c in config):
                self.warnings.append("⚠️  Network subnet not configured")
            else:
                print("✅ Network configuration looks good")
    
    def _validate_volumes(self):
        """Validate volume configuration."""
        if not hasattr(self, 'compose_config'):
            return
        
        volumes = self.compose_config.get('volumes', {})
        required_volumes = ['postgres_data', 'bge_model_cache', 'app_logs', 'redis_data']
        
        for volume in required_volumes:
            if volume not in volumes:
                self.errors.append(f"❌ Required volume '{volume}' not found")
            else:
                print(f"✅ Found required volume: {volume}")
    
    def _validate_gpu_configuration(self):
        """Validate GPU configuration for BGE service."""
        if not hasattr(self, 'compose_config'):
            return
        
        services = self.compose_config.get('services', {})
        bge_service = services.get('bge-service', {})
        
        if not bge_service:
            self.warnings.append("⚠️  BGE GPU service not configured")
            return
        
        # Check runtime configuration
        runtime = bge_service.get('runtime')
        if runtime != 'nvidia':
            self.errors.append("❌ BGE service missing 'nvidia' runtime")
        else:
            print("✅ BGE service has nvidia runtime configured")
        
        # Check environment variables
        env = bge_service.get('environment', {})
        gpu_env_vars = [
            'NVIDIA_VISIBLE_DEVICES',
            'NVIDIA_DRIVER_CAPABILITIES',
            'CUDA_VISIBLE_DEVICES'
        ]
        
        for var in gpu_env_vars:
            if var not in env:
                self.warnings.append(f"⚠️  Missing GPU environment variable: {var}")
            else:
                print(f"✅ GPU environment variable configured: {var}")
        
        # Check deploy configuration
        deploy = bge_service.get('deploy', {})
        resources = deploy.get('resources', {})
        reservations = resources.get('reservations', {})
        devices = reservations.get('devices', [])
        
        gpu_device_found = any(
            device.get('driver') == 'nvidia' and 'gpu' in device.get('capabilities', [])
            for device in devices
        )
        
        if not gpu_device_found:
            self.warnings.append("⚠️  GPU device reservation not properly configured")
        else:
            print("✅ GPU device reservation configured")
    
    def _validate_dependencies(self):
        """Validate service dependencies."""
        if not hasattr(self, 'compose_config'):
            return
        
        services = self.compose_config.get('services', {})
        dependency_issues = []
        
        for service_name, service_config in services.items():
            depends_on = service_config.get('depends_on', {})
            
            for dep_service, dep_config in depends_on.items():
                if dep_service not in services:
                    dependency_issues.append(
                        f"Service '{service_name}' depends on '{dep_service}' which doesn't exist"
                    )
                
                # Check if dependency has health check when condition is specified
                if isinstance(dep_config, dict) and dep_config.get('condition') == 'service_healthy':
                    dep_service_config = services.get(dep_service, {})
                    if 'healthcheck' not in dep_service_config:
                        dependency_issues.append(
                            f"Service '{dep_service}' needs healthcheck for condition 'service_healthy'"
                        )
        
        if dependency_issues:
            for issue in dependency_issues:
                self.errors.append(f"❌ Dependency issue: {issue}")
        else:
            print("✅ Service dependencies are properly configured")
    
    def _validate_docker_requirements(self):
        """Validate Docker and Docker Compose installation."""
        # Check Docker
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ Docker is installed: {result.stdout.strip()}")
            else:
                self.errors.append("❌ Docker is not properly installed")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.errors.append("❌ Docker command not found or not responding")
        
        # Check Docker Compose
        try:
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ Docker Compose is installed: {result.stdout.strip()}")
            else:
                # Try newer docker compose syntax
                result = subprocess.run(['docker', 'compose', 'version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"✅ Docker Compose is installed: {result.stdout.strip()}")
                else:
                    self.errors.append("❌ Docker Compose is not properly installed")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.errors.append("❌ Docker Compose command not found")
        
        # Check NVIDIA Docker runtime (if available)
        try:
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ NVIDIA GPU detected")
                
                # Check nvidia-docker
                result = subprocess.run(['docker', 'run', '--rm', '--gpus', 'all', 
                                       'nvidia/cuda:12.1-base-ubuntu22.04', 'nvidia-smi'], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print("✅ NVIDIA Docker runtime is working")
                else:
                    self.warnings.append("⚠️  NVIDIA Docker runtime may not be properly configured")
            else:
                self.warnings.append("⚠️  NVIDIA GPU not detected (CPU fallback will be used)")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.warnings.append("⚠️  NVIDIA tools not found (CPU fallback will be used)")
    
    def _report_results(self):
        """Report validation results."""
        print("\n" + "="*80)
        print("🔍 DOCKER COMPOSE VALIDATION RESULTS")
        print("="*80)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if not self.errors and not self.warnings:
            print("\n🎉 All validations passed! Configuration looks great.")
        elif not self.errors:
            print(f"\n✅ Validation passed with {len(self.warnings)} warnings.")
        else:
            print(f"\n❌ Validation failed with {len(self.errors)} errors and {len(self.warnings)} warnings.")
        
        print("\n💡 Quick Start Commands:")
        print("  # Development (CPU-only):")
        print("  docker-compose --profile dev up")
        print("\n  # Development with GPU:")
        print("  docker-compose --profile gpu up")
        print("\n  # Full stack with monitoring:")
        print("  docker-compose --profile dev --profile monitoring --profile full up")
        print("\n  # Production:")
        print("  docker-compose --profile prod up -d")
        
        print("\n📚 Service URLs (when running):")
        print("  - API Gateway: http://localhost:8000")
        print("  - BGE Embeddings: http://localhost:8007")
        print("  - PgAdmin: http://localhost:5050")
        print("  - Redis Commander: http://localhost:8081")
        print("="*80)


def main():
    """Main validation function."""
    # Determine project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    
    print(f"🚀 RevOps Automation Platform - Docker Compose Validator")
    print(f"📁 Project root: {project_root}")
    print()
    
    # Run validation
    validator = DockerComposeValidator(project_root)
    success = validator.validate_all()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()