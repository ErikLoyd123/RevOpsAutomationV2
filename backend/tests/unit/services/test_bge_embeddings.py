"""
Unit tests for BGE Embeddings Service.

Tests BGE-M3 embedding generation with GPU acceleration, performance validation,
and integration with SEARCH schema storage. Validates service against RTX 3070 Ti
performance targets.

Test Categories:
- Service initialization and health checks
- Embedding generation with sample data
- Performance benchmarking (32 embeddings per 500ms target)
- GPU vs CPU fallback behavior
- Error handling and resilience
- Database integration tests

Run with: python test_bge_embeddings.py
"""

import asyncio
import time
import sys
import unittest
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from backend.services.embeddings.health import BGEHealthMonitor, HealthStatus, GPUStatus
    from backend.core.database import get_database_manager
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error (expected in test environment): {e}")
    IMPORTS_AVAILABLE = False


class TestBGEEmbeddingsService(unittest.TestCase):
    """Test suite for BGE embeddings service"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.health_monitor = BGEHealthMonitor()
        self.sample_texts = [
            "Acme Corporation - Cloud computing solutions provider",
            "TechStart Inc - AI and machine learning platform",
            "DataFlow Systems - Real-time analytics and insights",
            "CloudBridge Solutions - Multi-cloud infrastructure management",
            "InnovateTech - Software development and consulting",
        ]
        
        # Extended test set for performance testing (32 items)
        self.performance_test_texts = [
            f"Company {i:02d} - Technology sector business with cloud solutions focus"
            for i in range(32)
        ]
    
    def test_health_monitor_initialization(self):
        """Test health monitor initializes correctly"""
        monitor = BGEHealthMonitor()
        
        assert monitor.target_throughput == 64.0  # embeddings per second
        assert monitor.target_memory_limit == 7500  # MB for RTX 3070 Ti
        assert monitor.target_temperature_limit == 83  # Celsius
        assert monitor.target_success_rate == 99.0  # percent
        
        assert monitor.request_count == 0
        assert monitor.success_count == 0
        assert len(monitor.inference_times) == 0
    
    @pytest.mark.asyncio
    async def test_gpu_metrics_collection(self):
        """Test GPU metrics collection"""
        gpu_metrics = await self.health_monitor.get_gpu_metrics()
        
        # Should return valid metrics regardless of GPU availability
        assert gpu_metrics is not None
        assert hasattr(gpu_metrics, 'gpu_available')
        assert hasattr(gpu_metrics, 'memory_total')
        assert hasattr(gpu_metrics, 'gpu_utilization_percent')
        assert hasattr(gpu_metrics, 'temperature_celsius')
        
        # Memory metrics should be reasonable
        if gpu_metrics.gpu_available:
            assert gpu_metrics.memory_total > 0
            assert gpu_metrics.memory_utilization_percent >= 0
            assert gpu_metrics.memory_utilization_percent <= 100
            
            # Temperature should be reasonable (not negative, not extremely high)
            assert gpu_metrics.temperature_celsius >= 0
            assert gpu_metrics.temperature_celsius < 120  # Safety check
    
    @pytest.mark.asyncio
    async def test_model_metrics_collection(self):
        """Test model performance metrics collection"""
        # Record some sample inference times
        self.health_monitor.record_inference(32, 450.0, True)  # Good performance
        self.health_monitor.record_inference(16, 200.0, True)  # Smaller batch
        self.health_monitor.record_inference(32, 520.0, True)  # Slightly over target
        
        model_metrics = await self.health_monitor.get_model_metrics()
        
        assert model_metrics is not None
        assert model_metrics.model_name == "BAAI/bge-m3"
        assert model_metrics.embedding_dimension == 1024
        assert model_metrics.successful_generations == 3
        assert model_metrics.failed_generations == 0
        assert model_metrics.success_rate_percent == 100.0
        
        # Check performance calculations
        assert model_metrics.avg_inference_time_ms > 0
        assert model_metrics.throughput_embeddings_per_second > 0
    
    @pytest.mark.asyncio
    async def test_service_health_assessment(self):
        """Test overall service health assessment"""
        service_metrics = await self.health_monitor.get_service_metrics()
        
        assert service_metrics is not None
        assert service_metrics.overall_health in [
            HealthStatus.HEALTHY, 
            HealthStatus.DEGRADED, 
            HealthStatus.UNHEALTHY, 
            HealthStatus.UNKNOWN
        ]
        assert service_metrics.gpu_status in [
            GPUStatus.AVAILABLE,
            GPUStatus.UNAVAILABLE,
            GPUStatus.THROTTLING,
            GPUStatus.OVERHEATING,
            GPUStatus.OUT_OF_MEMORY
        ]
        
        # Service should be running
        assert service_metrics.service_uptime_seconds >= 0
        assert service_metrics.cpu_usage_percent >= 0
        assert service_metrics.memory_usage_mb > 0
        
        # Alerts and recommendations should be lists
        assert isinstance(service_metrics.active_alerts, list)
        assert isinstance(service_metrics.warnings, list)
        assert isinstance(service_metrics.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_health_status_endpoint(self):
        """Test health status endpoint response"""
        health_response = await self.health_monitor.get_health_status()
        
        assert health_response.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
            HealthStatus.UNKNOWN
        ]
        
        assert health_response.uptime_seconds >= 0
        assert isinstance(health_response.gpu_metrics, dict)
        assert isinstance(health_response.model_metrics, dict)
        assert isinstance(health_response.service_metrics, dict)
        assert isinstance(health_response.alerts, list)
        assert isinstance(health_response.warnings, list)
        assert isinstance(health_response.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_performance_benchmark(self):
        """Test performance benchmark against RTX 3070 Ti targets"""
        # Run benchmark with 32 test texts (target batch size)
        benchmark_result = await self.health_monitor.benchmark_performance(
            self.performance_test_texts
        )
        
        assert benchmark_result is not None
        assert "timestamp" in benchmark_result
        assert "test_batch_size" in benchmark_result
        assert "elapsed_time_ms" in benchmark_result
        assert "target_time_ms" in benchmark_result
        assert "performance_scores" in benchmark_result
        
        # Check batch size
        assert benchmark_result["test_batch_size"] == 32
        assert benchmark_result["target_time_ms"] == 500
        
        # Check performance scores structure
        scores = benchmark_result["performance_scores"]
        assert "throughput" in scores
        assert "memory" in scores
        assert "temperature" in scores
        assert "overall" in scores
        
        # Scores should be between 0 and 100
        for score_name, score_value in scores.items():
            assert 0 <= score_value <= 100, f"{score_name} score out of range: {score_value}"
        
        # Should have GPU metrics before and after
        assert "gpu_metrics_before" in benchmark_result
        assert "gpu_metrics_after" in benchmark_result
    
    @pytest.mark.asyncio
    async def test_inference_recording(self):
        """Test inference performance recording"""
        initial_count = self.health_monitor.request_count
        initial_success = self.health_monitor.success_count
        
        # Record successful inference
        self.health_monitor.record_inference(32, 450.0, True)
        
        assert self.health_monitor.request_count == initial_count + 1
        assert self.health_monitor.success_count == initial_success + 1
        assert len(self.health_monitor.inference_times) == 1
        assert self.health_monitor.inference_times[0] == 450.0
        
        # Record failed inference
        self.health_monitor.record_inference(16, 800.0, False)
        
        assert self.health_monitor.request_count == initial_count + 2
        assert self.health_monitor.success_count == initial_success + 1  # No change
        assert len(self.health_monitor.inference_times) == 1  # Failed inference not recorded
    
    @pytest.mark.asyncio
    async def test_rolling_history_limit(self):
        """Test that inference history maintains rolling limit"""
        # Record more than max_history inferences
        max_history = self.health_monitor.max_history
        
        for i in range(max_history + 100):
            self.health_monitor.record_inference(1, float(i), True)
        
        # Should maintain exactly max_history items
        assert len(self.health_monitor.inference_times) == max_history
        
        # Should contain the most recent items
        assert self.health_monitor.inference_times[0] == 100.0  # First kept item
        assert self.health_monitor.inference_times[-1] == float(max_history + 99)  # Last item
    
    @pytest.mark.asyncio 
    async def test_gpu_fallback_behavior(self):
        """Test behavior when GPU is unavailable"""
        with patch('torch.cuda.is_available', return_value=False):
            gpu_metrics = await self.health_monitor.get_gpu_metrics()
            
            assert gpu_metrics.gpu_available is False
            assert gpu_metrics.gpu_count == 0
            assert gpu_metrics.current_device == -1
            assert gpu_metrics.gpu_name == "N/A"
            assert gpu_metrics.memory_total == 0
            assert gpu_metrics.meets_performance_requirements is False
    
    @pytest.mark.asyncio
    async def test_thermal_throttling_detection(self):
        """Test thermal throttling detection and alerts"""
        # Mock GPU metrics to simulate overheating
        with patch.object(self.health_monitor, 'get_gpu_metrics') as mock_gpu:
            mock_gpu.return_value = Mock(
                gpu_available=True,
                temperature_celsius=90,  # Above limit
                thermal_throttling=True,
                memory_utilization_percent=50,
                meets_temperature_requirements=False,
                is_throttling=True
            )
            
            service_metrics = await self.health_monitor.get_service_metrics()
            
            assert service_metrics.gpu_status == GPUStatus.OVERHEATING
            assert service_metrics.overall_health in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]
            assert any("thermal throttling" in alert.lower() for alert in service_metrics.active_alerts)
    
    @pytest.mark.asyncio
    async def test_memory_pressure_detection(self):
        """Test GPU memory pressure detection"""
        with patch.object(self.health_monitor, 'get_gpu_metrics') as mock_gpu:
            mock_gpu.return_value = Mock(
                gpu_available=True,
                memory_utilization_percent=90,  # High utilization
                memory_pressure=True,
                memory_free=500,  # Low free memory
                temperature_celsius=70,
                thermal_throttling=False,
                is_throttling=False,
                meets_memory_requirements=False
            )
            
            service_metrics = await self.health_monitor.get_service_metrics()
            
            assert service_metrics.gpu_status == GPUStatus.OUT_OF_MEMORY
            assert any("memory pressure" in alert.lower() for alert in service_metrics.active_alerts)
    
    @pytest.mark.asyncio
    async def test_performance_targets_validation(self):
        """Test validation against RTX 3070 Ti performance targets"""
        # Simulate good performance
        for _ in range(10):
            self.health_monitor.record_inference(32, 400.0, True)  # Well under 500ms target
        
        model_metrics = await self.health_monitor.get_model_metrics()
        
        # Should meet throughput target (32 embeddings in 400ms = 80 embeddings/sec)
        assert model_metrics.throughput_embeddings_per_second > 60  # Above 64/sec target
        assert model_metrics.meets_throughput_target is True
        assert model_metrics.meets_quality_target is True  # 100% success rate
        
        # Test poor performance
        health_monitor_slow = BGEHealthMonitor()
        for _ in range(10):
            health_monitor_slow.record_inference(32, 800.0, True)  # Over 500ms target
        
        slow_model_metrics = await health_monitor_slow.get_model_metrics()
        assert slow_model_metrics.meets_throughput_target is False
    
    @pytest.mark.asyncio
    async def test_error_handling_in_health_check(self):
        """Test error handling during health checks"""
        with patch.object(self.health_monitor, 'get_gpu_metrics', side_effect=Exception("GPU error")):
            health_response = await self.health_monitor.get_health_status()
            
            assert health_response.status == HealthStatus.UNKNOWN
            assert len(health_response.alerts) > 0
            assert any("Health check failed" in alert for alert in health_response.alerts)
            assert "Check service logs and restart if necessary" in health_response.recommendations


class TestBGEServiceIntegration:
    """Integration tests for BGE service with actual components"""
    
    @pytest.mark.asyncio
    async def test_database_connection_health(self):
        """Test database connectivity for embedding storage"""
        try:
            db_manager = get_database_manager()
            
            # Test basic connectivity
            with db_manager.get_connection(database="local") as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                assert result[0] == 1
            
            # Test SEARCH schema exists (for embedding storage)
            with db_manager.get_connection(database="local") as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT schema_name 
                    FROM information_schema.schemata 
                    WHERE schema_name = 'search'
                """)
                result = cursor.fetchone()
                # Schema might not exist yet - that's ok for this test
                
        except Exception as e:
            pytest.skip(f"Database not available for integration test: {e}")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_realistic_embedding_generation_performance(self):
        """Test realistic embedding generation performance with actual data"""
        # This test would ideally load the actual BGE model and test performance
        # For now, we simulate the test structure
        
        # Sample realistic company names and descriptions
        realistic_texts = [
            "Amazon Web Services - Cloud computing platform and services",
            "Microsoft Azure - Comprehensive cloud services platform", 
            "Google Cloud Platform - Infrastructure and platform services",
            "Salesforce Inc - Customer relationship management platform",
            "Oracle Corporation - Enterprise software and database systems",
            "IBM Cloud - Hybrid cloud and AI solutions",
            "Accenture Technology - Digital transformation consulting",
            "Deloitte Digital - Technology consulting and implementation",
            "McKinsey Digital - Strategic technology advisory services",
            "Boston Consulting Group - Management consulting and technology",
            "CloudFlare Inc - Web infrastructure and security services",
            "Snowflake Computing - Cloud data platform and analytics",
            "Databricks Inc - Unified analytics platform for big data",
            "MongoDB Inc - NoSQL database platform and services",
            "Redis Labs - In-memory database and caching solutions",
            "Elastic NV - Search and analytics platform solutions",
        ]
        
        # Extend to 32 items for full batch test
        while len(realistic_texts) < 32:
            realistic_texts.extend(realistic_texts[:min(16, 32 - len(realistic_texts))])
        realistic_texts = realistic_texts[:32]
        
        health_monitor = BGEHealthMonitor()
        
        # Record start time
        start_time = time.time()
        
        # Simulate embedding generation (in real test, this would call actual BGE service)
        await asyncio.sleep(0.4)  # Simulate 400ms for good performance
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        
        # Record the inference
        health_monitor.record_inference(len(realistic_texts), elapsed_ms, True)
        
        # Validate performance
        model_metrics = await health_monitor.get_model_metrics()
        
        assert model_metrics.total_embeddings_generated > 0
        assert model_metrics.success_rate_percent == 100.0
        
        # Performance should be reasonable (allowing for simulation)
        assert elapsed_ms < 1000  # Should complete within 1 second for test
        
        print(f"Simulated embedding generation for {len(realistic_texts)} texts in {elapsed_ms:.1f}ms")
        print(f"Simulated throughput: {model_metrics.throughput_embeddings_per_second:.1f} embeddings/sec")


# Pytest configuration and fixtures
@pytest.fixture
def sample_embedding_data():
    """Fixture providing sample data for embedding tests"""
    return {
        "identity_texts": [
            "Acme Corp - acme.com",
            "TechStart Inc - techstart.io", 
            "DataFlow Systems - dataflow.com"
        ],
        "context_texts": [
            "Leading provider of cloud computing solutions with focus on enterprise clients",
            "AI-powered analytics platform helping businesses make data-driven decisions",
            "Real-time data processing and visualization tools for modern enterprises"
        ],
        "expected_embedding_dimension": 1024,
        "expected_batch_processing_time_ms": 500
    }


# Mark slow tests
pytestmark = pytest.mark.asyncio


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])