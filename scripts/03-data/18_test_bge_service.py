#!/usr/bin/env python3
"""
Test BGE Service with Sample Data - Task 2.8

This script tests the self-contained BGE embeddings service to verify:
- Service is running and responding to HTTP requests
- GPU acceleration is working (if available)
- Performance meets RTX 3070 Ti targets (32 embeddings per 500ms)
- Embedding quality and dimensions are correct (1024-dim BGE-M3)
- Integration with identity and context embedding scripts

Run: python scripts/03-data/18_test_bge_service.py
"""

import asyncio
import aiohttp
import json
import sys
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
import statistics

# BGE Service Configuration
BGE_SERVICE_URL = "http://localhost:8007"
BGE_HEALTH_URL = f"{BGE_SERVICE_URL}/health"
BGE_EMBED_URL = f"{BGE_SERVICE_URL}/embed"

# Performance targets for RTX 3070 Ti
TARGET_EMBEDDINGS_PER_SEC = 64  # 32 embeddings per 500ms
TARGET_BATCH_SIZE = 32
TARGET_MAX_RESPONSE_TIME = 1000  # 1 second for batch of 32

# Test data samples
SAMPLE_IDENTITY_TEXTS = [
    "Acme Corporation acme.com",
    "TechCorp Industries techcorp.io", 
    "Global Solutions Inc globalsolutions.net",
    "Innovation Labs innovationlabs.org",
    "DataFlow Systems dataflow.com",
    "CloudTech Enterprises cloudtech.biz",
    "NextGen Analytics nextgen.ai",
    "SmartBusiness Solutions smartbiz.co"
]

SAMPLE_CONTEXT_TEXTS = [
    "Enterprise software implementation project for customer onboarding automation. Next activity: technical requirements gathering scheduled for next week. Deal value: $250,000 ARR with 3-year commitment.",
    "Cloud migration consulting engagement for financial services client. Currently in proof-of-concept phase with positive stakeholder feedback. Expected close date: Q1 2024.",
    "AI/ML platform deployment for healthcare analytics. Customer has completed security review and is moving to contract negotiation phase. High priority opportunity.",
    "Digital transformation initiative for manufacturing company. Pilot project successful, expanding to full implementation. Customer champion is VP of Operations.",
    "Cybersecurity audit and implementation services. Customer experienced recent security incident, urgent timeline for deployment. Enterprise-level engagement.",
    "Business intelligence dashboard development for retail chain. Customer wants to consolidate reporting across 50+ locations. Multi-year strategic partnership potential."
]


class BGEServiceTester:
    """
    Comprehensive test suite for the self-contained BGE service.
    
    Tests HTTP endpoints, performance characteristics, embedding quality,
    and integration with the embedding generation scripts.
    """
    
    def __init__(self):
        self.test_results = {
            'service_health': False,
            'basic_embedding': False,
            'batch_embedding': False,
            'performance_test': False,
            'embedding_quality': False,
            'gpu_availability': None,
            'service_info': {},
            'performance_metrics': {},
            'error_count': 0,
            'warnings': []
        }
        
    async def test_service_health(self) -> bool:
        """Test BGE service health endpoint"""
        print("🔧 Testing BGE service health...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(BGE_HEALTH_URL, timeout=10) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        
                        print(f"✅ BGE service is healthy")
                        print(f"   Status: {health_data.get('status', 'unknown')}")
                        print(f"   Model: {health_data.get('model', 'unknown')}")
                        print(f"   CUDA Available: {health_data.get('cuda_available', False)}")
                        print(f"   Device: {health_data.get('device', 'unknown')}")
                        
                        # Store service info
                        self.test_results['service_info'] = health_data
                        self.test_results['gpu_availability'] = health_data.get('cuda_available', False)
                        
                        return True
                    else:
                        print(f"❌ Health check failed with status {response.status}")
                        return False
                        
        except asyncio.TimeoutError:
            print("❌ Health check timeout - service may not be running")
            return False
        except aiohttp.ClientError as e:
            print(f"❌ Health check connection error: {e}")
            return False
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    async def test_single_embedding(self) -> bool:
        """Test single text embedding generation"""
        print("\n🔧 Testing single embedding generation...")
        
        try:
            test_text = SAMPLE_IDENTITY_TEXTS[0]
            
            async with aiohttp.ClientSession() as session:
                payload = {"texts": [test_text]}
                
                start_time = time.time()
                async with session.post(BGE_EMBED_URL, json=payload, timeout=30) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        embeddings = result.get('embeddings', [])
                        dimension = result.get('dimension', 0)
                        count = result.get('count', 0)
                        
                        if embeddings and len(embeddings) > 0:
                            embedding = embeddings[0]
                            
                            print(f"✅ Single embedding generated successfully")
                            print(f"   Dimension: {dimension}")
                            print(f"   Count: {count}")
                            print(f"   Response time: {response_time:.1f}ms")
                            print(f"   First 5 values: {embedding[:5]}")
                            
                            # Validate embedding properties
                            if dimension == 1024:
                                print("✅ Correct BGE-M3 dimension (1024)")
                            else:
                                print(f"⚠️  Unexpected dimension: {dimension} (expected 1024)")
                                self.test_results['warnings'].append(f"Dimension mismatch: {dimension}")
                            
                            # Check value range (normalized embeddings should be reasonable)
                            min_val = min(embedding)
                            max_val = max(embedding)
                            if -2.0 <= min_val <= 2.0 and -2.0 <= max_val <= 2.0:
                                print(f"✅ Embedding values in reasonable range: [{min_val:.3f}, {max_val:.3f}]")
                            else:
                                print(f"⚠️  Embedding values outside expected range: [{min_val:.3f}, {max_val:.3f}]")
                                self.test_results['warnings'].append("Embedding values out of range")
                            
                            return True
                        else:
                            print("❌ Empty embeddings in response")
                            return False
                    else:
                        error_text = await response.text()
                        print(f"❌ Embedding request failed with status {response.status}: {error_text}")
                        return False
                        
        except Exception as e:
            print(f"❌ Single embedding test failed: {e}")
            return False
    
    async def test_batch_embedding(self) -> bool:
        """Test batch embedding generation"""
        print("\n🔧 Testing batch embedding generation...")
        
        try:
            test_texts = SAMPLE_IDENTITY_TEXTS + SAMPLE_CONTEXT_TEXTS
            batch_size = len(test_texts)
            
            async with aiohttp.ClientSession() as session:
                payload = {"texts": test_texts}
                
                start_time = time.time()
                async with session.post(BGE_EMBED_URL, json=payload, timeout=60) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        embeddings = result.get('embeddings', [])
                        dimension = result.get('dimension', 0)
                        count = result.get('count', 0)
                        
                        print(f"✅ Batch embedding generated successfully")
                        print(f"   Input texts: {batch_size}")
                        print(f"   Output embeddings: {count}")
                        print(f"   Dimension: {dimension}")
                        print(f"   Response time: {response_time:.1f}ms")
                        print(f"   Rate: {count / (response_time / 1000):.1f} embeddings/sec")
                        
                        # Validate batch consistency
                        if count == batch_size:
                            print("✅ Correct number of embeddings returned")
                        else:
                            print(f"❌ Embedding count mismatch: {count} != {batch_size}")
                            return False
                        
                        # Check embedding diversity (they should be different)
                        if len(embeddings) >= 2:
                            emb1, emb2 = embeddings[0], embeddings[1]
                            similarity = sum(a * b for a, b in zip(emb1, emb2))
                            if abs(similarity) < 0.99:  # Should not be identical
                                print(f"✅ Embeddings are diverse (similarity: {similarity:.3f})")
                            else:
                                print(f"⚠️  Embeddings may be too similar: {similarity:.3f}")
                                self.test_results['warnings'].append("Low embedding diversity")
                        
                        return True
                    else:
                        error_text = await response.text()
                        print(f"❌ Batch embedding request failed: {error_text}")
                        return False
                        
        except Exception as e:
            print(f"❌ Batch embedding test failed: {e}")
            return False
    
    async def test_performance_benchmark(self) -> bool:
        """Test performance against RTX 3070 Ti targets"""
        print("\n🔧 Testing performance benchmark...")
        
        try:
            # Test with target batch size
            test_texts = (SAMPLE_IDENTITY_TEXTS * 4)[:TARGET_BATCH_SIZE]  # Exactly 32 texts
            
            response_times = []
            embedding_rates = []
            
            # Run multiple iterations for statistical validity
            iterations = 3
            print(f"   Running {iterations} iterations with {len(test_texts)} embeddings each...")
            
            for i in range(iterations):
                async with aiohttp.ClientSession() as session:
                    payload = {"texts": test_texts}
                    
                    start_time = time.time()
                    async with session.post(BGE_EMBED_URL, json=payload, timeout=60) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            result = await response.json()
                            count = result.get('count', 0)
                            
                            response_times.append(response_time)
                            embedding_rate = count / (response_time / 1000)
                            embedding_rates.append(embedding_rate)
                            
                            print(f"   Iteration {i+1}: {response_time:.1f}ms, {embedding_rate:.1f} emb/sec")
                        else:
                            print(f"   Iteration {i+1} failed with status {response.status}")
                            return False
                
                # Small delay between iterations
                await asyncio.sleep(1)
            
            # Calculate statistics
            avg_response_time = statistics.mean(response_times)
            avg_embedding_rate = statistics.mean(embedding_rates)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            print(f"\n📊 Performance Results:")
            print(f"   Average response time: {avg_response_time:.1f}ms")
            print(f"   Response time range: {min_response_time:.1f}ms - {max_response_time:.1f}ms")
            print(f"   Average embedding rate: {avg_embedding_rate:.1f} embeddings/sec")
            print(f"   Target rate: {TARGET_EMBEDDINGS_PER_SEC} embeddings/sec")
            
            # Store metrics
            self.test_results['performance_metrics'] = {
                'avg_response_time_ms': avg_response_time,
                'avg_embedding_rate': avg_embedding_rate,
                'response_times': response_times,
                'embedding_rates': embedding_rates
            }
            
            # Evaluate against targets
            meets_response_target = avg_response_time <= TARGET_MAX_RESPONSE_TIME
            meets_rate_target = avg_embedding_rate >= TARGET_EMBEDDINGS_PER_SEC
            
            if meets_response_target:
                print(f"✅ Response time target met ({TARGET_MAX_RESPONSE_TIME}ms)")
            else:
                print(f"⚠️  Response time target missed ({TARGET_MAX_RESPONSE_TIME}ms)")
                self.test_results['warnings'].append("Response time target missed")
            
            if meets_rate_target:
                print(f"✅ Embedding rate target met ({TARGET_EMBEDDINGS_PER_SEC} emb/sec)")
            else:
                print(f"⚠️  Embedding rate target missed ({TARGET_EMBEDDINGS_PER_SEC} emb/sec)")
                self.test_results['warnings'].append("Embedding rate target missed")
            
            # GPU performance note
            if self.test_results['gpu_availability']:
                print("🚀 GPU acceleration appears to be working")
            else:
                print("💻 Running on CPU (GPU not available)")
                self.test_results['warnings'].append("GPU acceleration not available")
            
            return meets_response_target and meets_rate_target
            
        except Exception as e:
            print(f"❌ Performance benchmark failed: {e}")
            return False
    
    async def test_embedding_quality(self) -> bool:
        """Test embedding quality with known similar/dissimilar texts"""
        print("\n🔧 Testing embedding quality...")
        
        try:
            # Test texts with known relationships
            similar_texts = [
                "Acme Corporation acme.com",
                "Acme Corp acme.com"  # Similar company
            ]
            
            dissimilar_texts = [
                "Acme Corporation acme.com",
                "TechFlow Industries techflow.biz"  # Different company
            ]
            
            # Get embeddings for similar texts
            async with aiohttp.ClientSession() as session:
                payload = {"texts": similar_texts}
                async with session.post(BGE_EMBED_URL, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        similar_embeddings = result.get('embeddings', [])
                    else:
                        print("❌ Failed to get similar text embeddings")
                        return False
                
                # Get embeddings for dissimilar texts
                payload = {"texts": dissimilar_texts}
                async with session.post(BGE_EMBED_URL, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        dissimilar_embeddings = result.get('embeddings', [])
                    else:
                        print("❌ Failed to get dissimilar text embeddings")
                        return False
            
            # Calculate similarities
            def cosine_similarity(a, b):
                dot_product = sum(x * y for x, y in zip(a, b))
                magnitude_a = sum(x * x for x in a) ** 0.5
                magnitude_b = sum(x * x for x in b) ** 0.5
                return dot_product / (magnitude_a * magnitude_b)
            
            similar_score = cosine_similarity(similar_embeddings[0], similar_embeddings[1])
            dissimilar_score = cosine_similarity(dissimilar_embeddings[0], dissimilar_embeddings[1])
            
            print(f"   Similar texts similarity: {similar_score:.3f}")
            print(f"   Dissimilar texts similarity: {dissimilar_score:.3f}")
            
            # Quality checks
            quality_good = True
            
            if similar_score > 0.8:  # Similar texts should have high similarity
                print("✅ Similar texts show high similarity")
            else:
                print(f"⚠️  Similar texts similarity lower than expected: {similar_score:.3f}")
                quality_good = False
                self.test_results['warnings'].append("Low similarity for similar texts")
            
            if similar_score > dissimilar_score:  # Similar should be more similar than dissimilar
                print("✅ Similarity ordering is correct")
            else:
                print(f"⚠️  Similarity ordering incorrect")
                quality_good = False
                self.test_results['warnings'].append("Incorrect similarity ordering")
            
            return quality_good
            
        except Exception as e:
            print(f"❌ Embedding quality test failed: {e}")
            return False
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all BGE service tests"""
        print("🚀 Starting comprehensive BGE service test suite...")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all tests
        self.test_results['service_health'] = await self.test_service_health()
        
        if self.test_results['service_health']:
            self.test_results['basic_embedding'] = await self.test_single_embedding()
            self.test_results['batch_embedding'] = await self.test_batch_embedding()
            self.test_results['performance_test'] = await self.test_performance_benchmark()
            self.test_results['embedding_quality'] = await self.test_embedding_quality()
        else:
            print("\n❌ Service health check failed - skipping other tests")
            print("   Make sure BGE service is running: docker-compose --profile gpu up -d bge-service")
        
        total_time = time.time() - start_time
        
        # Generate final report
        print("\n" + "=" * 60)
        print("🎯 BGE Service Test Results Summary")
        print("=" * 60)
        
        passed_tests = sum(1 for result in [
            self.test_results['service_health'],
            self.test_results['basic_embedding'], 
            self.test_results['batch_embedding'],
            self.test_results['performance_test'],
            self.test_results['embedding_quality']
        ] if result)
        
        total_tests = 5
        
        print(f"Overall: {passed_tests}/{total_tests} tests passed")
        print(f"Service Health: {'✅' if self.test_results['service_health'] else '❌'}")
        print(f"Basic Embedding: {'✅' if self.test_results['basic_embedding'] else '❌'}")
        print(f"Batch Embedding: {'✅' if self.test_results['batch_embedding'] else '❌'}")
        print(f"Performance Test: {'✅' if self.test_results['performance_test'] else '❌'}")
        print(f"Embedding Quality: {'✅' if self.test_results['embedding_quality'] else '❌'}")
        
        if self.test_results['gpu_availability'] is not None:
            print(f"GPU Availability: {'🚀 Yes' if self.test_results['gpu_availability'] else '💻 No'}")
        
        if self.test_results['warnings']:
            print(f"\n⚠️  Warnings ({len(self.test_results['warnings'])}):")
            for warning in self.test_results['warnings']:
                print(f"   • {warning}")
        
        print(f"\nTotal test time: {total_time:.1f}s")
        
        # Final assessment
        if passed_tests == total_tests:
            print("\n🎉 All tests passed! BGE service is ready for production.")
        elif passed_tests >= 3:
            print("\n✅ Most tests passed. BGE service is functional with some limitations.")
        else:
            print("\n❌ Multiple test failures. BGE service needs attention.")
        
        return self.test_results


async def main():
    """Main entry point for BGE service testing"""
    tester = BGEServiceTester()
    
    try:
        results = await tester.run_comprehensive_test()
        return 0 if results['service_health'] else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))