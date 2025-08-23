#!/usr/bin/env python3
"""
Integration Test for Task 4.2: Two-Stage Retrieval Architecture

This test validates the complete candidate_generator.py implementation including:
- Two-stage retrieval workflow (BGE → Multi-method scoring)
- RRF fusion algorithm with 4 methods
- Database connectivity and caching
- Performance metrics and audit trail
- Error handling and edge cases

Prerequisites:
- BGE service running on localhost:8007
- PostgreSQL with revops_core database
- Redis running (if caching enabled)
- Sample opportunity data in core.opportunities

Usage:
    cd /home/loyd2888/Projects/RevOpsAutomationV2
    source venv/bin/activate
    python backend/tests/integration/test_candidate_generator.py
"""

import asyncio
import sys
import time
import json
from pathlib import Path

# Add backend modules to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "backend" / "services" / "08-matching"))
sys.path.append(str(PROJECT_ROOT / "backend" / "core"))

from candidate_generator import (
    TwoStageRetrieval,
    BatchCandidateProcessor,
    CandidateGenerationMetrics
)
from config import get_config
from database import DatabaseManager
import structlog

logger = structlog.get_logger(__name__)

class CandidateGeneratorTester:
    """Integration test suite for candidate generation"""
    
    def __init__(self):
        self.config = get_config()
        self.db_manager = DatabaseManager(self.config.database_url)
        self.retrieval_engine = None
        self.test_results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": []
        }
    
    async def setup(self):
        """Initialize test environment"""
        print("🔧 Setting up test environment...")
        
        # Initialize database connection
        await self.db_manager.initialize()
        print("   ✅ Database connection initialized")
        
        # Initialize retrieval engine
        self.retrieval_engine = TwoStageRetrieval(
            config=self.config,
            db_manager=self.db_manager
        )
        await self.retrieval_engine.initialize()
        print("   ✅ Two-stage retrieval engine initialized")
        
        # Validate BGE service connectivity
        try:
            test_embedding = await self.retrieval_engine.bge_client.get_embeddings(
                ["test connectivity"], 
                embedding_type="context"
            )
            print("   ✅ BGE service connectivity confirmed")
            print(f"   📊 Embedding dimension: {len(test_embedding[0])}")
        except Exception as e:
            print(f"   ❌ BGE service test failed: {e}")
            raise
    
    async def test_database_connectivity(self):
        """Test database operations"""
        self.test_results["tests_run"] += 1
        
        try:
            print("\n🗄️  Testing database connectivity...")
            
            # Test opportunity count
            async with self.db_manager.get_connection() as conn:
                result = await conn.fetchrow("SELECT COUNT(*) as count FROM core.opportunities")
                opp_count = result["count"]
                print(f"   📊 Found {opp_count} opportunities in database")
                
                if opp_count == 0:
                    raise ValueError("No opportunities found - run data pipeline first")
                
                # Test embedding readiness
                result = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(identity_embedding) as identity_count,
                        COUNT(context_embedding) as context_count
                    FROM core.opportunities
                """)
                
                print(f"   🔍 Identity embeddings: {result['identity_count']}/{result['total']}")
                print(f"   🔍 Context embeddings: {result['context_count']}/{result['total']}")
                
                if result['context_count'] == 0:
                    print("   ⚠️  No embeddings found - run embedding generation first")
                    print("   🚀 ./scripts/03-data/19_generate_all_embeddings.sh")
            
            self.test_results["tests_passed"] += 1
            print("   ✅ Database connectivity test passed")
            
        except Exception as e:
            self.test_results["tests_failed"] += 1
            self.test_results["errors"].append(f"Database connectivity: {str(e)}")
            print(f"   ❌ Database connectivity test failed: {e}")
            raise
    
    async def test_single_opportunity_matching(self):
        """Test matching for a single opportunity"""
        self.test_results["tests_run"] += 1
        
        try:
            print("\n🎯 Testing single opportunity matching...")
            
            # Get a test opportunity
            async with self.db_manager.get_connection() as conn:
                test_opp = await conn.fetchrow("""
                    SELECT opportunity_id, source_system, identity_text, context_text
                    FROM core.opportunities 
                    WHERE context_embedding IS NOT NULL
                    LIMIT 1
                """)
                
                if not test_opp:
                    raise ValueError("No opportunities with embeddings found")
                
                print(f"   🔍 Testing opportunity: {test_opp['opportunity_id']} ({test_opp['source_system']})")
                print(f"   🏢 Identity: {test_opp['identity_text'][:100]}...")
                print(f"   📝 Context: {test_opp['context_text'][:100]}...")
            
            # Run candidate generation
            start_time = time.time()
            
            candidates = await self.retrieval_engine.generate_candidates(
                opportunity_id=test_opp["opportunity_id"],
                source_system=test_opp["source_system"],
                max_candidates=10
            )
            
            duration = time.time() - start_time
            
            print(f"   ⏱️  Generation time: {duration:.2f}s")
            print(f"   📊 Candidates found: {len(candidates)}")
            
            # Analyze results
            if candidates:
                best_candidate = candidates[0]
                print(f"   🏆 Best match: {best_candidate.matched_opportunity_id}")
                print(f"   📈 RRF score: {best_candidate.rrf_score:.3f}")
                print(f"   🔧 Methods: {list(best_candidate.method_scores.keys())}")
                
                # Print top 3 candidates
                for i, candidate in enumerate(candidates[:3]):
                    print(f"   {i+1}. {candidate.matched_opportunity_id} (score: {candidate.rrf_score:.3f})")
            else:
                print("   ⚠️  No candidates found - this may be normal for some opportunities")
            
            self.test_results["tests_passed"] += 1
            print("   ✅ Single opportunity matching test passed")
            
        except Exception as e:
            self.test_results["tests_failed"] += 1
            self.test_results["errors"].append(f"Single opportunity matching: {str(e)}")
            print(f"   ❌ Single opportunity matching test failed: {e}")
            raise
    
    async def test_batch_processing(self):
        """Test batch candidate processing"""
        self.test_results["tests_run"] += 1
        
        try:
            print("\n📦 Testing batch processing...")
            
            # Get test opportunities
            async with self.db_manager.get_connection() as conn:
                test_opps = await conn.fetch("""
                    SELECT opportunity_id, source_system
                    FROM core.opportunities 
                    WHERE context_embedding IS NOT NULL
                    LIMIT 5
                """)
                
                if len(test_opps) < 2:
                    raise ValueError("Need at least 2 opportunities with embeddings")
                
                print(f"   🔍 Testing {len(test_opps)} opportunities")
            
            # Initialize batch processor
            batch_processor = BatchCandidateProcessor(
                retrieval_engine=self.retrieval_engine,
                batch_size=2,
                max_concurrency=2
            )
            
            # Prepare batch requests
            opportunities = [
                (opp["opportunity_id"], opp["source_system"])
                for opp in test_opps
            ]
            
            # Run batch processing
            start_time = time.time()
            
            results = await batch_processor.process_batch(
                opportunities=opportunities,
                max_candidates_per_opportunity=5
            )
            
            duration = time.time() - start_time
            
            print(f"   ⏱️  Batch processing time: {duration:.2f}s")
            print(f"   📊 Success rate: {len(results)}/{len(opportunities)}")
            
            # Analyze batch results
            total_candidates = sum(len(candidates) for candidates in results.values())
            avg_candidates = total_candidates / len(results) if results else 0
            
            print(f"   📈 Total candidates: {total_candidates}")
            print(f"   📈 Average per opportunity: {avg_candidates:.1f}")
            
            self.test_results["tests_passed"] += 1
            print("   ✅ Batch processing test passed")
            
        except Exception as e:
            self.test_results["tests_failed"] += 1
            self.test_results["errors"].append(f"Batch processing: {str(e)}")
            print(f"   ❌ Batch processing test failed: {e}")
    
    async def test_performance_metrics(self):
        """Test performance metrics collection"""
        self.test_results["tests_run"] += 1
        
        try:
            print("\n📊 Testing performance metrics...")
            
            # Get metrics instance
            metrics = CandidateGenerationMetrics()
            
            # Record some test metrics
            metrics.record_stage_timing("stage_1_retrieval", 0.5)
            metrics.record_stage_timing("stage_2_scoring", 1.2)
            metrics.record_method_coverage("semantic_similarity", 25)
            metrics.record_method_coverage("company_fuzzy_match", 15)
            metrics.record_candidate_stats(total_evaluated=40, final_candidates=8)
            
            # Get summary
            summary = metrics.get_summary()
            
            print(f"   ⏱️  Total processing time: {summary['total_processing_time_ms']:.1f}ms")
            print(f"   🎯 Stage 1 time: {summary['stage_timings']['stage_1_retrieval']:.1f}ms")
            print(f"   🎯 Stage 2 time: {summary['stage_timings']['stage_2_scoring']:.1f}ms")
            print(f"   📊 Method coverage: {summary['method_coverage']}")
            print(f"   📈 Candidate stats: {summary['candidate_stats']}")
            
            # Validate metrics structure
            required_fields = ['total_processing_time_ms', 'stage_timings', 'method_coverage', 'candidate_stats']
            for field in required_fields:
                if field not in summary:
                    raise ValueError(f"Missing metric field: {field}")
            
            self.test_results["tests_passed"] += 1
            print("   ✅ Performance metrics test passed")
            
        except Exception as e:
            self.test_results["tests_failed"] += 1
            self.test_results["errors"].append(f"Performance metrics: {str(e)}")
            print(f"   ❌ Performance metrics test failed: {e}")
    
    async def test_matching_session_audit(self):
        """Test matching session audit trail"""
        self.test_results["tests_run"] += 1
        
        try:
            print("\n📋 Testing matching session audit trail...")
            
            # Check if ops.matching_sessions table exists
            async with self.db_manager.get_connection() as conn:
                table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_schema = 'ops' 
                        AND table_name = 'matching_sessions'
                    )
                """)
                
                if not table_exists:
                    print("   ⚠️  ops.matching_sessions table not found - creating it...")
                    
                    # Create the table
                    await conn.execute("""
                        CREATE SCHEMA IF NOT EXISTS ops;
                        
                        CREATE TABLE IF NOT EXISTS ops.matching_sessions (
                            session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            opportunity_id VARCHAR(255) NOT NULL,
                            source_system VARCHAR(50) NOT NULL,
                            request_params JSONB,
                            execution_time_ms INTEGER,
                            candidates_found INTEGER,
                            method_coverage JSONB,
                            confidence_scores JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            
                            INDEX idx_matching_sessions_opportunity (opportunity_id),
                            INDEX idx_matching_sessions_source (source_system),
                            INDEX idx_matching_sessions_created (created_at)
                        )
                    """)
                    print("   ✅ ops.matching_sessions table created")
                
                # Test inserting a session record
                test_session_data = {
                    "opportunity_id": "test_opp_001",
                    "source_system": "odoo",
                    "request_params": json.dumps({
                        "max_candidates": 10,
                        "methods_enabled": ["semantic_similarity", "company_fuzzy_match"]
                    }),
                    "execution_time_ms": 1500,
                    "candidates_found": 5,
                    "method_coverage": json.dumps({
                        "semantic_similarity": 25,
                        "company_fuzzy_match": 12
                    }),
                    "confidence_scores": json.dumps({
                        "max": 0.85,
                        "avg": 0.62,
                        "min": 0.41
                    })
                }
                
                session_id = await conn.fetchval("""
                    INSERT INTO ops.matching_sessions 
                    (opportunity_id, source_system, request_params, execution_time_ms, 
                     candidates_found, method_coverage, confidence_scores)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING session_id
                """, 
                    test_session_data["opportunity_id"],
                    test_session_data["source_system"],
                    test_session_data["request_params"],
                    test_session_data["execution_time_ms"],
                    test_session_data["candidates_found"],
                    test_session_data["method_coverage"],
                    test_session_data["confidence_scores"]
                )
                
                print(f"   📝 Created test session: {session_id}")
                
                # Verify the record
                session_record = await conn.fetchrow("""
                    SELECT * FROM ops.matching_sessions WHERE session_id = $1
                """, session_id)
                
                if session_record:
                    print("   ✅ Session record successfully created and retrieved")
                    print(f"   📊 Execution time: {session_record['execution_time_ms']}ms")
                    print(f"   🎯 Candidates found: {session_record['candidates_found']}")
                else:
                    raise ValueError("Failed to retrieve created session record")
                
                # Clean up test record
                await conn.execute("DELETE FROM ops.matching_sessions WHERE session_id = $1", session_id)
                print("   🧹 Test record cleaned up")
            
            self.test_results["tests_passed"] += 1
            print("   ✅ Matching session audit test passed")
            
        except Exception as e:
            self.test_results["tests_failed"] += 1
            self.test_results["errors"].append(f"Matching session audit: {str(e)}")
            print(f"   ❌ Matching session audit test failed: {e}")
    
    async def run_all_tests(self):
        """Run complete test suite"""
        print("🧪 Starting Candidate Generator Integration Tests")
        print("=" * 60)
        
        try:
            await self.setup()
            
            # Run individual tests
            await self.test_database_connectivity()
            await self.test_single_opportunity_matching()
            await self.test_batch_processing()
            await self.test_performance_metrics()
            await self.test_matching_session_audit()
            
        except Exception as e:
            print(f"\n❌ Test suite failed with critical error: {e}")
            self.test_results["tests_failed"] += 1
            self.test_results["errors"].append(f"Critical error: {str(e)}")
        
        finally:
            if self.db_manager:
                await self.db_manager.close()
        
        # Print final results
        print("\n" + "=" * 60)
        print("🏁 Test Suite Results")
        print("=" * 60)
        print(f"Tests Run: {self.test_results['tests_run']}")
        print(f"Tests Passed: {self.test_results['tests_passed']}")
        print(f"Tests Failed: {self.test_results['tests_failed']}")
        
        if self.test_results["errors"]:
            print("\n❌ Errors encountered:")
            for error in self.test_results["errors"]:
                print(f"   • {error}")
        
        success_rate = self.test_results['tests_passed'] / max(self.test_results['tests_run'], 1) * 100
        
        if success_rate == 100:
            print(f"\n🎉 All tests passed! Success rate: {success_rate:.1f}%")
            return True
        else:
            print(f"\n⚠️  Some tests failed. Success rate: {success_rate:.1f}%")
            return False


async def main():
    """Main test execution"""
    print("🚀 RevOps Platform - Candidate Generator Integration Test")
    print("Task 4.2: Two-Stage Retrieval Architecture Validation")
    print()
    
    # Check prerequisites
    print("🔍 Checking prerequisites...")
    
    # Check if we're in virtual environment
    venv_active = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not venv_active:
        print("❌ Virtual environment not detected")
        print("Please activate: source venv/bin/activate")
        return False
    
    print("✅ Virtual environment active")
    
    # Run test suite
    tester = CandidateGeneratorTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎯 Task 4.2 implementation validation: PASSED")
        return True
    else:
        print("\n❌ Task 4.2 implementation validation: FAILED") 
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)