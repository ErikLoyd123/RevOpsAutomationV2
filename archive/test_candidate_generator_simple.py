#!/usr/bin/env python3
"""
Simple test for Task 4.2: Two-Stage Retrieval Architecture
"""

import asyncio
import sys
import os
import time
from pathlib import Path

# Add paths
sys.path.append('/home/loyd2888/Projects/RevOpsAutomationV2/backend/services/08-matching')

# Set environment for database connection
os.environ['LOCAL_DB_HOST'] = 'localhost'
os.environ['LOCAL_DB_PORT'] = '5432'
os.environ['LOCAL_DB_NAME'] = 'revops_core'
os.environ['LOCAL_DB_USER'] = 'revops_user'
os.environ['LOCAL_DB_PASSWORD'] = 'RevOps2024Secure!'
os.environ['BGE_SERVICE_URL'] = 'http://localhost:8007'

async def test_simple():
    """Simple test to verify the candidate generator can be imported and initialized"""
    
    print("🧪 Simple Candidate Generator Test")
    print("=" * 50)
    
    try:
        # Test import
        print("📦 Testing import...")
        from config import get_config, get_database_url
        print("   ✅ Config imported successfully")
        
        # Test config loading
        print("🔧 Testing configuration...")
        config = get_config()
        db_url = get_database_url()
        print(f"   ✅ Configuration loaded")
        print(f"   📊 RRF k-value: {config.rrf.k_value}")
        print(f"   🎯 Method weights: {config.rrf.method_weights}")
        print(f"   🔗 Database URL: postgresql://{config.database_url.split('@')[1] if '@' in config.database_url else 'configured'}")
        
        # Test BGE service connectivity
        print("🌐 Testing BGE service connectivity...")
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get("http://localhost:8007/health")
                if response.status_code == 200:
                    print("   ✅ BGE service is healthy")
                    print(f"   📊 Response: {response.text}")
                else:
                    print(f"   ⚠️  BGE service returned status {response.status_code}")
            except Exception as e:
                print(f"   ❌ BGE service not reachable: {e}")
                print("   💡 Start BGE service: docker-compose --profile gpu up -d bge-service")
        
        # Test database connectivity
        print("🗄️  Testing database connectivity...")
        import asyncpg
        try:
            conn = await asyncpg.connect(db_url)
            
            # Check opportunity count
            result = await conn.fetchrow("SELECT COUNT(*) as count FROM core.opportunities")
            opp_count = result["count"]
            print(f"   ✅ Database connected successfully")
            print(f"   📊 Found {opp_count} opportunities")
            
            # Check embeddings
            result = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(context_embedding) as context_count
                FROM core.opportunities
            """)
            print(f"   📊 Context embeddings: {result['context_count']}/{result['total']}")
            
            if result['context_count'] == 0:
                print("   ⚠️  No embeddings found - generate them first:")
                print("   🚀 ./scripts/03-data/19_generate_all_embeddings.sh")
            
            # Get sample opportunity
            if opp_count > 0:
                sample = await conn.fetchrow("""
                    SELECT source_id, source_system, identity_text
                    FROM core.opportunities 
                    LIMIT 1
                """)
                print(f"   🔍 Sample opportunity: {sample['source_id']} ({sample['source_system']})")
                print(f"   🏢 Identity: {sample['identity_text'][:80]}...")
            
            await conn.close()
            
        except Exception as e:
            print(f"   ❌ Database connection failed: {e}")
        
        # Try to import the candidate generator
        print("🎯 Testing candidate generator import...")
        try:
            from candidate_generator import TwoStageRetrieval
            print("   ✅ TwoStageRetrieval imported successfully")
            print("   📚 Available methods:")
            print("      - Stage 1: BGE similarity search")
            print("      - Stage 2: Multi-method RRF fusion")
            print("      - Methods: semantic, fuzzy match, domain, context")
        except Exception as e:
            print(f"   ❌ Candidate generator import failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n🎉 Simple test completed!")
        print("\n💡 To run full matching test:")
        print("   1. Start BGE service: docker-compose --profile gpu up -d bge-service")
        print("   2. Generate embeddings: ./scripts/03-data/19_generate_all_embeddings.sh")
        print("   3. Test individual opportunity matching")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple())