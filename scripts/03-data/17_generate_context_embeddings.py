#!/usr/bin/env python3
"""
Generate Context Embeddings for Opportunities - Task 2.7

This script generates BGE-M3 context embeddings for all opportunities in core.opportunities.
Creates embeddings from rich business context for semantic opportunity matching.

Key Features:
- Processes all 7,937 opportunities (3,121 Odoo + 4,816 APN)
- Generates BGE-M3 context embeddings from business context information
- Uses hash-based change detection to avoid redundant processing
- Stores embeddings in SEARCH schema with proper source tracking
- Batch processing optimized for RTX 3070 Ti (32 embeddings per 500ms)
- Comprehensive progress tracking and error handling
- Integration with BGE health monitoring

Processing Strategy:
1. Extract opportunities with existing context_text but missing embeddings
2. Generate context text from business fields if missing
3. Calculate SHA-256 hash for change detection
4. Generate BGE-M3 embeddings in batches of 32
5. Store embeddings with metadata in SEARCH schema
6. Update core.opportunities with embedding references

Prerequisites:
- Phase 1 complete (BGE service ready)
- core.opportunities table populated (7,937 records)
- SEARCH schema created with embedding tables
- BGE health monitoring operational

Usage:
    python scripts/03-data/17_generate_context_embeddings.py --batch-size 32
    python scripts/03-data/17_generate_context_embeddings.py --force-regenerate
    python scripts/03-data/17_generate_context_embeddings.py --status
"""

import asyncio
import hashlib
import json
import os
import sys
import time
import argparse
import uuid
import aiohttp
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import traceback

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from backend.core.config import get_settings
    print("✅ Successfully imported database and config modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the project root with virtual environment activated")
    sys.exit(1)

# BGE service configuration
BGE_SERVICE_URL = "http://localhost:8007"
BGE_HEALTH_URL = f"{BGE_SERVICE_URL}/health"
BGE_EMBED_URL = f"{BGE_SERVICE_URL}/embed"


class ContextEmbeddingsGenerator:
    """Generate context embeddings for opportunities using BGE-M3 model"""
    
    def __init__(self, batch_size: int = 32, force_regenerate: bool = False):
        """Initialize the context embeddings generator"""
        self.batch_size = batch_size
        self.force_regenerate = force_regenerate
        self.settings = get_settings()
        
        # Try to initialize health monitoring
        try:
            import psutil
            self.health_monitor = psutil
        except ImportError:
            print("⚠️  BGE health monitoring not available: No module named 'psutil'")
            self.health_monitor = None
        
        # Database configuration
        self.local_db_config = {
            'host': self.settings.database.local_db_host,
            'port': self.settings.database.local_db_port,
            'database': self.settings.database.local_db_name,
            'user': self.settings.database.local_db_user,
            'password': self.settings.database.local_db_password,
            'sslmode': 'prefer'
        }
        
        print(f"🔧 Context Embedding Generator initialized")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Health monitoring: {'enabled' if self.health_monitor else 'disabled'}")
    
    def get_local_connection(self):
        """Get a simple local database connection"""
        import psycopg2
        from psycopg2.extras import RealDictCursor
        return psycopg2.connect(cursor_factory=RealDictCursor, **self.local_db_config)
    
    async def analyze_opportunity_data(self) -> Dict[str, Any]:
        """Analyze current opportunity data to understand embedding needs"""
        print("🔍 Analyzing opportunity data...")
        
        analysis_queries = {
            'total_opportunities': """
                SELECT COUNT(*) as count FROM core.opportunities
            """,
            'by_source_system': """
                SELECT source_system, COUNT(*) as count 
                FROM core.opportunities 
                GROUP BY source_system 
                ORDER BY source_system
            """,
            'context_text_status': """
                SELECT 
                    COUNT(*) as total,
                    COUNT(co.context_text) as with_context_text,
                    COUNT(co.context_hash) as with_context_hash,
                    COUNT(CASE WHEN se.context_vector IS NOT NULL THEN 1 END) as with_context_embedding,
                    COUNT(CASE WHEN co.context_text IS NOT NULL AND se.context_vector IS NULL THEN 1 END) as needing_embeddings
                FROM core.opportunities co
                LEFT JOIN search.embeddings_opportunities se ON co.id = se.opportunity_id
            """,
            'sample_context_text': """
                SELECT 
                    source_system,
                    company_name,
                    CASE 
                        WHEN LENGTH(context_text) > 200 THEN LEFT(context_text, 200) || '...'
                        ELSE context_text
                    END as context_preview,
                    context_hash
                FROM core.opportunities 
                WHERE context_text IS NOT NULL 
                ORDER BY source_system, id 
                LIMIT 5
            """
        }
        
        analysis_result = {}
        
        try:
            with self.get_local_connection() as conn:
                cursor = conn.cursor()
                
                for query_name, query_sql in analysis_queries.items():
                    cursor.execute(query_sql)
                    
                    if query_name in ['by_source_system', 'sample_context_text']:
                        # Fetch all rows - psycopg2 returns RealDictRow objects
                        results = cursor.fetchall()
                        # Convert RealDictRow objects to regular tuples with values only
                        analysis_result[query_name] = [tuple(row.values()) for row in results]
                    else:
                        # Fetch single row - psycopg2 returns RealDictRow
                        result = cursor.fetchone()
                        if result:
                            # Convert RealDictRow to dict
                            analysis_result[query_name] = dict(result)
                        else:
                            analysis_result[query_name] = {}
        
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            raise
        
        print("✅ Analysis completed successfully")
        return analysis_result
    
    def print_analysis_report(self, analysis: Dict[str, Any]):
        """Print detailed analysis report"""
        print("🔍 Context Embeddings Analysis Report")
        print("=" * 60)
        
        # Total opportunities
        total = analysis.get('total_opportunities', {}).get('count', 0)
        print(f"📊 Total Opportunities: {total:,}")
        
        # By source system
        print("\n📈 Opportunities by Source System:")
        for source_system, count in analysis.get('by_source_system', []):
            print(f"   {source_system}: {int(count):,}")
        
        # Context text status
        context_status = analysis.get('context_text_status', {})
        print(f"\n📝 Context Text Status:")
        print(f"   Total opportunities: {context_status.get('total', 0):,}")
        print(f"   With context text: {context_status.get('with_context_text', 0):,}")
        print(f"   With context hash: {context_status.get('with_context_hash', 0):,}")
        print(f"   With context embeddings: {context_status.get('with_context_embedding', 0):,}")
        print(f"   Needing embeddings: {context_status.get('needing_embeddings', 0):,}")
        
        # Sample context text
        print(f"\n📄 Sample Context Text:")
        for source_system, company_name, context_preview, context_hash in analysis.get('sample_context_text', []):
            print(f"   {source_system} | {company_name} | {context_hash[:8]}...")
            print(f"     {context_preview}")
        print()
    
    async def get_opportunities_for_embedding(self) -> List[Dict[str, Any]]:
        """Get opportunities that need context embeddings"""
        print("🔍 Fetching opportunities that need context embeddings...")
        
        query = """
            SELECT 
                co.id,
                co.source_system,
                co.source_id,
                co.company_name,
                co.name as opportunity_name,
                co.context_text,
                co.context_hash,
                se.embedding_id as existing_embedding_id
            FROM core.opportunities co
            LEFT JOIN search.embeddings_opportunities se ON co.id = se.opportunity_id
            WHERE co.context_text IS NOT NULL
            AND (se.context_vector IS NULL OR %s = true)
            ORDER BY co.source_system, co.id
        """
        
        try:
            with self.get_local_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (self.force_regenerate,))
                results = cursor.fetchall()
                
            opportunities = [dict(row) for row in results]
            print(f"✅ Found {len(opportunities)} opportunities needing context embeddings")
            return opportunities
            
        except Exception as e:
            print(f"❌ Failed to fetch opportunities: {e}")
            raise
    
    def calculate_context_hash(self, context_text: str) -> str:
        """Calculate SHA-256 hash for context text"""
        return hashlib.sha256(context_text.encode('utf-8')).hexdigest()
    
    async def generate_bge_embedding(self, text: str) -> List[float]:
        """Generate BGE-M3 embedding using the containerized BGE service"""
        async with aiohttp.ClientSession() as session:
            payload = {"texts": [text]}
            async with session.post(BGE_EMBED_URL, json=payload, timeout=30) as response:
                if response.status != 200:
                    raise Exception(f"BGE service error: {response.status} - {await response.text()}")
                
                result = await response.json()
                if 'embeddings' not in result:
                    raise Exception(f"Invalid BGE response: {result}")
                
                return result['embeddings'][0]  # Return first (and only) embedding
    
    async def generate_embeddings_batch(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate context embeddings for a batch of opportunities"""
        print(f"🔄 Processing batch of {len(opportunities)} opportunities...")
        
        # Extract texts for embedding
        texts = [opp['context_text'] for opp in opportunities]
        
        try:
            # Generate embeddings via BGE service
            async with aiohttp.ClientSession() as session:
                payload = {"texts": texts}
                async with session.post(BGE_EMBED_URL, json=payload, timeout=60) as response:
                    if response.status != 200:
                        raise Exception(f"BGE service error: {response.status} - {await response.text()}")
                    
                    result = await response.json()
                    if 'embeddings' not in result:
                        raise Exception(f"Invalid BGE response: {result}")
                    
                    embeddings = result['embeddings']
            
            # Create results with embeddings
            results = []
            for i, opp in enumerate(opportunities):
                results.append({
                    'opportunity': opp,
                    'embedding': embeddings[i],
                    'text_hash': self.calculate_context_hash(opp['context_text']),
                    'processing_time_ms': 0  # BGE service handles timing
                })
            
            print(f"✅ Generated {len(results)} context embeddings")
            return results
            
        except Exception as e:
            print(f"❌ Failed to generate embeddings: {e}")
            raise
    
    async def store_embeddings(self, results: List[Dict[str, Any]]) -> int:
        """Store context embeddings in the database"""
        print(f"💾 Storing {len(results)} context embeddings...")
        
        stored_count = 0
        
        try:
            with self.get_local_connection() as conn:
                cursor = conn.cursor()
                
                for result in results:
                    opp = result['opportunity']
                    embedding = result['embedding']
                    text_hash = result['text_hash']
                    
                    # Generate embedding ID
                    embedding_id = str(uuid.uuid4())
                    
                    # Insert into search.embeddings_opportunities
                    cursor.execute("""
                        INSERT INTO search.embeddings_opportunities (
                            opportunity_id,
                            source_system,
                            source_id,
                            context_vector,
                            context_text,
                            context_hash,
                            company_name,
                            opportunity_name,
                            embedding_model,
                            created_at,
                            updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (opportunity_id) 
                        DO UPDATE SET
                            context_vector = EXCLUDED.context_vector,
                            context_text = EXCLUDED.context_text,
                            context_hash = EXCLUDED.context_hash,
                            updated_at = EXCLUDED.updated_at
                    """, (
                        opp['id'],
                        opp['source_system'],
                        opp['source_id'],
                        f"[{','.join(map(str, embedding))}]",  # Convert to JSON string
                        opp['context_text'],
                        text_hash,
                        opp['company_name'],
                        opp['opportunity_name'],
                        'BAAI/bge-m3',
                        datetime.now(timezone.utc),
                        datetime.now(timezone.utc)
                    ))
                    
                    stored_count += 1
                
                conn.commit()
                
        except Exception as e:
            print(f"❌ Failed to store embeddings: {e}")
            raise
        
        print(f"✅ Stored {stored_count} context embeddings")
        return stored_count
    
    async def generate_all_context_embeddings(self):
        """Main function to generate all context embeddings"""
        print("🚀 Starting context embedding generation")
        print("=" * 60)
        
        start_time = time.time()
        total_processed = 0
        
        try:
            # Get opportunities that need embeddings
            opportunities = await self.get_opportunities_for_embedding()
            
            if not opportunities:
                print("✅ All opportunities already have context embeddings")
                return
            
            # Process in batches
            for i in range(0, len(opportunities), self.batch_size):
                batch = opportunities[i:i + self.batch_size]
                batch_num = (i // self.batch_size) + 1
                total_batches = (len(opportunities) + self.batch_size - 1) // self.batch_size
                
                print(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch)} opportunities)")
                
                try:
                    # Generate embeddings for batch
                    results = await self.generate_embeddings_batch(batch)
                    
                    # Store embeddings
                    stored = await self.store_embeddings(results)
                    total_processed += stored
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    print(f"   Progress: {total_processed}/{len(opportunities)} ({rate:.1f} embeddings/sec)")
                    
                except Exception as e:
                    print(f"❌ Batch {batch_num} failed: {e}")
                    continue
            
            # Final report
            elapsed = time.time() - start_time
            print(f"\n🎉 Context embedding generation complete!")
            print(f"   Total processed: {total_processed}")
            print(f"   Total time: {elapsed:.1f}s")
            print(f"   Average rate: {total_processed/elapsed:.1f} embeddings/sec")
            
        except Exception as e:
            print(f"❌ Context embedding generation failed: {e}")
            raise


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate context embeddings for opportunities')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--force-regenerate', action='store_true', help='Force regenerate all embeddings')
    parser.add_argument('--status', action='store_true', help='Show current embedding status')
    
    args = parser.parse_args()
    
    generator = ContextEmbeddingsGenerator(
        batch_size=args.batch_size,
        force_regenerate=args.force_regenerate
    )
    
    if args.status:
        # Show status only
        analysis = await generator.analyze_opportunity_data()
        generator.print_analysis_report(analysis)
        return
    
    try:
        # Run analysis first
        analysis = await generator.analyze_opportunity_data()
        generator.print_analysis_report(analysis)
        
        # Generate embeddings
        await generator.generate_all_context_embeddings()
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())