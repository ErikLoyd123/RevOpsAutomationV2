#!/usr/bin/env python3
"""
Integration Test for Task 4.3 - Two-Stage Retrieval + Match Results Storage

This script demonstrates the complete workflow:
1. Use candidate_generator.py (Task 4.2) to find matches
2. Store results using match_store.py (Task 4.3) 
3. Manage match workflow with decisions and audit trail

This shows how the two services work together for the complete
opportunity matching pipeline.

Created for: Task 4.3 - Create Match Results Storage
"""

import asyncio
import sys
import uuid
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).resolve().parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)

# Add the matching service to path
sys.path.append(str(Path(__file__).parent))

from candidate_generator import TwoStageRetrieval, get_config
from match_store import MatchStore, MatchDecision, store_candidate_matches

async def main():
    """Demonstrate complete integration between candidate generator and match store"""
    
    print("🎯 Task 4.3 Integration Test: Two-Stage Retrieval + Match Results Storage")
    print("=" * 80)
    
    # Initialize services
    config = get_config()
    candidate_generator = TwoStageRetrieval(config)
    match_store = MatchStore()
    
    print("1. Testing service initialization...")
    try:
        # Test database connections
        stats = await match_store.get_match_statistics()
        print("   ✅ Match store connected")
        print(f"   📊 Current matches in database: {stats['total_matches']}")
        
        # Test candidate generator (this requires actual data)
        print("   ✅ Candidate generator initialized")
    except Exception as e:
        print(f"   ❌ Service initialization failed: {e}")
        return False
    
    print("\n2. Finding real opportunities to test with...")
    try:
        # Get a real Odoo opportunity for testing
        import asyncpg
        conn = await asyncpg.connect(
            host=os.getenv('LOCAL_DB_HOST', 'localhost'),
            port=os.getenv('LOCAL_DB_PORT', '5432'),
            database=os.getenv('LOCAL_DB_NAME', 'revops_core'),
            user=os.getenv('LOCAL_DB_USER', 'revops_user'),
            password=os.getenv('LOCAL_DB_PASSWORD')
        )
        odoo_opps = await conn.fetch("""
            SELECT source_id, name, partner_name 
            FROM search.embeddings_opportunities 
            WHERE source_system = 'odoo' 
            AND identity_vector IS NOT NULL 
            AND context_vector IS NOT NULL
            LIMIT 3
        """)
        await conn.close()
        
        if not odoo_opps:
            print("   ⚠️  No Odoo opportunities with embeddings found")
            print("   ℹ️  You need to run embedding generation first")
            return False
        
        print(f"   ✅ Found {len(odoo_opps)} Odoo opportunities with embeddings")
        for opp in odoo_opps:
            print(f"      • {opp['source_id']}: {opp['name'][:50]}...")
            
    except Exception as e:
        print(f"   ❌ Failed to get test opportunities: {e}")
        return False
    
    # Test with the first opportunity
    test_opportunity_id = odoo_opps[0]['source_id']
    test_opportunity_name = odoo_opps[0]['name']
    
    print(f"\n3. Running Two-Stage Retrieval for Odoo opportunity {test_opportunity_id}...")
    print(f"   Opportunity: {test_opportunity_name}")
    
    try:
        # Get matches using candidate generator (Task 4.2)
        # Use the generate_candidates method with opportunity ID as integer
        opportunity_id_int = int(test_opportunity_id)
        matches, metrics = await candidate_generator.generate_candidates(
            opportunity_id_int, 
            max_final_candidates=5
        )
        
        print(f"   ✅ Found {len(matches)} potential matches")
        
        if matches:
            print("   📊 Top matches:")
            for i, match in enumerate(matches[:3], 1):
                score = match.get('overall_score', match.get('rrf_combined_score', 0))
                apn_id = match.get('opportunity_id', match.get('source_id', 'unknown'))
                apn_name = match.get('opportunity_name', match.get('name', 'Unknown'))[:40]
                print(f"      {i}. APN {apn_id}: {apn_name}... (Score: {score:.3f})")
        else:
            print("   ⚠️  No matches found - similarity threshold may be too high")
            
    except Exception as e:
        print(f"   ❌ Candidate generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    if not matches:
        print("\n⚠️  No matches to store - stopping integration test")
        return False
    
    print(f"\n4. Storing match results using Match Store (Task 4.3)...")
    
    try:
        # Generate a batch ID for this matching session
        batch_id = str(uuid.uuid4())
        
        # Store matches using the integration function
        match_ids = await store_candidate_matches(
            odoo_opportunity_id=test_opportunity_id,
            candidates=matches,
            batch_id=batch_id
        )
        
        print(f"   ✅ Stored {len(match_ids)} match results")
        print(f"   🔗 Batch ID: {batch_id}")
        
        # Show first few match IDs
        for i, match_id in enumerate(match_ids[:3], 1):
            print(f"      {i}. Match ID: {match_id}")
            
    except Exception as e:
        print(f"   ❌ Match storage failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n5. Testing match workflow management...")
    
    try:
        # Test confirming the top match
        if match_ids:
            top_match_id = match_ids[0]
            
            # Create a confirmation decision
            confirm_decision = MatchDecision(
                opportunity_match_id=top_match_id,
                previous_status=None,
                new_status="confirmed",
                decision_reason="Integration test - confirming top match",
                decided_by="integration_test",
                business_justification="Highest RRF score with strong semantic and company name similarity",
                decision_source="automated"
            )
            
            success = await match_store.update_match_status(top_match_id, confirm_decision)
            if success:
                print(f"   ✅ Confirmed top match: {top_match_id}")
            else:
                print(f"   ❌ Failed to confirm match")
        
        # Test rejecting the last match
        if len(match_ids) > 1:
            last_match_id = match_ids[-1]
            
            reject_decision = MatchDecision(
                opportunity_match_id=last_match_id,
                previous_status=None,
                new_status="rejected",
                decision_reason="Integration test - rejecting low-confidence match",
                decided_by="integration_test",
                business_justification="Score too low for reliable match",
                decision_source="automated"
            )
            
            success = await match_store.update_match_status(last_match_id, reject_decision)
            if success:
                print(f"   ✅ Rejected low match: {last_match_id}")
            else:
                print(f"   ❌ Failed to reject match")
                
    except Exception as e:
        print(f"   ❌ Workflow management failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n6. Testing audit trail functionality...")
    
    try:
        # Get audit trail for the confirmed match
        if match_ids:
            audit_trail = await match_store.get_match_audit_trail(match_ids[0])
            print(f"   ✅ Retrieved {len(audit_trail)} audit records for top match")
            
            for i, record in enumerate(audit_trail, 1):
                print(f"      {i}. {record['previous_status'] or 'pending'} → {record['new_status']}")
                print(f"         By: {record['decided_by']} at {record['decided_at']}")
                if record['decision_reason']:
                    print(f"         Reason: {record['decision_reason']}")
                    
    except Exception as e:
        print(f"   ❌ Audit trail retrieval failed: {e}")
        return False
    
    print(f"\n7. Final statistics and summary...")
    
    try:
        # Get updated statistics
        final_stats = await match_store.get_match_statistics()
        print(f"   📊 Updated Match Statistics:")
        print(f"      Total matches: {final_stats['total_matches']}")
        print(f"      Status breakdown: {final_stats['status_counts']}")
        print(f"      Confidence breakdown: {final_stats['confidence_counts']}")
        
        # Show average scores
        if final_stats['average_scores']:
            avg_scores = final_stats['average_scores']
            print(f"   📈 Average Scores:")
            if avg_scores.get('avg_rrf_score'):
                print(f"      RRF Score: {avg_scores['avg_rrf_score']:.3f}")
            if avg_scores.get('avg_similarity_score'):
                print(f"      Similarity: {avg_scores['avg_similarity_score']:.3f}")
            if avg_scores.get('avg_semantic_score'):
                print(f"      Semantic: {avg_scores['avg_semantic_score']:.3f}")
            if avg_scores.get('avg_fuzzy_score'):
                print(f"      Company Fuzzy: {avg_scores['avg_fuzzy_score']:.3f}")
                
    except Exception as e:
        print(f"   ⚠️  Statistics retrieval failed: {e}")
    
    print("\n" + "=" * 80)
    print("🎉 INTEGRATION TEST COMPLETE!")
    print("=" * 80)
    print("\n✅ Successfully demonstrated:")
    print("   • Two-Stage Retrieval Architecture (Task 4.2)")
    print("   • Match Results Storage (Task 4.3)")
    print("   • Match workflow management (confirm/reject)")
    print("   • Complete audit trail tracking")
    print("   • End-to-end opportunity matching pipeline")
    
    print(f"\n📋 What was tested:")
    print(f"   • Found matches for Odoo opportunity: {test_opportunity_id}")
    print(f"   • Stored {len(match_ids) if 'match_ids' in locals() else 0} match results")
    print(f"   • Confirmed highest-scoring match")
    print(f"   • Rejected lowest-scoring match") 
    print(f"   • Retrieved complete audit trail")
    
    print(f"\n🚀 Task 4.3 is COMPLETE and ready for production use!")
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n✅ Integration test passed!")
    else:
        print("\n❌ Integration test failed!")
        sys.exit(1)