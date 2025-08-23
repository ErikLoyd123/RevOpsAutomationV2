#!/usr/bin/env python3
"""
Complete APN→Odoo Matching - ALL APN opportunities with optimized processing
"""

import asyncio
import asyncpg
import json
import csv
import numpy as np
from datetime import datetime
from fuzzywuzzy import fuzz
import time

def vectorized_cosine_similarity(query_vec, candidate_matrix):
    """Calculate cosine similarity between one query vector and many candidate vectors efficiently"""
    try:
        query = np.array(query_vec).reshape(1, -1)
        candidates = np.array(candidate_matrix)
        
        # Normalize vectors
        query_norm = query / np.linalg.norm(query, axis=1, keepdims=True)
        candidates_norm = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarities = np.dot(query_norm, candidates_norm.T).flatten()
        return similarities
    except:
        return np.zeros(len(candidate_matrix))

async def main():
    print("🎯 Complete APN→Odoo Matching - ALL APN Opportunities")
    print("Optimized for speed with vectorized similarity calculations")
    print("=" * 70)
    
    conn = await asyncpg.connect('postgresql://revops_user:RevOps2024Secure!@localhost:5432/revops_core')
    
    try:
        # Get ALL APN opportunities
        print("📊 Loading ALL APN opportunities...")
        apn_opps = await conn.fetch("""
            SELECT 
                e.source_id,
                o.name,
                o.partner_name,
                e.identity_vector,
                e.context_vector,
                e.opportunity_value
            FROM search.embeddings_opportunities e
            JOIN core.opportunities o ON o.source_id = e.source_id AND o.source_system = e.source_system
            WHERE e.source_system = 'apn' 
            AND e.context_vector IS NOT NULL
            AND e.identity_vector IS NOT NULL
            AND LENGTH(e.context_vector) > 1000
            ORDER BY e.opportunity_value DESC NULLS LAST
        """)
        
        print(f"✅ Loaded {len(apn_opps)} APN opportunities")
        
        # Get ALL Odoo opportunities and pre-process vectors
        print("📊 Loading and pre-processing ALL Odoo opportunities...")
        odoo_raw = await conn.fetch("""
            SELECT 
                e.source_id,
                o.name,
                o.partner_name,
                e.identity_vector,
                e.context_vector,
                e.opportunity_value
            FROM search.embeddings_opportunities e
            JOIN core.opportunities o ON o.source_id = e.source_id AND o.source_system = e.source_system
            WHERE e.source_system = 'odoo'
            AND e.context_vector IS NOT NULL
            AND e.identity_vector IS NOT NULL
        """)
        
        # Pre-process all Odoo vectors into matrices for vectorized operations
        print("🔧 Building Odoo vector matrices...")
        odoo_data = []
        odoo_context_matrix = []
        odoo_identity_matrix = []
        
        for odoo in odoo_raw:
            try:
                context_vec = json.loads(odoo['context_vector'])
                identity_vec = json.loads(odoo['identity_vector'])
                
                if len(context_vec) == 1024 and len(identity_vec) == 1024:
                    odoo_data.append({
                        'source_id': odoo['source_id'],
                        'name': odoo['name'],
                        'partner_name': odoo['partner_name'],
                        'opportunity_value': odoo['opportunity_value']
                    })
                    odoo_context_matrix.append(context_vec)
                    odoo_identity_matrix.append(identity_vec)
            except:
                continue
        
        odoo_context_matrix = np.array(odoo_context_matrix)
        odoo_identity_matrix = np.array(odoo_identity_matrix)
        
        print(f"✅ Built matrices for {len(odoo_data)} Odoo opportunities")
        print(f"📐 Context matrix shape: {odoo_context_matrix.shape}")
        print(f"📐 Identity matrix shape: {odoo_identity_matrix.shape}")
        
        # CSV setup
        csv_data = []
        csv_headers = [
            'APN_ID', 'APN_Name', 'APN_Partner', 'APN_Value',
            'Rank', 'Odoo_ID', 'Odoo_Name', 'Odoo_Partner', 'Odoo_Value',
            'Context_Similarity', 'Identity_Similarity', 'Company_Fuzzy', 'Overall_Score'
        ]
        
        # Process each APN opportunity with vectorized similarity
        print(f"\n🚀 Processing {len(apn_opps)} APN opportunities...")
        start_time = time.time()
        
        for idx, apn in enumerate(apn_opps, 1):
            if idx % 100 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                remaining = (len(apn_opps) - idx) / rate
                print(f"   📊 Progress: {idx}/{len(apn_opps)} ({idx/len(apn_opps)*100:.1f}%) - "
                      f"Rate: {rate:.1f}/s - ETA: {remaining/60:.1f}min")
            
            try:
                # Parse APN vectors
                apn_context = json.loads(apn['context_vector'])
                apn_identity = json.loads(apn['identity_vector'])
                
                if len(apn_context) != 1024 or len(apn_identity) != 1024:
                    continue
                
                # Vectorized similarity calculations
                context_similarities = vectorized_cosine_similarity(apn_context, odoo_context_matrix)
                identity_similarities = vectorized_cosine_similarity(apn_identity, odoo_identity_matrix)
                
                # Calculate company fuzzy scores for all Odoo opportunities
                apn_company = (apn['partner_name'] or "").lower()
                fuzzy_scores = np.array([
                    fuzz.ratio(apn_company, (odoo['partner_name'] or "").lower()) / 100.0
                    for odoo in odoo_data
                ])
                
                # Calculate overall scores
                overall_scores = (0.4 * context_similarities + 
                                0.3 * identity_similarities + 
                                0.3 * fuzzy_scores)
                
                # Get top 5 matches
                top_5_indices = np.argsort(overall_scores)[-5:][::-1]
                
                # Add to CSV data
                for rank, odoo_idx in enumerate(top_5_indices, 1):
                    odoo = odoo_data[odoo_idx]
                    csv_data.append([
                        apn['source_id'],
                        apn['name'],
                        apn['partner_name'],
                        apn['opportunity_value'] or 0,
                        rank,
                        odoo['source_id'],
                        odoo['name'],
                        odoo['partner_name'],
                        odoo['opportunity_value'] or 0,
                        f"{context_similarities[odoo_idx]:.3f}",
                        f"{identity_similarities[odoo_idx]:.3f}",
                        f"{fuzzy_scores[odoo_idx]:.3f}",
                        f"{overall_scores[odoo_idx]:.3f}"
                    ])
                
            except Exception as e:
                print(f"   ❌ Error processing APN {apn['source_id']}: {e}")
                continue
        
        # Write CSV
        print(f"\n💾 Writing CSV file...")
        with open("apn_odoo_matches.csv", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
            writer.writerows(csv_data)
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 COMPLETE SUCCESS!")
        print(f"📁 File: apn_odoo_matches.csv")
        print(f"📊 Total APN opportunities: {len(apn_opps)}")
        print(f"📊 Total matches: {len(csv_data)} rows ({len(csv_data)/len(apn_opps):.1f} per APN)")
        print(f"⏱️  Total processing time: {total_time/60:.1f} minutes")
        print(f"🎯 Using real BGE-M3 1024-dimensional embeddings!")
        print(f"✅ Ready for your manual testing and analysis!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await conn.close()

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🎉 Complete APN→Odoo matching analysis ready!")
    else:
        print("\n❌ Failed")