#!/usr/bin/env python3
"""
Complete Odoo→APN Matching - ALL Odoo opportunities with optimized processing
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
    print("🎯 Complete Odoo→APN Matching - ALL Odoo Opportunities")
    print("Optimized for speed with vectorized similarity calculations")
    print("=" * 70)
    
    conn = await asyncpg.connect('postgresql://revops_user:RevOps2024Secure!@localhost:5432/revops_core')
    
    try:
        # Get ALL Odoo opportunities
        print("📊 Loading ALL Odoo opportunities...")
        odoo_opps = await conn.fetch("""
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
            AND LENGTH(e.context_vector) > 1000
            ORDER BY e.opportunity_value DESC NULLS LAST
        """)
        
        print(f"✅ Loaded {len(odoo_opps)} Odoo opportunities")
        
        # Get ALL APN opportunities and pre-process vectors
        print("📊 Loading and pre-processing ALL APN opportunities...")
        apn_raw = await conn.fetch("""
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
        """)
        
        # Pre-process all APN vectors into matrices for vectorized operations
        print("🔧 Building APN vector matrices...")
        apn_data = []
        apn_context_matrix = []
        apn_identity_matrix = []
        
        for apn in apn_raw:
            try:
                context_vec = json.loads(apn['context_vector'])
                identity_vec = json.loads(apn['identity_vector'])
                
                if len(context_vec) == 1024 and len(identity_vec) == 1024:
                    apn_data.append({
                        'source_id': apn['source_id'],
                        'name': apn['name'],
                        'partner_name': apn['partner_name'],
                        'opportunity_value': apn['opportunity_value']
                    })
                    apn_context_matrix.append(context_vec)
                    apn_identity_matrix.append(identity_vec)
            except:
                continue
        
        apn_context_matrix = np.array(apn_context_matrix)
        apn_identity_matrix = np.array(apn_identity_matrix)
        
        print(f"✅ Built matrices for {len(apn_data)} APN opportunities")
        print(f"📐 Context matrix shape: {apn_context_matrix.shape}")
        print(f"📐 Identity matrix shape: {apn_identity_matrix.shape}")
        
        # CSV setup
        csv_data = []
        csv_headers = [
            'Odoo_ID', 'Odoo_Name', 'Odoo_Partner', 'Odoo_Value',
            'Rank', 'APN_ID', 'APN_Name', 'APN_Partner', 'APN_Value',
            'Context_Similarity', 'Identity_Similarity', 'Company_Fuzzy', 'Overall_Score'
        ]
        
        # Process each Odoo opportunity with vectorized similarity
        print(f"\n🚀 Processing {len(odoo_opps)} Odoo opportunities...")
        start_time = time.time()
        
        for idx, odoo in enumerate(odoo_opps, 1):
            if idx % 100 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                remaining = (len(odoo_opps) - idx) / rate
                print(f"   📊 Progress: {idx}/{len(odoo_opps)} ({idx/len(odoo_opps)*100:.1f}%) - "
                      f"Rate: {rate:.1f}/s - ETA: {remaining/60:.1f}min")
            
            try:
                # Parse Odoo vectors
                odoo_context = json.loads(odoo['context_vector'])
                odoo_identity = json.loads(odoo['identity_vector'])
                
                if len(odoo_context) != 1024 or len(odoo_identity) != 1024:
                    continue
                
                # Vectorized similarity calculations
                context_similarities = vectorized_cosine_similarity(odoo_context, apn_context_matrix)
                identity_similarities = vectorized_cosine_similarity(odoo_identity, apn_identity_matrix)
                
                # Calculate company fuzzy scores for all APN opportunities
                odoo_company = (odoo['partner_name'] or "").lower()
                fuzzy_scores = np.array([
                    fuzz.ratio(odoo_company, (apn['partner_name'] or "").lower()) / 100.0
                    for apn in apn_data
                ])
                
                # Calculate overall scores
                overall_scores = (0.4 * context_similarities + 
                                0.3 * identity_similarities + 
                                0.3 * fuzzy_scores)
                
                # Get top 5 matches
                top_5_indices = np.argsort(overall_scores)[-5:][::-1]
                
                # Add to CSV data
                for rank, apn_idx in enumerate(top_5_indices, 1):
                    apn = apn_data[apn_idx]
                    csv_data.append([
                        odoo['source_id'],
                        odoo['name'],
                        odoo['partner_name'],
                        odoo['opportunity_value'] or 0,
                        rank,
                        apn['source_id'],
                        apn['name'],
                        apn['partner_name'],
                        apn['opportunity_value'] or 0,
                        f"{context_similarities[apn_idx]:.3f}",
                        f"{identity_similarities[apn_idx]:.3f}",
                        f"{fuzzy_scores[apn_idx]:.3f}",
                        f"{overall_scores[apn_idx]:.3f}"
                    ])
                
            except Exception as e:
                print(f"   ❌ Error processing Odoo {odoo['source_id']}: {e}")
                continue
        
        # Write CSV
        print(f"\n💾 Writing CSV file...")
        with open("odoo_apn_matches.csv", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
            writer.writerows(csv_data)
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 COMPLETE SUCCESS!")
        print(f"📁 File: odoo_apn_matches.csv")
        print(f"📊 Total Odoo opportunities: {len(odoo_opps)}")
        print(f"📊 Total matches: {len(csv_data)} rows ({len(csv_data)/len(odoo_opps):.1f} per Odoo)")
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
        print("\n🎉 Complete Odoo→APN matching analysis ready!")
    else:
        print("\n❌ Failed")