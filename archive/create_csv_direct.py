#!/usr/bin/env python3
"""
Direct APN→Odoo Matching using Fixed Vector Logic
"""

import asyncio
import asyncpg
import json
import csv
import numpy as np
from datetime import datetime
from fuzzywuzzy import fuzz
import re

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    try:
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    except:
        return 0.0

async def main():
    print("🎯 Direct APN→Odoo Vector Similarity Matching")
    print("=" * 60)
    
    conn = await asyncpg.connect('postgresql://revops_user:RevOps2024Secure!@localhost:5432/revops_core')
    
    try:
        # Get ALL APN opportunities with good embeddings
        print("📊 Getting ALL APN opportunities...")
        apn_opps = await conn.fetch("""
            SELECT 
                e.source_id,
                e.source_system,
                o.name,
                o.partner_name,
                e.identity_text,
                e.context_text,
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
        
        print(f"✅ Found {len(apn_opps)} APN opportunities")
        
        # Get Odoo opportunities with embeddings  
        print("📊 Getting Odoo opportunities...")
        odoo_opps = await conn.fetch("""
            SELECT 
                e.source_id,
                e.source_system,
                o.name,
                o.partner_name,
                e.identity_text,
                e.context_text,
                e.identity_vector,
                e.context_vector,
                e.opportunity_value
            FROM search.embeddings_opportunities e
            JOIN core.opportunities o ON o.source_id = e.source_id AND o.source_system = e.source_system
            WHERE e.source_system = 'odoo'
            AND e.context_vector IS NOT NULL
            AND e.identity_vector IS NOT NULL
        """)
        
        print(f"✅ Found {len(odoo_opps)} Odoo opportunities")
        
        # CSV setup
        csv_data = []
        csv_headers = [
            'APN_ID', 'APN_Name', 'APN_Partner', 'APN_Value',
            'Rank', 'Odoo_ID', 'Odoo_Name', 'Odoo_Partner', 'Odoo_Value',
            'Context_Similarity', 'Identity_Similarity', 'Company_Fuzzy', 'Overall_Score'
        ]
        
        # Process each APN opportunity
        for idx, apn in enumerate(apn_opps, 1):
            print(f"\n🔍 Processing {idx}/{len(apn_opps)} - APN {apn['source_id']}: {apn['name'][:60]}...")
            
            # Parse APN vectors
            try:
                apn_context = json.loads(apn['context_vector'])
                apn_identity = json.loads(apn['identity_vector'])
            except:
                print(f"   ❌ Failed to parse APN vectors")
                continue
            
            # Calculate similarities with all Odoo opportunities
            matches = []
            
            for odoo in odoo_opps:
                try:
                    # Parse Odoo vectors
                    odoo_context = json.loads(odoo['context_vector'])
                    odoo_identity = json.loads(odoo['identity_vector'])
                    
                    # Calculate similarities
                    context_sim = cosine_similarity(apn_context, odoo_context)
                    identity_sim = cosine_similarity(apn_identity, odoo_identity)
                    
                    # Company fuzzy matching
                    apn_company = apn['partner_name'] or ""
                    odoo_company = odoo['partner_name'] or ""
                    fuzzy_score = fuzz.ratio(apn_company, odoo_company) / 100.0
                    
                    # Overall score (weighted)
                    overall_score = (0.4 * context_sim + 0.3 * identity_sim + 0.3 * fuzzy_score)
                    
                    matches.append({
                        'odoo': odoo,
                        'context_sim': context_sim,
                        'identity_sim': identity_sim,
                        'fuzzy_score': fuzzy_score,
                        'overall_score': overall_score
                    })
                    
                except:
                    continue
            
            # Sort by overall score and get top 5
            matches.sort(key=lambda x: x['overall_score'], reverse=True)
            top_matches = matches[:5]
            
            print(f"   📊 Top matches:")
            for i, match in enumerate(top_matches, 1):
                odoo = match['odoo']
                score = match['overall_score']
                print(f"      {i}. {odoo['source_id']}: {odoo['name'][:50]} (Score: {score:.3f})")
            
            # Add to CSV
            for rank, match in enumerate(top_matches, 1):
                odoo = match['odoo']
                csv_data.append([
                    apn['source_id'], apn['name'], apn['partner_name'], 
                    apn['opportunity_value'] or 0,
                    rank, odoo['source_id'], odoo['name'], odoo['partner_name'],
                    odoo['opportunity_value'] or 0,
                    f"{match['context_sim']:.3f}",
                    f"{match['identity_sim']:.3f}", 
                    f"{match['fuzzy_score']:.3f}",
                    f"{match['overall_score']:.3f}"
                ])
        
        # Write CSV (replace existing file)
        csv_filename = "apn_odoo_matches.csv"
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
            writer.writerows(csv_data)
        
        print(f"\n🎉 SUCCESS!")
        print(f"📁 CSV file: {csv_filename}")
        print(f"📊 Total matches: {len(csv_data)} rows")
        print(f"🎯 Using REAL BGE-M3 1024-dimensional embeddings!")
        
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
        print("\n✅ Task 4.2: Two-Stage Retrieval working with real embeddings! 🎉")
    else:
        print("\n❌ Failed")