#!/usr/bin/env python3
"""
Fast APN→Odoo Matching - Top 100 APN opportunities only
"""

import asyncio
import asyncpg
import json
import csv
import numpy as np
from datetime import datetime
from fuzzywuzzy import fuzz

def cosine_similarity(vec1, vec2):
    try:
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    except:
        return 0.0

async def main():
    print("🎯 Fast APN→Odoo Matching - Top 100 APN Opportunities")
    print("=" * 60)
    
    conn = await asyncpg.connect('postgresql://revops_user:RevOps2024Secure!@localhost:5432/revops_core')
    
    try:
        # Get top 100 highest-value APN opportunities
        print("📊 Getting top 100 APN opportunities...")
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
            LIMIT 100
        """)
        
        print(f"✅ Found {len(apn_opps)} APN opportunities")
        
        # Get all Odoo opportunities (pre-parse vectors)
        print("📊 Getting Odoo opportunities...")
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
        
        # Pre-parse all Odoo vectors for speed
        print("🔧 Pre-parsing Odoo vectors...")
        odoo_opps = []
        for odoo in odoo_raw:
            try:
                odoo_data = dict(odoo)
                odoo_data['context_vec'] = json.loads(odoo['context_vector'])
                odoo_data['identity_vec'] = json.loads(odoo['identity_vector'])
                odoo_opps.append(odoo_data)
            except:
                continue
        
        print(f"✅ Parsed {len(odoo_opps)} Odoo opportunities")
        
        # CSV setup
        csv_data = []
        csv_headers = [
            'APN_ID', 'APN_Name', 'APN_Partner', 'APN_Value',
            'Rank', 'Odoo_ID', 'Odoo_Name', 'Odoo_Partner', 'Odoo_Value',
            'Context_Similarity', 'Identity_Similarity', 'Company_Fuzzy', 'Overall_Score'
        ]
        
        # Process APN opportunities
        for idx, apn in enumerate(apn_opps, 1):
            print(f"🔍 {idx}/100 - APN {apn['source_id']}: {(apn['name'] or '')[:50]}...")
            
            try:
                apn_context = json.loads(apn['context_vector'])
                apn_identity = json.loads(apn['identity_vector'])
            except:
                continue
            
            # Calculate similarities with all Odoo opportunities
            matches = []
            
            for odoo in odoo_opps:
                try:
                    context_sim = cosine_similarity(apn_context, odoo['context_vec'])
                    identity_sim = cosine_similarity(apn_identity, odoo['identity_vec'])
                    
                    apn_company = apn['partner_name'] or ""
                    odoo_company = odoo['partner_name'] or ""
                    fuzzy_score = fuzz.ratio(apn_company, odoo_company) / 100.0
                    
                    overall_score = (0.4 * context_sim + 0.3 * identity_sim + 0.3 * fuzzy_score)
                    
                    if overall_score > 0.3:  # Only keep decent matches
                        matches.append({
                            'odoo': odoo,
                            'context_sim': context_sim,
                            'identity_sim': identity_sim,
                            'fuzzy_score': fuzzy_score,
                            'overall_score': overall_score
                        })
                except:
                    continue
            
            # Sort and get top 5
            matches.sort(key=lambda x: x['overall_score'], reverse=True)
            top_matches = matches[:5]
            
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
        
        # Write CSV (replace existing)
        with open("apn_odoo_matches.csv", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
            writer.writerows(csv_data)
        
        print(f"\n🎉 SUCCESS!")
        print(f"📁 CSV file: apn_odoo_matches.csv")
        print(f"📊 Total matches: {len(csv_data)} rows")
        print(f"🎯 Top 100 APN opportunities with BGE-M3 similarity!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    finally:
        await conn.close()

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n✅ Fast CSV generation complete!")
    else:
        print("\n❌ Failed")