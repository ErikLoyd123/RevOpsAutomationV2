#!/usr/bin/env python3
"""
Interactive CLI Testing Tool for Opportunity Matching - Task 4.6

Interactive command-line tool for testing specific opportunity matches.
Features:
- Display similarity scores and matching explanations
- Support manual match confirmation/rejection  
- Export results to CSV for analysis
- Real-time testing of matching algorithms

Prerequisites: Task 4.5 (Matching API Endpoints)
"""

import asyncio
import csv
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# Add backend paths for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "backend"))
sys.path.append(str(Path(__file__).parent.parent.parent / "backend" / "services" / "08-matching"))

from core.database import DatabaseManager
from candidate_generator import TwoStageRetrieval
from match_store import MatchStore

class MatchingTestCLI:
    """Interactive CLI for testing opportunity matching"""
    
    def __init__(self):
        """Initialize the CLI tool"""
        self.db = DatabaseManager()
        self.retrieval_engine = TwoStageRetrieval()
        self.match_store = MatchStore()
        self.session_results = []
        
    async def initialize(self):
        """Initialize database connections"""
        await self.db.initialize()
        print("🔗 Database connection established")
        
    async def list_opportunities(self, source_system: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """List available opportunities for testing"""
        
        query = """
        SELECT 
            id, source_system, external_id, company_name, 
            opportunity_name, stage, amount_usd, created_date
        FROM core.opportunities 
        """
        
        params = []
        if source_system:
            query += " WHERE source_system = $1"
            params.append(source_system)
            
        query += " ORDER BY created_date DESC LIMIT ${}".format(len(params) + 1)
        params.append(limit)
        
        async with self.db.get_connection() as conn:
            rows = await conn.fetch(query, *params)
            
        opportunities = []
        for row in rows:
            opportunities.append({
                'id': row['id'],
                'source_system': row['source_system'],
                'external_id': row['external_id'],
                'company_name': row['company_name'],
                'opportunity_name': row['opportunity_name'],
                'stage': row['stage'],
                'amount_usd': float(row['amount_usd']) if row['amount_usd'] else 0,
                'created_date': row['created_date'].strftime('%Y-%m-%d') if row['created_date'] else None
            })
            
        return opportunities
        
    async def get_opportunity_details(self, opportunity_id: str) -> Optional[Dict]:
        """Get detailed information about a specific opportunity"""
        
        query = """
        SELECT 
            id, source_system, external_id, company_name, opportunity_name,
            description, stage, amount_usd, close_date_est, salesperson_name,
            primary_contact_name, primary_contact_email, next_activity,
            identity_text, context_text, created_date, updated_date
        FROM core.opportunities 
        WHERE id = $1 OR external_id = $1
        """
        
        async with self.db.get_connection() as conn:
            row = await conn.fetchrow(query, opportunity_id)
            
        if not row:
            return None
            
        return {
            'id': row['id'],
            'source_system': row['source_system'],
            'external_id': row['external_id'],
            'company_name': row['company_name'],
            'opportunity_name': row['opportunity_name'],
            'description': row['description'],
            'stage': row['stage'],
            'amount_usd': float(row['amount_usd']) if row['amount_usd'] else 0,
            'close_date_est': row['close_date_est'].strftime('%Y-%m-%d') if row['close_date_est'] else None,
            'salesperson_name': row['salesperson_name'],
            'primary_contact_name': row['primary_contact_name'],
            'primary_contact_email': row['primary_contact_email'],
            'next_activity': row['next_activity'],
            'identity_text': row['identity_text'],
            'context_text': row['context_text'],
            'created_date': row['created_date'].strftime('%Y-%m-%d %H:%M:%S') if row['created_date'] else None,
            'updated_date': row['updated_date'].strftime('%Y-%m-%d %H:%M:%S') if row['updated_date'] else None
        }
        
    async def run_matching_test(self, opportunity_id: str, max_candidates: int = 5) -> Dict:
        """Run matching test for a specific opportunity"""
        
        print(f"\n🎯 Running matching test for opportunity: {opportunity_id}")
        print("=" * 60)
        
        # Get opportunity details
        opportunity = await self.get_opportunity_details(opportunity_id)
        if not opportunity:
            return {'error': f'Opportunity {opportunity_id} not found'}
            
        print(f"📋 Target Opportunity: {opportunity['company_name']} - {opportunity['opportunity_name']}")
        print(f"💰 Amount: ${opportunity['amount_usd']:,.2f} | Stage: {opportunity['stage']}")
        print(f"🏷️  Source: {opportunity['source_system']} | ID: {opportunity['external_id']}")
        
        # Run two-stage retrieval
        try:
            candidates = await self.retrieval_engine.find_candidates(
                opportunity_id=opportunity['id'],
                max_candidates=max_candidates,
                store_results=True
            )
            
            print(f"\n🔍 Found {len(candidates)} candidate matches:")
            print("-" * 60)
            
            match_results = []
            for i, candidate in enumerate(candidates, 1):
                print(f"\n{i}. {candidate['company_name']} - {candidate['opportunity_name']}")
                print(f"   💰 Amount: ${candidate['amount_usd']:,.2f} | Stage: {candidate['stage']}")
                print(f"   🏷️  Source: {candidate['source_system']} | ID: {candidate['external_id']}")
                print(f"   📊 Scores: Semantic={candidate['semantic_score']:.3f}, Exact={candidate['exact_score']:.3f}, Final={candidate['final_score']:.3f}")
                
                if candidate.get('explanation'):
                    print(f"   💡 Explanation: {candidate['explanation']}")
                    
                match_results.append({
                    'target_id': opportunity['id'],
                    'target_company': opportunity['company_name'],
                    'target_opportunity': opportunity['opportunity_name'],
                    'candidate_id': candidate['id'],
                    'candidate_company': candidate['company_name'],
                    'candidate_opportunity': candidate['opportunity_name'],
                    'semantic_score': candidate['semantic_score'],
                    'exact_score': candidate['exact_score'],
                    'final_score': candidate['final_score'],
                    'explanation': candidate.get('explanation', ''),
                    'timestamp': datetime.now().isoformat()
                })
                
            return {
                'opportunity': opportunity,
                'candidates': candidates,
                'match_results': match_results
            }
            
        except Exception as e:
            print(f"❌ Error during matching: {e}")
            return {'error': str(e)}
            
    async def interactive_confirmation(self, test_result: Dict) -> None:
        """Interactive confirmation/rejection of matches"""
        
        if 'error' in test_result:
            return
            
        candidates = test_result['candidates']
        opportunity = test_result['opportunity']
        
        print(f"\n🤔 Manual Match Review for: {opportunity['company_name']}")
        print("=" * 60)
        
        confirmed_matches = []
        
        for i, candidate in enumerate(candidates, 1):
            print(f"\nCandidate {i}: {candidate['company_name']} - {candidate['opportunity_name']}")
            print(f"Final Score: {candidate['final_score']:.3f}")
            
            while True:
                choice = input("Confirm this match? (y)es/(n)o/(s)kip/(q)uit: ").lower().strip()
                
                if choice in ['y', 'yes']:
                    reason = input("Confirmation reason (optional): ").strip()
                    
                    # Store confirmation in database
                    try:
                        await self.match_store.confirm_match(
                            match_id=candidate.get('match_id', f"{opportunity['id']}-{candidate['id']}"),
                            status='confirmed',
                            decided_by='cli_user',
                            decision_reason=reason or f"CLI confirmed - Score: {candidate['final_score']:.3f}"
                        )
                        
                        confirmed_matches.append({
                            **candidate,
                            'decision': 'confirmed',
                            'decision_reason': reason
                        })
                        print("✅ Match confirmed and saved")
                        
                    except Exception as e:
                        print(f"❌ Error saving confirmation: {e}")
                        
                    break
                    
                elif choice in ['n', 'no']:
                    reason = input("Rejection reason (optional): ").strip()
                    
                    # Store rejection in database
                    try:
                        await self.match_store.confirm_match(
                            match_id=candidate.get('match_id', f"{opportunity['id']}-{candidate['id']}"),
                            status='rejected',
                            decided_by='cli_user',
                            decision_reason=reason or f"CLI rejected - Score: {candidate['final_score']:.3f}"
                        )
                        
                        confirmed_matches.append({
                            **candidate,
                            'decision': 'rejected',
                            'decision_reason': reason
                        })
                        print("❌ Match rejected and saved")
                        
                    except Exception as e:
                        print(f"❌ Error saving rejection: {e}")
                        
                    break
                    
                elif choice in ['s', 'skip']:
                    print("⏭️  Match skipped")
                    confirmed_matches.append({
                        **candidate,
                        'decision': 'skipped'
                    })
                    break
                    
                elif choice in ['q', 'quit']:
                    print("🚪 Exiting confirmation process")
                    return
                    
                else:
                    print("Invalid choice. Please enter y/n/s/q")
                    
        # Add confirmations to session results
        self.session_results.extend(confirmed_matches)
        
        print(f"\n✅ Review complete: {len(confirmed_matches)} matches processed")
        
    def export_session_results(self, filename: Optional[str] = None) -> str:
        """Export session results to CSV"""
        
        if not self.session_results:
            print("No results to export")
            return ""
            
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"matching_test_results_{timestamp}.csv"
            
        filepath = Path(__file__).parent / filename
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'target_company', 'target_opportunity', 'candidate_company', 'candidate_opportunity',
                'semantic_score', 'exact_score', 'final_score', 'decision', 'decision_reason', 'timestamp'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.session_results:
                writer.writerow({
                    'target_company': result.get('target_company', ''),
                    'target_opportunity': result.get('target_opportunity', ''),
                    'candidate_company': result.get('company_name', ''),
                    'candidate_opportunity': result.get('opportunity_name', ''),
                    'semantic_score': result.get('semantic_score', 0),
                    'exact_score': result.get('exact_score', 0),
                    'final_score': result.get('final_score', 0),
                    'decision': result.get('decision', ''),
                    'decision_reason': result.get('decision_reason', ''),
                    'timestamp': result.get('timestamp', datetime.now().isoformat())
                })
                
        print(f"📄 Results exported to: {filepath}")
        return str(filepath)
        
    async def run_interactive_session(self):
        """Run interactive CLI session"""
        
        print("🎯 Interactive Opportunity Matching Test Tool")
        print("=" * 60)
        
        await self.initialize()
        
        while True:
            print("\n📋 Main Menu:")
            print("1. List opportunities")
            print("2. Test matching for specific opportunity")
            print("3. Export session results")
            print("4. Exit")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                source_filter = input("Filter by source system (odoo/apn) or press Enter for all: ").strip().lower()
                source_filter = source_filter if source_filter in ['odoo', 'apn'] else None
                
                limit = input("Number of opportunities to show (default 20): ").strip()
                limit = int(limit) if limit.isdigit() else 20
                
                opportunities = await self.list_opportunities(source_filter, limit)
                
                print(f"\n📋 Found {len(opportunities)} opportunities:")
                print("-" * 60)
                
                for opp in opportunities:
                    print(f"{opp['id']:<10} | {opp['company_name']:<30} | {opp['source_system']:<6} | ${opp['amount_usd']:>10,.0f}")
                    
            elif choice == '2':
                opp_id = input("Enter opportunity ID or external ID: ").strip()
                if not opp_id:
                    print("❌ Opportunity ID required")
                    continue
                    
                max_candidates = input("Max candidates to find (default 5): ").strip()
                max_candidates = int(max_candidates) if max_candidates.isdigit() else 5
                
                # Run matching test
                test_result = await self.run_matching_test(opp_id, max_candidates)
                
                if 'error' not in test_result:
                    # Ask for interactive confirmation
                    confirm = input("\nReview matches interactively? (y/n): ").lower().strip()
                    if confirm in ['y', 'yes']:
                        await self.interactive_confirmation(test_result)
                        
            elif choice == '3':
                filename = input("Export filename (press Enter for auto-generated): ").strip()
                filename = filename if filename else None
                self.export_session_results(filename)
                
            elif choice == '4':
                print("👋 Goodbye!")
                break
                
            else:
                print("Invalid choice. Please select 1-4.")


async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Interactive Opportunity Matching Test Tool')
    parser.add_argument('--opportunity-id', help='Test specific opportunity ID')
    parser.add_argument('--max-candidates', type=int, default=5, help='Maximum candidates to find')
    parser.add_argument('--export', help='Export filename for results')
    parser.add_argument('--batch-mode', action='store_true', help='Run in batch mode (no interactive prompts)')
    
    args = parser.parse_args()
    
    cli = MatchingTestCLI()
    
    if args.opportunity_id:
        # Single opportunity test mode
        await cli.initialize()
        
        test_result = await cli.run_matching_test(args.opportunity_id, args.max_candidates)
        
        if 'error' not in test_result and not args.batch_mode:
            await cli.interactive_confirmation(test_result)
            
        if args.export:
            cli.export_session_results(args.export)
            
    else:
        # Interactive session mode
        await cli.run_interactive_session()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Session interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)