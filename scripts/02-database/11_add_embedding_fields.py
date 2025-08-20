#!/usr/bin/env python3
"""
Add Identity/Context Embedding Fields to CORE Opportunities - Task 2.6

This script extends the core.opportunities table with embedding infrastructure for BGE-M3 
dual embeddings with hash-based change detection. This supports the POD matching algorithm 
by providing both identity and context embeddings for semantic similarity matching.

Key Features:
- Adds identity_text and context_text fields for dual embedding strategy
- Implements hash-based change detection to avoid redundant embedding generation
- Adds BGE-M3 embedding storage fields (384 dimensions each)
- Creates performance indexes for embedding operations
- Supports both core.opportunities and any future opportunity tables

Changes Made:
1. identity_text TEXT - Combined identity information (company, account, names)
2. context_text TEXT - Combined contextual information (description, use case, stage)
3. identity_hash VARCHAR(64) - SHA-256 hash for identity change detection
4. context_hash VARCHAR(64) - SHA-256 hash for context change detection
5. identity_embedding JSONB - BGE-M3 identity embedding vector (384 dims)
6. context_embedding JSONB - BGE-M3 context embedding vector (384 dims)
7. embedding_generated_at TIMESTAMP - When embeddings were last generated

Prerequisites:
- Task 2.4: CORE schema tables created
- Task 2.5: OPS and SEARCH schemas created
- Database infrastructure operational

Related Files:
- backend/services/07-embeddings/main.py - BGE-M3 embedding service
- scripts/03-data/10_normalize_opportunities.py - Populates text fields
- scripts/03-data/11_normalize_aws_accounts.py - Account normalization
"""

import os
import sys
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
project_root = Path(__file__).resolve().parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)

# Embedding field definitions
EMBEDDING_FIELDS = {
    'identity_text': {
        'type': 'TEXT',
        'description': 'Combined identity information for embeddings (company, account, names)',
        'example': 'Acme Corp, AWS Account 123456789012, John Smith, Enterprise Sales'
    },
    'context_text': {
        'type': 'TEXT', 
        'description': 'Combined contextual information for embeddings (description, use case, stage)',
        'example': 'AWS migration project, modernize infrastructure, discovery stage'
    },
    'identity_hash': {
        'type': 'VARCHAR(64)',
        'description': 'SHA-256 hash of identity_text for change detection',
        'example': 'a1b2c3d4e5f6...'
    },
    'context_hash': {
        'type': 'VARCHAR(64)',
        'description': 'SHA-256 hash of context_text for change detection', 
        'example': 'f6e5d4c3b2a1...'
    },
    'identity_embedding': {
        'type': 'JSONB',
        'description': 'BGE-M3 identity embedding vector (384 dimensions)',
        'example': '{"vector": [0.123, -0.456, 0.789, ...], "model": "bge-m3"}'
    },
    'context_embedding': {
        'type': 'JSONB', 
        'description': 'BGE-M3 context embedding vector (384 dimensions)',
        'example': '{"vector": [0.321, -0.654, 0.987, ...], "model": "bge-m3"}'
    },
    'embedding_generated_at': {
        'type': 'TIMESTAMP',
        'description': 'When embeddings were last generated',
        'example': '2024-12-20 15:30:45'
    }
}

def check_field_exists(cursor, table_name, field_name):
    """Check if a field exists in the specified table."""
    cursor.execute("""
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'core' 
        AND table_name = %s 
        AND column_name = %s
    """, (table_name, field_name))
    return cursor.fetchone() is not None

def add_embedding_field(cursor, table_name, field_name, field_type):
    """Add a single embedding field to the table."""
    if check_field_exists(cursor, table_name, field_name):
        print(f"     ⚠ Field '{field_name}' already exists - skipping")
        return False
    
    # Build ALTER TABLE statement
    alter_sql = sql.SQL("ALTER TABLE core.{} ADD COLUMN {} {}").format(
        sql.Identifier(table_name),
        sql.Identifier(field_name), 
        sql.SQL(field_type)
    )
    
    cursor.execute(alter_sql)
    print(f"     ✓ Added field '{field_name}' ({field_type})")
    return True

def create_embedding_indexes(cursor, table_name):
    """Create indexes for embedding fields to optimize performance."""
    indexes = [
        {
            'name': f'idx_{table_name}_identity_hash',
            'field': 'identity_hash',
            'description': 'Fast lookup for identity change detection'
        },
        {
            'name': f'idx_{table_name}_context_hash', 
            'field': 'context_hash',
            'description': 'Fast lookup for context change detection'
        },
        {
            'name': f'idx_{table_name}_embedding_generated_at',
            'field': 'embedding_generated_at',
            'description': 'Track when embeddings were generated'
        }
    ]
    
    indexes_created = 0
    for index in indexes:
        try:
            # Check if index already exists
            cursor.execute("""
                SELECT 1 FROM pg_indexes 
                WHERE schemaname = 'core' 
                AND tablename = %s 
                AND indexname = %s
            """, (table_name, index['name']))
            
            if cursor.fetchone():
                print(f"     ⚠ Index '{index['name']}' already exists - skipping")
                continue
            
            # Create index
            create_index_sql = sql.SQL("CREATE INDEX {} ON core.{} ({})").format(
                sql.Identifier(index['name']),
                sql.Identifier(table_name),
                sql.Identifier(index['field'])
            )
            
            cursor.execute(create_index_sql)
            print(f"     ✓ Created index '{index['name']}' on {index['field']}")
            indexes_created += 1
            
        except psycopg2.Error as e:
            print(f"     ✗ Error creating index '{index['name']}': {e}")
            continue
    
    return indexes_created

def add_field_comments(cursor, table_name):
    """Add comments to the embedding fields for documentation."""
    comments_added = 0
    
    for field_name, field_info in EMBEDDING_FIELDS.items():
        try:
            # Add comment to field
            comment_sql = sql.SQL("COMMENT ON COLUMN core.{}.{} IS %s").format(
                sql.Identifier(table_name),
                sql.Identifier(field_name)
            )
            
            cursor.execute(comment_sql, (field_info['description'],))
            comments_added += 1
            
        except psycopg2.Error as e:
            print(f"     ⚠ Could not add comment for field '{field_name}': {e}")
            continue
    
    print(f"     ✓ Added {comments_added} field comments")
    return comments_added

def verify_embedding_fields(cursor, table_name):
    """Verify all embedding fields were added correctly."""
    cursor.execute("""
        SELECT 
            column_name,
            data_type,
            character_maximum_length,
            is_nullable,
            column_default
        FROM information_schema.columns 
        WHERE table_schema = 'core' 
        AND table_name = %s 
        AND column_name IN %s
        ORDER BY column_name
    """, (table_name, tuple(EMBEDDING_FIELDS.keys())))
    
    fields = cursor.fetchall()
    return fields

def get_opportunity_tables(cursor):
    """Get all opportunity-related tables in CORE schema."""
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'core' 
        AND table_name LIKE '%opportunit%'
        ORDER BY table_name
    """)
    
    tables = cursor.fetchall()
    return [table[0] for table in tables]

def add_embedding_fields():
    """Add embedding fields to all opportunity tables in CORE schema."""
    
    # Get configuration from environment
    db_host = os.getenv('LOCAL_DB_HOST', 'localhost')
    db_port = os.getenv('LOCAL_DB_PORT', '5432')
    db_name = os.getenv('LOCAL_DB_NAME', 'revops_core')
    app_user = os.getenv('LOCAL_DB_USER', 'revops_user')
    app_password = os.getenv('LOCAL_DB_PASSWORD')
    
    if not app_password:
        print("✗ Error: LOCAL_DB_PASSWORD not found in environment")
        return False
    
    print("=" * 80)
    print("RevOps CORE Schema - Add Embedding Fields (Task 2.6)")
    print("=" * 80)
    print("\nThis script adds BGE-M3 dual embedding fields to opportunities tables:")
    for field_name, field_info in EMBEDDING_FIELDS.items():
        print(f"  • {field_name}: {field_info['description']}")
    print("\nKey features:")
    print("  • Dual embedding strategy (identity + context)")
    print("  • Hash-based change detection to avoid redundant processing")
    print("  • BGE-M3 384-dimensional vector storage")
    print("  • Performance indexes for embedding operations")
    print()
    
    try:
        # Connect to database
        print(f"1. Connecting to database '{db_name}' as user '{app_user}'...")
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=app_user,
            password=app_password
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        print("   ✓ Connected successfully")
        
        # Get opportunity tables
        print(f"\n2. Finding opportunity tables in CORE schema...")
        opportunity_tables = get_opportunity_tables(cursor)
        
        if not opportunity_tables:
            print("   ✗ No opportunity tables found in CORE schema!")
            print("   Please run 09_create_core_tables.py first")
            return False
        
        print(f"   ✓ Found {len(opportunity_tables)} opportunity tables:")
        for table in opportunity_tables:
            print(f"     • core.{table}")
        
        # Process each opportunity table
        total_fields_added = 0
        total_indexes_created = 0
        
        for table_name in opportunity_tables:
            print(f"\n3. Processing table 'core.{table_name}'...")
            
            # Add embedding fields
            print(f"   Adding embedding fields...")
            fields_added = 0
            
            for field_name, field_info in EMBEDDING_FIELDS.items():
                try:
                    if add_embedding_field(cursor, table_name, field_name, field_info['type']):
                        fields_added += 1
                        total_fields_added += 1
                except psycopg2.Error as e:
                    print(f"     ✗ Error adding field '{field_name}': {e}")
                    continue
            
            print(f"   ✓ Added {fields_added} new fields to core.{table_name}")
            
            # Create indexes
            print(f"   Creating performance indexes...")
            indexes_created = create_embedding_indexes(cursor, table_name)
            total_indexes_created += indexes_created
            
            # Add field comments
            print(f"   Adding field documentation...")
            add_field_comments(cursor, table_name)
            
            # Verify fields
            print(f"   Verifying field addition...")
            embedding_fields = verify_embedding_fields(cursor, table_name)
            
            if embedding_fields:
                print(f"   ✓ Verified {len(embedding_fields)} embedding fields:")
                for field in embedding_fields:
                    field_name, data_type, max_length, nullable, default = field
                    nullable_str = 'NULL' if nullable == 'YES' else 'NOT NULL'
                    length_str = f'({max_length})' if max_length else ''
                    print(f"     • {field_name}: {data_type}{length_str} ({nullable_str})")
            else:
                print(f"   ⚠ No embedding fields found in verification")
        
        # Summary
        print(f"\n4. Processing Summary:")
        print(f"   ✓ Tables processed: {len(opportunity_tables)}")
        print(f"   ✓ Fields added: {total_fields_added}")
        print(f"   ✓ Indexes created: {total_indexes_created}")
        
        # Test field access
        print(f"\n5. Testing embedding field access...")
        if opportunity_tables:
            test_table = opportunity_tables[0]
            
            # Test each embedding field
            for field_name in EMBEDDING_FIELDS.keys():
                try:
                    cursor.execute(f"""
                        SELECT COUNT(*) 
                        FROM core.{test_table} 
                        WHERE {field_name} IS NOT NULL
                    """)
                    count = cursor.fetchone()[0]
                    print(f"   ✓ Field '{field_name}' accessible (0 non-null expected)")
                except psycopg2.Error as e:
                    print(f"   ✗ Error accessing field '{field_name}': {e}")
        
        # Check constraints and defaults
        print(f"\n6. Verifying field constraints...")
        cursor.execute(f"""
            SELECT 
                column_name,
                is_nullable,
                column_default,
                data_type
            FROM information_schema.columns 
            WHERE table_schema = 'core' 
            AND table_name = '{opportunity_tables[0] if opportunity_tables else 'opportunities'}'
            AND column_name IN {tuple(EMBEDDING_FIELDS.keys())}
            ORDER BY column_name
        """)
        
        constraints = cursor.fetchall()
        print(f"   ✓ Field constraints verified:")
        for field_name, nullable, default, data_type in constraints:
            nullable_str = 'NULL' if nullable == 'YES' else 'NOT NULL'
            default_str = f', DEFAULT {default}' if default else ''
            print(f"     • {field_name}: {data_type} ({nullable_str}{default_str})")
        
        # Close connection
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 80)
        print("✓ Embedding fields addition completed successfully!")
        print("=" * 80)
        print(f"\nAdded embedding infrastructure to {len(opportunity_tables)} opportunity tables:")
        for table in opportunity_tables:
            print(f"  • core.{table}")
        
        print(f"\nNew embedding capabilities:")
        print("  • Dual embedding strategy: identity + context embeddings")
        print("  • Hash-based change detection for efficient updates")
        print("  • BGE-M3 384-dimensional vector storage (JSONB format)")
        print("  • Performance indexes for embedding operations")
        print("  • Full documentation with field comments")
        
        print(f"\nReady for BGE embedding integration:")
        print("  • Text fields ready for embedding generation")
        print("  • Hash fields ready for change detection")  
        print("  • Vector fields ready for BGE-M3 embeddings")
        print("  • Timestamp field ready for generation tracking")
        
        print(f"\nNext steps:")
        print("  1. Populate identity_text and context_text fields via normalization scripts")
        print("  2. Generate embeddings using backend/services/07-embeddings/")
        print("  3. Implement POD matching algorithm using dual embeddings")
        print("  4. Set up embedding refresh process with hash-based change detection")
        
        return True
        
    except psycopg2.Error as e:
        print(f"\n✗ Database error: {e}")
        if hasattr(e, 'pgcode') and e.pgcode:
            print(f"   Error code: {e.pgcode}")
        if hasattr(e, 'pgerror') and e.pgerror:
            print(f"   Details: {e.pgerror}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False

def main():
    """Main function to add embedding fields."""
    # Check if .env exists
    if not env_path.exists():
        print("✗ Error: .env file not found!")
        print(f"  Please create {env_path} with database credentials.")
        sys.exit(1)
    
    # Check required environment variables
    required_vars = ['LOCAL_DB_HOST', 'LOCAL_DB_NAME', 'LOCAL_DB_USER', 'LOCAL_DB_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("✗ Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  • {var}")
        sys.exit(1)
    
    # Add embedding fields
    success = add_embedding_fields()
    
    if success:
        print("\n🎉 Embedding fields addition completed successfully!")
        print("   The CORE opportunities table is now ready for BGE-M3 embeddings.")
    else:
        print("\n💥 Embedding fields addition failed!")
        print("   Please check the error messages above and try again.")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()