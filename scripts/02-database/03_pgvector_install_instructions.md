# pgvector Extension Installation Instructions

## Prerequisites
- PostgreSQL 15 must be installed and running
- sudo privileges required for installation

## Installation Steps

### Option 1: Automated Installation (Recommended)
```bash
sudo /home/loyd2888/Projects/RevOpsAutomation/scripts/02-database/03_install_pgvector.sh
```

This script will:
1. Check PostgreSQL installation
2. Try to install pgvector from package manager
3. If not available, build from source
4. Enable the extension in PostgreSQL
5. Run verification tests

### Option 2: Manual Installation from Package Manager
```bash
# Update package list
sudo apt-get update

# Install pgvector for PostgreSQL 15
sudo apt-get install -y postgresql-15-pgvector

# Enable extension in PostgreSQL
sudo -u postgres psql -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Option 3: Manual Installation from Source
```bash
# Install build dependencies
sudo apt-get install -y build-essential postgresql-server-dev-15 git

# Clone pgvector repository
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector

# Build and install
make
sudo make install

# Enable extension in PostgreSQL
sudo -u postgres psql -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Verification

After installation, verify pgvector is working:

```bash
bash /home/loyd2888/Projects/RevOpsAutomation/scripts/02-database/04_verify_pgvector.sh
```

Or manually test:
```sql
-- Connect to PostgreSQL
PGPASSWORD=postgres psql -U postgres -h localhost

-- Check if extension is available
SELECT * FROM pg_available_extensions WHERE name = 'vector';

-- Create extension if not exists
CREATE EXTENSION IF NOT EXISTS vector;

-- Test vector operations
CREATE TEMP TABLE test (embedding vector(3));
INSERT INTO test VALUES ('[1,2,3]'), ('[4,5,6]');
SELECT embedding <=> '[1,2,3]' AS distance FROM test;
```

## Expected Output

When successfully installed:
- pgvector extension available in PostgreSQL
- Version 0.5.0 or higher
- Vector operations working
- HNSW index support available

## Using pgvector

### Creating Tables with Vectors
```sql
CREATE TABLE items (
    id SERIAL PRIMARY KEY,
    name TEXT,
    embedding vector(1024)  -- 1024-dimensional vector
);
```

### Similarity Search Operations
```sql
-- Cosine similarity (most common for embeddings)
SELECT * FROM items ORDER BY embedding <=> '[...]' LIMIT 10;

-- Euclidean distance
SELECT * FROM items ORDER BY embedding <-> '[...]' LIMIT 10;

-- Inner product
SELECT * FROM items ORDER BY embedding <#> '[...]' LIMIT 10;
```

### Creating Indexes for Performance
```sql
-- HNSW index for approximate nearest neighbor search
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops);

-- IVFFlat index (alternative method)
CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops);
```

## Troubleshooting

### Error: "could not open extension control file"
- pgvector is not installed. Run the installation script.

### Error: "operator does not exist: vector <=> unknown"
- The vector extension is not enabled in the current database
- Run: `CREATE EXTENSION vector;`

### Build fails with "pg_config not found"
- Install postgresql-server-dev-15: `sudo apt-get install postgresql-server-dev-15`

### Performance issues with large vectors
- Create appropriate indexes (HNSW recommended)
- Consider using smaller vector dimensions if possible
- Tune PostgreSQL memory settings

## Next Steps

After pgvector is installed:
1. Create the revops_core database (TASK-003)
2. Enable pgvector in the new database
3. Create SEARCH schema tables with vector columns
4. Implement BGE-M3 embedding generation (future task)