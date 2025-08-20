#!/bin/bash
# Create the project directory structure as defined in Project_Plan.md

echo "Creating RevOps Automation Platform directory structure..."

# Create main directories
mkdir -p infrastructure/postgres
mkdir -p infrastructure/bge

# Backend microservices
mkdir -p backend/services/01-ingestion
mkdir -p backend/services/02-transformation
mkdir -p backend/services/03-embedding
mkdir -p backend/services/04-matching
mkdir -p backend/services/05-rules
mkdir -p backend/services/06-api

# Backend core components
mkdir -p backend/models
mkdir -p backend/core
mkdir -p backend/tests

# Frontend structure
mkdir -p frontend/src/components
mkdir -p frontend/src/pages
mkdir -p frontend/src/services
mkdir -p frontend/src/hooks

# Scripts organized by purpose (max 15 per directory)
mkdir -p scripts/01-setup
mkdir -p scripts/02-database
mkdir -p scripts/03-data
mkdir -p scripts/04-deployment

# Documentation (max 5 files)
mkdir -p docs

# Data directory for samples and exports
mkdir -p data

echo "✓ Directory structure created successfully"

# Create .gitignore
cat > .gitignore << 'EOF'
# Environment variables
.env
*.env.local

# Python
__pycache__/
*.py[cod]
*$py.class
venv/
.venv/
*.egg-info/
dist/
build/

# Node
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Database
*.db
*.sqlite
*.sqlite3

# Logs
logs/
*.log

# Data files (keep structure, ignore contents)
data/*
!data/.gitkeep

# Docker
*.pid
EOF

echo "✓ .gitignore created"

# Create placeholder files to preserve structure
touch infrastructure/postgres/.gitkeep
touch infrastructure/bge/.gitkeep
touch backend/models/.gitkeep
touch backend/core/.gitkeep
touch backend/tests/.gitkeep
touch frontend/src/components/.gitkeep
touch frontend/src/pages/.gitkeep
touch frontend/src/services/.gitkeep
touch frontend/src/hooks/.gitkeep
touch docs/.gitkeep
touch data/.gitkeep

echo "✓ Placeholder files created"
echo "Project structure setup complete!"