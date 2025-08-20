#!/bin/bash

# PostgreSQL 15+ Installation Script for Ubuntu
# This script installs PostgreSQL 15 and configures it for development

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}PostgreSQL 15+ Installation Script${NC}"
echo "======================================"

# Check if running on Ubuntu
if [[ ! -f /etc/os-release ]] || ! grep -q "Ubuntu" /etc/os-release; then
    echo -e "${RED}Error: This script is designed for Ubuntu systems${NC}"
    exit 1
fi

# Check if PostgreSQL is already installed
if command -v psql &> /dev/null; then
    CURRENT_VERSION=$(psql --version | awk '{print $3}' | sed 's/\..*//g')
    echo -e "${YELLOW}PostgreSQL is already installed (version $CURRENT_VERSION)${NC}"
    
    if [[ $CURRENT_VERSION -ge 15 ]]; then
        echo -e "${GREEN}PostgreSQL 15+ is already installed${NC}"
        exit 0
    else
        echo -e "${YELLOW}Upgrading to PostgreSQL 15...${NC}"
    fi
fi

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install required packages
echo "Installing required packages..."
sudo apt-get install -y wget ca-certificates

# Add PostgreSQL APT repository
echo "Adding PostgreSQL APT repository..."
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'

# Import repository signing key
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -

# Update package list again
sudo apt-get update

# Install PostgreSQL 15
echo "Installing PostgreSQL 15..."
sudo apt-get install -y postgresql-15 postgresql-client-15 postgresql-contrib-15

# Start and enable PostgreSQL service
echo "Starting PostgreSQL service..."
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Configure PostgreSQL for development
echo "Configuring PostgreSQL..."

# Update postgresql.conf for better development performance
PG_CONFIG="/etc/postgresql/15/main/postgresql.conf"
sudo cp $PG_CONFIG "${PG_CONFIG}.backup"

# Apply basic performance tuning
echo "Applying performance tuning..."
sudo sed -i "s/#shared_buffers = 128MB/shared_buffers = 256MB/" $PG_CONFIG
sudo sed -i "s/#work_mem = 4MB/work_mem = 8MB/" $PG_CONFIG
sudo sed -i "s/#maintenance_work_mem = 64MB/maintenance_work_mem = 128MB/" $PG_CONFIG
sudo sed -i "s/#effective_cache_size = 4GB/effective_cache_size = 1GB/" $PG_CONFIG

# Update pg_hba.conf to allow local connections with password
PG_HBA="/etc/postgresql/15/main/pg_hba.conf"
sudo cp $PG_HBA "${PG_HBA}.backup"

# Allow password authentication for local connections
echo "Configuring authentication..."
sudo sed -i "s/local   all             all                                     peer/local   all             all                                     md5/" $PG_HBA

# Restart PostgreSQL to apply changes
echo "Restarting PostgreSQL..."
sudo systemctl restart postgresql

# Set password for postgres user
echo "Setting up postgres user..."
sudo -u postgres psql <<EOF
ALTER USER postgres PASSWORD 'postgres';
EOF

# Test connection
echo "Testing PostgreSQL connection..."
PGPASSWORD=postgres psql -U postgres -h localhost -c "SELECT version();" &> /dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PostgreSQL 15 installed and configured successfully!${NC}"
    echo ""
    echo "PostgreSQL is running on port 5432"
    echo "Default superuser: postgres"
    echo "Default password: postgres (change this in production!)"
    echo ""
    echo "You can connect with:"
    echo "  psql -U postgres -h localhost"
else
    echo -e "${RED}✗ Connection test failed${NC}"
    exit 1
fi

# Create script to check PostgreSQL status
cat > check_postgresql_status.sh << 'EOF'
#!/bin/bash
echo "PostgreSQL Status:"
sudo systemctl status postgresql | head -n 10
echo ""
echo "PostgreSQL Version:"
psql --version
echo ""
echo "Test Connection:"
PGPASSWORD=postgres psql -U postgres -h localhost -c "SELECT current_database(), current_user, version();"
EOF

chmod +x check_postgresql_status.sh
echo -e "${GREEN}Created check_postgresql_status.sh for future status checks${NC}"