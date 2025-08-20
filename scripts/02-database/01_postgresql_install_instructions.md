# PostgreSQL 15 Installation Instructions

## Manual Installation Required

Since automated installation requires sudo privileges, please run the following commands manually:

### Step 1: Run the Installation Script
```bash
sudo /home/loyd2888/Projects/RevOpsAutomation/scripts/02-database/01_install_postgresql.sh
```

### Alternative: Manual Installation Commands

If the script doesn't work, run these commands one by one:

#### 1. Update package list
```bash
sudo apt-get update
```

#### 2. Install required packages
```bash
sudo apt-get install -y wget ca-certificates
```

#### 3. Add PostgreSQL APT repository
```bash
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
```

#### 4. Import repository signing key
```bash
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
```

#### 5. Update package list again
```bash
sudo apt-get update
```

#### 6. Install PostgreSQL 15
```bash
sudo apt-get install -y postgresql-15 postgresql-client-15 postgresql-contrib-15
```

#### 7. Start and enable PostgreSQL
```bash
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

#### 8. Set postgres user password
```bash
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
```

### Step 2: Verify Installation

After installation, test the connection:
```bash
PGPASSWORD=postgres psql -U postgres -h localhost -c "SELECT version();"
```

### Step 3: Create Test Script

We've created a verification script at:
`/home/loyd2888/Projects/RevOpsAutomation/scripts/02-database/02_verify_postgresql.sh`

Run it to verify the installation:
```bash
bash /home/loyd2888/Projects/RevOpsAutomation/scripts/02-database/02_verify_postgresql.sh
```

## Expected Output

When successfully installed, you should see:
- PostgreSQL 15.x installed
- Service running on port 5432
- Connection test successful

## Docker Alternative

If you prefer Docker, we can use PostgreSQL in a container instead. Let us know if you'd like to proceed with Docker.