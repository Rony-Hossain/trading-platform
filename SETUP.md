# Trading Platform - Complete Setup Guide

A comprehensive guide to set up and run the complete trading platform with frontend, backend APIs, and database infrastructure.

## 🏗️ Architecture Overview

**Development Environment:**
- Windows 11 (Your development machine)
- 2 Ubuntu Servers (Production deployment)

**Technology Stack:**
- **Frontend**: Next.js 15 + TypeScript (Nx workspace)
- **Backend**: FastAPI (Python) + PostgreSQL + Redis
- **Deployment**: Docker Swarm
- **APIs**: Market Data API + Analysis API
- **Database**: PostgreSQL with Redis caching

---

## 📋 Prerequisites

### On Windows 11 (Development)

**Required Software:**
```bash
# Node.js 20+
winget install OpenJS.NodeJS

# Docker Desktop
winget install Docker.DockerDesktop

# Git
winget install Git.Git

# Python 3.11+
winget install Python.Python.3.11

# PostgreSQL Client (optional, for direct DB access)
winget install PostgreSQL.PostgreSQL
```

**IDE Recommendations:**
- **VS Code** (recommended): `winget install Microsoft.VisualStudioCode`
- **PyCharm Professional**: `winget install JetBrains.PyCharm.Professional`
- **WebStorm**: `winget install JetBrains.WebStorm`

### On Ubuntu Servers

**Server 1 (Database Server):**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Log out and back in for group changes
```

**Server 2 (API Server):**
```bash
# Same as Server 1
sudo apt update && sudo apt upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

---

## 🚀 Quick Start (Development)

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url> trading-platform
cd trading-platform

# Create environment file
cp .env.example .env

# Edit .env with your settings
notepad .env  # Windows
# or
code .env     # VS Code
```

### 2. Start Database Services

```bash
# Start PostgreSQL and Redis
docker compose up -d postgres redis

# Verify services are running
docker compose ps
docker compose logs postgres
```

### 3. Run Database Migrations

```bash
# Option 1: Using Docker
docker compose exec postgres psql -U trading_user -d trading_db -f /docker-entrypoint-initdb.d/001_initial_schema.sql

# Option 2: Direct connection
psql "postgresql://trading_user:trading_pass@localhost:5432/trading_db" -f migrations/001_initial_schema.sql

# Load seed data (optional)
psql "postgresql://trading_user:trading_pass@localhost:5432/trading_db" -f migrations/seed_data.sql
```

### 4. Start Backend APIs

**Terminal 1 - Market Data API:**
```bash
cd services/market-data
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

**Terminal 2 - Analysis API:**
```bash
cd services/analysis
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload
```

### 5. Start Frontend

```bash
cd trading-frontend
npm install
npm run dev
```

### 6. Verify Everything Works

**Check Services:**
- Frontend: http://localhost:3000
- Market Data API: http://localhost:8002/docs
- Analysis API: http://localhost:8003/docs
- Database: Direct connection on port 5432
- Redis: Port 6379

**Test API Endpoints:**
```bash
# Test market data
curl http://localhost:8002/health
curl http://localhost:8002/stocks/AAPL

# Test analysis
curl http://localhost:8003/health
curl "http://localhost:8003/analyze/AAPL?period=6mo"
```

---

## 🛠️ IDE Setup & Configuration

### VS Code (Recommended)

**Essential Extensions:**
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.flake8",
    "ms-python.black-formatter",
    "bradlc.vscode-tailwindcss",
    "esbenp.prettier-vscode",
    "ms-vscode.vscode-typescript-next",
    "ms-vscode-remote.remote-containers",
    "redhat.vscode-yaml",
    "ms-vscode.docker"
  ]
}
```

**Workspace Settings (`.vscode/settings.json`):**
```json
{
  "python.defaultInterpreterPath": "./services/market-data/.venv/Scripts/python",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "typescript.preferences.importModuleSpecifier": "relative",
  "tailwindCSS.includeLanguages": {
    "typescript": "javascript",
    "typescriptreact": "javascript"
  }
}
```

**Tasks (`.vscode/tasks.json`):**
```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Start Database",
      "type": "shell",
      "command": "docker compose up -d postgres redis",
      "group": "build",
      "problemMatcher": []
    },
    {
      "label": "Start Market Data API",
      "type": "shell",
      "command": "uvicorn app.main:app --reload",
      "group": "build",
      "options": {
        "cwd": "${workspaceFolder}/services/market-data"
      }
    },
    {
      "label": "Start Analysis API",
      "type": "shell",
      "command": "uvicorn app.main:app --reload",
      "group": "build",
      "options": {
        "cwd": "${workspaceFolder}/services/analysis"
      }
    },
    {
      "label": "Start Frontend",
      "type": "shell",
      "command": "npm run dev",
      "group": "build",
      "options": {
        "cwd": "${workspaceFolder}/trading-frontend"
      }
    }
  ]
}
```

**Debug Configuration (`.vscode/launch.json`):**
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Market Data API",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/services/market-data/.venv/Scripts/uvicorn",
      "args": ["app.main:app", "--reload", "--port", "8002"],
      "cwd": "${workspaceFolder}/services/market-data",
      "console": "integratedTerminal"
    },
    {
      "name": "Debug Analysis API",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/services/analysis/.venv/Scripts/uvicorn",
      "args": ["app.main:app", "--reload", "--port", "8003"],
      "cwd": "${workspaceFolder}/services/analysis",
      "console": "integratedTerminal"
    }
  ]
}
```

### PyCharm Professional

**Project Setup:**
1. Open PyCharm → Open → Select `trading-platform` folder
2. Configure Python Interpreters:
   - File → Settings → Project → Python Interpreter
   - Add New Interpreter → Virtualenv → Existing
   - Point to `services/market-data/.venv/Scripts/python.exe`
   - Repeat for `services/analysis/.venv/Scripts/python.exe`

**Run Configurations:**
1. Add Configuration → FastAPI
2. Script path: `services/market-data/app/main.py`
3. Parameters: `--host 0.0.0.0 --port 8002`

---

## 🐳 Docker Development

### Full Docker Development Setup

**Start everything with Docker:**
```bash
# Start all services (when docker-compose.override.yml is present)
docker compose up -d

# View logs
docker compose logs -f

# Stop all services
docker compose down
```

**Development Tools (included in override):**
- **Adminer**: http://localhost:8080 (Database GUI)
- **Redis Commander**: http://localhost:8081 (Redis GUI)

### Individual Service Development

**Market Data API:**
```bash
cd services/market-data
docker build -f Dockerfile.dev -t market-data-dev .
docker run -p 8002:8002 -v $(pwd):/app market-data-dev
```

**Analysis API:**
```bash
cd services/analysis
docker build -f Dockerfile.dev -t analysis-dev .
docker run -p 8003:8003 -v $(pwd):/app analysis-dev
```

---

## 🚀 Production Deployment

### Deploy to 2 Ubuntu Servers

**Step 1: Setup Docker Swarm (on Ubuntu Server 1)**
```bash
# Clone repository
git clone <your-repo-url> trading-platform
cd trading-platform

# Make scripts executable
chmod +x scripts/*.sh

# Initialize Docker Swarm
./scripts/setup-swarm.sh
```

**Step 2: Join Second Server (on Ubuntu Server 2)**
```bash
# Use the command output from step 1
curl -fsSL <your-repo>/scripts/join-swarm.sh | bash -s worker <manager-ip> <token>
```

**Step 3: Label Nodes (on Manager)**
```bash
# Label database node (Server 1)
docker node update --label-add role=database $(docker node ls -q -f role=manager)

# Label API node (Server 2)
NODE_ID=$(docker node ls -q -f name=<server2-hostname>)
docker node update --label-add role=api $NODE_ID

# Verify labels
docker node ls
```

**Step 4: Deploy Production Stack**
```bash
# Create production environment
cp .env.example .env.prod
# Edit .env.prod with production values

# Deploy stack
./scripts/deploy-production.sh
```

**Step 5: Monitor Deployment**
```bash
# Check services
docker stack services trading
docker stack ps trading

# View logs
docker service logs trading_postgres -f
docker service logs trading_market-data-api -f

# Check health
curl http://<server-ip>:8002/health
curl http://<server-ip>:8003/health
```

### SSL Setup (Production)

**Generate SSL Certificates:**
```bash
# Option 1: Self-signed (development)
openssl req -x509 -newkey rsa:4096 -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem -days 365 -nodes

# Option 2: Let's Encrypt (production)
sudo apt install certbot
sudo certbot certonly --standalone -d yourdomain.com -d api.yourdomain.com
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem nginx/ssl/key.pem
```

**Update Domain Names:**
Edit `nginx/nginx.conf` and `docker-compose.prod.yml` with your actual domain names.

---

## 🔧 Development Workflow

### Daily Development

1. **Start Development Environment:**
   ```bash
   # Terminal 1: Database
   docker compose up -d postgres redis
   
   # Terminal 2: Market Data API
   cd services/market-data && .venv\Scripts\activate && uvicorn app.main:app --reload
   
   # Terminal 3: Analysis API
   cd services/analysis && .venv\Scripts\activate && uvicorn app.main:app --reload
   
   # Terminal 4: Frontend
   cd trading-frontend && npm run dev
   ```

2. **Development URLs:**
   - Frontend: http://localhost:3000
   - Market Data API Docs: http://localhost:8002/docs
   - Analysis API Docs: http://localhost:8003/docs
   - Database GUI: http://localhost:8080 (if using Docker override)

### Code Changes

**Backend API Changes:**
- FastAPI has hot-reload enabled in development
- Changes to `.py` files will automatically restart the server
- Database changes require migration files in `migrations/`

**Frontend Changes:**
- Next.js has hot-reload enabled
- Changes to React components update instantly
- TypeScript compilation happens automatically

### Testing

**Backend Testing:**
```bash
# Unit tests (when implemented)
cd services/market-data
python -m pytest tests/

cd services/analysis
python -m pytest tests/
```

**API Testing:**
```bash
# Test all health endpoints
curl http://localhost:8002/health
curl http://localhost:8003/health

# Test stock data
curl "http://localhost:8002/stocks/AAPL"
curl "http://localhost:8003/analyze/AAPL?period=6mo"
```

**Database Testing:**
```bash
# Connect to test database
psql "postgresql://trading_user:trading_pass@localhost:5432/trading_db"

# Run test queries
SELECT * FROM users LIMIT 5;
SELECT * FROM alerts WHERE symbol = 'AAPL';
```

---

## 📊 Monitoring & Observability

### Development Monitoring

**Database Monitoring:**
```bash
# PostgreSQL logs
docker compose logs postgres -f

# Redis logs
docker compose logs redis -f

# Connection stats
docker compose exec postgres psql -U trading_user -d trading_db -c "SELECT * FROM pg_stat_activity;"
```

**API Monitoring:**
- Market Data API Metrics: http://localhost:8002/metrics
- Analysis API Metrics: http://localhost:8003/metrics
- Health Checks: `/health` endpoint on both APIs

### Production Monitoring

**Prometheus Metrics:**
- http://server1:9090 (Prometheus UI)
- Grafana: http://server1:3001 (admin/admin123)

**Log Aggregation:**
```bash
# View service logs
docker service logs trading_market-data-api -f
docker service logs trading_analysis-api -f
docker service logs trading_postgres -f

# System resources
docker stats
```

---

## 🛠️ Troubleshooting

### Common Development Issues

**Port Already in Use:**
```bash
# Find process using port
netstat -ano | findstr :8002
taskkill /F /PID <process-id>

# Or use different ports in .env
```

**Database Connection Issues:**
```bash
# Check if PostgreSQL is running
docker compose ps postgres

# Check logs
docker compose logs postgres

# Test connection
psql "postgresql://trading_user:trading_pass@localhost:5432/trading_db" -c "SELECT 1;"
```

**API Import Errors:**
```bash
# Verify virtual environment
cd services/market-data
.venv\Scripts\python -c "import fastapi; print('FastAPI OK')"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Frontend Build Errors:**
```bash
# Clear cache
cd trading-frontend
rm -rf node_modules .next .nx
npm install

# Check Node version
node --version  # Should be 20+
```

### Production Issues

**Service Won't Start:**
```bash
# Check service status
docker service ls
docker service ps trading_market-data-api

# Check logs
docker service logs trading_market-data-api --tail 50

# Restart service
docker service update --force trading_market-data-api
```

**Database Issues:**
```bash
# Check database health
docker service ps trading_postgres

# Connect to database
docker exec -it $(docker ps -qf name=trading_postgres) psql -U trading_user -d trading_db

# Check disk space
df -h
```

**Network Issues:**
```bash
# Check Docker networks
docker network ls

# Test internal connectivity
docker exec -it $(docker ps -qf name=trading_market-data) ping postgres
```

---

## 📚 Additional Resources

### API Documentation
- Market Data API: http://localhost:8002/docs (Swagger UI)
- Analysis API: http://localhost:8003/docs (Swagger UI)
- OpenAPI Specs: `documentation/contracts/`

### Database Documentation
- Schema: `migrations/001_initial_schema.sql`
- ERD: `documentation/database-schema.md` (create if needed)
- Sample Queries: `scripts/test-db-connection.sql`

### Deployment Documentation
- Docker Swarm: `docker-compose.prod.yml`
- Scripts: `scripts/` directory
- Network Configuration: `nginx/nginx.conf`

### Performance Tuning
- Database: Connection pooling in SQLAlchemy settings
- Redis: Memory configuration in `docker-compose.yml`
- APIs: Worker processes in production Dockerfiles

---

**🎉 You're all set!** Your trading platform should now be running with full backend, database, and frontend integration.

For issues or questions, check the troubleshooting section or create an issue in the repository.