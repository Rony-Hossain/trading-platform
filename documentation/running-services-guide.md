# Trading Platform - Running Services Guide

This guide provides step-by-step instructions for running all components of the trading platform, including Docker services, microservices, Redis, frontend, and more.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Docker Services](#docker-services)
3. [Database Setup](#database-setup)
4. [Redis Cache](#redis-cache)
5. [Backend Microservices](#backend-microservices)
6. [Frontend Application](#frontend-application)
7. [Monitoring Services](#monitoring-services)
8. [Complete Startup Sequence](#complete-startup-sequence)
9. [Health Checks](#health-checks)
10. [Troubleshooting](#troubleshooting)

## System Requirements

- **Docker & Docker Compose**: Latest version
- **Python**: 3.10 or higher
- **Node.js**: 16 or higher
- **Git**: For version control
- **RAM**: Minimum 8GB recommended
- **Storage**: At least 10GB free space

## Docker Services

### 1. Start Core Infrastructure
The platform uses Docker Compose for core infrastructure services.

```bash
# Navigate to project root
cd E:\rony-data\trading-platform

# Start PostgreSQL database with TimescaleDB
docker-compose up -d postgres

# Start Redis cache
docker-compose up -d redis

# Start all infrastructure services at once
docker-compose up -d
```

### 2. Available Docker Services
```bash
# PostgreSQL with TimescaleDB extension
docker-compose up -d postgres

# Redis for caching and session management
docker-compose up -d redis

# Optional: Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d
```

### 3. Check Docker Service Status
```bash
# View running containers
docker ps

# Check service logs
docker-compose logs postgres
docker-compose logs redis

# View all service logs
docker-compose logs -f
```

## Database Setup

### 1. Database Initialization
```bash
# Wait for PostgreSQL to be ready (about 30 seconds)
timeout 30

# Connect to database to verify
docker exec -it trading-platform-postgres-1 psql -U trading_user -d trading_db

# Run database migrations
cd E:\rony-data\trading-platform
python scripts/run_migrations.py

# Or manually run migration files
docker exec -it trading-platform-postgres-1 psql -U trading_user -d trading_db -f /migrations/001_initial_schema.sql
```

### 2. Seed Sample Data (Optional)
```bash
# Run seed data script
docker exec -it trading-platform-postgres-1 psql -U trading_user -d trading_db -f /migrations/99-seed.sql
```

## Redis Cache

### 1. Start Redis
```bash
# Start Redis container
docker-compose up -d redis

# Connect to Redis CLI
docker exec -it trading-platform-redis-1 redis-cli

# Test Redis connection
docker exec -it trading-platform-redis-1 redis-cli ping
```

### 2. Redis Configuration
- **Port**: 6379
- **Password**: None (default config)
- **Databases**: 0-15 available

## Backend Microservices

### 1. Market Data Service (Port 8002)
```bash
# Navigate to service directory
cd E:\rony-data\trading-platform\services\market-data-service

# Install dependencies
pip install -r requirements.txt

# Start service
python -m uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload

# Alternative: Run in background
python -m uvicorn app.main:app --host 0.0.0.0 --port 8002 &
```

### 2. Analysis Service (Port 8003)
```bash
# Navigate to service directory
cd E:\rony-data\trading-platform\services\analysis-service

# Install dependencies
pip install -r requirements.txt

# Start service
python -m uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload
```

### 3. Sentiment Service (Port 8004)
```bash
# Navigate to service directory
cd E:\rony-data\trading-platform\services\sentiment-service

# Install dependencies
pip install -r requirements.txt

# Start service (working version)
python -m uvicorn app.main_working:app --host 0.0.0.0 --port 8004 --reload

# Alternative: Minimal version for testing
python -m uvicorn app.main_minimal:app --host 0.0.0.0 --port 8004 --reload
```

### 4. Portfolio Service (Port 8005)
```bash
# Navigate to service directory
cd E:\rony-data\trading-platform\services\portfolio-service

# Install dependencies
pip install -r requirements.txt

# Start service
python -m uvicorn app.main:app --host 0.0.0.0 --port 8005 --reload
```

### 5. Strategy Service (Port 8006)
```bash
# Navigate to service directory
cd E:\rony-data\trading-platform\services\strategy-service

# Install dependencies
pip install -r requirements.txt

# Start service
python -m uvicorn app.main:app --host 0.0.0.0 --port 8006 --reload
```

### 6. Fundamentals Service (Port 8007)
```bash
# Navigate to service directory
cd E:\rony-data\trading-platform\services\fundamentals-service

# Install dependencies
pip install -r requirements.txt

# Start service
python -m uvicorn app.main:app --host 0.0.0.0 --port 8007 --reload
```

### 7. Event Data Service (Port 8008)
```bash
# Navigate to service directory
cd E:\rony-data\trading-platform\services\event-data-service

# Install dependencies
pip install -r requirements.txt

# Start service
python -m uvicorn app.main:app --host 0.0.0.0 --port 8008 --reload
```

## Frontend Application

### 1. Start React Frontend (Port 3000)
```bash
# Navigate to frontend directory
cd E:\rony-data\trading-platform\trading-frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev

# Alternative: Start production build
npm run build
npm run start
```

### 2. Frontend Configuration
- **Development URL**: http://localhost:3000
- **Production Build**: Use `npm run build` then serve static files
- **Environment Variables**: Configure in `.env.local`

## Monitoring Services

### 1. Start Monitoring Stack
```bash
# Start monitoring services
cd E:\rony-data\trading-platform
docker-compose -f docker-compose.monitoring.yml up -d

# Available monitoring services:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3001
# - AlertManager: http://localhost:9093
```

### 2. Monitoring Script
```bash
# Use the monitoring startup script
chmod +x scripts/start-monitoring.sh
./scripts/start-monitoring.sh
```

## Complete Startup Sequence

### Option 1: Manual Startup (Recommended for Development)
```bash
# 1. Start Docker infrastructure
cd E:\rony-data\trading-platform
docker-compose up -d

# 2. Wait for services to be ready
timeout 30

# 3. Start microservices (open separate terminal windows/tabs)
# Terminal 1 - Market Data Service
cd services\market-data-service && python -m uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload

# Terminal 2 - Analysis Service
cd services\analysis-service && python -m uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload

# Terminal 3 - Sentiment Service
cd services\sentiment-service && python -m uvicorn app.main_working:app --host 0.0.0.0 --port 8004 --reload

# Terminal 4 - Portfolio Service
cd services\portfolio-service && python -m uvicorn app.main:app --host 0.0.0.0 --port 8005 --reload

# Terminal 5 - Strategy Service
cd services\strategy-service && python -m uvicorn app.main:app --host 0.0.0.0 --port 8006 --reload

# Terminal 6 - Frontend
cd trading-frontend && npm run dev
```

### Option 2: Automated Startup Script
```bash
# Use the provided startup script
cd E:\rony-data\trading-platform
start-services.bat

# Or create your own script based on start-services.bat
```

### Option 3: Background Services
```bash
# Start all services in background
cd E:\rony-data\trading-platform

# Infrastructure
docker-compose up -d

# Microservices (Windows)
start "Market Data" cmd /k "cd services\market-data-service && python -m uvicorn app.main:app --host 0.0.0.0 --port 8002"
start "Analysis" cmd /k "cd services\analysis-service && python -m uvicorn app.main:app --host 0.0.0.0 --port 8003"
start "Sentiment" cmd /k "cd services\sentiment-service && python -m uvicorn app.main_working:app --host 0.0.0.0 --port 8004"
start "Portfolio" cmd /k "cd services\portfolio-service && python -m uvicorn app.main:app --host 0.0.0.0 --port 8005"
start "Strategy" cmd /k "cd services\strategy-service && python -m uvicorn app.main:app --host 0.0.0.0 --port 8006"
start "Frontend" cmd /k "cd trading-frontend && npm run dev"
```

## Health Checks

### 1. Service Health Check Script
```bash
# Run comprehensive health check
cd E:\rony-data\trading-platform
python test-all-services.py

# Check individual services
curl http://localhost:8002/health  # Market Data
curl http://localhost:8003/health  # Analysis
curl http://localhost:8004/health  # Sentiment
curl http://localhost:8005/health  # Portfolio
curl http://localhost:8006/health  # Strategy
```

### 2. Database Health Check
```bash
# Test database connection
docker exec -it trading-platform-postgres-1 psql -U trading_user -d trading_db -c "SELECT version();"

# Test Redis connection
docker exec -it trading-platform-redis-1 redis-cli ping
```

### 3. Frontend Health Check
```bash
# Check if frontend is running
curl http://localhost:3000
```

## Service URLs and Ports

| Service | URL | Port | Description |
|---------|-----|------|-------------|
| Frontend | http://localhost:3000 | 3000 | React application |
| Market Data | http://localhost:8002 | 8002 | Market data and quotes |
| Analysis | http://localhost:8003 | 8003 | Technical analysis |
| Sentiment | http://localhost:8004 | 8004 | Sentiment analysis |
| Portfolio | http://localhost:8005 | 8005 | Portfolio management |
| Strategy | http://localhost:8006 | 8006 | Trading strategies |
| Fundamentals | http://localhost:8007 | 8007 | Fundamental data |
| Event Data | http://localhost:8008 | 8008 | Event data processing |
| PostgreSQL | localhost:5432 | 5432 | Database |
| Redis | localhost:6379 | 6379 | Cache |
| Prometheus | http://localhost:9090 | 9090 | Monitoring |
| Grafana | http://localhost:3001 | 3001 | Dashboards |

## Environment Variables

### Common Environment Variables
```bash
# Database Configuration
DATABASE_URL=postgresql://trading_user:trading_password@localhost:5432/trading_db

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# API Keys (configure as needed)
FINNHUB_API_KEY=your_finnhub_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
TWITTER_BEARER_TOKEN=your_twitter_token

# Service URLs
MARKET_DATA_SERVICE_URL=http://localhost:8002
ANALYSIS_SERVICE_URL=http://localhost:8003
SENTIMENT_SERVICE_URL=http://localhost:8004
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Port Already in Use
```bash
# Find process using port
netstat -ano | findstr :8002

# Kill process (Windows)
taskkill /PID <process_id> /F

# Kill process (Linux/Mac)
kill -9 <process_id>
```

#### 2. Database Connection Issues
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Restart PostgreSQL
docker-compose restart postgres

# Check database logs
docker-compose logs postgres
```

#### 3. Redis Connection Issues
```bash
# Check Redis status
docker ps | grep redis

# Restart Redis
docker-compose restart redis

# Test Redis connectivity
docker exec -it trading-platform-redis-1 redis-cli ping
```

#### 4. Service Startup Failures
```bash
# Check service logs
docker-compose logs [service_name]

# Verify Python dependencies
cd services/[service-name]
pip install -r requirements.txt

# Check if virtual environment is activated
python --version
```

#### 5. Frontend Build Issues
```bash
# Clear npm cache
cd trading-frontend
npm cache clean --force

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install

# Check Node.js version
node --version
npm --version
```

### Performance Optimization

#### 1. Memory Usage
```bash
# Monitor Docker memory usage
docker stats

# Limit container memory (add to docker-compose.yml)
mem_limit: 512m
```

#### 2. Database Performance
```bash
# Monitor database connections
docker exec -it trading-platform-postgres-1 psql -U trading_user -d trading_db -c "SELECT * FROM pg_stat_activity;"

# Optimize PostgreSQL settings in docker-compose.yml
command: postgres -c shared_preload_libraries=timescaledb -c max_connections=200
```

## Production Deployment

### 1. Environment Setup
```bash
# Set production environment variables
export NODE_ENV=production
export ENVIRONMENT=production

# Build frontend for production
cd trading-frontend
npm run build
```

### 2. Service Configuration
- Use production database URLs
- Configure proper logging levels
- Set up SSL certificates
- Configure load balancing
- Set up monitoring and alerting

### 3. Security Considerations
- Change default passwords
- Configure firewalls
- Set up API rate limiting
- Use HTTPS everywhere
- Regular security updates

## Maintenance

### Daily Operations
```bash
# Check service health
python test-all-services.py

# Monitor logs for errors
docker-compose logs --tail=100

# Check disk space
df -h

# Monitor memory usage
free -h
```

### Weekly Maintenance
```bash
# Update Docker images
docker-compose pull
docker-compose up -d

# Clean up unused Docker resources
docker system prune -f

# Backup database
docker exec trading-platform-postgres-1 pg_dump -U trading_user trading_db > backup_$(date +%Y%m%d).sql
```

---

## Quick Reference Commands

```bash
# Start everything
docker-compose up -d && python start-all-services.py

# Stop everything
docker-compose down && pkill -f uvicorn

# Restart a specific service
docker-compose restart postgres

# View all service status
python check-status.py

# Emergency stop all
docker stop $(docker ps -q) && pkill -f "uvicorn\|npm"
```

For additional help or issues, refer to the project's README.md or create an issue in the project repository.