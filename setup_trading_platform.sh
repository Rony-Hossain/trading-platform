#!/bin/bash

# One-command Trading Platform Setup
# This script creates the entire project structure and sets up the development environment

set -e

PROJECT_NAME="trading-platform"
PYTHON_VERSION="3.11"

echo "🚀 Trading Platform - One-Command Setup"
echo "========================================"

# Check Python version
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 is not installed. Please install Python $PYTHON_VERSION or higher."
        exit 1
    fi
    
    PYTHON_VER=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    if [ "$(printf '%s\n' "$PYTHON_VERSION" "$PYTHON_VER" | sort -V | head -n1)" != "$PYTHON_VERSION" ]; then
        echo "⚠️  Warning: Python $PYTHON_VER detected. Recommended: $PYTHON_VERSION or higher"
    fi
}

# Check Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed. Please install Docker first."
        echo "   Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
}

# Check Git
check_git() {
    if ! command -v git &> /dev/null; then
        echo "❌ Git is not installed. Please install Git first."
        exit 1
    fi
}

echo "🔍 Checking prerequisites..."
check_python
check_docker
check_git

# Create project directory
echo "📁 Creating project structure..."
if [ -d "$PROJECT_NAME" ]; then
    echo "⚠️  Directory $PROJECT_NAME already exists. Do you want to continue? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "❌ Setup cancelled."
        exit 1
    fi
fi

mkdir -p $PROJECT_NAME
cd $PROJECT_NAME

# Create complete directory structure (from previous script)
echo "📁 Creating service directories..."
services=(
    "api-gateway"
    "market-data-service" 
    "analysis-service"
    "sentiment-service"
    "trading-engine-service"
    "portfolio-service"
    "strategy-service"
    "notification-service"
    "user-service"
    "monitoring-service"
)

for service in "${services[@]}"; do
    mkdir -p "services/$service/app/"{core,models,schemas,services,api/v1/endpoints,utils}
    mkdir -p "services/$service/tests/"{unit,integration,e2e}
    mkdir -p "services/$service/migrations"
    
    # Create __init__.py files
    find "services/$service" -type d -exec touch {}/__init__.py \;
done

# Create specialized directories
mkdir -p "services/market-data-service/app/"{clients,api/websocket}
mkdir -p "services/analysis-service/app/"{algorithms/indicators,algorithms/patterns,algorithms/ml,clients}
mkdir -p "services/analysis-service/notebooks"
mkdir -p "services/sentiment-service/app/"{scrapers/news,scrapers/social,nlp}
mkdir -p "services/sentiment-service/models"
mkdir -p "services/trading-engine-service/app/"{brokers,risk}
mkdir -p "services/portfolio-service/app/analytics"
mkdir -p "services/strategy-service/app/"{strategies/mean_reversion,strategies/momentum,strategies/arbitrage,strategies/ml_strategies,backtesting}
mkdir -p "services/strategy-service/notebooks"
mkdir -p "services/notification-service/app/"{channels,templates/email,templates/sms}
mkdir -p "services/monitoring-service/app/collectors"

# Create all __init__.py files
find services/ -type d -exec touch {}/__init__.py \; 2>/dev/null || true

# Create shared, infrastructure, frontend, scripts, tests, docs, notebooks, data, logs, config directories
mkdir -p "shared/"{schemas,utils,models,clients,middleware,constants,proto}
mkdir -p "infrastructure/"{docker,kubernetes,terraform,monitoring}
mkdir -p "infrastructure/kubernetes/"{namespaces,secrets,configmaps,deployments,services,ingress,persistent-volumes,jobs,monitoring}
mkdir -p "infrastructure/terraform/"{modules,environments,scripts}
mkdir -p "infrastructure/monitoring/"{prometheus,grafana,alertmanager,jaeger,elasticsearch}
mkdir -p "frontend/web/"{public,src}
mkdir -p "frontend/web/src/"{components,pages,services,hooks,store,utils,types}
mkdir -p "frontend/mobile/"{src,android,ios}
mkdir -p "scripts/"{setup,deployment,maintenance,data,monitoring,development}
mkdir -p "tests/"{integration,e2e,load,security,fixtures}
mkdir -p "docs/"{api,architecture,development,operations,user,compliance,research}
mkdir -p "notebooks/"{research,backtesting,data_exploration,monitoring}
mkdir -p "data/"{raw,processed,models,exports}
mkdir -p "logs/"{application,system,audit,trading,archived}
mkdir -p "config/"{environments,database,message_queue,monitoring,nginx,ssl,secrets}

find shared/ -type d -exec touch {}/__init__.py \; 2>/dev/null || true

echo "📄 Creating configuration files..."

# Create pyproject.toml
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "trading-platform"
version = "0.1.0"
description = "Microservices-based trading platform"
readme = "README.md"
requires-python = ">=3.11"

dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.4.0",
    "sqlalchemy>=2.0.0",
    "alembic>=1.12.0",
    "psycopg2-binary>=2.9.0",
    "redis>=5.0.0",
    "celery>=5.3.0",
    "pandas>=2.1.0",
    "numpy>=1.25.0",
    "ta-lib>=0.4.0",
    "yfinance>=0.2.0",
    "httpx>=0.25.0",
    "websockets>=11.0",
    "pyjwt>=2.8.0",
    "passlib[bcrypt]>=1.7.0",
    "python-multipart>=0.0.6",
    "prometheus-client>=0.17.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.9.0",
    "isort>=5.12.0",
    "flake8>=6.1.0",
    "mypy>=1.6.0",
    "pre-commit>=3.4.0",
    "jupyterlab>=4.0.0",
]

analysis = [
    "scipy>=1.11.0",
    "scikit-learn>=1.3.0",
    "nltk>=3.8.0",
    "textblob>=0.17.0",
    "beautifulsoup4>=4.12.0",
    "requests>=2.31.0",
]
EOF

# Create .env.example
cat > .env.example << 'EOF'
# Environment
ENVIRONMENT=development
DEBUG=true

# Database
DATABASE_URL=postgresql://trading_user:trading_pass@localhost:5432/trading_db
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET_KEY=your-jwt-secret-key-change-in-production

# External APIs (Free tiers available)
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key
NEWS_API_KEY=your-news-api-key

# Services Ports
API_GATEWAY_PORT=8000
MARKET_DATA_SERVICE_PORT=8001
ANALYSIS_SERVICE_PORT=8002
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
.Python
build/
dist/
*.egg-info/
.env
.venv/
venv/
*.log
logs/
.DS_Store
.vscode/
.idea/
*.db
*.sqlite3
.ipynb_checkpoints
data/raw/
data/processed/
*.csv
*.parquet
tmp/
temp/
*.tfstate*
.terraform/
config/secrets/
*.pem
*.key
*.crt
EOF

# Create docker-compose.dev.yml
cat > docker-compose.dev.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: trading_db
      POSTGRES_USER: trading_user
      POSTGRES_PASSWORD: trading_pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
EOF

# Create Makefile
cat > Makefile << 'EOF'
.PHONY: help install dev-setup test clean docker-up docker-down

help:
	@echo 'Trading Platform - Available Commands:'
	@echo ''
	@echo '  install      Install Python dependencies'
	@echo '  dev-setup    Setup development environment'
	@echo '  test         Run tests'
	@echo '  docker-up    Start infrastructure services'
	@echo '  docker-down  Stop all services'
	@echo '  clean        Clean generated files'

install:
	pip install -e ".[dev,analysis]"

dev-setup:
	cp .env.example .env
	docker-compose -f docker-compose.dev.yml up -d
	@echo "✅ Development environment ready!"

test:
	pytest tests/ -v

docker-up:
	docker-compose -f docker-compose.dev.yml up -d

docker-down:
	docker-compose -f docker-compose.dev.yml down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
EOF

# Create README.md
cat > README.md << 'EOF'
# Trading Platform

A microservices-based trading platform for algorithmic trading and market analysis.

## Quick Start

1. **Prerequisites**
   - Python 3.11+
   - Docker & Docker Compose
   - Git

2. **Setup**
   ```bash
   # Setup development environment
   make dev-setup
   
   # Install Python dependencies
   make install
   ```

3. **Configure**
   ```bash
   # Edit .env with your API keys
   nano .env
   ```

4. **Start Services**
   ```bash
   # Start infrastructure
   make docker-up
   ```

5. **Access**
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090

## Development

- `make help` - Show available commands
- `make test` - Run tests
- `make clean` - Clean generated files

## Next Steps

1. Get free API keys:
   - Alpha Vantage: https://www.alphavantage.co/support/#api-key
   - News API: https://newsapi.org/register

2. Start building services:
   - Begin with market-data-service
   - Add analysis capabilities
   - Implement trading strategies

## Architecture

```
API Gateway → Microservices → Databases
    ↓
Message Queue → Background Tasks
    ↓
Monitoring Stack
```
EOF

# Create first service template
echo "🔧 Creating first service template..."
mkdir -p services/market-data-service/app

cat > services/market-data-service/app/main.py << 'EOF'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Market Data Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Market Data Service is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "market-data-service"}

@app.get("/stocks/{symbol}/price")
async def get_stock_price(symbol: str):
    # TODO: Implement real market data fetching
    return {
        "symbol": symbol.upper(),
        "price": 150.00,
        "change": 2.50,
        "change_percent": 1.69,
        "timestamp": "2024-08-22T10:30:00Z"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
EOF

cat > services/market-data-service/requirements.txt << 'EOF'
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.4.0
yfinance>=0.2.0
httpx>=0.25.0
python-multipart>=0.0.6
EOF

cat > services/market-data-service/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
EOF

# Create basic test
cat > tests/test_basic.py << 'EOF'
def test_basic():
    """Basic test to ensure testing works"""
    assert True

def test_math():
    """Test basic math operations"""
    assert 2 + 2 == 4
    assert 10 / 2 == 5
EOF

# Create pytest.ini
cat > pytest.ini << 'EOF'
[tool:pytest]
testpaths = tests
python_files = test_*.py
addopts = -v --cov=services --cov=shared
asyncio_mode = auto
EOF

echo "🐍 Setting up Python environment..."

# Create virtual environment
if command -v python3 &> /dev/null; then
    python3 -m venv .venv
    source .venv/bin/activate
    
    echo "📦 Installing dependencies..."
    pip install --upgrade pip
    pip install -e ".[dev,analysis]"
    
    echo "🔧 Setting up pre-commit hooks..."
    pip install pre-commit
    
    # Create basic pre-commit config
    cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
EOF
    
    pre-commit install
fi

echo "🐳 Starting Docker services..."
docker-compose -f docker-compose.dev.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

echo "✅ Setup completed successfully!"
echo ""
echo "🎉 Trading Platform is ready!"
echo "================================"
echo ""
echo "📊 Access Points:"
echo "  • Grafana:    http://localhost:3000 (admin/admin)"
echo "  • Prometheus: http://localhost:9090"
echo "  • PostgreSQL: localhost:5432 (trading_user/trading_pass)"
echo "  • Redis:      localhost:6379"
echo ""
echo "🚀 Next Steps:"
echo "  1. Edit .env file with your API keys:"
echo "     nano .env"
echo ""
echo "  2. Test the market data service:"
echo "     cd services/market-data-service"
echo "     python -m app.main"
echo "     # Visit: http://localhost:8001"
echo ""
echo "  3. Run tests:"
echo "     make test"
echo ""
echo "  4. Start building your trading strategies!"
echo ""
echo "📚 Documentation:"
echo "  • README.md - Getting started guide"
echo "  • docs/ - Detailed documentation"
echo "  • notebooks/ - Research and analysis"
echo ""
echo "🆘 Need help?"
echo "  • make help - Show available commands"
echo "  • Check logs: docker-compose -f docker-compose.dev.yml logs"