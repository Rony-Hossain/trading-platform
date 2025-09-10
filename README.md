# Trading Platform

A comprehensive trading platform built with modern technologies including Next.js, PostgreSQL, Redis, and microservices architecture.

## Database Implementation

The database has been implemented with the following components:

### ✅ Completed Features

- **SQL Schema**: Complete database schema with users, portfolios, alerts, candles, and caching tables
- **Protobuf Definitions**: Type-safe service interfaces for internal communication
- **Docker Setup**: PostgreSQL and Redis with healthchecks and auto-initialization
- **Migrations**: Automated database setup with initial schema
- **Seed Data**: Test data for development and testing

### Database Structure

```
📁 migrations/
├── 001_initial_schema.sql    # Main database schema
└── seed_data.sql            # Development test data

📁 proto/
├── market_data.proto        # Market data service definitions
├── analysis.proto          # Analysis service definitions
└── alerts.proto            # Alert engine definitions

📁 scripts/
├── init-db.sh              # Database initialization script
└── test-db-connection.sql   # Connection and schema verification
```

### Quick Start

1. **Copy environment file**:
   ```bash
   cp .env.example .env
   ```

2. **Start databases** (requires Docker):
   ```bash
   docker compose up -d postgres redis
   ```

3. **Verify setup**:
   ```bash
   # Check container health
   docker compose ps
   
   # Test database connection
   psql "postgresql://trading_user:trading_pass@localhost:5432/trading_db" -f scripts/test-db-connection.sql
   ```

### Schema Highlights

- **Users & Authentication**: UUID-based user management with bcrypt password hashing
- **Portfolio Management**: Multi-portfolio support with position tracking
- **Alert System**: Flexible alert types (price, volume, technical signals) with cooldowns
- **Time-Series Data**: Optimized candle storage with proper indexing
- **Caching Layer**: Redis integration with PostgreSQL JSONB for analysis results
- **Audit Trail**: Alert triggers and timestamp tracking

### API Interfaces

The platform uses a hybrid approach:
- **REST + OpenAPI**: Frontend ↔ Backend communication (see `documentation/contracts/`)
- **gRPC + Protobuf**: Internal service communication for performance
- **WebSocket/SSE**: Real-time data streaming (planned)

## Multi-Server Production Setup

### Architecture: Windows Dev + 2 Ubuntu Servers

**Windows 11 (Development)**:
- Frontend development (Next.js)
- Local PostgreSQL/Redis for testing
- Docker Desktop for local stack

**Ubuntu Server 1 (Database)**:
- PostgreSQL primary
- Redis primary
- Alert Engine worker
- Prometheus monitoring

**Ubuntu Server 2 (APIs/Web)**:
- Market Data API
- Analysis API
- Frontend (production)
- Nginx reverse proxy
- Redis replica

### Deployment Steps

1. **Setup Docker Swarm** (on Ubuntu Server 1):
   ```bash
   chmod +x scripts/setup-swarm.sh
   ./scripts/setup-swarm.sh
   ```

2. **Join Second Server** (on Ubuntu Server 2):
   ```bash
   # Use the command provided by setup-swarm.sh output
   curl -fsSL <your-repo>/scripts/join-swarm.sh | bash -s worker <manager-ip> <token>
   ```

3. **Deploy to Production**:
   ```bash
   chmod +x scripts/deploy-production.sh
   ./scripts/deploy-production.sh
   ```

### Development vs Production

- **Development**: Use `docker-compose up -d` (includes Adminer, Redis Commander)
- **Production**: Use `docker stack deploy -c docker-compose.prod.yml trading`

### Monitoring & Access

- **Frontend**: https://yourdomain.com
- **API**: https://api.yourdomain.com
- **Prometheus**: http://server1:9090
- **Grafana**: http://server1:3001
- **Database**: Direct access via PostgreSQL client

### Next Steps

See `documentation/suggestions.md` for the complete implementation roadmap including:
- Backend API services (Market Data, Analysis)
- Frontend improvements and React Query integration  
- Alert Engine worker implementation
- ML forecasting and trading algorithms

---

For detailed commands and setup instructions, see `documentation/commands.md`.

