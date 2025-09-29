# Trading Platform - Project Flow & Structure Guide

This document outlines the project structure, development workflow, and coding patterns for the trading platform.

## 📁 Project Structure Overview

```
trading-platform/
├── 🎨 frontend/                     # React Native (future mobile app)
├── 💻 trading-frontend/             # Next.js 15 Web Application
│   ├── apps/trading-web/           # Main web app
│   ├── libs/                       # Shared libraries
│   └── node_modules/               # Dependencies
├── ⚡ services/                     # FastAPI Microservices
│   ├── analysis-service/           # Technical analysis
│   ├── api-gateway/               # Request routing & auth
│   ├── market-data-service/       # Real-time stock data
│   ├── notification-service/      # Alerts & notifications
│   ├── portfolio-service/         # Portfolio management
│   ├── sentiment-service/         # News/social sentiment
│   ├── strategy-service/          # Trading strategies
│   ├── trading-engine-service/    # Order execution
│   └── user-service/              # User management
├── 🗄️ migrations/                  # Database schema & seed data
├── 📊 proto/                       # gRPC protobuf definitions
├── 🛠️ scripts/                     # Deployment & maintenance
├── 🌐 nginx/                       # Reverse proxy config
├── 📚 documentation/               # Architecture & API docs
├── 🔄 shared/                      # Common utilities
└── 🧪 tests/                       # Testing suites
```

## 🏗️ Architecture Patterns

### **Frontend Architecture (Next.js 15)**
```
trading-frontend/apps/trading-web/
├── app/                    # Next.js App Router
│   ├── (auth)/            # Authentication routes
│   ├── dashboard/         # Main trading dashboard
│   ├── portfolio/         # Portfolio management
│   └── layout.tsx         # Root layout
├── components/            # Reusable UI components
│   ├── ui/               # Base UI components (Radix)
│   ├── charts/           # Trading charts (Recharts)
│   └── forms/            # Form components
├── contexts/             # React contexts for state
├── lib/                  # Utilities & configurations
│   ├── api.ts           # API client setup
│   ├── utils.ts         # Helper functions
│   └── types.ts         # TypeScript definitions
└── public/              # Static assets
```

**Frontend Tech Stack:**
- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS v4
- **UI Library**: Radix UI primitives
- **State Management**: TanStack React Query
- **Charts**: Recharts for financial visualizations
- **Icons**: Lucide React

### **Backend Architecture (FastAPI Microservices)**

**Common Service Structure:**
```
services/[service-name]/
├── app/
│   ├── api/v1/endpoints/     # API route handlers
│   ├── core/                # Core functionality & config
│   ├── models/              # Database models (SQLAlchemy)
│   ├── schemas/             # Pydantic models for validation
│   ├── services/            # Business logic
│   └── main.py             # FastAPI app entry point
├── tests/                  # Unit & integration tests
├── Dockerfile             # Container configuration
└── requirements.txt       # Python dependencies
```

**Backend Tech Stack:**
- **Framework**: FastAPI (Python 3.11+)
- **Database ORM**: SQLAlchemy
- **Validation**: Pydantic
- **Authentication**: JWT tokens
- **Async**: asyncio/await patterns
- **Documentation**: OpenAPI/Swagger auto-generated

## 🔄 Development Workflow

### **1. Development Environment Setup**

```bash
# 1. Start Database Services
docker compose up -d postgres redis

# 2. Start Backend APIs (separate terminals)
cd services/market-data-service && uvicorn app.main:app --reload --port 8002
cd services/analysis-service && uvicorn app.main:app --reload --port 8003

# 3. Start Frontend
cd trading-frontend && npm run dev
```

### **2. Development URLs**
- **Frontend**: http://localhost:3000
- **Market Data API**: http://localhost:8002/docs
- **Analysis API**: http://localhost:8003/docs
- **Database GUI**: http://localhost:8080 (Adminer)
- **Redis GUI**: http://localhost:8081 (Redis Commander)

### **3. Code Development Patterns**

#### **Frontend Patterns**
```typescript
// API Client Pattern (lib/api.ts)
import { useQuery, useMutation } from '@tanstack/react-query'

export const useStockPrice = (symbol: string) => {
  return useQuery({
    queryKey: ['stock', symbol],
    queryFn: () => fetchStockPrice(symbol),
    refetchInterval: 5000 // Real-time updates
  })
}

// Component Pattern
export function StockCard({ symbol }: { symbol: string }) {
  const { data, isLoading, error } = useStockPrice(symbol)
  // Component logic
}
```

#### **Backend Patterns**
```python
# FastAPI Route Pattern
@app.get("/stocks/{symbol}/price")
async def get_stock_price(
    symbol: str,
    db: Session = Depends(get_db)
) -> StockPriceResponse:
    service = MarketDataService(db)
    return await service.get_current_price(symbol)

# Service Layer Pattern  
class MarketDataService:
    def __init__(self, db: Session):
        self.db = db
        
    async def get_current_price(self, symbol: str) -> StockPriceResponse:
        # Business logic implementation
        pass
```

## 🗄️ Database Design Patterns

### **Schema Structure**
```sql
-- UUID-based primary keys
id UUID PRIMARY KEY DEFAULT gen_random_uuid()

-- Audit fields on all tables
created_at TIMESTAMPTZ NOT NULL DEFAULT now()
updated_at TIMESTAMPTZ NOT NULL DEFAULT now()

-- Proper foreign key relationships
user_id UUID REFERENCES users(id) ON DELETE CASCADE

-- Enums for controlled values
CREATE TYPE alert_type AS ENUM ('price_above', 'price_below', ...)
```

### **Key Tables**
- **users** - Authentication & user profiles
- **portfolios** - User investment portfolios
- **portfolio_positions** - Stock positions within portfolios
- **alerts** - Price & technical alerts
- **candles** - Time-series market data
- **watchlists** - User stock watchlists

## 🚀 Deployment Patterns

### **Development Deployment**
```bash
# Single command development start
docker compose up -d

# Individual service development
cd services/[service] && uvicorn app.main:app --reload
```

### **Production Deployment (Docker Swarm)**
```bash
# Server 1 (Database Server)
./scripts/setup-swarm.sh

# Server 2 (API Server) 
curl -fsSL <repo>/scripts/join-swarm.sh | bash -s worker <ip> <token>

# Deploy production stack
./scripts/deploy-production.sh
```

## 🧪 Testing Patterns

### **API Testing**
```bash
# Health checks
curl http://localhost:8002/health
curl http://localhost:8003/health

# Functional testing
curl "http://localhost:8002/stocks/AAPL"
curl "http://localhost:8003/analyze/AAPL?period=6mo"
```

### **Database Testing**
```bash
# Connection test
psql "postgresql://trading_user:trading_pass@localhost:5432/trading_db"

# Schema validation
psql -f scripts/test-db-connection.sql
```

## 📋 Code Standards

### **TypeScript/React Standards**
- Use TypeScript strict mode
- Functional components with hooks
- TanStack Query for server state
- Tailwind CSS for styling
- Radix UI for accessible components

### **Python/FastAPI Standards**
- Type hints on all functions
- Pydantic models for validation
- SQLAlchemy for database operations
- Async/await for I/O operations
- OpenAPI documentation

### **File Naming Conventions**
- **Frontend**: kebab-case for files, PascalCase for components
- **Backend**: snake_case for Python files and functions
- **Database**: snake_case for tables and columns
- **API Routes**: REST conventions (/api/v1/resource)

## 🔄 Git Workflow

```bash
# Feature development
git checkout -b feature/stock-analysis
git add .
git commit -m "feat: add technical analysis indicators"
git push origin feature/stock-analysis

# Database changes
git add migrations/
git commit -m "feat: add portfolio performance tracking schema"
```

## 📈 Performance Patterns

### **Frontend Optimization**
- React Query for caching & background updates
- Code splitting with Next.js dynamic imports
- Image optimization with Next.js Image component
- WebSocket connections for real-time data

### **Backend Optimization**
- Redis caching for frequently accessed data
- Database connection pooling
- Async I/O for external API calls
- Background tasks for data processing

---

## 🗓️ Development Phases

### **Phase 1: Core Infrastructure** ✅
- Database schema & migrations
- Basic API services structure
- Frontend foundation with Next.js
- Docker development environment

### **Phase 2: Market Data Integration** 🔄
- Real-time stock price feeds
- Historical data storage
- WebSocket connections
- Chart visualization

### **Phase 3: Portfolio Management** 📋
- Portfolio CRUD operations
- Position tracking
- Performance analytics
- Alert system implementation

### **Phase 4: Advanced Features** 🚀
- Technical analysis indicators
- Trading strategy backtesting
- News sentiment analysis
- Mobile app development

---

## 🔄 **Change Tracking & Version Control**

### **Recent Changes**
```
2025-01-22: Created initial project structure documentation
- Added comprehensive architecture overview
- Documented all 9 microservices
- Established development workflow patterns
```

### **File Dependencies Map**
```
Trading Platform Dependencies:
├── Frontend (trading-frontend/) → Backend APIs (services/)
├── Backend APIs → Database (PostgreSQL + Redis)  
├── Docker Compose → All Services (development)
├── Docker Swarm → All Services (production)
└── Documentation → All Code Changes
```

### **Change Impact Analysis**
When modifying files, consider impact on:
- **Frontend changes** → May need API updates
- **API changes** → Update OpenAPI docs, frontend client
- **Database schema** → Create migration, update models
- **Docker configs** → Rebuild containers, update deployment

### **Code Review Checklist**
- [ ] Tests updated/added
- [ ] Documentation updated
- [ ] API docs regenerated (if backend changes)
- [ ] Environment variables added to `.env.example`
- [ ] Docker configurations updated if needed
- [ ] Change logged in `claude-chat.md`

---

## 📋 **Development Todos & Roadmap**

### **Immediate Next Steps**
- [ ] Complete market data service implementation
- [ ] Set up frontend-backend API integration  
- [ ] Implement user authentication flow
- [ ] Create portfolio dashboard components

### **Feature Development Pipeline**
1. **Core Features** (Current Phase)
   - [ ] Real-time stock price feeds
   - [ ] User portfolio management
   - [ ] Basic alerting system

2. **Advanced Features** (Next Phase)  
   - [ ] Technical analysis indicators
   - [ ] Strategy backtesting
   - [ ] News sentiment integration

3. **Production Features** (Future)
   - [ ] Mobile app development
   - [ ] Advanced trading algorithms
   - [ ] Machine learning models

---

*Last Updated: 2025-01-22*
*Next Update: [Will be updated by Claude in next session]*
*Change Log: See claude-chat.md for detailed change history*