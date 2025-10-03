# Trading Platform Frontend - Comprehensive Revamp Plan

**Date:** 2025-10-03
**Version:** 2.0.0
**Tech Stack:** Next.js 15 + React 19 + TypeScript + Tailwind CSS + Material-UI

---

## Executive Summary

Complete frontend revamp with:
- ✅ **Beginner & Expert Dashboards** (dual-mode interface)
- ✅ **Theme System** (Dark/Light mode with custom colors)
- ✅ **Advanced Charting** (TradingView-style with multiple chart types)
- ✅ **Indicator Selector** (50+ technical indicators)
- ✅ **Settings Management** (comprehensive user preferences)
- ✅ **Journal Management** (trade logging and analysis)
- ✅ **Alert Management** (real-time notifications)
- ✅ **Responsive Design** (mobile, tablet, desktop)

**Estimated Timeline:** 4-6 weeks
**Files to Create/Modify:** ~80 files
**Lines of Code:** ~15,000 lines

---

## Architecture Overview

```
trading-frontend/
├── apps/
│   └── trading-web/
│       ├── app/                          # Next.js 15 App Router
│       │   ├── (auth)/                   # Auth routes
│       │   ├── (dashboard)/              # Main app routes
│       │   │   ├── beginner/             # Beginner dashboard
│       │   │   ├── expert/               # Expert dashboard
│       │   │   ├── journal/              # Trade journal
│       │   │   ├── alerts/               # Alert management
│       │   │   └── settings/             # Settings
│       │   └── layout.tsx                # Root layout
│       ├── components/
│       │   ├── charts/                   # Chart components
│       │   │   ├── TradingChart.tsx      # Main chart component
│       │   │   ├── ChartTypeSelector.tsx # Candlestick, Line, etc.
│       │   │   ├── IndicatorPanel.tsx    # Indicator management
│       │   │   └── DrawingTools.tsx      # Trend lines, etc.
│       │   ├── dashboard/
│       │   │   ├── beginner/             # Beginner components
│       │   │   └── expert/               # Expert components
│       │   ├── journal/                  # Journal components
│       │   ├── alerts/                   # Alert components
│       │   ├── settings/                 # Settings components
│       │   └── ui/                       # Reusable UI components
│       ├── lib/
│       │   ├── api/                      # API client
│       │   ├── hooks/                    # Custom React hooks
│       │   ├── stores/                   # State management (Zustand)
│       │   ├── theme/                    # Theme configuration
│       │   └── utils/                    # Utility functions
│       └── styles/
│           └── globals.css               # Global styles
└── libs/                                 # Shared libraries
    ├── ui-components/                    # Reusable components
    └── utils/                            # Shared utilities
```

---

## Feature Specifications

### 1. Dashboard System (Beginner vs Expert)

#### **Beginner Dashboard** 👶
**Target Users:** New traders, casual investors

**Layout:**
```
┌─────────────────────────────────────────────────────┐
│  Top Bar: Portfolio Value | Daily P&L | Alerts     │
├─────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────────────┐ │
│  │  Watchlist      │  │  Recommended Trades      │ │
│  │  - AAPL         │  │  ┌────────────────────┐  │ │
│  │  - MSFT         │  │  │ BUY AAPL           │  │ │
│  │  - GOOGL        │  │  │ Confidence: 85%    │  │ │
│  │  (Simple list)  │  │  │ Reason: Strong...  │  │ │
│  └─────────────────┘  │  └────────────────────┘  │ │
│                       └──────────────────────────┘ │
├─────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────┐  │
│  │  Simple Chart (Candlestick only)            │  │
│  │  + Basic indicators (SMA, Volume)           │  │
│  └──────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │ Recent      │  │ Performance │  │ Quick      │ │
│  │ Trades      │  │ Summary     │  │ Actions    │ │
│  └─────────────┘  └─────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────┘
```

**Features:**
- ✅ Simplified interface (minimal clutter)
- ✅ AI-powered trade recommendations with explanations
- ✅ One-click trade execution
- ✅ Performance metrics (simple)
- ✅ Educational tooltips
- ✅ Guided tours for new users

#### **Expert Dashboard** 🚀
**Target Users:** Professional traders, quants

**Layout:**
```
┌──────────────────────────────────────────────────────────────┐
│  Top Bar: Multi-account | Real-time P&L | System Status    │
├──────────────────────────────────────────────────────────────┤
│  ┌────────┐  ┌──────────────────────────────────────────┐   │
│  │Watch   │  │  Advanced Multi-Chart View              │   │
│  │list    │  │  ┌──────────┐  ┌──────────┐             │   │
│  │        │  │  │ Chart 1  │  │ Chart 2  │             │   │
│  │+ Heat  │  │  │ (1min)   │  │ (Daily)  │             │   │
│  │  map   │  │  └──────────┘  └──────────┘             │   │
│  └────────┘  └──────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ Order    │ │ Position │ │ Options  │ │ Greeks & │      │
│  │ Book     │ │ Manager  │ │ Chain    │ │ Analytics│      │
│  │ (L2)     │ │          │ │          │ │          │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
├──────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌────────────────────────────────┐  │
│  │ Terminal/Logs    │  │ Custom Indicators & Alerts    │  │
│  │ > Order filled   │  │ - VWAP cross                  │  │
│  │ > Alert: RSI >70 │  │ - Bollinger squeeze           │  │
│  └──────────────────┘  └────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

**Features:**
- ✅ Multi-chart layout (up to 6 charts)
- ✅ Level 2 market data
- ✅ Advanced order types (IOC, FOK, Iceberg, etc.)
- ✅ Custom indicator builder
- ✅ Algorithmic trading interface
- ✅ Real-time Greeks for options
- ✅ Heatmaps and correlation matrices
- ✅ Terminal/console for logs

---

### 2. Theme System 🎨

#### **Color Modes:**
1. **Dark Mode** (default for traders)
2. **Light Mode**
3. **Auto** (system preference)

#### **Color Customization:**
```typescript
interface ThemeSettings {
  mode: 'dark' | 'light' | 'auto';

  // Custom colors
  primaryColor: string;      // Accent color
  bullishColor: string;      // Green (up)
  bearishColor: string;      // Red (down)
  chartBackground: string;   // Chart canvas
  gridColor: string;         // Chart grid

  // Chart styles
  candleStyle: 'solid' | 'hollow';
  chartType: 'candlestick' | 'line' | 'area' | 'heikin-ashi';

  // Font
  fontSize: 'small' | 'medium' | 'large';
  fontFamily: 'Inter' | 'Roboto' | 'SF Pro';
}
```

#### **Presets:**
- 🌙 **Dark Classic** (Black background, green/red)
- ☀️ **Light Modern** (White background, blue/orange)
- 🌃 **Night Blue** (Navy background, cyan/magenta)
- 🎯 **High Contrast** (Accessibility-focused)
- 🔥 **Custom** (User-defined)

#### **Implementation:**
```typescript
// Using Tailwind CSS + CSS Variables
// lib/theme/theme-provider.tsx

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useLocalStorage('theme', defaultTheme);

  useEffect(() => {
    document.documentElement.style.setProperty('--primary', theme.primaryColor);
    document.documentElement.style.setProperty('--bullish', theme.bullishColor);
    // ... etc
  }, [theme]);

  return <ThemeContext.Provider value={{theme, setTheme}}>{children}</ThemeContext.Provider>;
}
```

---

### 3. Advanced Charting 📊

#### **Chart Library:** Lightweight Charts (TradingView library)
**Alternative:** Recharts (currently installed) + Custom overlays

#### **Chart Types:**
1. **Candlestick** ✅ (default)
2. **Line Chart** ✅
3. **Area Chart** ✅
4. **Heikin-Ashi** ✅
5. **Renko** 🔄 (advanced)
6. **Point & Figure** 🔄 (advanced)
7. **Volume Profile** ✅

#### **Timeframes:**
- 1 minute, 5 min, 15 min, 30 min, 1 hour
- 4 hour, Daily, Weekly, Monthly

#### **Features:**
- ✅ **Multi-pane layout** (Price + Volume + Indicators)
- ✅ **Drawing tools** (Trend lines, channels, Fibonacci)
- ✅ **Crosshair with data tooltip**
- ✅ **Zoom & Pan**
- ✅ **Save chart layouts**
- ✅ **Screenshot/export**
- ✅ **Real-time updates** (WebSocket)

#### **Chart Component Structure:**
```typescript
// components/charts/TradingChart.tsx

interface TradingChartProps {
  symbol: string;
  timeframe: Timeframe;
  chartType: ChartType;
  indicators: Indicator[];
  height: number;
  mode: 'beginner' | 'expert';
}

export function TradingChart({ symbol, timeframe, chartType, indicators, mode }: TradingChartProps) {
  // Main chart + Indicator overlays + Volume pane
  return (
    <div className="chart-container">
      <ChartToolbar />
      <MainChart data={priceData} type={chartType} />
      <IndicatorOverlays indicators={indicators} />
      <VolumePane data={volumeData} />
      <TimeAxis />
    </div>
  );
}
```

---

### 4. Indicator Selector 📈

#### **Categories:**

**Trend Indicators:**
- Moving Averages (SMA, EMA, WMA, VWAP)
- Bollinger Bands
- Ichimoku Cloud
- Parabolic SAR

**Momentum Indicators:**
- RSI (Relative Strength Index)
- MACD
- Stochastic Oscillator
- Williams %R
- Momentum
- ROC (Rate of Change)

**Volume Indicators:**
- On-Balance Volume (OBV)
- Volume Weighted Average Price (VWAP)
- Accumulation/Distribution
- Chaikin Money Flow

**Volatility Indicators:**
- Average True Range (ATR)
- Bollinger Bandwidth
- Keltner Channels
- Donchian Channels

**Custom Indicators:**
- User-defined formulas
- Import from TradingView Pine Script (future)

#### **Indicator Panel UI:**
```
┌─────────────────────────────────────────┐
│  Indicators                        [+]  │
├─────────────────────────────────────────┤
│  ✓ SMA (20)        [⚙️] [👁️] [🗑️]      │
│  ✓ RSI (14)        [⚙️] [👁️] [🗑️]      │
│  ✓ MACD            [⚙️] [👁️] [🗑️]      │
├─────────────────────────────────────────┤
│  [+ Add Indicator]                      │
│                                         │
│  Popular:                               │
│  • Bollinger Bands                      │
│  • Volume                               │
│  • Stochastic                          │
└─────────────────────────────────────────┘
```

#### **Implementation:**
```typescript
// components/charts/IndicatorPanel.tsx

interface Indicator {
  id: string;
  type: IndicatorType;
  params: Record<string, number>;
  visible: boolean;
  color: string;
  pane: 'main' | 'separate';
}

export function IndicatorPanel({ indicators, onAdd, onRemove, onUpdate }) {
  return (
    <div className="indicator-panel">
      {indicators.map(indicator => (
        <IndicatorRow
          key={indicator.id}
          indicator={indicator}
          onSettings={() => openSettingsModal(indicator)}
          onToggle={() => toggleVisibility(indicator.id)}
          onDelete={() => onRemove(indicator.id)}
        />
      ))}
      <AddIndicatorButton onClick={() => openIndicatorLibrary()} />
    </div>
  );
}
```

---

### 5. Settings Management ⚙️

#### **Settings Categories:**

**1. General**
- Language
- Timezone
- Date/Time format
- Default currency

**2. Trading**
- Default order type
- Order confirmation (on/off)
- Risk warnings
- Position size calculator settings
- Stop-loss defaults

**3. Appearance**
- Theme (Dark/Light/Auto)
- Color customization
- Chart defaults
- Font size
- Layout density (compact/comfortable/spacious)

**4. Notifications**
- Email alerts
- Push notifications
- Sound alerts
- Alert types (price, volume, indicator)

**5. Data & Privacy**
- Data retention
- Analytics opt-in/out
- Export data

**6. Advanced**
- API keys
- WebSocket settings
- Performance mode (reduce animations)
- Debug mode

#### **Settings Page Layout:**
```
┌─────────────────────────────────────────────────────┐
│  Settings                                      [✕]  │
├──────────┬──────────────────────────────────────────┤
│ General  │  General Settings                        │
│ Trading  │  ┌────────────────────────────────────┐  │
│ Appear.  │  │ Language:        [English ▼]      │  │
│ Notif.   │  │ Timezone:        [UTC-5 ▼]        │  │
│ Data     │  │ Date Format:     [MM/DD/YYYY ▼]   │  │
│ Advanced │  │ Currency:        [USD ▼]          │  │
│          │  └────────────────────────────────────┘  │
│          │                                          │
│          │  [Save Changes]  [Reset to Defaults]    │
└──────────┴──────────────────────────────────────────┘
```

#### **Implementation:**
```typescript
// components/settings/SettingsPage.tsx

interface UserSettings {
  general: GeneralSettings;
  trading: TradingSettings;
  appearance: AppearanceSettings;
  notifications: NotificationSettings;
  data: DataSettings;
  advanced: AdvancedSettings;
}

export function SettingsPage() {
  const [settings, setSettings] = useSettings();
  const [activeTab, setActiveTab] = useState('general');

  return (
    <div className="settings-page">
      <SettingsSidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <SettingsContent>
        {activeTab === 'general' && <GeneralSettings settings={settings.general} />}
        {activeTab === 'trading' && <TradingSettings settings={settings.trading} />}
        {/* ... */}
      </SettingsContent>
    </div>
  );
}
```

---

### 6. Trade Journal Management 📔

#### **Features:**
- ✅ **Trade Logging** (automatic + manual)
- ✅ **P&L Tracking** (realized/unrealized)
- ✅ **Trade Analytics** (win rate, avg profit, drawdown)
- ✅ **Tags & Categories** (strategy, asset class, setup type)
- ✅ **Screenshots** (auto-capture chart on entry/exit)
- ✅ **Notes & Lessons** (post-trade review)
- ✅ **Performance Charts** (equity curve, calendar heatmap)
- ✅ **Export** (CSV, PDF report)

#### **Journal Entry Structure:**
```typescript
interface JournalEntry {
  id: string;
  timestamp: Date;
  symbol: string;
  side: 'long' | 'short';
  entry: {
    price: number;
    quantity: number;
    time: Date;
    screenshot?: string;
  };
  exit: {
    price: number;
    quantity: number;
    time: Date;
    screenshot?: string;
  };
  pnl: {
    gross: number;
    net: number;
    percentReturn: number;
  };
  tags: string[];
  strategy: string;
  setup: string;
  notes: string;
  mistakes: string[];
  lessons: string[];
}
```

#### **Journal Page Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  Trade Journal                     [+ New Entry] [Export]  │
├─────────────────────────────────────────────────────────────┤
│  Filters: [All ▼] [Strategy ▼] [Date Range ▼] [🔍 Search] │
├─────────────────────────────────────────────────────────────┤
│  Performance Summary                                        │
│  ┌──────────┬──────────┬──────────┬──────────┐            │
│  │ Win Rate │  Avg P&L │ Sharpe   │ Max DD   │            │
│  │   67%    │  $234    │   1.82   │  -$456   │            │
│  └──────────┴──────────┴──────────┴──────────┘            │
├─────────────────────────────────────────────────────────────┤
│  Trade List                                                 │
│  ┌──────┬────────┬──────┬──────┬─────────┬────────┐       │
│  │ Date │ Symbol │ Side │ P&L  │ Strategy│ [View] │       │
│  ├──────┼────────┼──────┼──────┼─────────┼────────┤       │
│  │ 10/3 │ AAPL   │ LONG │ +$120│ Breakout│  [👁️] │       │
│  │ 10/2 │ MSFT   │ SHORT│ -$45 │ Reversal│  [👁️] │       │
│  └──────┴────────┴──────┴──────┴─────────┴────────┘       │
└─────────────────────────────────────────────────────────────┘
```

#### **Detailed Entry View:**
```
┌───────────────────────────────────────────────────┐
│  Trade Details - AAPL Long (10/03/2025)     [✕]  │
├───────────────────────────────────────────────────┤
│  ┌─────────────────┬─────────────────┐           │
│  │ Entry           │ Exit            │           │
│  │ Price: $185.50  │ Price: $187.30  │           │
│  │ Qty: 100        │ Qty: 100        │           │
│  │ Time: 09:45     │ Time: 14:32     │           │
│  └─────────────────┴─────────────────┘           │
│                                                   │
│  P&L: +$120.00 (+0.97%)                          │
│  Duration: 4h 47m                                │
│                                                   │
│  Strategy: Breakout                              │
│  Setup: Bull flag                                │
│  Tags: #tech #high-confidence                    │
│                                                   │
│  [Entry Chart Screenshot]                        │
│  [Exit Chart Screenshot]                         │
│                                                   │
│  Notes:                                          │
│  Clean breakout above resistance. Good volume.   │
│                                                   │
│  Lessons Learned:                                │
│  - Waited for retest confirmation               │
│  - Proper stop placement                        │
│                                                   │
│  [Edit] [Delete] [Duplicate]                    │
└───────────────────────────────────────────────────┘
```

---

### 7. Alert Management 🔔

#### **Alert Types:**

1. **Price Alerts**
   - Above/Below price
   - Crosses level
   - Percentage change

2. **Indicator Alerts**
   - RSI overbought/oversold
   - MACD crossover
   - Bollinger Band breakout
   - Custom conditions

3. **Pattern Alerts**
   - Chart patterns (head & shoulders, triangles)
   - Candlestick patterns
   - Support/resistance breaks

4. **Volume Alerts**
   - Unusual volume
   - Volume breakout

5. **News Alerts**
   - Earnings announcements
   - SEC filings
   - News sentiment

6. **Portfolio Alerts**
   - Position profit/loss threshold
   - Margin call warning
   - Unusual account activity

#### **Alert UI:**
```
┌─────────────────────────────────────────────────────┐
│  Alerts                             [+ Create Alert]│
├─────────────────────────────────────────────────────┤
│  Active (5)  |  Triggered (12)  |  History         │
├─────────────────────────────────────────────────────┤
│  🔴 AAPL > $190                          [Edit] [×]│
│     Price: $187.45 (90% there)                     │
│     Created: 2 hours ago                           │
│                                                     │
│  🟢 MSFT RSI < 30                        [Edit] [×]│
│     RSI: 45.2                                      │
│     Created: 1 day ago                             │
│                                                     │
│  🟡 TSLA Volume > 2x avg                 [Edit] [×]│
│     Volume: 1.5x avg                               │
│     Created: 3 days ago                            │
└─────────────────────────────────────────────────────┘
```

#### **Create Alert Modal:**
```
┌───────────────────────────────────────┐
│  Create New Alert              [✕]    │
├───────────────────────────────────────┤
│  Symbol:  [AAPL______]                │
│                                       │
│  Type:    [Price ▼]                   │
│                                       │
│  Condition: [Above ▼]                 │
│  Value:     [190.00_______]           │
│                                       │
│  Notification:                        │
│  ☑ Push notification                 │
│  ☑ Email                              │
│  ☑ Sound                              │
│                                       │
│  Expires: [Never ▼]                   │
│                                       │
│  [Cancel]          [Create Alert]    │
└───────────────────────────────────────┘
```

#### **Implementation:**
```typescript
// components/alerts/AlertManager.tsx

interface Alert {
  id: string;
  symbol: string;
  type: AlertType;
  condition: AlertCondition;
  value: number;
  status: 'active' | 'triggered' | 'expired';
  notifications: {
    push: boolean;
    email: boolean;
    sound: boolean;
  };
  createdAt: Date;
  triggeredAt?: Date;
  expiresAt?: Date;
}

export function AlertManager() {
  const { alerts, createAlert, deleteAlert } = useAlerts();

  return (
    <div className="alert-manager">
      <AlertList alerts={alerts} onDelete={deleteAlert} />
      <CreateAlertButton onClick={openCreateModal} />
    </div>
  );
}
```

---

## State Management Strategy

### **Zustand Stores:**

```typescript
// lib/stores/user-store.ts
interface UserStore {
  user: User | null;
  settings: UserSettings;
  updateSettings: (settings: Partial<UserSettings>) => void;
}

// lib/stores/market-store.ts
interface MarketStore {
  prices: Record<string, Price>;
  subscriptions: string[];
  subscribe: (symbol: string) => void;
  unsubscribe: (symbol: string) => void;
}

// lib/stores/chart-store.ts
interface ChartStore {
  symbols: string[];
  timeframes: Record<string, Timeframe>;
  indicators: Record<string, Indicator[]>;
  layouts: ChartLayout[];
}

// lib/stores/journal-store.ts
interface JournalStore {
  entries: JournalEntry[];
  filters: JournalFilters;
  addEntry: (entry: JournalEntry) => void;
  updateEntry: (id: string, entry: Partial<JournalEntry>) => void;
}

// lib/stores/alert-store.ts
interface AlertStore {
  alerts: Alert[];
  createAlert: (alert: Alert) => void;
  deleteAlert: (id: string) => void;
  checkAlerts: () => void;
}
```

---

## API Integration Layer

### **API Client:**

```typescript
// lib/api/client.ts

class APIClient {
  private baseURL: string;
  private token?: string;

  // Services
  market = new MarketAPI(this);
  signals = new SignalAPI(this);
  journal = new JournalAPI(this);
  alerts = new AlertAPI(this);

  async get<T>(endpoint: string): Promise<T> { /* ... */ }
  async post<T>(endpoint: string, data: any): Promise<T> { /* ... */ }
}

// lib/api/market.ts
class MarketAPI {
  async getPrice(symbol: string): Promise<Price> {
    return this.client.get(`/api/v1/market/${symbol}/price`);
  }

  async getOHLCV(symbol: string, timeframe: string): Promise<OHLCV[]> {
    return this.client.get(`/api/v1/market/${symbol}/ohlcv?timeframe=${timeframe}`);
  }
}

// lib/api/signals.ts
class SignalAPI {
  async getTodaysPlan(): Promise<TradingPlan> {
    return this.client.get('/api/v1/plan');
  }

  async executeAction(action: TradeAction): Promise<ExecutionResult> {
    return this.client.post('/api/v1/actions/execute', action);
  }
}
```

---

## File Structure (Complete)

```
trading-frontend/apps/trading-web/
├── app/
│   ├── (auth)/
│   │   ├── login/page.tsx
│   │   └── signup/page.tsx
│   ├── (dashboard)/
│   │   ├── layout.tsx                    # Dashboard layout with sidebar
│   │   ├── beginner/
│   │   │   └── page.tsx                  # Beginner dashboard
│   │   ├── expert/
│   │   │   └── page.tsx                  # Expert dashboard
│   │   ├── journal/
│   │   │   ├── page.tsx                  # Journal list
│   │   │   └── [id]/page.tsx             # Journal entry detail
│   │   ├── alerts/
│   │   │   └── page.tsx                  # Alert management
│   │   └── settings/
│   │       └── page.tsx                  # Settings page
│   └── layout.tsx                        # Root layout
├── components/
│   ├── charts/
│   │   ├── TradingChart.tsx              # Main chart component
│   │   ├── ChartToolbar.tsx              # Chart controls
│   │   ├── ChartTypeSelector.tsx         # Candlestick, Line, etc.
│   │   ├── TimeframeSelector.tsx         # 1m, 5m, 1h, etc.
│   │   ├── IndicatorPanel.tsx            # Indicator list
│   │   ├── IndicatorLibrary.tsx          # Indicator selector modal
│   │   ├── DrawingTools.tsx              # Trend lines, etc.
│   │   └── VolumePane.tsx                # Volume chart
│   ├── dashboard/
│   │   ├── beginner/
│   │   │   ├── BeginnerDashboard.tsx
│   │   │   ├── Watchlist.tsx
│   │   │   ├── RecommendedTrades.tsx
│   │   │   ├── SimpleChart.tsx
│   │   │   └── QuickActions.tsx
│   │   └── expert/
│   │       ├── ExpertDashboard.tsx
│   │       ├── MultiChartLayout.tsx
│   │       ├── OrderBook.tsx
│   │       ├── PositionManager.tsx
│   │       ├── OptionsChain.tsx
│   │       └── Terminal.tsx
│   ├── journal/
│   │   ├── JournalList.tsx
│   │   ├── JournalEntry.tsx
│   │   ├── JournalFilters.tsx
│   │   ├── PerformanceMetrics.tsx
│   │   ├── EquityCurve.tsx
│   │   └── CalendarHeatmap.tsx
│   ├── alerts/
│   │   ├── AlertList.tsx
│   │   ├── AlertItem.tsx
│   │   ├── CreateAlertModal.tsx
│   │   └── AlertNotification.tsx
│   ├── settings/
│   │   ├── SettingsLayout.tsx
│   │   ├── GeneralSettings.tsx
│   │   ├── TradingSettings.tsx
│   │   ├── AppearanceSettings.tsx
│   │   ├── NotificationSettings.tsx
│   │   └── AdvancedSettings.tsx
│   ├── layout/
│   │   ├── Sidebar.tsx
│   │   ├── TopBar.tsx
│   │   └── MobileNav.tsx
│   └── ui/                               # Reusable components
│       ├── Button.tsx
│       ├── Card.tsx
│       ├── Modal.tsx
│       ├── Tabs.tsx
│       └── ...
├── lib/
│   ├── api/
│   │   ├── client.ts
│   │   ├── market.ts
│   │   ├── signals.ts
│   │   ├── journal.ts
│   │   └── alerts.ts
│   ├── hooks/
│   │   ├── useMarketData.ts
│   │   ├── useWebSocket.ts
│   │   ├── useSettings.ts
│   │   ├── useJournal.ts
│   │   └── useAlerts.ts
│   ├── stores/
│   │   ├── user-store.ts
│   │   ├── market-store.ts
│   │   ├── chart-store.ts
│   │   ├── journal-store.ts
│   │   └── alert-store.ts
│   ├── theme/
│   │   ├── theme-provider.tsx
│   │   ├── theme-config.ts
│   │   └── presets.ts
│   └── utils/
│       ├── indicators.ts                 # Indicator calculations
│       ├── formatters.ts
│       └── validators.ts
└── styles/
    └── globals.css
```

---

## Implementation Phases

### **Phase 1: Foundation** (Week 1)
- ✅ Set up theme system (dark/light mode)
- ✅ Create layout components (Sidebar, TopBar)
- ✅ Set up state management (Zustand stores)
- ✅ Create API client
- ✅ Set up routing structure

### **Phase 2: Charting** (Week 2)
- ✅ Integrate charting library (Lightweight Charts)
- ✅ Build TradingChart component
- ✅ Implement chart types (candlestick, line, area)
- ✅ Add timeframe selector
- ✅ Create indicator system
- ✅ Build indicator panel

### **Phase 3: Dashboards** (Week 2-3)
- ✅ Build beginner dashboard
- ✅ Build expert dashboard
- ✅ Implement watchlist
- ✅ Create trade recommendations UI
- ✅ Build multi-chart layout

### **Phase 4: Journal & Alerts** (Week 3-4)
- ✅ Build journal list and entry views
- ✅ Implement performance metrics
- ✅ Create alert management UI
- ✅ Implement alert notifications
- ✅ Add WebSocket for real-time alerts

### **Phase 5: Settings & Polish** (Week 4)
- ✅ Build settings page
- ✅ Implement theme customization
- ✅ Add user preferences
- ✅ Polish UI/UX
- ✅ Responsive design
- ✅ Performance optimization

### **Phase 6: Testing & Deployment** (Week 5-6)
- ✅ Unit tests
- ✅ Integration tests
- ✅ E2E tests (Playwright)
- ✅ Performance testing
- ✅ Accessibility audit
- ✅ Deploy to production

---

## Technology Stack

### **Core:**
- **Framework:** Next.js 15 (App Router)
- **UI Library:** React 19
- **Language:** TypeScript
- **Styling:** Tailwind CSS 4 + Material-UI
- **State Management:** Zustand
- **Data Fetching:** TanStack Query (React Query)

### **Charting:**
- **Primary:** Lightweight Charts (by TradingView)
- **Fallback:** Recharts (already installed)

### **UI Components:**
- **Radix UI** (headless components)
- **Material-UI** (pre-built components)
- **Lucide React** (icons)

### **Utilities:**
- **clsx** + **tailwind-merge** (className management)
- **date-fns** (date formatting)
- **zod** (validation)

---

## Performance Targets

- ✅ **Initial Load:** < 2 seconds
- ✅ **Chart Render:** < 100ms
- ✅ **Indicator Update:** < 50ms
- ✅ **Route Transition:** < 200ms
- ✅ **WebSocket Latency:** < 100ms
- ✅ **Lighthouse Score:** > 90

---

## Next Steps

1. Review and approve this plan
2. Set up project structure
3. Start with Phase 1 (Foundation)
4. Iterate and refine

**Ready to start implementation?** 🚀
