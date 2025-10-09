# VizTracer Performance Profiling Guide

Complete guide to using VizTracer for performance analysis in the Market Data Service.

---

## 📊 What is VizTracer?

**VizTracer** is a low-overhead Python profiler that generates interactive **flame graphs** to visualize code execution. It helps you:

- Identify performance bottlenecks
- Understand function call hierarchy
- Measure execution time
- Find slow database queries
- Optimize hot paths

**Output:** Interactive HTML flame graphs viewable in any browser

---

## 🚀 Quick Start

### 1. Install VizTracer

```bash
pip install viztracer
```

### 2. Enable in Configuration

Edit `.env`:

```bash
VIZTRACER_ENABLED=true
VIZ_OUT_DIR=./traces
```

### 3. Start Service

```bash
uvicorn app.main:app --reload
```

### 4. Generate Traces

Make API requests:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/stats/cadence
```

### 5. View Traces

```bash
# Traces are saved as HTML files
ls traces/

# Open in browser (Windows)
start traces/trace_fetch_bars_1728400000000.html

# Or Mac
open traces/trace_fetch_bars_1728400000000.html

# Or Linux
xdg-open traces/trace_fetch_bars_1728400000000.html
```

---

## 🎯 How to Use VizTracer in Code

### Automatic Tracing (Built-in)

VizTracer is already integrated into key operations via `maybe_trace()`:

```python
# File: app/observability/trace.py
@contextmanager
def maybe_trace(label: str):
    """Context manager for optional tracing."""
    if os.getenv("VIZTRACER_ENABLED", "false").lower() in ("1", "true", "yes"):
        # Capture trace
        ...
    else:
        # Zero overhead when disabled
        yield
```

### Add Tracing to Your Code

```python
from app.observability.trace import maybe_trace

async def fetch_bars(self, provider: str, symbols: list):
    """Fetch bars with performance tracing."""

    # Trace this operation
    with maybe_trace(f"fetch_bars_{provider}"):
        # This entire block will be profiled
        result = await self.provider.get_bars(symbols)
        await self.db.write_bars(result)
        return result
```

### Nested Traces

```python
async def process_market_data():
    """Process market data with nested tracing."""

    with maybe_trace("process_market_data"):
        # Outer trace

        with maybe_trace("fetch_quotes"):
            quotes = await fetch_quotes()

        with maybe_trace("calculate_indicators"):
            indicators = calculate_indicators(quotes)

        with maybe_trace("write_to_db"):
            await db.write(indicators)
```

---

## 📁 Trace File Structure

### File Naming

Traces are saved with this format:

```
traces/trace_{label}_{timestamp}.html
```

**Examples:**
- `trace_fetch_bars_polygon_1728400123456.html`
- `trace_process_backfill_1728400789012.html`
- `trace_gap_detection_1728401234567.html`

### Trace Contents

Each HTML file contains:
- **Timeline view** - Chronological execution
- **Flame graph** - Call hierarchy
- **Statistics** - Function call counts, total time
- **Search** - Find specific functions
- **Zoom** - Drill down into specific time ranges

---

## 🔍 Reading Flame Graphs

### Understanding the Visualization

```
┌────────────────────────────────────────┐
│ fetch_bars (1000ms total)              │  ← Top-level function
├────────────────────────────────────────┤
│ ├─ _make_request (800ms)               │  ← Most time spent here
│ │  ├─ httpx.post (700ms)               │  ← HTTP request
│ │  └─ json.loads (100ms)               │  ← JSON parsing
│ ├─ _parse_response (150ms)             │  ← Response parsing
│ └─ _write_to_db (50ms)                 │  ← Database write
└────────────────────────────────────────┘
```

### Color Coding

- **Wide bars** = Functions that take a lot of time
- **Many children** = Functions that call many other functions
- **Red/Orange** = Hot paths (slow)
- **Green/Blue** = Fast operations

### What to Look For

1. **Wide bars at the top** - Main bottlenecks
2. **Unexpected depth** - Too many nested calls
3. **Repeated patterns** - Inefficient loops
4. **Database icons** - Slow queries
5. **Network icons** - Slow API calls

---

## 🛠️ Common Use Cases

### 1. Profile Provider API Calls

**Scenario:** Check if Polygon API is slow

```python
# In app/providers/polygon_provider.py
async def fetch_bars(self, symbols: list):
    with maybe_trace("polygon_fetch_bars"):
        response = await self._make_request("/v2/aggs/ticker", symbols)
        return self._parse_response(response)
```

**View trace:**
- Open `traces/trace_polygon_fetch_bars_*.html`
- Look for `_make_request` - should be <500ms
- Check `_parse_response` - should be <100ms

**Fix slow operations:**
- If `_make_request` is slow: Check network latency
- If `_parse_response` is slow: Optimize JSON parsing

---

### 2. Profile Database Writes

**Scenario:** Bulk inserts are slow

```python
# In app/services/database.py
async def upsert_bars_bulk(self, bars: list):
    with maybe_trace("db_upsert_bars_bulk"):
        async with self.pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO candles_intraday ...",
                bars
            )
```

**View trace:**
- Open trace and find `db_upsert_bars_bulk`
- Check `executemany` time
- Expected: <50ms for 1000 bars

**Optimizations:**
- Batch size too large? Reduce to 500
- Missing indexes? Add indexes on (symbol, ts)
- Use `COPY` instead of `INSERT` for large batches

---

### 3. Profile Gap Detection

**Scenario:** Gap detection is slow

```python
# In app/services/data_collector.py
async def _write_with_gaps(self, symbol, interval, provider, bars, tier):
    with maybe_trace(f"gap_detection_{symbol}_{interval}"):
        # Detect gaps
        last_ts = await self.db.get_cursor(symbol, interval, provider)
        gaps = self._find_gaps(bars, last_ts)

        # Enqueue gaps
        for gap in gaps:
            await self._enqueue_gap(gap)
```

**View trace:**
- Find `gap_detection_*` operations
- Check `get_cursor` time
- Check `_find_gaps` time
- Check `_enqueue_gap` time

**Optimizations:**
- Cache cursors to avoid repeated DB queries
- Batch gap enqueuing
- Use calendar preloading (already implemented)

---

### 4. Profile WebSocket Publishing

**Scenario:** WebSocket publish latency is high

```python
# In app/services/websocket.py
async def broadcast(self, message: dict):
    with maybe_trace("ws_broadcast"):
        for client in self.clients:
            with maybe_trace("ws_send_single"):
                await client.send_json(message)
```

**View trace:**
- Find `ws_broadcast` operations
- Count `ws_send_single` children
- Check individual send times

**Expected:**
- Total broadcast time: <250ms (p99)
- Individual send: <50ms

**Optimizations:**
- Concurrent sends instead of sequential
- Remove slow clients from broadcast
- Batch messages before broadcasting

---

## 📈 Performance Benchmarks

### Expected Trace Times

Based on SLO targets:

| Operation | Target (p95) | Target (p99) |
|-----------|--------------|--------------|
| Provider API call | 400ms | 800ms |
| DB write (1k bars) | 50ms | 100ms |
| Gap detection | 20ms | 50ms |
| WebSocket publish | 150ms | 250ms |
| Cursor lookup | 5ms | 10ms |
| Calendar align | 0.05ms | 0.1ms |

### How to Measure

1. **Generate traces** - Run operations multiple times
2. **Open traces** - View in browser
3. **Check statistics** - Use VizTracer's stats view
4. **Compare to targets** - Identify operations exceeding targets

---

## 🔧 Advanced Features

### 1. Filtering Traces

Show only specific functions:

```python
# Only trace database operations
with maybe_trace("critical_operation"):
    # This will be traced
    await slow_db_query()
```

### 2. Adding Metadata

Add context to traces:

```python
with maybe_trace(f"fetch_bars_{provider}_{len(symbols)}_symbols"):
    # Trace includes provider name and symbol count
    ...
```

### 3. Programmatic Analysis

Parse trace JSON for automation:

```python
import json

# VizTracer can output JSON instead of HTML
# Set: tracer = VizTracer(output_file="trace.json")

with open("trace.json") "r") as f:
    trace_data = json.load(f)

# Analyze trace data
for event in trace_data["traceEvents"]:
    if event["dur"] > 1000000:  # > 1 second
        print(f"Slow operation: {event['name']}")
```

---

## 🚨 Production Considerations

### Overhead

VizTracer adds **~5-10% overhead** when enabled:

- Acceptable in development/staging
- **NOT recommended in production**
- Use selective tracing in production

### Selective Tracing (Production Mode)

Enable only for specific operations:

```python
# Only trace if explicitly enabled per-operation
TRACE_SPECIFIC = os.getenv("TRACE_SPECIFIC", "").split(",")

def should_trace(operation: str) -> bool:
    return operation in TRACE_SPECIFIC

async def fetch_bars(self, provider: str):
    if should_trace(f"fetch_{provider}"):
        with maybe_trace(f"fetch_bars_{provider}"):
            ...
    else:
        # No tracing overhead
        ...
```

### Disk Usage

Traces can be large:

- **~10MB per 30-second trace**
- **~100MB per 5-minute trace**

**Solutions:**
- Set retention policy (delete traces older than 7 days)
- Compress traces: `gzip traces/*.html`
- Store in S3/GCS with lifecycle policies

### Trace Rotation

Auto-delete old traces:

```python
# In app/observability/trace.py
import time
from pathlib import Path

def cleanup_old_traces(max_age_hours: int = 24):
    """Delete traces older than max_age_hours."""
    traces_dir = Path(os.getenv("VIZ_OUT_DIR", "/tmp/traces"))
    now = time.time()

    for trace_file in traces_dir.glob("trace_*.html"):
        age_hours = (now - trace_file.stat().st_mtime) / 3600
        if age_hours > max_age_hours:
            trace_file.unlink()
```

---

## 🎓 Example Analysis Session

### Scenario: Investigate Slow Backfill

1. **Enable tracing:**
   ```bash
   export VIZTRACER_ENABLED=true
   export VIZ_OUT_DIR=./traces
   ```

2. **Trigger backfill:**
   ```bash
   curl -X POST "http://localhost:8000/ops/backfill" \
     -d '{"symbol":"AAPL","interval":"1m","start":"2025-10-08T13:00:00Z","end":"2025-10-08T14:00:00Z"}'
   ```

3. **Find trace:**
   ```bash
   ls -lht traces/ | head -1
   # trace_backfill_worker_1728400000000.html
   ```

4. **Open in browser:**
   ```bash
   start traces/trace_backfill_worker_1728400000000.html
   ```

5. **Analyze:**
   - **Wide bars:** `polygon.fetch_bars` taking 2000ms (too slow!)
   - **Many calls:** 60 API calls (one per minute - should be batched)
   - **Solution:** Batch fetch 60 minutes at once

6. **Fix:**
   ```python
   # Before: One API call per minute
   for minute in time_range:
       bars = await provider.fetch_bars([symbol], minute, minute + 1m)

   # After: One API call for entire range
   bars = await provider.fetch_bars([symbol], start, end)  # All 60 minutes
   ```

7. **Re-test:**
   - New trace: `polygon.fetch_bars` now 300ms
   - **60x fewer API calls**
   - **Backfill 6x faster**

---

## 📚 Resources

### VizTracer Documentation

- **Official Docs:** https://viztracer.readthedocs.io/
- **GitHub:** https://github.com/gaogaotiantian/viztracer
- **Examples:** https://viztracer.readthedocs.io/en/stable/basic_usage.html

### Chrome DevTools

VizTracer traces can be viewed in Chrome DevTools:

1. Open Chrome DevTools (F12)
2. Go to "Performance" tab
3. Click "Load profile"
4. Select trace JSON file

---

## 🔍 Troubleshooting

### Issue: Traces not being generated

**Check:**
```bash
# Verify environment variable
echo $VIZTRACER_ENABLED  # Should be "true"

# Check traces directory exists
ls -la traces/

# Check service logs
tail -f logs/market-data-service.log | grep viztracer
```

**Fix:**
```bash
export VIZTRACER_ENABLED=true
mkdir -p traces
```

### Issue: Traces are empty

**Cause:** No operations executed during trace

**Fix:** Make API requests to trigger operations:
```bash
curl http://localhost:8000/stats/cadence
curl http://localhost:8000/providers/polygon/health-history
```

### Issue: Traces are too large

**Cause:** Long-running operations

**Fix:** Trace specific operations instead of entire requests:
```python
# Don't trace entire request handler
# @app.get("/stats/cadence")
# async def get_cadence():
#     with maybe_trace("get_cadence"):  # Too broad
#         ...

# Instead, trace specific operations
@app.get("/stats/cadence")
async def get_cadence():
    # Only trace slow operation
    with maybe_trace("db_query_cadence"):
        stats = await db.get_cadence_stats()
    return stats
```

---

## ✅ Quick Reference

### Enable/Disable

```bash
# Enable
export VIZTRACER_ENABLED=true
export VIZ_OUT_DIR=./traces

# Disable
export VIZTRACER_ENABLED=false
```

### Add Trace

```python
from app.observability.trace import maybe_trace

with maybe_trace("operation_name"):
    # code to profile
    ...
```

### View Trace

```bash
# List traces
ls -lht traces/

# Open most recent
start traces/$(ls -t traces/ | head -1)
```

### Clean Up

```bash
# Delete traces older than 1 day
find traces/ -name "*.html" -mtime +1 -delete

# Delete all traces
rm -rf traces/*.html
```

---

## 🎯 Summary

VizTracer is a powerful tool for performance optimization:

- **Zero overhead when disabled** - Safe to leave code in production
- **Selective tracing** - Profile specific operations
- **Interactive visualization** - Easy to understand flame graphs
- **Low learning curve** - View in any browser
- **Actionable insights** - Identify exact bottlenecks

**Use VizTracer to:**
- Find slow provider API calls
- Optimize database queries
- Reduce WebSocket latency
- Speed up backfill operations
- Meet SLO targets (p99 < 250ms for WS publish)

**Next steps:** Run `python deploy_local.py` and follow the VizTracer setup!
