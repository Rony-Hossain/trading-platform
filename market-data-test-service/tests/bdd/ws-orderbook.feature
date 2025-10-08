@optional
Feature: Realtime orderbook via WebSocket
  Validate snapshot+delta coherence and throttling.

  Scenario: Snapshot followed by deltas apply cleanly
    When I connect to "ws://localhost:8002/ws/orderbook?symbol=AAPL&depth=5"
    Then I should receive a "snapshot" event
    And then receive at least 3 "delta" events

  Scenario: Throttle/resolution parameter respected
    When I connect to "ws://localhost:8002/ws/orderbook?symbol=AAPL&depth=5&throttle_ms=200"
    Then the average message rate should be <= 5 msgs per second
