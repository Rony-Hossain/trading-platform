Feature: Historical OHLCV
  Validate intervals, ranges, adjustments, and tricky calendars.

  @required
  Scenario: Daily candles for a short range
    When I GET "/api/v1/ohlcv" with params:
      | symbol  | AAPL   |
      | interval| 1d     |
      | start   | 2025-09-15 |
      | end     | 2025-09-22 |
      | adjust  | split  |
    Then the response status should be 200
    And the response should match the "ohlcv" schema
    And the array "candles" length should be >= 3
    And every item in "candles" should have fields: open, high, low, close, volume, ts

  @required
  Scenario: Intraday returns empty on weekend
    When I GET "/api/v1/ohlcv" with params:
      | symbol  | AAPL  |
      | interval| 1m    |
      | start   | 2025-09-20T10:00:00Z |
      | end     | 2025-09-20T16:00:00Z |
    Then the response status should be 200
    And the array "candles" length should be 0

  @required
  Scenario: Bad interval is rejected
    When I GET "/api/v1/ohlcv" with params:
      | symbol  | AAPL  |
      | interval| 7m    |
      | start   | 2025-09-22T13:30:00Z |
      | end     | 2025-09-22T14:00:00Z |
    Then the response status should be 400

  @required
  Scenario: Start after end is rejected
    When I GET "/api/v1/ohlcv" with params:
      | symbol  | AAPL  |
      | interval| 1d    |
      | start   | 2025-10-10 |
      | end     | 2025-09-10 |
    Then the response status should be 422

  @optional
  Scenario: Large range is truncated by max limit
    When I GET "/api/v1/ohlcv" with params:
      | symbol  | AAPL  |
      | interval| 1m    |
      | start   | 2025-09-01T13:30:00Z |
      | end     | 2025-09-05T20:00:00Z |
      | limit   | 10000 |
    Then the response status should be 200
    And the array "candles" length should be <= 10000
