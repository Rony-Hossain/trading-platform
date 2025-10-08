Feature: Order book snapshot
  Validate top-of-book integrity and edge cases.

  @required
  Scenario: Normal book has bids and asks
    When I GET "/api/v1/orderbook" with params:
      | symbol | AAPL |
      | depth  | 5    |
    Then the response status should be 200
    And the response should match the "orderbook" schema
    And the array "bids" length should be > 0
    And the array "asks" length should be > 0
    And the best bid price should be < the best ask price

  @required
  Scenario: Zero/negative depth is rejected
    When I GET "/api/v1/orderbook" with params:
      | symbol | AAPL |
      | depth  | 0    |
    Then the response status should be 422

  @required
  Scenario: Illiquid or halted symbol returns empty book
    When I GET "/api/v1/orderbook" with params:
      | symbol | __HALTED__ |
      | depth  | 10 |
    Then the response status should be 200
    And the array "bids" length should be 0
    And the array "asks" length should be 0
