@profile
Feature: Options chain
  Scenario: AAPL has options
    Given the Market Data API base URL "http://localhost:8002"
    When I GET "/options/AAPL/chain"
    Then the response status is 200
    And the json has a non-empty array "expiries"
