@optional
Feature: Symbol search
  Quick lookup for tickers by name/partial code.

  Scenario: Basic search returns results
    When I GET "/api/v1/symbols/search" with params:
      | q | apple |
    Then the response status should be 200
    And the response should match the "search_result" schema
    And the array "symbols" length should be >= 1

  Scenario: Query too short is rejected
    When I GET "/api/v1/symbols/search" with params:
      | q | ap |
    Then the response status should be 422

  Scenario: Pagination works
    When I GET "/api/v1/symbols/search" with params:
      | q     | bank |
      | limit | 20   |
      | page  | 2    |
    Then the response status should be 200
    And the array "symbols" length should be <= 20
