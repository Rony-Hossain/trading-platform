@profile
Feature: Latest price
  Scenario: Price endpoint returns a number for INTC
    Given the Market Data API base URL "http://localhost:8002"
    When I GET "/stocks/INTC/price"
    Then the response status is 200
    And the json has a numeric field "price"
