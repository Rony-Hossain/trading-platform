Feature: Latest quote
  Validate current price endpoint across symbols, venues, and odd inputs.

  Background:
    Given the default symbol is "AAPL"

  @required
  Scenario: Get a normal quote
    When I GET "/api/v1/quote" with params:
      | symbol | AAPL |
    Then the response status should be 200
    And the response should match the "quote" schema
    And the field "symbol" should equal "AAPL"
    And the field "ts" should be within 30 seconds of now

  @required
  Scenario: Symbol normalization (lowercase -> uppercase)
    When I GET "/api/v1/quote" with params:
      | symbol | aapl |
    Then the response status should be 200
    And the field "symbol" should equal "AAPL"

  @required
  Scenario Outline: Edge symbol forms
    When I GET "/api/v1/quote" with params:
      | symbol | <sym> |
    Then the response status should be 200
    And the response should match the "quote" schema
    Examples:
      | sym     |
      | GOOG    |
      | GOOGL   |
      | BRK.B   |
      | RY.TO   |
      | DLR-U.TO |

  @required
  Scenario: Unknown symbol is rejected
    When I GET "/api/v1/quote" with params:
      | symbol | __NOPE__ |
    Then the response status should be 422

  @required
  Scenario: Timestamp freshness
    When I GET "/api/v1/quote" with params:
      | symbol | AAPL |
    Then the response status should be 200
    And the field "ts" should be within 120 seconds of now

  @optional
  Scenario: Unknown venue is rejected
    When I GET "/api/v1/quote" with params:
      | symbol | AAPL |
      | venue  | MADEUP |
    Then the response status should be 400

  @optional @rate
  Scenario: Client-side rate limit surfaces as 429
    Given I perform 200 GET requests to "/api/v1/quote" with params:
      | symbol | AAPL |
    Then at least one response status should be 429
