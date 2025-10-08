@optional
Feature: OpenAPI availability

  Scenario: OpenAPI is served
    When I GET "/openapi.json"
    Then the response status should be 200
    And the field "$.openapi" should start with "3."
