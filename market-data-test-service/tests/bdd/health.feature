@required
Feature: Service health
  As an operator
  I want a health signal
  So that I can verify the service and its deps are ready

  Scenario: Health is OK
    When I GET "/health"
    Then the response status should be 200
    And the response should match the "health" schema
    And the field "status" should equal "ok"

  Scenario: Health eventually becomes OK after startup
    Given I retry GET "/health" every 2s for up to 60s
    Then the response status should be 200
