@profile
Feature: Live ticks
  Scenario: First tick arrives quickly
    Given the WS URL "ws://localhost:8002/ws/AAPL"
    When I connect and read one message
    Then the json includes "price"
