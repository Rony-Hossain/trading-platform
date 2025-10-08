Feature: Realtime quotes via WebSocket
  Ensure subscribe, keepalive, and error handling work.

  Background:
    Given a WebSocket URL "ws://localhost:8002/ws/quotes"

  @required
  Scenario: Subscribe to one symbol and receive updates
    When I connect to the WebSocket
    And I send a subscribe message for:
      | symbol |
      | AAPL   |
    Then I should receive a message within 10 seconds matching "quote"
    And the field "symbol" in the last message should equal "AAPL"

  @required
  Scenario: Subscribe to multiple symbols
    When I connect to the WebSocket
    And I send a subscribe message for:
      | symbol |
      | AAPL   |
      | TSLA   |
    Then I should receive messages for all of:
      | symbol |
      | AAPL   |
      | TSLA   |

  @required
  Scenario: Unknown symbol yields an error event
    When I connect to the WebSocket
    And I send a subscribe message for:
      | symbol |
      | __NOPE__ |
    Then I should receive an error event with code "INVALID_SYMBOL"

  @required
  Scenario: Ping/pong keeps the connection alive
    When I connect to the WebSocket
    And I wait for 30 seconds
    Then the connection should still be open

  @required
  Scenario: Server closes; client can reconnect and resubscribe
    When I connect to the WebSocket
    And I send a subscribe message for:
      | symbol |
      | AAPL   |
    And the server closes the connection
    Then the client reconnects within 5 seconds
    And I should receive a message within 10 seconds matching "quote"

  @optional
  Scenario: Unsubscribe stops messages
    When I connect to the WebSocket
    And I subscribe to "AAPL"
    And I unsubscribe from "AAPL"
    Then I should not receive any "AAPL" messages for 5 seconds
