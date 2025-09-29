"""Real-Time Endpoints for Event Streaming

FastAPI endpoints for WebSocket and Server-Sent Events connections
to provide real-time event streaming to clients.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass

from fastapi import WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import StreamingResponse
from starlette.endpoints import WebSocketEndpoint

from .event_streaming import EventStreamingService, EventType

logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """Information about a real-time connection."""
    id: str
    type: str  # "websocket" or "sse"
    connected_at: datetime
    client_ip: str
    user_agent: str
    filters: Optional[Dict[str, Any]] = None
    subscriptions: Optional[Set[str]] = None


class WebSocketManager:
    """Manages WebSocket connections for real-time event streaming."""
    
    def __init__(self, streaming_service: EventStreamingService):
        self.streaming_service = streaming_service
        self.connections: Dict[str, WebSocket] = {}
        self.connection_info: Dict[str, ConnectionInfo] = {}
    
    async def connect(self, websocket: WebSocket, connection_id: str, client_info: Dict[str, Any]):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        
        # Check connection limits
        if not await self.streaming_service.add_websocket_connection(websocket):
            await websocket.close(code=1013, reason="Connection limit reached")
            return False
        
        self.connections[connection_id] = websocket
        self.connection_info[connection_id] = ConnectionInfo(
            id=connection_id,
            type="websocket",
            connected_at=datetime.utcnow(),
            client_ip=client_info.get("client_ip", "unknown"),
            user_agent=client_info.get("user_agent", "unknown"),
            subscriptions=set()
        )
        
        logger.info(f"WebSocket connection established: {connection_id}")
        
        # Send welcome message
        await self.send_to_connection(connection_id, {
            "type": "connection.established",
            "connection_id": connection_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Connected to Event Data Service real-time stream"
        })
        
        return True
    
    def disconnect(self, connection_id: str):
        """Disconnect and clean up a WebSocket connection."""
        if connection_id in self.connections:
            websocket = self.connections[connection_id]
            self.streaming_service.remove_websocket_connection(websocket)
            del self.connections[connection_id]
            del self.connection_info[connection_id]
            
            logger.info(f"WebSocket connection disconnected: {connection_id}")
    
    async def send_to_connection(self, connection_id: str, data: Dict[str, Any]) -> bool:
        """Send data to a specific WebSocket connection."""
        if connection_id not in self.connections:
            return False
        
        try:
            websocket = self.connections[connection_id]
            await websocket.send_text(json.dumps(data, default=str))
            return True
        except Exception as e:
            logger.error(f"Failed to send to WebSocket {connection_id}: {e}")
            self.disconnect(connection_id)
            return False
    
    async def broadcast(self, data: Dict[str, Any], filter_func=None):
        """Broadcast data to all connected WebSocket clients."""
        disconnected = []
        
        for connection_id, websocket in self.connections.items():
            # Apply filter if provided
            if filter_func and not filter_func(self.connection_info[connection_id]):
                continue
            
            try:
                await websocket.send_text(json.dumps(data, default=str))
            except Exception as e:
                logger.error(f"Failed to broadcast to WebSocket {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    async def handle_client_message(self, connection_id: str, message: Dict[str, Any]):
        """Handle incoming message from WebSocket client."""
        try:
            message_type = message.get("type")
            
            if message_type == "subscribe":
                await self._handle_subscribe(connection_id, message)
            elif message_type == "unsubscribe":
                await self._handle_unsubscribe(connection_id, message)
            elif message_type == "ping":
                await self._handle_ping(connection_id)
            elif message_type == "get_status":
                await self._handle_get_status(connection_id)
            else:
                await self.send_to_connection(connection_id, {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
        
        except Exception as e:
            logger.error(f"Error handling client message from {connection_id}: {e}")
            await self.send_to_connection(connection_id, {
                "type": "error",
                "message": f"Failed to process message: {str(e)}"
            })
    
    async def _handle_subscribe(self, connection_id: str, message: Dict[str, Any]):
        """Handle subscription request."""
        topics = message.get("topics", [])
        filters = message.get("filters", {})
        
        if connection_id in self.connection_info:
            conn_info = self.connection_info[connection_id]
            conn_info.subscriptions.update(topics)
            conn_info.filters = filters
            
            await self.send_to_connection(connection_id, {
                "type": "subscribed",
                "topics": list(conn_info.subscriptions),
                "filters": filters,
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def _handle_unsubscribe(self, connection_id: str, message: Dict[str, Any]):
        """Handle unsubscription request."""
        topics = message.get("topics", [])
        
        if connection_id in self.connection_info:
            conn_info = self.connection_info[connection_id]
            conn_info.subscriptions -= set(topics)
            
            await self.send_to_connection(connection_id, {
                "type": "unsubscribed",
                "topics": topics,
                "remaining_subscriptions": list(conn_info.subscriptions),
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def _handle_ping(self, connection_id: str):
        """Handle ping request."""
        await self.send_to_connection(connection_id, {
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def _handle_get_status(self, connection_id: str):
        """Handle status request."""
        if connection_id in self.connection_info:
            conn_info = self.connection_info[connection_id]
            
            await self.send_to_connection(connection_id, {
                "type": "status",
                "connection_id": connection_id,
                "connected_at": conn_info.connected_at.isoformat(),
                "subscriptions": list(conn_info.subscriptions or []),
                "total_connections": len(self.connections),
                "timestamp": datetime.utcnow().isoformat()
            })
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        return {
            "total_connections": len(self.connections),
            "connections": [
                {
                    "id": conn_info.id,
                    "connected_at": conn_info.connected_at.isoformat(),
                    "client_ip": conn_info.client_ip,
                    "subscriptions": list(conn_info.subscriptions or []),
                    "has_filters": bool(conn_info.filters)
                }
                for conn_info in self.connection_info.values()
            ]
        }


class SSEManager:
    """Manages Server-Sent Events connections for real-time event streaming."""
    
    def __init__(self, streaming_service: EventStreamingService):
        self.streaming_service = streaming_service
        self.connections: Dict[str, Any] = {}
        self.connection_info: Dict[str, ConnectionInfo] = {}
    
    async def create_sse_response(
        self, 
        request: Request, 
        connection_id: str,
        topics: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> StreamingResponse:
        """Create a Server-Sent Events response."""
        
        # Check connection limits
        if len(self.connections) >= self.streaming_service.sse_max_connections:
            raise Exception("SSE connection limit reached")
        
        # Store connection info
        self.connection_info[connection_id] = ConnectionInfo(
            id=connection_id,
            type="sse",
            connected_at=datetime.utcnow(),
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown"),
            filters=filters,
            subscriptions=set(topics or [])
        )
        
        return StreamingResponse(
            self._sse_generator(connection_id, topics, filters),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
    
    async def _sse_generator(
        self, 
        connection_id: str, 
        topics: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ):
        """Generate Server-Sent Events stream."""
        try:
            # Send initial connection message
            yield f"id: {connection_id}\n"
            yield f"event: connection\n"
            yield f"data: {json.dumps({'type': 'connection.established', 'connection_id': connection_id, 'timestamp': datetime.utcnow().isoformat()})}\n\n"
            
            # Create a queue for this connection
            message_queue = asyncio.Queue()
            self.connections[connection_id] = message_queue
            
            # Add to streaming service
            await self.streaming_service.add_sse_connection(self)
            
            logger.info(f"SSE connection established: {connection_id}")
            
            # Send heartbeat and process messages
            last_heartbeat = datetime.utcnow()
            
            while True:
                try:
                    # Send heartbeat every 30 seconds
                    now = datetime.utcnow()
                    if (now - last_heartbeat).total_seconds() > 30:
                        yield f"event: heartbeat\n"
                        yield f"data: {json.dumps({'timestamp': now.isoformat()})}\n\n"
                        last_heartbeat = now
                    
                    # Check for new messages (with timeout)
                    try:
                        message = await asyncio.wait_for(message_queue.get(), timeout=1.0)
                        
                        # Apply filters if specified
                        if self._message_matches_filters(message, topics, filters):
                            event_type = message.get("type", "event")
                            event_id = message.get("id", "")
                            
                            yield f"id: {event_id}\n"
                            yield f"event: {event_type}\n"
                            yield f"data: {json.dumps(message, default=str)}\n\n"
                    
                    except asyncio.TimeoutError:
                        # No message received, continue loop for heartbeat
                        continue
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"SSE generator error for {connection_id}: {e}")
                    break
        
        finally:
            # Clean up connection
            self.disconnect(connection_id)
    
    def disconnect(self, connection_id: str):
        """Disconnect and clean up an SSE connection."""
        if connection_id in self.connections:
            del self.connections[connection_id]
        
        if connection_id in self.connection_info:
            del self.connection_info[connection_id]
        
        self.streaming_service.remove_sse_connection(self)
        logger.info(f"SSE connection disconnected: {connection_id}")
    
    async def send_to_connection(self, connection_id: str, data: Dict[str, Any]) -> bool:
        """Send data to a specific SSE connection."""
        if connection_id not in self.connections:
            return False
        
        try:
            queue = self.connections[connection_id]
            await queue.put(data)
            return True
        except Exception as e:
            logger.error(f"Failed to send to SSE {connection_id}: {e}")
            self.disconnect(connection_id)
            return False
    
    async def broadcast(self, data: Dict[str, Any]):
        """Broadcast data to all connected SSE clients."""
        disconnected = []
        
        for connection_id, queue in self.connections.items():
            try:
                await queue.put(data)
            except Exception as e:
                logger.error(f"Failed to broadcast to SSE {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    def _message_matches_filters(
        self, 
        message: Dict[str, Any], 
        topics: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if message matches SSE connection filters."""
        
        # Check topic filter
        if topics:
            message_type = message.get("type", "")
            if not any(topic in message_type for topic in topics):
                return False
        
        # Check additional filters
        if filters:
            # Symbol filter
            if "symbols" in filters:
                message_data = message.get("data", {})
                symbol = message_data.get("symbol", "")
                if symbol not in filters["symbols"]:
                    return False
            
            # Priority filter
            if "min_priority" in filters:
                priority = message.get("priority", 1)
                if priority > filters["min_priority"]:
                    return False
            
            # Source filter
            if "sources" in filters:
                source = message.get("source", "")
                if source not in filters["sources"]:
                    return False
        
        return True
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get SSE connection statistics."""
        return {
            "total_connections": len(self.connections),
            "connections": [
                {
                    "id": conn_info.id,
                    "connected_at": conn_info.connected_at.isoformat(),
                    "client_ip": conn_info.client_ip,
                    "subscriptions": list(conn_info.subscriptions or []),
                    "has_filters": bool(conn_info.filters)
                }
                for conn_info in self.connection_info.values()
            ]
        }


def build_websocket_manager(streaming_service: EventStreamingService) -> WebSocketManager:
    """Factory function to create WebSocket manager."""
    return WebSocketManager(streaming_service)


def build_sse_manager(streaming_service: EventStreamingService) -> SSEManager:
    """Factory function to create SSE manager."""
    return SSEManager(streaming_service)