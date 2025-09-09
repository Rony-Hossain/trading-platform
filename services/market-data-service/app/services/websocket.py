import json
import logging
from fastapi import WebSocket
from typing import Dict, Set

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manage WebSocket connections for real-time data"""
    def __init__(self):
        self.active_connections: Dict[WebSocket, Set[str]] = {}
        self.symbol_subscribers: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, symbol: str):
        await websocket.accept()
        
        if websocket not in self.active_connections:
            self.active_connections[websocket] = set()
        
        self.active_connections[websocket].add(symbol)
        
        if symbol not in self.symbol_subscribers:
            self.symbol_subscribers[symbol] = set()
        
        self.symbol_subscribers[symbol].add(websocket)
        
        logger.info(f"Client connected to {symbol}. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            symbols = self.active_connections[websocket]
            for symbol in symbols:
                if symbol in self.symbol_subscribers:
                    self.symbol_subscribers[symbol].discard(websocket)
                    if not self.symbol_subscribers[symbol]:
                        del self.symbol_subscribers[symbol]
            
            del self.active_connections[websocket]
            logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast_to_symbol(self, symbol: str, message: Dict):
        if symbol in self.symbol_subscribers:
            disconnected_clients = []
            for websocket in self.symbol_subscribers[symbol]:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error sending to client: {e}")
                    disconnected_clients.append(websocket)
            
            # Clean up disconnected clients
            for websocket in disconnected_clients:
                self.disconnect(websocket)
    
    def stats(self) -> Dict:
        return {
            "total_connections": len(self.active_connections),
            "subscribed_symbols": list(self.symbol_subscribers.keys()),
            "subscribers_per_symbol": {
                symbol: len(clients) 
                for symbol, clients in self.symbol_subscribers.items()
            }
        }