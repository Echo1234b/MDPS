from typing import Dict, Any, List
from datetime import datetime
from ..data.database import Database
from ..data.cache import Cache
from ..data.models.market_model import MarketData
from ..data.models.trade_model import Trade

class DataService:
    def __init__(self, db_path: str, cache_host='localhost', cache_port=6379):
        self.db = Database(db_path)
        self.cache = Cache(cache_host, cache_port)

    def save_market_data(self, data: MarketData):
        """Save market data to database and cache"""
        self.db.insert_market_data(data.to_dict())
        self.cache.set(f"market_data:{data.symbol}", data.to_dict(), expiration=60)

    def get_market_data(self, symbol: str) -> MarketData:
        """Get market data from cache or database"""
        cached_data = self.cache.get(f"market_data:{symbol}")
        if cached_data:
            return MarketData.from_dict(cached_data)
        
        # TODO: Implement database query if not in cache
        return None

    def save_trade(self, trade: Trade):
        """Save trade to database and cache"""
        self.db.insert_trade(trade.to_dict())
        self.cache.set(f"trade:{trade.symbol}:{trade.timestamp}", trade.to_dict())

    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Get recent trades for a symbol"""
        # TODO: Implement database query
        return []
