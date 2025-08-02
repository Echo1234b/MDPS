from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Trade:
    symbol: str
    timestamp: datetime
    side: str
    price: float
    quantity: float
    pnl: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'side': self.side,
            'price': self.price,
            'quantity': self.quantity,
            'pnl': self.pnl
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create instance from dictionary"""
        return cls(
            symbol=data['symbol'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            side=data['side'],
            price=float(data['price']),
            quantity=float(data['quantity']),
            pnl=float(data['pnl']) if data.get('pnl') else None
        )
