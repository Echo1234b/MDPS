from typing import Dict, List
import pandas as pd

class MarketData:
    def __init__(self):
        self.symbols: Dict[str, pd.DataFrame] = {}
        self.order_books: Dict[str, Dict] = {}

    def update_symbol_data(self, symbol: str, data: pd.DataFrame):
        self.symbols[symbol] = data

    def update_order_book(self, symbol: str, order_book: Dict):
        self.order_books[symbol] = order_book

    def get_symbol_data(self, symbol: str) -> pd.DataFrame:
        return self.symbols.get(symbol, pd.DataFrame())

    def get_order_book(self, symbol: str) -> Dict:
        return self.order_books.get(symbol, {})
