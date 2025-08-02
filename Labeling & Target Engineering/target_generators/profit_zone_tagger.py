import pandas as pd
import numpy as np

class ProfitZoneTagger:
    def __init__(self, price_data, breakout_threshold=0.02):
        self.price_data = price_data
        self.breakout_threshold = breakout_threshold

    def tag_profit_zones(self, lookforward_periods=5):
        profit_zones = []
        for i in range(len(self.price_data) - lookforward_periods):
            future_prices = self.price_data['close'].iloc[i:i+lookforward_periods]
            current_price = self.price_data['close'].iloc[i]
            
            max_profit = (future_prices.max() - current_price) / current_price
            min_profit = (future_prices.min() - current_price) / current_price
            
            if max_profit > self.breakout_threshold:
                profit_zones.append('High Profit Zone')
            elif abs(min_profit) > self.breakout_threshold:
                profit_zones.append('High Loss Zone')
            else:
                profit_zones.append('Neutral Zone')
        
        return pd.Series(profit_zones, index=self.price_data.index[:-lookforward_periods])

    def generate_labels(self):
        zones = self.tag_profit_zones()
        label_map = {'High Profit Zone': 2, 'Neutral Zone': 1, 'High Loss Zone': 0}
        return zones.map(label_map)
