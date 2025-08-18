import pandas as pd
import numpy as np

class RiskRewardLabeler:
    def __init__(self, price_data, atr_period=14, risk_multiple=1.0, reward_multiple=2.0):
        self.price_data = price_data
        self.atr_period = atr_period
        self.risk_multiple = risk_multiple
        self.reward_multiple = reward_multiple

    def calculate_rr_labels(self):
        # Calculate ATR for stop loss/target levels
        high = self.price_data['high']
        low = self.price_data['low']
        close = self.price_data['close']
        
        atr = self._calculate_atr()
        
        labels = []
        for i in range(len(self.price_data)):
            current_price = close.iloc[i]
            stop_loss = current_price - (atr.iloc[i] * self.risk_multiple)
            take_profit = current_price + (atr.iloc[i] * self.reward_multiple)
            
            # Look forward to see if price hits targets
            future_prices = close.iloc[i+1:] if i < len(close)-1 else []
            hit_take_profit = False
            hit_stop_loss = False
            
            for price in future_prices:
                if price >= take_profit:
                    hit_take_profit = True
                    break
                elif price <= stop_loss:
                    hit_stop_loss = True
                    break
            
            if hit_take_profit:
                labels.append('High-RR')
            elif hit_stop_loss:
                labels.append('Low-RR')
            else:
                labels.append('Negative-RR')
        
        return pd.Series(labels, index=self.price_data.index)

    def _calculate_atr(self):
        high = self.price_data['high']
        low = self.price_data['low']
        close = self.price_data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        return tr.rolling(window=self.atr_period).mean()
