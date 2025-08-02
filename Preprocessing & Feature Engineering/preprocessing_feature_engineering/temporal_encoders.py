import pandas as pd
import numpy as np
from datetime import datetime
import pytz

class TemporalEncoders:
    def __init__(self):
        self.sessions = {
            'Tokyo': {'start': '00:00', 'end': '09:00', 'timezone': 'Asia/Tokyo'},
            'London': {'start': '08:00', 'end': '17:00', 'timezone': 'Europe/London'},
            'New York': {'start': '13:00', 'end': '22:00', 'timezone': 'America/New_York'}
        }
    
    def encode_time_of_day(self, timestamps: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame()
        hours = timestamps.dt.hour
        df['sin_hour'] = np.sin(2 * np.pi * hours / 24)
        df['cos_hour'] = np.cos(2 * np.pi * hours / 24)
        return df
    
    def track_session(self, timestamps: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame()
        for session, times in self.sessions.items():
            tz = pytz.timezone(times['timezone'])
            local_times = timestamps.dt.tz_convert(tz)
            start = pd.to_datetime(times['start']).time()
            end = pd.to_datetime(times['end']).time()
            df[f'session_{session}'] = ((local_times.dt.time >= start) & 
                                       (local_times.dt.time <= end)).astype(int)
        return df
    
    def tag_trend_context(self, data: pd.DataFrame, lookback: int = 20) -> pd.Series:
        sma = data['close'].rolling(window=lookback).mean()
        price_change = data['close'].pct_change(lookback)
        
        conditions = [
            (data['close'] > sma) & (price_change > 0.02),
            (data['close'] > sma) & (price_change <= 0.02) & (price_change > -0.02),
            (data['close'] <= sma) & (price_change >= -0.02) & (price_change < 0.02),
            (data['close'] <= sma) & (price_change < -0.02)
        ]
        choices = ['uptrend', 'range_up', 'range_down', 'downtrend']
        
        return pd.Series(np.select(conditions, choices, default='unknown'), index=data.index)
