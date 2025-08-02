import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from gym import spaces
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TradingEnvironment(gym.Env):
    """
    A custom trading environment for reinforcement learning agents.
    """
    metadata = {'render.modes': ['human', 'system', 'none']}
    
    def __init__(self, df, initial_balance=10000, commission=0.001, 
                 window_size=10, reward_function='sharpe', max_steps=1000,
                 add_indicators=True, render_freq=None):
        """
        Initialize the trading environment.
        
        Args:
            df: DataFrame with OHLCV data
            initial_balance: Initial account balance
            commission: Commission rate for trades
            window_size: Size of the observation window
            reward_function: Reward function to use ('sharpe', 'profit', 'sortino', 'calmar')
            max_steps: Maximum steps per episode
            add_indicators: Whether to add technical indicators
            render_freq: Frequency of rendering (None for no rendering)
        """
        super(TradingEnvironment, self).__init__()
        
        # Store parameters
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        self.reward_function = reward_function
        self.max_steps = max_steps
        self.add_indicators = add_indicators
        self.render_freq = render_freq
        
        # Add technical indicators if requested
        if self.add_indicators:
            self._add_technical_indicators()
        
        # Define action and observation space
        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observations: window_size * number of features
        # Features include OHLCV, indicators, balance, and shares held
        n_features = len(self.df.columns) + 2  # +2 for balance and shares held
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(window_size, n_features), 
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def _add_technical_indicators(self):
        """Add technical indicators to the dataframe."""
        df = self.df
        
        # Simple Moving Average (SMA)
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        
        # Exponential Moving Average (EMA)
        df['EMA_10'] = df['Close'].ewm(span=10).mean()
        df['EMA_30'] = df['Close'].ewm(span=30).mean()
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        df['Upper_BB'] = sma_20 + (std_20 * 2)
        df['Lower_BB'] = sma_20 - (std_20 * 2)
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['%D'] = df['%K'].rolling(window=3).mean()
        
        # Commodity Channel Index (CCI)
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.fabs(x - x.mean()).mean())
        df['CCI'] = (tp - sma_tp) / (0.015 * mad)
        
        # Average True Range (ATR)
        df['TR1'] = df['High'] - df['Low']
        df['TR2'] = abs(df['High'] - df['Close'].shift(1))
        df['TR3'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Drop NaN values
        self.df = df.dropna()
    
    def reset(self):
        """Reset the environment to initial state."""
        # Reset account
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        
        # Reset data pointer
        self.current_step = self.window_size
        
        # Reset history
        self.balance_history = [self.balance]
        self.net_worth_history = [self.net_worth]
        self.shares_held_history = [self.shares_held]
        self.action_history = []
        self.reward_history = []
        
        # Get initial observation
        observation = self._next_observation()
        
        # Reset render data
        self.render_data = {
            'dates': self.df.index[self.current_step-self.window_size:self.current_step].tolist(),
            'prices': self.df['Close'].iloc[self.current_step-self.window_size:self.current_step].tolist(),
            'actions': [],
            'portfolio_values': [self.initial_balance]
        }
        
        return observation
    
    def _next_observation(self):
        """Get the next observation."""
        # Get the data window
        frame = self.df.iloc[self.current_step - self.window_size:self.current_step].copy()
        
        # Normalize data
        for col in frame.columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                # Normalize indicators to [0, 1]
                min_val = frame[col].min()
                max_val = frame[col].max()
                if max_val > min_val:
                    frame[col] = (frame[col] - min_val) / (max_val - min_val)
            else:
                # Normalize OHLCV to [0, 1]
                if col == 'Volume':
                    frame[col] = frame[col] / frame[col].max()
                else:
                    frame[col] = frame[col] / frame[col].max()
        
        # Add account information
        frame['Balance'] = self.balance / self.initial_balance
        frame['Shares_Held'] = self.shares_held / (self.initial_balance / frame['Close'].iloc[-1])
        
        return frame.values.astype(np.float32)
    
    def step(self, action):
        """Take a step in the environment."""
        # Get current price
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Execute action
        if action == 1:  # Buy
            # Calculate maximum shares we can buy
            max_shares = (self.balance * (1 - self.commission)) // current_price
            
            # Buy shares
            if max_shares > 0:
                self.shares_held += max_shares
                self.balance -= max_shares * current_price * (1 + self.commission)
        
        elif action == 2:  # Sell
            # Sell all shares
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price * (1 - self.commission)
                self.shares_held = 0
        
        # Update net worth
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Update history
        self.balance_history.append(self.balance)
        self.net_worth_history.append(self.net_worth)
        self.shares_held_history.append(self.shares_held)
        self.action_history.append(action)
        
        # Calculate reward
        reward = self._calculate_reward()
        self.reward_history.append(reward)
        
        # Update render data
        if self.render_freq and len(self.action_history) % self.render_freq == 0:
            self.render_data['dates'].append(self.df.index[self.current_step])
            self.render_data['prices'].append(current_price)
            self.render_data['actions'].append(action)
            self.render_data['portfolio_values'].append(self.net_worth)
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.df) - 1 or self.current_step >= self.max_steps
        
        # Get next observation
        obs = self._next_observation() if not done else None
        
        return obs, reward, done, {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': current_price
        }
    
    def _calculate_reward(self):
        """Calculate reward based on the reward function."""
        if self.reward_function == 'profit':
            # Simple profit reward
            return (self.net_worth - self.initial_balance) / self.initial_balance
        
        elif self.reward_function == 'sharpe':
            # Sharpe ratio reward
            if len(self.net_worth_history) > 1:
                returns = pd.Series(self.net_worth_history).pct_change().dropna()
                if returns.std() != 0:
                    return returns.mean() / returns.std()
            return 0
        
        elif self.reward_function == 'sortino':
            # Sortino ratio reward
            if len(self.net_worth_history) > 1:
                returns = pd.Series(self.net_worth_history).pct_change().dropna()
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0 and downside_returns.std() != 0:
                    return returns.mean() / downside_returns.std()
            return 0
        
        elif self.reward_function == 'calmar':
            # Calmar ratio reward
            if len(self.net_worth_history) > 1:
                returns = pd.Series(self.net_worth_history).pct_change().dropna()
                max_drawdown = (pd.Series(self.net_worth_history) / pd.Series(self.net_worth_history).expanding().max() - 1).min()
                if max_drawdown != 0:
                    return returns.mean() / abs(max_drawdown)
            return 0
        
        else:
            raise ValueError(f"Unknown reward function: {self.reward_function}")
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            print(f'Step: {self.current_step}')
            print(f'Balance: {self.balance:.2f}')
            print(f'Shares held: {self.shares_held}')
            print(f'Net worth: {self.net_worth:.2f}')
            print(f'Current price: {self.df.iloc[self.current_step]["Close"]:.2f}')
            print(f'Reward: {self.reward_history[-1]:.4f}')
            print('---')
        
        elif mode == 'system':
            # System mode for programmatic access to render data
            return self.render_data
        
        return None
    
    def close(self):
        """Close the environment."""
        pass


class MultiAssetTradingEnvironment(gym.Env):
    """
    A multi-asset trading environment for reinforcement learning agents.
    """
    metadata = {'render.modes': ['human', 'system', 'none']}
    
    def __init__(self, dfs, initial_balance=10000, commission=0.001, 
                 window_size=10, reward_function='sharpe', max_steps=1000,
                 add_indicators=True, max_positions=5, render_freq=None):
        """
        Initialize the multi-asset trading environment.
        
        Args:
            dfs: Dictionary of asset_name: DataFrame with OHLCV data
            initial_balance: Initial account balance
            commission: Commission rate for trades
            window_size: Size of the observation window
            reward_function: Reward function to use
            max_steps: Maximum steps per episode
            add_indicators: Whether to add technical indicators
            max_positions: Maximum number of positions to hold simultaneously
            render_freq: Frequency of rendering (None for no rendering)
        """
        super(MultiAssetTradingEnvironment, self).__init__()
        
        # Store parameters
        self.dfs = {name: df.copy() for name, df in dfs.items()}
        self.asset_names = list(dfs.keys())
        self.n_assets = len(self.asset_names)
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        self.reward_function = reward_function
        self.max_steps = max_steps
        self.add_indicators = add_indicators
        self.max_positions = max_positions
        self.render_freq = render_freq
        
        # Add technical indicators if requested
        if self.add_indicators:
            for name in self.asset_names:
                self._add_technical_indicators(name)
        
        # Define action and observation space
        # Actions: For each asset, 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.MultiDiscrete([3] * self.n_assets)
        
        # Observations: window_size * (sum of features for all assets + portfolio features)
        n_features_per_asset = len(self.dfs[self.asset_names[0]].columns)
        portfolio_features = 2  # Balance and allocation percentages
        
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(window_size, n_features_per_asset * self.n_assets + portfolio_features), 
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def _add_technical_indicators(self, asset_name):
        """Add technical indicators to the dataframe."""
        df = self.dfs[asset_name]
        
        # Simple Moving Average (SMA)
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        
        # Exponential Moving Average (EMA)
        df['EMA_10'] = df['Close'].ewm(span=10).mean()
        df['EMA_30'] = df['Close'].ewm(span=30).mean()
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        df['Upper_BB'] = sma_20 + (std_20 * 2)
        df['Lower_BB'] = sma_20 - (std_20 * 2)
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['%D'] = df['%K'].rolling(window=3).mean()
        
        # Commodity Channel Index (CCI)
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.fabs(x - x.mean()).mean())
        df['CCI'] = (tp - sma_tp) / (0.015 * mad)
        
        # Average True Range (ATR)
        df['TR1'] = df['High'] - df['Low']
        df['TR2'] = abs(df['High'] - df['Close'].shift(1))
        df['TR3'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Drop NaN values
        self.dfs[asset_name] = df.dropna()
    
    def reset(self):
        """Reset the environment to initial state."""
        # Reset account
        self.balance = self.initial_balance
        self.positions = {name: 0 for name in self.asset_names}
        self.net_worth = self.initial_balance
        
        # Reset data pointer
        self.current_step = self.window_size
        
        # Reset history
        self.balance_history = [self.balance]
        self.net_worth_history = [self.net_worth]
        self.positions_history = [self.positions.copy()]
        self.action_history = []
        self.reward_history = []
        
        # Get initial observation
        observation = self._next_observation()
        
        # Reset render data
        self.render_data = {
            'dates': self.dfs[self.asset_names[0]].index[self.current_step-self.window_size:self.current_step].tolist(),
            'prices': {name: self.dfs[name]['Close'].iloc[self.current_step-self.window_size:self.current_step].tolist() 
                      for name in self.asset_names},
            'actions': [],
            'portfolio_values': [self.initial_balance],
            'positions': {name: [] for name in self.asset_names}
        }
        
        return observation
    
    def _next_observation(self):
        """Get the next observation."""
        # Initialize observation frame
        n_features_per_asset = len(self.dfs[self.asset_names[0]].columns)
        obs_frame = np.zeros((self.window_size, n_features_per_asset * self.n_assets + 2))
        
        # Add asset data
        for i, name in enumerate(self.asset_names):
            # Get the data window
            frame = self.dfs[name].iloc[self.current_step - self.window_size:self.current_step].copy()
            
            # Normalize data
            for col in frame.columns:
                if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    # Normalize indicators to [0, 1]
                    min_val = frame[col].min()
                    max_val = frame[col].max()
                    if max_val > min_val:
                        frame[col] = (frame[col] - min_val) / (max_val - min_val)
                else:
                    # Normalize OHLCV to [0, 1]
                    if col == 'Volume':
                        frame[col] = frame[col] / frame[col].max()
                    else:
                        frame[col] = frame[col] / frame[col].max()
            
            # Add to observation
            obs_frame[:, i*n_features_per_asset:(i+1)*n_features_per_asset] = frame.values
        
        # Add portfolio information
        obs_frame[:, -2] = self.balance / self.initial_balance
        
        # Add position information as percentage of portfolio
        total_value = self.balance + sum(
            self.positions[name] * self.dfs[name]['Close'].iloc[self.current_step-1]
            for name in self.asset_names
        )
        
        for i, name in enumerate(self.asset_names):
            position_value = self.positions[name] * self.dfs[name]['Close'].iloc[self.current_step-1]
            obs_frame[:, -1] = position_value / total_value if total_value > 0 else 0
        
        return obs_frame.astype(np.float32)
    
    def step(self, actions):
        """Take a step in the environment."""
        # Get current prices
        current_prices = {name: self.dfs[name]['Close'].iloc[self.current_step] for name in self.asset_names}
        
        # Execute actions
        for i, name in enumerate(self.asset_names):
            action = actions[i]
            current_price = current_prices[name]
            
            if action == 1:  # Buy
                # Calculate maximum shares we can buy
                max_shares = (self.balance * (1 - self.commission)) // current_price
                
                # Buy shares if we have room for more positions
                if max_shares > 0 and sum(1 for pos in self.positions.values() if pos > 0) < self.max_positions:
                    self.positions[name] += max_shares
                    self.balance -= max_shares * current_price * (1 + self.commission)
            
            elif action == 2:  # Sell
                # Sell all shares of this asset
                if self.positions[name] > 0:
                    self.balance += self.positions[name] * current_price * (1 - self.commission)
                    self.positions[name] = 0
        
        # Update net worth
        self.net_worth = self.balance + sum(
            self.positions[name] * current_prices[name] for name in self.asset_names
        )
        
        # Update history
        self.balance_history.append(self.balance)
        self.net_worth_history.append(self.net_worth)
        self.positions_history.append(self.positions.copy())
        self.action_history.append(actions)
        
        # Calculate reward
        reward = self._calculate_reward()
        self.reward_history.append(reward)
        
        # Update render data
        if self.render_freq and len(self.action_history) % self.render_freq == 0:
            self.render_data['dates'].append(self.dfs[self.asset_names[0]].index[self.current_step])
            for name in self.asset_names:
                self.render_data['prices'][name].append(current_prices[name])
                self.render_data['positions'][name].append(self.positions[name])
            self.render_data['actions'].append(actions)
            self.render_data['portfolio_values'].append(self.net_worth)
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.dfs[self.asset_names[0]]) - 1 or self.current_step >= self.max_steps
        
        # Get next observation
        obs = self._next_observation() if not done else None
        
        return obs, reward, done, {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'current_prices': current_prices
        }
    
    def _calculate_reward(self):
        """Calculate reward based on the reward function."""
        if self.reward_function == 'profit':
            # Simple profit reward
            return (self.net_worth - self.initial_balance) / self.initial_balance
        
        elif self.reward_function == 'sharpe':
            # Sharpe ratio reward
            if len(self.net_worth_history) > 1:
                returns = pd.Series(self.net_worth_history).pct_change().dropna()
                if returns.std() != 0:
                    return returns.mean() / returns.std()
            return 0
        
        elif self.reward_function == 'sortino':
            # Sortino ratio reward
            if len(self.net_worth_history) > 1:
                returns = pd.Series(self.net_worth_history).pct_change().dropna()
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0 and downside_returns.std() != 0:
                    return returns.mean() / downside_returns.std()
            return 0
        
        elif self.reward_function == 'calmar':
            # Calmar ratio reward
            if len(self.net_worth_history) > 1:
                returns = pd.Series(self.net_worth_history).pct_change().dropna()
                max_drawdown = (pd.Series(self.net_worth_history) / pd.Series(self.net_worth_history).expanding().max() - 1).min()
                if max_drawdown != 0:
                    return returns.mean() / abs(max_drawdown)
            return 0
        
        else:
            raise ValueError(f"Unknown reward function: {self.reward_function}")
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            print(f'Step: {self.current_step}')
            print(f'Balance: {self.balance:.2f}')
            print(f'Positions: {self.positions}')
            print(f'Net worth: {self.net_worth:.2f}')
            print(f'Current prices: { {name: self.dfs[name]["Close"].iloc[self.current_step]:.2f} for name in self.asset_names} }')
            print(f'Reward: {self.reward_history[-1]:.4f}')
            print('---')
        
        elif mode == 'system':
            # System mode for programmatic access to render data
            return self.render_data
        
        return None
    
    def close(self):
        """Close the environment."""
        pass


class EnvironmentSimulator:
    """
    A simulator for creating and managing trading environments.
    """
    def __init__(self, data_dir='./data'):
        """
        Initialize the environment simulator.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
        self.environments = {}
        self.current_env = None
    
    def load_data(self, filename, asset_name=None):
        """
        Load data from a file.
        
        Args:
            filename: Name of the data file
            asset_name: Name of the asset (if None, use filename without extension)
            
        Returns:
            DataFrame with OHLCV data
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        # Set asset name
        if asset_name is None:
            asset_name = os.path.splitext(filename)[0]
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Set index if datetime column exists
        datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if datetime_cols:
            df[datetime_cols[0]] = pd.to_datetime(df[datetime_cols[0]])
            df.set_index(datetime_cols[0], inplace=True)
        
        return df
    
    def create_single_asset_env(self, data, env_name, **kwargs):
        """
        Create a single-asset trading environment.
        
        Args:
            data: DataFrame with OHLCV data or filename of data file
            env_name: Name of the environment
            **kwargs: Additional arguments for TradingEnvironment
            
        Returns:
            Created environment
        """
        # Load data if filename is provided
        if isinstance(data, str):
            data = self.load_data(data)
        
        # Create environment
        env = TradingEnvironment(data, **kwargs)
        
        # Store environment
        self.environments[env_name] = env
        
        # Set as current environment
        self.current_env = env
        
        return env
    
    def create_multi_asset_env(self, data_dict, env_name, **kwargs):
        """
        Create a multi-asset trading environment.
        
        Args:
            data_dict: Dictionary of asset_name: DataFrame or filename
            env_name: Name of the environment
            **kwargs: Additional arguments for MultiAssetTradingEnvironment
            
        Returns:
            Created environment
        """
        # Load data if filenames are provided
        dfs = {}
        for asset_name, data in data_dict.items():
            if isinstance(data, str):
                dfs[asset_name] = self.load_data(data, asset_name)
            else:
                dfs[asset_name] = data
        
        # Create environment
        env = MultiAssetTradingEnvironment(dfs, **kwargs)
        
        # Store environment
        self.environments[env_name] = env
        
        # Set as current environment
        self.current_env = env
        
        return env
    
    def get_environment(self, env_name):
        """
        Get an environment by name.
        
        Args:
            env_name: Name of the environment
            
        Returns:
            Environment
        """
        if env_name not in self.environments:
            raise ValueError(f"Environment '{env_name}' not found")
        
        return self.environments[env_name]
    
    def set_current_environment(self, env_name):
        """
        Set the current environment.
        
        Args:
            env_name: Name of the environment
        """
        self.current_env = self.get_environment(env_name)
    
    def list_environments(self):
        """
        List all available environments.
        
        Returns:
            List of environment names
        """
        return list(self.environments.keys())
    
    def save_environment(self, env_name, filepath):
        """
        Save an environment to a file.
        
        Args:
            env_name: Name of the environment
            filepath: Path to save the environment
        """
        env = self.get_environment(env_name)
        
        # Create save data
        save_data = {
            'env_type': type(env).__name__,
            'env_name': env_name,
            'params': {
                'initial_balance': env.initial_balance,
                'commission': env.commission,
                'window_size': env.window_size,
                'reward_function': env.reward_function,
                'max_steps': env.max_steps,
                'add_indicators': env.add_indicators
            }
        }
        
        # Add data for single-asset environment
        if isinstance(env, TradingEnvironment):
            save_data['data'] = env.df.to_dict()
        else:
            # Add data for multi-asset environment
            save_data['data'] = {
                name: df.to_dict() for name, df in env.dfs.items()
            }
            save_data['params']['max_positions'] = env.max_positions
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(save_data, f)
    
    def load_environment(self, filepath):
        """
        Load an environment from a file.
        
        Args:
            filepath: Path to the environment file
            
        Returns:
            Loaded environment
        """
        # Load from file
        with open(filepath, 'r') as f:
            save_data = json.load(f)
        
        # Get parameters
        env_name = save_data['env_name']
        params = save_data['params']
        
        # Load data
        if save_data['env_type'] == 'TradingEnvironment':
            # Single-asset environment
            df = pd.DataFrame.from_dict(save_data['data'])
            env = self.create_single_asset_env(df, env_name, **params)
        else:
            # Multi-asset environment
            dfs = {
                name: pd.DataFrame.from_dict(data_dict)
                for name, data_dict in save_data['data'].items()
            }
            env = self.create_multi_asset_env(dfs, env_name, **params)
        
        return env
