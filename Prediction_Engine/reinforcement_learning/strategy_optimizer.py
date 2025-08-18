import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import gym
from gym import spaces
from stable_baselines3 import PPO, A2C, DQN, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import os
import json
import joblib
from datetime import datetime
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

class TradingEnvironment(gym.Env):
    """
    Custom trading environment for reinforcement learning.
    """
    def __init__(self, data, initial_balance=10000, commission=0.001, 
                 window_size=10, reward_function='sharpe', max_steps=1000):
        """
        Initialize the trading environment.
        
        Args:
            data: DataFrame with OHLCV data
            initial_balance: Initial account balance
            commission: Commission rate for trades
            window_size: Size of the observation window
            reward_function: Reward function to use ('sharpe', 'profit', 'sortino')
            max_steps: Maximum steps per episode
        """
        super(TradingEnvironment, self).__init__()
        
        # Store data
        self.data = data
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        self.reward_function = reward_function
        self.max_steps = max_steps
        
        # Define action and observation space
        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observations: window_size * number of features
        # Features: Open, High, Low, Close, Volume, Balance, Shares Held
        n_features = 6  # OHLCV + Balance + Shares Held
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(window_size, n_features), 
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
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
        
        # Get initial observation
        observation = self._next_observation()
        
        return observation
    
    def _next_observation(self):
        """Get the next observation."""
        # Get the data window
        frame = self.data.iloc[self.current_step - self.window_size:self.current_step]
        
        # Normalize data
        frame = frame.copy()
        for col in ['Open', 'High', 'Low', 'Close']:
            frame[col] = frame[col] / frame[col].max()
        
        frame['Volume'] = frame['Volume'] / frame['Volume'].max()
        
        # Add account information
        frame['Balance'] = self.balance / self.initial_balance
        frame['Shares_Held'] = self.shares_held / (self.initial_balance / frame['Close'].iloc[-1])
        
        return frame.values.astype(np.float32)
    
    def step(self, action):
        """Take a step in the environment."""
        # Get current price
        current_price = self.data.iloc[self.current_step]['Close']
        
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
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        done = self.current_step >= len(self.data) - 1 or self.current_step >= self.max_steps
        
        # Get next observation
        obs = self._next_observation() if not done else None
        
        return obs, reward, done, {}
    
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
        
        else:
            raise ValueError(f"Unknown reward function: {self.reward_function}")
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            print(f'Step: {self.current_step}')
            print(f'Balance: {self.balance:.2f}')
            print(f'Shares held: {self.shares_held}')
            print(f'Net worth: {self.net_worth:.2f}')
            print(f'Current price: {self.data.iloc[self.current_step]["Close"]:.2f}')
    
    def close(self):
        """Close the environment."""
        pass


class TensorboardCallback(BaseCallback):
    """
    Callback for logging additional metrics to Tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_net_worths = []
    
    def _on_step(self) -> bool:
        # Log reward
        if 'rewards' in self.locals:
            self.episode_rewards.append(self.locals['rewards'][0])
        
        # Log episode length
        if self.locals['dones'][0]:
            self.episode_lengths.append(self.locals['episode_lengths'][0])
            
            # Log net worth
            if hasattr(self.training_env, 'envs'):
                env = self.training_env.envs[0]
                if hasattr(env, 'net_worth_history'):
                    self.episode_net_worths.append(env.net_worth_history[-1])
        
        return True
    
    def _on_rollout_end(self) -> None:
        if self.episode_rewards:
            # Log mean reward
            self.logger.record('rollout/ep_reward_mean', np.mean(self.episode_rewards))
            self.logger.record('rollout/ep_reward_std', np.std(self.episode_rewards))
            self.episode_rewards = []
        
        if self.episode_lengths:
            # Log mean episode length
            self.logger.record('rollout/ep_length_mean', np.mean(self.episode_lengths))
            self.episode_lengths = []
        
        if self.episode_net_worths:
            # Log mean net worth
            self.logger.record('rollout/ep_net_worth_mean', np.mean(self.episode_net_worths))
            self.episode_net_worths = []


class RLStrategyOptimizer:
    def __init__(self, model_type='PPO', tensorboard_log='./tensorboard/', 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the RL Strategy Optimizer.
        
        Args:
            model_type: Type of RL model ('PPO', 'A2C', 'DQN', 'TD3')
            tensorboard_log: Path to Tensorboard log directory
            device: Device to use for training
        """
        self.model_type = model_type
        self.tensorboard_log = tensorboard_log
        self.device = device
        
        # Initialize model
        self.model = None
        self.env = None
        
        # Training history
        self.training_history = []
        
        # Set up logging
        self.logger = logging.getLogger("RLStrategyOptimizer")
        self.logger.setLevel(logging.INFO)
        
        # Create log directory if it doesn't exist
        os.makedirs(tensorboard_log, exist_ok=True)
    
    def create_environment(self, data, initial_balance=10000, commission=0.001, 
                          window_size=10, reward_function='sharpe', max_steps=1000,
                          n_envs=1):
        """
        Create a trading environment.
        
        Args:
            data: DataFrame with OHLCV data
            initial_balance: Initial account balance
            commission: Commission rate for trades
            window_size: Size of the observation window
            reward_function: Reward function to use
            max_steps: Maximum steps per episode
            n_envs: Number of parallel environments
            
        Returns:
            Created environment
        """
        def make_env():
            def _init():
                env = TradingEnvironment(
                    data=data,
                    initial_balance=initial_balance,
                    commission=commission,
                    window_size=window_size,
                    reward_function=reward_function,
                    max_steps=max_steps
                )
                env = Monitor(env, filename=None)
                return env
            return _init
        
        if n_envs == 1:
            self.env = DummyVecEnv([make_env()])
        else:
            self.env = SubprocVecEnv([make_env() for _ in range(n_envs)])
        
        return self.env
    
    def create_model(self, policy='MlpPolicy', learning_rate=0.0003, 
                    n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, 
                    vf_coef=0.5, max_grad_norm=0.5, **kwargs):
        """
        Create an RL model.
        
        Args:
            policy: Policy network architecture
            learning_rate: Learning rate
            n_steps: Number of steps to run for each environment per update
            batch_size: Minibatch size
            n_epochs: Number of epochs when optimizing the surrogate loss
            gamma: Discount factor
            gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            clip_range: Clipping parameter
            ent_coef: Entropy coefficient for the loss calculation
            vf_coef: Value function coefficient for the loss calculation
            max_grad_norm: Maximum value for gradient clipping
            **kwargs: Additional model-specific parameters
            
        Returns:
            Created model
        """
        if self.env is None:
            raise ValueError("Environment not created. Call create_environment() first.")
        
        # Create model based on type
        if self.model_type == 'PPO':
            self.model = PPO(
                policy=policy,
                env=self.env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                tensorboard_log=self.tensorboard_log,
                device=self.device,
                **kwargs
            )
        elif self.model_type == 'A2C':
            self.model = A2C(
                policy=policy,
                env=self.env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                tensorboard_log=self.tensorboard_log,
                device=self.device,
                **kwargs
            )
        elif self.model_type == 'DQN':
            self.model = DQN(
                policy=policy,
                env=self.env,
                learning_rate=learning_rate,
                batch_size=batch_size,
                gamma=gamma,
                tensorboard_log=self.tensorboard_log,
                device=self.device,
                **kwargs
            )
        elif self.model_type == 'TD3':
            self.model = TD3(
                policy=policy,
                env=self.env,
                learning_rate=learning_rate,
                batch_size=batch_size,
                gamma=gamma,
                tensorboard_log=self.tensorboard_log,
                device=self.device,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def train(self, total_timesteps=10000, callback=None, log_interval=100, 
             eval_env=None, eval_freq=1000, n_eval_episodes=5, 
             tb_log_name='run', reset_num_timesteps=True, 
             progress_bar=True):
        """
        Train the RL model.
        
        Args:
            total_timesteps: Total number of timesteps to train
            callback: Callback(s) called at every step with state of the algorithm
            log_interval: Log training progress every log_interval timesteps
            eval_env: Environment to use for evaluation
            eval_freq: Evaluate the model every eval_freq timesteps
            n_eval_episodes: Number of episodes to evaluate
            tb_log_name: Name of the Tensorboard log
            reset_num_timesteps: Whether to reset or not the num_timesteps attribute
            progress_bar: Display a progress bar using tqdm and rich
            
        Returns:
            Trained model
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Create callback if not provided
        if callback is None:
            callback = TensorboardCallback()
        
        # Train model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar
        )
        
        # Add to training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'total_timesteps': total_timesteps,
            'final_reward': callback.episode_rewards[-1] if callback.episode_rewards else None
        })
        
        return self.model
    
    def evaluate(self, env=None, n_episodes=10, deterministic=True, render=False):
        """
        Evaluate the trained model.
        
        Args:
            env: Environment to evaluate in (if None, use training environment)
            n_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic or stochastic actions
            render: Whether to render the environment
            
        Returns:
            Evaluation results
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Use training environment if not provided
        if env is None:
            env = self.env
        
        # Evaluate model
        episode_rewards = []
        episode_lengths = []
        episode_net_worths = []
        
        for i in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _states = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if render:
                    env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Get net worth
            if hasattr(env, 'envs'):
                env_obj = env.envs[0]
                if hasattr(env_obj, 'net_worth_history'):
                    episode_net_worths.append(env_obj.net_worth_history[-1])
        
        # Calculate statistics
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
        }
        
        if episode_net_worths:
            results['mean_net_worth'] = np.mean(episode_net_worths)
            results['std_net_worth'] = np.std(episode_net_worths)
        
        return results
    
    def predict(self, observation, deterministic=True):
        """
        Get the model's action(s) for a given observation(s).
        
        Args:
            observation: The input observation
            deterministic: Whether to use deterministic or stochastic actions
            
        Returns:
            Model's action and the next state
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        action, states = self.model.predict(observation, deterministic=deterministic)
        return action, states
    
    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        self.model.save(filepath)
        
        # Save training history
        history_path = f"{filepath}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        # Load model based on type
        if self.model_type == 'PPO':
            self.model = PPO.load(filepath, device=self.device)
        elif self.model_type == 'A2C':
            self.model = A2C.load(filepath, device=self.device)
        elif self.model_type == 'DQN':
            self.model = DQN.load(filepath, device=self.device)
        elif self.model_type == 'TD3':
            self.model = TD3.load(filepath, device=self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load training history if exists
        history_path = f"{filepath}_history.json"
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
        
        self.logger.info(f"Model loaded from {filepath}")
        
        return self.model
    
    def plot_training_history(self, figsize=(15, 5)):
        """
        Plot training history.
        
        Args:
            figsize: Figure size
        """
        if not self.training_history:
            print("No training history to plot.")
            return
        
        # Create DataFrame from training history
        df = pd.DataFrame(self.training_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot final reward
        axes[0].plot(df['timestamp'], df['final_reward'], marker='o')
        axes[0].set_title('Final Reward')
        axes[0].set_xlabel('Timestamp')
        axes[0].set_ylabel('Reward')
        axes[0].grid(True)
        
        # Plot total timesteps
        axes[1].plot(df['timestamp'], df['total_timesteps'], marker='o')
        axes[1].set_title('Total Timesteps')
        axes[1].set_xlabel('Timestamp')
        axes[1].set_ylabel('Timesteps')
        axes[1].grid(True)
        
        # Plot model type distribution
        model_counts = df['model_type'].value_counts()
        axes[2].pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%')
        axes[2].set_title('Model Type Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def backtest(self, data, initial_balance=10000, commission=0.001, 
                window_size=10, reward_function='sharpe', render=False):
        """
        Backtest the trained model on historical data.
        
        Args:
            data: DataFrame with OHLCV data
            initial_balance: Initial account balance
            commission: Commission rate for trades
            window_size: Size of the observation window
            reward_function: Reward function to use
            render: Whether to render the environment
            
        Returns:
            Backtest results
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Create environment
        env = TradingEnvironment(
            data=data,
            initial_balance=initial_balance,
            commission=commission,
            window_size=window_size,
            reward_function=reward_function,
            max_steps=len(data)
        )
        
        # Reset environment
        obs = env.reset()
        
        # Run backtest
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            if render:
                env.render()
        
        # Calculate results
        results = {
            'initial_balance': initial_balance,
            'final_balance': env.balance,
            'final_net_worth': env.net_worth,
            'total_return': (env.net_worth - initial_balance) / initial_balance,
            'balance_history': env.balance_history,
            'net_worth_history': env.net_worth_history,
            'shares_held_history': env.shares_held_history,
            'action_history': env.action_history
        }
        
        return results
    
    def plot_backtest_results(self, results, figsize=(15, 10)):
        """
        Plot backtest results.
        
        Args:
            results: Backtest results from backtest() method
            figsize: Figure size
        """
        # Create figure
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # Plot net worth
        axes[0].plot(results['net_worth_history'])
        axes[0].set_title('Net Worth')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Net Worth')
        axes[0].grid(True)
        
        # Plot balance
        axes[1].plot(results['balance_history'])
        axes[1].set_title('Balance')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Balance')
        axes[1].grid(True)
        
        # Plot shares held
        axes[2].plot(results['shares_held_history'])
        axes[2].set_title('Shares Held')
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('Shares')
        axes[2].grid(True)
        
        # Plot actions
        actions = results['action_history']
        action_names = ['Hold', 'Buy', 'Sell']
        action_colors = ['gray', 'green', 'red']
        
        for i, action in enumerate(actions):
            axes[3].scatter(i, action, color=action_colors[action], alpha=0.5)
        
        axes[3].set_title('Actions')
        axes[3].set_xlabel('Step')
        axes[3].set_ylabel('Action')
        axes[3].set_yticks([0, 1, 2])
        axes[3].set_yticklabels(action_names)
        axes[3].grid(True)
        
        plt.tight_layout()
        plt.show()
