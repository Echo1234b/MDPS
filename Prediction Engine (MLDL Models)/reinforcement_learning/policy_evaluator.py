import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import warnings
warnings.filterwarnings('ignore')

class PolicyEvaluator:
    """
    A comprehensive evaluator for reinforcement learning policies.
    """
    def __init__(self, env, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the policy evaluator.
        
        Args:
            env: Environment to evaluate policies in
            device: Device to use for evaluation
        """
        self.env = env
        self.device = device
        self.evaluation_results = {}
        self.comparison_results = {}
    
    def evaluate_policy(self, policy, num_episodes=10, deterministic=True, 
                       render=False, verbose=True, return_trajectory=False):
        """
        Evaluate a policy in the environment.
        
        Args:
            policy: Policy to evaluate (function or model with predict method)
            num_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic actions
            render: Whether to render the environment
            verbose: Whether to print progress
            return_trajectory: Whether to return full trajectory data
            
        Returns:
            Evaluation results
        """
        episode_rewards = []
        episode_lengths = []
        trajectory_data = [] if return_trajectory else None
        
        for episode in range(num_episodes):
            if verbose:
                print(f"Evaluating episode {episode + 1}/{num_episodes}")
            
            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            # Initialize trajectory data
            if return_trajectory:
                trajectory = {
                    'states': [state],
                    'actions': [],
                    'rewards': [],
                    'next_states': [],
                    'dones': [],
                    'infos': []
                }
            
            # Run episode
            done = False
            while not done:
                # Get action from policy
                if hasattr(policy, 'predict'):
                    # Stable Baselines3 model
                    action, _ = policy.predict(state, deterministic=deterministic)
                elif callable(policy):
                    # Custom policy function
                    action = policy(state)
                else:
                    raise ValueError("Policy must be a callable function or have a predict method")
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Update metrics
                episode_reward += reward
                episode_length += 1
                
                # Store trajectory data
                if return_trajectory:
                    trajectory['actions'].append(action)
                    trajectory['rewards'].append(reward)
                    trajectory['next_states'].append(next_state)
                    trajectory['dones'].append(done)
                    trajectory['infos'].append(info)
                
                # Update state
                state = next_state
                
                # Render if requested
                if render:
                    self.env.render()
            
            # Store episode results
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if return_trajectory:
                trajectory['states'].append(state)  # Add final state
                trajectory_data.append(trajectory)
        
        # Calculate statistics
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
        
        if return_trajectory:
            results['trajectory_data'] = trajectory_data
        
        return results
    
    def evaluate_with_metrics(self, policy, metrics=None, num_episodes=10, 
                            baseline=None, verbose=True):
        """
        Evaluate a policy with custom metrics.
        
        Args:
            policy: Policy to evaluate
            metrics: Dictionary of metric_name: metric_function pairs
            num_episodes: Number of episodes to evaluate
            baseline: Baseline policy to compare against
            verbose: Whether to print progress
            
        Returns:
            Evaluation results with custom metrics
        """
        # Default metrics
        if metrics is None:
            metrics = {
                'sharpe_ratio': self._calculate_sharpe_ratio,
                'sortino_ratio': self._calculate_sortino_ratio,
                'max_drawdown': self._calculate_max_drawdown,
                'win_rate': self._calculate_win_rate,
                'profit_factor': self._calculate_profit_factor
            }
        
        # Evaluate policy
        results = self.evaluate_policy(
            policy, num_episodes=num_episodes, 
            return_trajectory=True, verbose=verbose
        )
        
        # Calculate custom metrics
        for metric_name, metric_func in metrics.items():
            try:
                metric_values = []
                
                for trajectory in results['trajectory_data']:
                    metric_value = metric_func(trajectory)
                    metric_values.append(metric_value)
                
                results[f'mean_{metric_name}'] = np.mean(metric_values)
                results[f'std_{metric_name}'] = np.std(metric_values)
                
            except Exception as e:
                if verbose:
                    print(f"Error calculating metric {metric_name}: {str(e)}")
                results[f'mean_{metric_name}'] = None
                results[f'std_{metric_name}'] = None
        
        # Compare with baseline if provided
        if baseline is not None:
            baseline_results = self.evaluate_with_metrics(
                baseline, metrics=metrics, num_episodes=num_episodes, verbose=False
            )
            
            comparison = {}
            for metric_name in metrics.keys():
                mean_key = f'mean_{metric_name}'
                if mean_key in results and mean_key in baseline_results:
                    comparison[f'{metric_name}_improvement'] = (
                        results[mean_key] - baseline_results[mean_key]
                    ) / baseline_results[mean_key]
            
            results['baseline_comparison'] = comparison
        
        return results
    
    def _calculate_sharpe_ratio(self, trajectory):
        """
        Calculate Sharpe ratio for a trajectory.
        
        Args:
            trajectory: Trajectory data
            
        Returns:
            Sharpe ratio
        """
        rewards = trajectory['rewards']
        if len(rewards) < 2:
            return 0.0
        
        returns = pd.Series(rewards).pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        
        return returns.mean() / returns.std()
    
    def _calculate_sortino_ratio(self, trajectory):
        """
        Calculate Sortino ratio for a trajectory.
        
        Args:
            trajectory: Trajectory data
            
        Returns:
            Sortino ratio
        """
        rewards = trajectory['rewards']
        if len(rewards) < 2:
            return 0.0
        
        returns = pd.Series(rewards).pct_change().dropna()
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return returns.mean() / downside_returns.std()
    
    def _calculate_max_drawdown(self, trajectory):
        """
        Calculate maximum drawdown for a trajectory.
        
        Args:
            trajectory: Trajectory data
            
        Returns:
            Maximum drawdown
        """
        rewards = trajectory['rewards']
        cumulative = np.cumsum(rewards)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-8)
        
        return np.min(drawdown)
    
    def _calculate_win_rate(self, trajectory):
        """
        Calculate win rate for a trajectory.
        
        Args:
            trajectory: Trajectory data
            
        Returns:
            Win rate
        """
        rewards = trajectory['rewards']
        if len(rewards) == 0:
            return 0.0
        
        wins = sum(1 for r in rewards if r > 0)
        return wins / len(rewards)
    
    def _calculate_profit_factor(self, trajectory):
        """
        Calculate profit factor for a trajectory.
        
        Args:
            trajectory: Trajectory data
            
        Returns:
            Profit factor
        """
        rewards = trajectory['rewards']
        if len(rewards) == 0:
            return 0.0
        
        gross_profit = sum(r for r in rewards if r > 0)
        gross_loss = abs(sum(r for r in rewards if r < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def compare_policies(self, policies, policy_names=None, num_episodes=10, 
                        metrics=None, statistical_test=True, significance_level=0.05):
        """
        Compare multiple policies.
        
        Args:
            policies: List of policies to compare
            policy_names: List of policy names (if None, use indices)
            num_episodes: Number of episodes to evaluate each policy
            metrics: Dictionary of metric_name: metric_function pairs
            statistical_test: Whether to perform statistical tests
            significance_level: Significance level for statistical tests
            
        Returns:
            Comparison results
        """
        if policy_names is None:
            policy_names = [f"Policy {i}" for i in range(len(policies))]
        
        if len(policies) != len(policy_names):
            raise ValueError("Number of policies must match number of policy names")
        
        # Evaluate all policies
        results = {}
        for policy, name in zip(policies, policy_names):
            results[name] = self.evaluate_with_metrics(
                policy, metrics=metrics, num_episodes=num_episodes, verbose=False
            )
        
        # Create comparison DataFrame
        comparison_data = []
        
        for name, result in results.items():
            row = {'Policy': name}
            
            # Add basic metrics
            row['Mean Reward'] = result['mean_reward']
            row['Std Reward'] = result['std_reward']
            row['Min Reward'] = result['min_reward']
            row['Max Reward'] = result['max_reward']
            
            # Add custom metrics
            if metrics:
                for metric_name in metrics.keys():
                    mean_key = f'mean_{metric_name}'
                    std_key = f'std_{metric_name}'
                    if mean_key in result:
                        row[metric_name.title()] = result[mean_key]
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Perform statistical tests if requested
        statistical_results = None
        if statistical_test and len(policies) >= 2:
            statistical_results = self._perform_statistical_tests(
                results, policy_names, significance_level
            )
        
        return {
            'comparison_df': comparison_df,
            'statistical_results': statistical_results,
            'detailed_results': results
        }
    
    def _perform_statistical_tests(self, results, policy_names, significance_level):
        """
        Perform statistical tests to compare policies.
        
        Args:
            results: Evaluation results for each policy
            policy_names: Names of the policies
            significance_level: Significance level for tests
            
        Returns:
            Statistical test results
        """
        from scipy import stats
        
        test_results = {}
        
        # Get reward data for each policy
        reward_data = {
            name: result['episode_rewards'] 
            for name, result in results.items()
        }
        
        # Perform pairwise t-tests
        t_test_results = {}
        for i, name1 in enumerate(policy_names):
            for j, name2 in enumerate(policy_names):
                if i < j:  # Avoid duplicate tests
                    stat, p_value = stats.ttest_ind(
                        reward_data[name1], reward_data[name2]
                    )
                    
                    t_test_results[f"{name1} vs {name2}"] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'significant': p_value < significance_level
                    }
        
        test_results['t_tests'] = t_test_results
        
        # Perform ANOVA if more than 2 policies
        if len(policy_names) > 2:
            # Prepare data for ANOVA
            anova_data = []
            for name in policy_names:
                for reward in reward_data[name]:
                    anova_data.append({'Policy': name, 'Reward': reward})
            
            anova_df = pd.DataFrame(anova_data)
            
            # Perform one-way ANOVA
            groups = [group['Reward'].values for name, group in anova_df.groupby('Policy')]
            f_stat, p_value = stats.f_oneway(*groups)
            
            test_results['anova'] = {
                'statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < significance_level
            }
        
        # Perform Kruskal-Wallis test (non-parametric alternative to ANOVA)
        kruskal_data = [reward_data[name] for name in policy_names]
        h_stat, p_value = stats.kruskal(*kruskal_data)
        
        test_results['kruskal_wallis'] = {
            'statistic': h_stat,
            'p_value': p_value,
            'significant': p_value < significance_level
        }
        
        return test_results
    
    def plot_comparison(self, comparison_results, figsize=(15, 10)):
        """
        Plot comparison results.
        
        Args:
            comparison_results: Results from compare_policies method
            figsize: Figure size
        """
        comparison_df = comparison_results['comparison_df']
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot mean rewards
        axes[0, 0].bar(comparison_df['Policy'], comparison_df['Mean Reward'])
        axes[0, 0].set_title('Mean Reward')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot reward distributions
        reward_data = []
        labels = []
        
        for _, row in comparison_df.iterrows():
            policy_name = row['Policy']
            policy_results = comparison_results['detailed_results'][policy_name]
            reward_data.append(policy_results['episode_rewards'])
            labels.append(policy_name)
        
        axes[0, 1].boxplot(reward_data, labels=labels)
        axes[0, 1].set_title('Reward Distribution')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot custom metrics if available
        custom_metrics = [col for col in comparison_df.columns 
                         if col not in ['Policy', 'Mean Reward', 'Std Reward', 'Min Reward', 'Max Reward']]
        
        if custom_metrics:
            # Plot first custom metric
            metric = custom_metrics[0]
            axes[1, 0].bar(comparison_df['Policy'], comparison_df[metric])
            axes[1, 0].set_title(metric)
            axes[1, 0].set_ylabel(metric)
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot second custom metric if available
            if len(custom_metrics) > 1:
                metric = custom_metrics[1]
                axes[1, 1].bar(comparison_df['Policy'], comparison_df[metric])
                axes[1, 1].set_title(metric)
                axes[1, 1].set_ylabel(metric)
                axes[1, 1].tick_params(axis='x', rotation=45)
            else:
                # Hide empty subplot
                axes[1, 1].axis('off')
        else:
            # Hide empty subplots
            axes[1, 0].axis('off')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curves(self, learning_data, figsize=(15, 5)):
        """
        Plot learning curves.
        
        Args:
            learning_data: Dictionary of policy_name: {'steps': list, 'rewards': list}
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot reward curves
        for policy_name, data in learning_data.items():
            axes[0].plot(data['steps'], data['rewards'], label=policy_name)
        
        axes[0].set_title('Learning Curves')
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Reward')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot smoothed reward curves
        for policy_name, data in learning_data.items():
            smoothed_rewards = self._smooth_curve(data['rewards'])
            axes[1].plot(data['steps'], smoothed_rewards, label=policy_name)
        
        axes[1].set_title('Smoothed Learning Curves')
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('Reward')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def _smooth_curve(self, data, window=10):
        """
        Smooth a curve using a moving average.
        
        Args:
            data: Data to smooth
            window: Window size for moving average
            
        Returns:
            Smoothed data
        """
        if len(data) < window:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            smoothed.append(np.mean(data[start:end]))
        
        return smoothed
    
    def save_results(self, results, filepath):
        """
        Save evaluation results to a file.
        
        Args:
            results: Evaluation results
            filepath: Path to save the results
        """
        # Convert non-serializable objects
        serializable_results = self._make_serializable(results)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
    
    def load_results(self, filepath):
        """
        Load evaluation results from a file.
        
        Args:
            filepath: Path to the results file
            
        Returns:
            Loaded results
        """
        # Load from file
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        return results
    
    def _make_serializable(self, obj):
        """
        Convert non-serializable objects to serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            Serializable object
        """
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
