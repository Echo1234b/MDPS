import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

class PolicyNetwork(nn.Module):
    """
    Neural network for policy gradient methods.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2):
        """
        Initialize the policy network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output (number of actions)
            dropout: Dropout rate
        """
        super(PolicyNetwork, self).__init__()
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)


class ValueNetwork(nn.Module):
    """
    Neural network for value function approximation.
    """
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        """
        Initialize the value network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super(ValueNetwork, self).__init__()
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)


class PolicyGradient:
    """
    Policy gradient methods for reinforcement learning.
    """
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64], 
                 lr_policy=0.001, lr_value=0.001, gamma=0.99, 
                 gae_lambda=0.95, clip_ratio=0.2, target_kl=0.01,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the policy gradient agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            lr_policy: Learning rate for policy network
            lr_value: Learning rate for value network
            gamma: Discount factor
            gae_lambda: Lambda for Generalized Advantage Estimation
            clip_ratio: Clipping ratio for PPO
            target_kl: Target KL divergence for PPO
            device: Device to use for training
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.device = device
        
        # Initialize networks
        self.policy_net = PolicyNetwork(state_dim, hidden_dims, action_dim).to(device)
        self.value_net = ValueNetwork(state_dim, hidden_dims).to(device)
        
        # Initialize old policy network for PPO
        self.old_policy_net = PolicyNetwork(state_dim, hidden_dims, action_dim).to(device)
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)
        
        # Training history
        self.policy_losses = []
        self.value_losses = []
        self.kl_divergences = []
        self.entropy_losses = []
        self.episode_rewards = []
    
    def get_action(self, state, deterministic=False):
        """
        Get action from policy network.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic action
            
        Returns:
            Action and log probability
        """
        state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            logits = self.policy_net(state)
            
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                # Sample action from distribution
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                return action.item(), log_prob.item()
        
        return action.item()
    
    def compute_returns(self, rewards, values, dones, next_value):
        """
        Compute returns and advantages.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value of next state
            
        Returns:
            Returns and advantages
        """
        # Compute returns
        returns = []
        R = next_value
        
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Compute advantages using GAE
        advantages = torch.zeros_like(returns).to(self.device)
        
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages[step] = gae
            next_value = values[step]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update_policy(self, states, actions, log_probs_old, returns, advantages):
        """
        Update policy network using PPO.
        
        Args:
            states: List of states
            actions: List of actions
            log_probs_old: List of old log probabilities
            returns: List of returns
            advantages: List of advantages
            
        Returns:
            Policy loss, value loss, KL divergence, entropy
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Get current policy logits and values
        logits = self.policy_net(states)
        values = self.value_net(states).squeeze()
        
        # Compute current log probabilities
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Compute KL divergence
        with torch.no_grad():
            old_logits = self.old_policy_net(states)
            old_dist = Categorical(logits=old_logits)
            kl_divergence = (old_dist.probs * (old_dist.logits - dist.logits)).sum(dim=-1).mean()
        
        # Compute policy loss (PPO)
        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss
        value_loss = F.mse_loss(values, returns)
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        (policy_loss - 0.01 * entropy).backward()
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Store losses
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        self.kl_divergences.append(kl_divergence.item())
        self.entropy_losses.append(entropy.item())
        
        return policy_loss.item(), value_loss.item(), kl_divergence.item(), entropy.item()
    
    def train(self, env, num_episodes=100, max_steps=1000, update_every=10, 
              print_every=10, save_every=100, save_path=None):
        """
        Train the policy gradient agent.
        
        Args:
            env: Environment to train in
            num_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            update_every: Update policy every N episodes
            print_every: Print progress every N episodes
            save_every: Save model every N episodes
            save_path: Path to save model
            
        Returns:
            Training history
        """
        # Training loop
        for episode in range(num_episodes):
            # Reset environment
            state = env.reset()
            episode_reward = 0
            
            # Collect trajectory
            states = []
            actions = []
            log_probs = []
            rewards = []
            values = []
            dones = []
            
            for step in range(max_steps):
                # Get action
                action, log_prob = self.get_action(state)
                
                # Take action
                next_state, reward, done, _ = env.step(action)
                
                # Get value estimate
                state_tensor = torch.FloatTensor(state).to(self.device)
                value = self.value_net(state_tensor).item()
                
                # Store transition
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)
                dones.append(done)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                
                # Check if done
                if done:
                    break
            
            # Get value of next state
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            next_value = self.value_net(next_state_tensor).item()
            
            # Compute returns and advantages
            returns, advantages = self.compute_returns(rewards, values, dones, next_value)
            
            # Update policy
            if episode % update_every == 0:
                # Copy current policy to old policy
                self.old_policy_net.load_state_dict(self.policy_net.state_dict())
                
                # Update policy multiple times
                for _ in range(10):
                    policy_loss, value_loss, kl_divergence, entropy = self.update_policy(
                        states, actions, log_probs, returns, advantages
                    )
                    
                    # Early stopping if KL divergence is too high
                    if kl_divergence > self.target_kl:
                        break
            
            # Store episode reward
            self.episode_rewards.append(episode_reward)
            
            # Print progress
            if episode % print_every == 0:
                avg_reward = np.mean(self.episode_rewards[-print_every:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
            
            # Save model
            if save_path and episode % save_every == 0:
                self.save_model(f"{save_path}_episode_{episode}.pt")
        
        # Save final model
        if save_path:
            self.save_model(f"{save_path}_final.pt")
        
        return {
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'kl_divergences': self.kl_divergences,
            'entropy_losses': self.entropy_losses,
            'episode_rewards': self.episode_rewards
        }
    
    def evaluate(self, env, num_episodes=10, max_steps=1000, render=False):
        """
        Evaluate the policy gradient agent.
        
        Args:
            env: Environment to evaluate in
            num_episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode
            render: Whether to render the environment
            
        Returns:
            Evaluation results
        """
        episode_rewards = []
        
        for episode in range(num_episodes):
            # Reset environment
            state = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                # Get action
                action = self.get_action(state, deterministic=True)
                
                # Take action
                next_state, reward, done, _ = env.step(action)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                
                # Render if requested
                if render:
                    env.render()
                
                # Check if done
                if done:
                    break
            
            episode_rewards.append(episode_reward)
        
        # Calculate statistics
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }
        
        return results
    
    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'kl_divergences': self.kl_divergences,
            'entropy_losses': self.entropy_losses,
            'episode_rewards': self.episode_rewards
        }
        
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath):
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        self.policy_losses = checkpoint['policy_losses']
        self.value_losses = checkpoint['value_losses']
        self.kl_divergences = checkpoint['kl_divergences']
        self.entropy_losses = checkpoint['entropy_losses']
        self.episode_rewards = checkpoint['episode_rewards']
        
        # Copy current policy to old policy
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        
        return self
    
    def plot_training_history(self, figsize=(15, 10)):
        """
        Plot training history.
        
        Args:
            figsize: Figure size
        """
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot policy loss
        axes[0, 0].plot(self.policy_losses)
        axes[0, 0].set_title('Policy Loss')
        axes[0, 0].set_xlabel('Update Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Plot value loss
        axes[0, 1].plot(self.value_losses)
        axes[0, 1].set_title('Value Loss')
        axes[0, 1].set_xlabel('Update Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Plot KL divergence
        axes[1, 0].plot(self.kl_divergences)
        axes[1, 0].set_title('KL Divergence')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('KL Divergence')
        axes[1, 0].grid(True)
        
        # Plot episode rewards
        axes[1, 1].plot(self.episode_rewards)
        axes[1, 1].set_title('Episode Rewards')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
