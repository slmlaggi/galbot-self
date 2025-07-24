#####################################################################################
# Copyright (c) 2023-2025 Galbot. All Rights Reserved.
#
# This software contains confidential and proprietary information of Galbot, Inc.
# ("Confidential Information"). You shall not disclose such Confidential Information
# and shall use it only in accordance with the terms of the license agreement you
# entered into with Galbot, Inc.
#
# UNAUTHORIZED COPYING, USE, OR DISTRIBUTION OF THIS SOFTWARE, OR ANY PORTION OR
# DERIVATIVE THEREOF, IS STRICTLY PROHIBITED. IF YOU HAVE RECEIVED THIS SOFTWARE IN
# ERROR, PLEASE NOTIFY GALBOT, INC. IMMEDIATELY AND DELETE IT FROM YOUR SYSTEM.
#####################################################################################
#          _____             _   _       _   _
#         / ____|           | | | |     | \ | |
#        | (___  _   _ _ __ | |_| |__   |  \| | _____   ____ _
#         \___ \| | | | '_ \| __| '_ \  | . ` |/ _ \ \ / / _` |
#         ____) | |_| | | | | |_| | | | | |\  | (_) \ V / (_| |
#        |_____/ \__, |_| |_|\__|_| |_| |_| \_|\___/ \_/ \__,_|
#                 __/ |
#                |___/
#
#####################################################################################
#
# Description: PPO training script for Galbot navigation environment
# Author: IOAI Training System
# Date: 2025-01-24
#
#####################################################################################

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque
import random
import json
from datetime import datetime
from pathlib import Path

from environment import IoaiNavEnv


class PPOActor(nn.Module):
    """PPO Actor Network for policy approximation"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PPOActor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state):
        return self.network(state)
    
    def get_action(self, state):
        """Sample action from policy"""
        probs = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


class PPOCritic(nn.Module):
    """PPO Critic Network for value function approximation"""
    
    def __init__(self, state_dim, hidden_dim=256):
        super(PPOCritic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        return self.network(state)


class PPOAgent:
    """Proximal Policy Optimization Agent"""
    
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=3e-4, 
                 gamma=0.99, eps_clip=0.2, k_epochs=4, hidden_dim=256):
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        self.actor = PPOActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = PPOCritic(state_dim, hidden_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Storage for training data
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        
    def select_action(self, state):
        """Select action using current policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob = self.actor.get_action(state_tensor)
            
        return action, log_prob.cpu().numpy()
    
    def store_transition(self, state, action, log_prob, reward, done):
        """Store transition for batch training"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.is_terminals.append(done)
    
    def clear_memory(self):
        """Clear stored transitions"""
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]
    
    def compute_discounted_rewards(self):
        """Compute discounted rewards for the trajectory"""
        discounted_rewards = []
        discounted_reward = 0
        
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)
            
        return discounted_rewards
    
    def update(self):
        """Update the policy and value networks using PPO"""
        old_states = torch.FloatTensor(np.array(self.states)).to(self.device)
        old_actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        discounted_rewards = self.compute_discounted_rewards()
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        state_values = self.critic(old_states).squeeze()
        advantages = discounted_rewards - state_values.detach()
        
        for _ in range(self.k_epochs):
            action_probs = self.actor(old_states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(old_actions)
            entropy = dist.entropy()
            
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Update critic
            state_values = self.critic(old_states).squeeze()
            critic_loss = F.mse_loss(state_values, discounted_rewards)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
    
    def save_model(self, filepath):
        """Save model weights"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
        
    def load_model(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


class TrainingManager:
    """Manages the training process and logging"""
    
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        self.setup_logging()
        
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rate = deque(maxlen=100)
        
        self.best_reward = -float('inf')
        self.best_success_rate = 0.0
        
    def setup_directories(self):
        """Create necessary directories for saving models and logs"""
        self.run_dir = Path(f"runs/ppo_nav_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_dir = self.run_dir / "models"
        self.model_dir.mkdir(exist_ok=True)
        
        self.log_dir = self.run_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
    def setup_logging(self):
        """Setup logging for training metrics"""
        self.metrics = {
            'episode': [],
            'reward': [],
            'length': [], 
            'success': [],
            'avg_reward': [],
            'success_rate': []
        }
        
    def log_episode(self, episode, reward, length, success):
        """Log episode results"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.success_rate.append(1.0 if success else 0.0)
        
        avg_reward = np.mean(self.episode_rewards)
        success_rate = np.mean(self.success_rate)
        
        self.metrics['episode'].append(episode)
        self.metrics['reward'].append(reward)
        self.metrics['length'].append(length)
        self.metrics['success'].append(success)
        self.metrics['avg_reward'].append(avg_reward)
        self.metrics['success_rate'].append(success_rate)
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode:4d} | Reward: {reward:7.2f} | Avg Reward: {avg_reward:7.2f} | "
                  f"Length: {length:3d} | Success Rate: {success_rate:.2f}")
        
        # Save best models
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            
    def save_metrics(self):
        """Save training metrics to file"""
        with open(self.log_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def plot_metrics(self):
        """Plot training metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        ax1.plot(self.metrics['episode'], self.metrics['reward'], alpha=0.6, label='Episode Reward')
        ax1.plot(self.metrics['episode'], self.metrics['avg_reward'], label='Average Reward', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.metrics['episode'], self.metrics['length'], alpha=0.6)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Lengths')
        ax2.grid(True)
        
        ax3.plot(self.metrics['episode'], self.metrics['success_rate'], linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Success Rate')
        ax3.set_title('Success Rate (100-episode average)')
        ax3.grid(True)
        
        ax4_twin = ax4.twinx()
        ax4.plot(self.metrics['episode'], self.metrics['avg_reward'], 'b-', label='Avg Reward')
        ax4_twin.plot(self.metrics['episode'], self.metrics['success_rate'], 'r-', label='Success Rate')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Average Reward', color='b')
        ax4_twin.set_ylabel('Success Rate', color='r')
        ax4.set_title('Training Progress')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()


def train_ppo_navigation():
    """Main training function"""
    
    config = {
        'total_episodes': 2000,
        'max_steps_per_episode': 60,
        'update_frequency': 20,  # Update every N episodes
        'save_frequency': 100,   # Save model every N episodes
        'headless': True,        # Run simulation in headless mode for speed
        'random_seed': 42,
        
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'gamma': 0.99,
        'eps_clip': 0.2,
        'k_epochs': 4,
        'hidden_dim': 256,
    }
    
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    random.seed(config['random_seed'])
    
    manager = TrainingManager(config)
    
    with open(manager.run_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("="*60)
    print("Starting PPO Training for Galbot Navigation")
    print("="*60)
    print(f"Total Episodes: {config['total_episodes']}")
    print(f"Max Steps per Episode: {config['max_steps_per_episode']}")
    print(f"Update Frequency: {config['update_frequency']}")
    print(f"Model Directory: {manager.model_dir}")
    print("="*60)
    
    # Initialize environment and agent
    try:
        env = IoaiNavEnv(headless=config['headless'], seed=config['random_seed'])
        print("Environment initialized successfully")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        return
    
    # Get environment dimensions
    state_dim = 7  # Based on observation function: [x, y, yaw, goal_x, goal_y, dist_to_goal, normalized_steps]
    action_dim = 6  # 6 possible actions: forward, backward, left, right, yaw+, yaw-
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=config['lr_actor'],
        lr_critic=config['lr_critic'],
        gamma=config['gamma'],
        eps_clip=config['eps_clip'],
        k_epochs=config['k_epochs'],
        hidden_dim=config['hidden_dim']
    )
    
    print(f"Agent initialized - State dim: {state_dim}, Action dim: {action_dim}")
    
    try:
        for episode in range(1, config['total_episodes'] + 1):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(config['max_steps_per_episode']):
                action, log_prob = agent.select_action(state)
                
                next_state, reward, done, info = env.step(action)
                
                agent.store_transition(state, action, log_prob, reward, done)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # Determine if episode was successful (goal reached)
            success = info.get('distance_to_goal', float('inf')) < 0.15
            
            manager.log_episode(episode, episode_reward, episode_length, success)
            
            if episode % config['update_frequency'] == 0:
                agent.update()
                agent.clear_memory()
                print(f"Agent updated at episode {episode}")
            
            # Save model periodically
            if episode % config['save_frequency'] == 0:
                model_path = manager.model_dir / f"ppo_model_episode_{episode}.pt"
                agent.save_model(model_path)
                print(f"Model saved: {model_path}")
                
                # Save and plot metrics
                manager.save_metrics()
                manager.plot_metrics()
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        final_model_path = manager.model_dir / "ppo_model_final.pt"
        agent.save_model(final_model_path)
        manager.save_metrics()
        manager.plot_metrics()
        
        print(f"\nTraining completed!")
        print(f"Final model saved: {final_model_path}")
        print(f"Best average reward: {manager.best_reward:.2f}")
        print(f"Best success rate: {manager.best_success_rate:.2f}")
        
        # Clean up environment
        try:
            env.simulator.close()
        except:
            pass


def test_trained_model(model_path, num_episodes=10, headless=False):
    """Test a trained model"""
    
    # Initialize environment and agent
    env = IoaiNavEnv(headless=headless, seed=42)
    state_dim = 7
    action_dim = 6
    
    agent = PPOAgent(state_dim, action_dim)
    agent.load_model(model_path)
    
    print(f"Testing model: {model_path}")
    print(f"Number of test episodes: {num_episodes}")
    
    successes = 0
    total_reward = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(60):  # Max steps
            action, _ = agent.select_action(state)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        success = info.get('distance_to_goal', float('inf')) < 0.15
        if success:
            successes += 1
            
        total_reward += episode_reward
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Success = {success}")
    
    success_rate = successes / num_episodes
    avg_reward = total_reward / num_episodes
    
    print(f"\nTest Results:")
    print(f"Success Rate: {success_rate:.2f} ({successes}/{num_episodes})")
    print(f"Average Reward: {avg_reward:.2f}")
    
    env.simulator.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PPO Training for Galbot Navigation')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                        help='Mode: train or test')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model for testing')
    parser.add_argument('--test_episodes', type=int, default=10,
                        help='Number of episodes for testing')
    parser.add_argument('--headless', action='store_true',
                        help='Run in headless mode')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_ppo_navigation()
    elif args.mode == 'test':
        if args.model_path is None:
            print("Error: --model_path required for testing mode")
        else:
            test_trained_model(args.model_path, args.test_episodes, args.headless)
