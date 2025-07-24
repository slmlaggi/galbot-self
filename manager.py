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
# Description: Manager script for Galbot navigation training system
# Author: IOAI Training System  
# Date: 2025-01-24
#
#####################################################################################

import sys
import os
import json
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime


class TrainingManager:
    """
    High-level manager for controlling the training process.
    Handles experiment configuration, monitoring, and coordination.
    """
    
    def __init__(self):
        self.config_dir = Path("configs")
        self.config_dir.mkdir(exist_ok=True)
        
    def create_experiment_config(self, experiment_name, **kwargs):
        """Create a new experiment configuration"""
        
        default_config = {
            "experiment_name": experiment_name,
            "total_episodes": 2000,
            "max_steps_per_episode": 60,
            "update_frequency": 20,
            "save_frequency": 100,
            "headless": True,
            "random_seed": 42,
            
            # PPO Hyperparameters
            "lr_actor": 3e-4,
            "lr_critic": 3e-4, 
            "gamma": 0.99,
            "eps_clip": 0.2,
            "k_epochs": 4,
            "hidden_dim": 256,
            
            # Environment settings
            "step_distance": 0.2,
            "goal_tolerance": 0.15,
            "wall_boundaries": {
                "x_min": -3.5,
                "x_max": 5.5,
                "y_min": -4.5, 
                "y_max": 4.5
            },
            
            # Training optimizations
            "early_stopping_episodes": 100,  # Stop if no improvement
            "target_success_rate": 0.8,      # Success rate to aim for
            "min_episodes_before_stop": 500  # Minimum episodes before early stopping
        }
        
        # Update with user provided parameters
        default_config.update(kwargs)
        
        # Save config
        config_path = self.config_dir / f"{experiment_name}.json"
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
            
        print(f"Experiment config created: {config_path}")
        return config_path
    
    def list_experiments(self):
        """List available experiment configurations"""
        configs = list(self.config_dir.glob("*.json"))
        
        if not configs:
            print("No experiment configurations found.")
            return []
            
        print("Available experiments:")
        for i, config_path in enumerate(configs, 1):
            print(f"  {i}. {config_path.stem}")
            
        return configs
    
    def run_experiment(self, config_path):
        """Run a training experiment with the given configuration"""
        
        if not Path(config_path).exists():
            print(f"Configuration file not found: {config_path}")
            return False
            
        print(f"Starting experiment with config: {config_path}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run training script with config
        cmd = [
            sys.executable, "training.py",
            "--mode", "train",
            "--config", str(config_path)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print("Experiment completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Experiment failed with error code: {e.returncode}")
            return False
        except KeyboardInterrupt:
            print("Experiment interrupted by user")
            return False
    
    def monitor_training(self, run_dir):
        """Monitor an ongoing training run"""
        
        run_path = Path(run_dir)
        if not run_path.exists():
            print(f"Run directory not found: {run_dir}")
            return
            
        metrics_file = run_path / "logs" / "metrics.json"
        
        print(f"Monitoring training run: {run_dir}")
        print("Press Ctrl+C to stop monitoring\n")
        
        last_episode = 0
        
        try:
            while True:
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    if metrics['episode']:
                        current_episode = metrics['episode'][-1]
                        
                        if current_episode > last_episode:
                            last_episode = current_episode
                            
                            # Display latest metrics
                            latest_reward = metrics['reward'][-1]
                            latest_success_rate = metrics['success_rate'][-1]
                            avg_reward = metrics['avg_reward'][-1]
                            
                            print(f"Episode {current_episode:4d} | "
                                  f"Reward: {latest_reward:7.2f} | "
                                  f"Avg Reward: {avg_reward:7.2f} | "
                                  f"Success Rate: {latest_success_rate:.2f}")
                
                time.sleep(5)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            print("\nStopped monitoring")
        except FileNotFoundError:
            print("Metrics file not found. Training may not have started yet.")
        except json.JSONDecodeError:
            print("Error reading metrics file. Training may be in progress.")


def main():
    """Main function for the training manager"""
    
    parser = argparse.ArgumentParser(description='Galbot Navigation Training Manager')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create experiment command
    create_parser = subparsers.add_parser('create', help='Create a new experiment configuration')
    create_parser.add_argument('name', help='Experiment name')
    create_parser.add_argument('--episodes', type=int, default=2000, help='Total training episodes')
    create_parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    create_parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    create_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # List experiments command
    list_parser = subparsers.add_parser('list', help='List available experiments')
    
    # Run experiment command
    run_parser = subparsers.add_parser('run', help='Run an experiment')
    run_parser.add_argument('config', help='Path to experiment configuration file')
    
    # Monitor training command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor ongoing training')
    monitor_parser.add_argument('run_dir', help='Path to training run directory')
    
    # Quick start command
    quick_parser = subparsers.add_parser('quick', help='Quick start with default settings')
    quick_parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    quick_parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = TrainingManager()
    
    if args.command == 'create':
        # Create new experiment configuration
        config_kwargs = {
            'total_episodes': args.episodes,
            'lr_actor': args.lr,
            'lr_critic': args.lr,
            'headless': args.headless,
            'random_seed': args.seed
        }
        manager.create_experiment_config(args.name, **config_kwargs)
        
    elif args.command == 'list':
        # List available experiments
        manager.list_experiments()
        
    elif args.command == 'run':
        # Run specified experiment
        manager.run_experiment(args.config)
        
    elif args.command == 'monitor':
        # Monitor training progress
        manager.monitor_training(args.run_dir)
        
    elif args.command == 'quick':
        # Quick start with default configuration
        config_name = f"quick_start_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config_kwargs = {
            'total_episodes': args.episodes,
            'headless': args.headless
        }
        config_path = manager.create_experiment_config(config_name, **config_kwargs)
        
        print("Starting quick training...")
        manager.run_experiment(config_path)


if __name__ == "__main__":
    main()
