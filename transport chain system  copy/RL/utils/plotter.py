import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import os

class Plotter:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_training_curves(self, metrics: Dict[str, List[float]]) -> None:
        """绘制训练曲线"""
        # 奖励曲线
        plt.figure(figsize=(10, 5))
        plt.plot(metrics['steps'], metrics['train_rewards'], label='Train')
        if 'eval_rewards' in metrics:
            plt.plot(metrics['steps'], metrics['eval_rewards'], label='Eval')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.title('Training Rewards')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'rewards.png'))
        plt.close()
        
        # 损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(metrics['steps'], metrics['actor_losses'], label='Actor')
        plt.plot(metrics['steps'], metrics['critic_losses'], label='Critic')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'losses.png'))
        plt.close()
        
    def plot_episode_stats(self, 
                          episode_lengths: List[int], 
                          episode_rewards: List[float]) -> None:
        """绘制回合统计信息"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # 回合长度
        ax1.plot(episode_lengths)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Length')
        ax1.set_title('Episode Lengths')
        
        # 回合奖励
        ax2.plot(episode_rewards)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.set_title('Episode Rewards')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'episode_stats.png'))
        plt.close()