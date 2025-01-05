from typing import Dict, List, Any
import numpy as np
import torch

class Evaluator:
    def __init__(self, env: Any, agent: Any):
        self.env = env
        self.agent = agent
        
    def evaluate(self, 
                num_episodes: int = 10, 
                render: bool = False) -> Dict[str, float]:
        """评估智能体性能"""
        rewards = []
        lengths = []
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                if render:
                    self.env.render()
                    
                action = self.agent.select_action(obs, evaluate=True)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                
            rewards.append(episode_reward)
            lengths.append(episode_length)
            
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths)
        }