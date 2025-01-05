import time
from typing import Dict, Any
import json
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.start_time = time.time()
        self.metrics = {
            'train_rewards': [],
            'eval_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'steps': []
        }
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
    def log_step(self, step: int, info: Dict[str, Any]) -> None:
        """记录单步训练信息"""
        for key, value in info.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        self.metrics['steps'].append(step)
        
    def log_episode(self, episode: int, rewards: float, info: Dict[str, Any]) -> None:
        """记录回合信息"""
        print(f"Episode {episode}: Reward = {rewards:.2f}")
        for key, value in info.items():
            print(f"{key}: {value:.4f}")
            
    def save_metrics(self) -> None:
        """保存训练指标"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.log_dir, f"metrics_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=4)