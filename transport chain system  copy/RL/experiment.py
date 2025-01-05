"""
Experiment management for RL training
"""

from typing import Dict, Any,Tuple
import os
import json
from .config import Config, TrainingConfig, SystemConfig, AgentConfig, EnvironmentConfig
from .agent import train_agent
from .utils.logger import Logger
from .utils.plotter import Plotter
from .evaluation import Evaluator
from .utils.helpers import create_experiment_dir, save_config

class Experiment:
    def __init__(self, 
                 config: Config,
                 env: Any,
                 agent: Any):
        self.config = config
        self.env = env
        self.agent = agent
        self.exp_dir = create_experiment_dir(config.save_dir)
        
        # 初始化组件
        self.logger = Logger(os.path.join(self.exp_dir, 'logs'))
        self.plotter = Plotter(os.path.join(self.exp_dir, 'plots'))
        self.evaluator = Evaluator(env, agent)
        
        # 保存配置
        save_config(config, os.path.join(self.exp_dir, 'config.json'))
        
    def run(self) -> Dict[str, Any]:
        """运行实验"""
        metrics = train_agent(
            env=self.env,
            agent=self.agent,
            total_steps=self.config.total_steps,
            eval_interval=self.config.eval_interval,
            logger=self.logger,
            plotter=self.plotter
        )
        
        # 最终评估
        eval_results = self.evaluator.evaluate(
            num_episodes=self.config.num_eval_episodes
        )
        
        # 保存结果
        results = {
            'training_metrics': metrics,
            'eval_results': eval_results
        }
        
        with open(os.path.join(self.exp_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        return results