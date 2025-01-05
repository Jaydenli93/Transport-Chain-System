"""
Reinforcement Learning Module for Transport Chain Control
======================================================

This module provides RL components for transport chain optimization:
- Environment: OpenAI Gym compatible environment
- Agent: TD3 (Twin Delayed Deep Deterministic Policy Gradient) implementation
- Training utilities and configurations
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import os
from models import SystemModel

# 导入所有子模块
from .environment import TransportEnv
from .agent import TD3Agent
from .buffer import ReplayBuffer
from .config import TrainingConfig, SystemConfig, AgentConfig, EnvironmentConfig
from .experiment import Experiment, create_rl_system, train_agent, evaluate_agent
from .evaluation import Evaluator
from .exceptions import RLError, TrainingError, EnvironmentError, ModelError
from .utils.helpers import set_random_seed, create_experiment_dir, save_config
from .utils.logger import Logger
from .utils.plotter import Plotter

# Version info
__version__ = '1.0.0'
__author__ = 'Jayden Li'
__email__ = 'jaydenlii93@gmail.com'

__all__ = [
    'TransportEnv',
    'TD3Agent',
    'ReplayBuffer',
    'Experiment',
    'Evaluator',
    'TrainingConfig',
    'SystemConfig',
    'AgentConfig',
    'EnvironmentConfig',
    'create_rl_system',
    'train_agent',
    'evaluate_agent'
]

def create_rl_system(
    system_model: Any,
    price_model: Any,
    history: Any,
    env_config: Optional[Dict[str, Any]] = None,
    agent_config: Optional[Dict[str, Any]] = None,
    system_config: Optional[SystemConfig] = None
) -> Tuple[TransportEnv, TD3Agent]:
    """创建 RL 系统"""
    
    # 创建环境
    env = TransportEnv(
        system_model=system_model,
        price_model=price_model,
        history=history,
        params=EnvironmentConfig(**(env_config or {}))
    )
    
    # 如果没有提供系统配置，创建默认配置
    if system_config is None:
        system_config = SystemConfig(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0]
        )
    
    # 创建智能体
    agent = TD3Agent(
        obs_dim=system_config.obs_dim,
        action_dim=system_config.action_dim,
        params=AgentConfig(**(agent_config or {}))
    )
    
    return env, agent

def train_agent(
    env: TransportEnv,
    agent: TD3Agent,
    config: TrainingConfig,
    experiment_dir: Optional[str] = None
) -> Dict[str, Any]:
    """训练智能体"""
    try:
        # 设置随机种子
        set_random_seed(config.random_seed, env)
        
        # 创建实验目录
        if experiment_dir is None:
            experiment_dir = create_experiment_dir('experiments')
            
        # 创建实验
        experiment = Experiment(
            config=config,
            env=env,
            agent=agent
        )
        
        # 运行实验
        results = experiment.run()
        return results
        
    except Exception as e:
        raise TrainingError(f"Training failed: {str(e)}")

def evaluate_agent(
    env: TransportEnv,
    agent: TD3Agent,
    num_episodes: int = 10,
    render: bool = False
) -> Dict[str, float]:
    """评估智能体"""
    try:
        evaluator = Evaluator(env, agent)
        results = evaluator.evaluate(
            num_episodes=num_episodes,
            render=render
        )
        return results
        
    except Exception as e:
        raise EnvironmentError(f"Evaluation failed: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # 配置参数
    training_config = TrainingConfig()
    system_config = SystemConfig()
    
    try:
        # 创建系统
        env, agent = create_rl_system(
            system_model=None,  # 需要提供具体的模型
            price_model=None,   # 需要提供具体的模型
            history=None,       # 需要提供具体的历史记录管理器
            system_config=system_config
        )
        
        # 训练智能体
        results = train_agent(env, agent, training_config)
        
        # 评估智能体
        eval_results = evaluate_agent(env, agent)
        
        print("Training completed successfully!")
        print(f"Final evaluation results: {eval_results}")
        
    except RLError as e:
        print(f"Error occurred: {str(e)}")