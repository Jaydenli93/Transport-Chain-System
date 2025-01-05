"""
Configuration classes for the RL module
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class Config:
    """Base configuration class"""
    pass

@dataclass
class TrainingConfig(Config):
    """Training configuration"""
    num_episodes: int
    max_steps: int
    learning_rate: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    batch_size: int
    memory_size: int
    target_update: int

@dataclass
class SystemConfig(Config):
    """System configuration"""
    state_dim: int
    action_dim: int
    hidden_dim: int
    num_layers: int

@dataclass
class AgentConfig:
    """智能体配置"""
    learning_rate: float = 3e-4
    batch_size: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2

@dataclass
class EnvironmentConfig:
    """环境配置"""
    env_name: str = 'CartPole-v1'
    max_episode_steps: int = 500
    frame_stack: int = 3
