from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from dataclasses import dataclass
from collections import deque
import random

@dataclass
class AgentParameters:
    """Agent parameters for TD3"""
    hidden_dim: int = 256        # Hidden layer dimension
    learning_rate: float = 3e-4  # Learning rate
    gamma: float = 0.99         # Discount factor
    tau: float = 0.005         # Target network update rate
    batch_size: int = 256      # Batch size for training
    buffer_size: int = 1000000  # Replay buffer size
    policy_noise: float = 0.2   # Noise added to target actions
    noise_clip: float = 0.5     # Range to clip target policy noise
    policy_delay: int = 2       # Frequency of delayed policy updates
    exploration_noise: float = 0.1  # Exploration noise
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class Actor(nn.Module):
    """Actor network for TD3"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.net(obs)

class Critic(nn.Module):
    """Critic network for TD3"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)

class TD3Agent:
    """Twin Delayed DDPG agent"""
    
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 params: AgentParameters):
        """Initialize TD3 agent"""
        self.params = params
        self.device = torch.device(params.device)
        self.action_dim = action_dim
        
        # Initialize actor networks
        self.actor = Actor(obs_dim, action_dim, params.hidden_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim, params.hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Initialize critic networks
        self.critic_1 = Critic(obs_dim, action_dim, params.hidden_dim).to(self.device)
        self.critic_2 = Critic(obs_dim, action_dim, params.hidden_dim).to(self.device)
        self.critic_1_target = Critic(obs_dim, action_dim, params.hidden_dim).to(self.device)
        self.critic_2_target = Critic(obs_dim, action_dim, params.hidden_dim).to(self.device)
        
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=params.learning_rate)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=params.learning_rate)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=params.learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=params.buffer_size)
        
        # Initialize training variables
        self.total_steps = 0
        self.training_info = {
            'actor_loss': [],
            'critic_loss': [],
            'episode_rewards': []
        }
        
    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action using policy"""
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            action = self.actor(obs).cpu().numpy()
            
            if not evaluate:
                noise = np.random.normal(0, self.params.exploration_noise, 
                                       size=self.action_dim)
                action = np.clip(action + noise, -1, 1)
                
        return action
        
    def train(self, batch: Tuple) -> Dict[str, float]:
        """Train agent on batch of experiences"""
        obs, action, reward, next_obs, done = [torch.FloatTensor(x).to(self.device) 
                                             for x in batch]
        
        # Select next action with noise
        noise = (torch.randn_like(action) * self.params.policy_noise
                ).clamp(-self.params.noise_clip, self.params.noise_clip)
                
        next_action = (self.actor_target(next_obs) + noise).clamp(-1, 1)
        
        # Compute target Q-values
        target_q1 = self.critic_1_target(next_obs, next_action)
        target_q2 = self.critic_2_target(next_obs, next_action)
        target_q = torch.min(target_q1, target_q2)
        target_q = reward + (1 - done) * self.params.gamma * target_q
        
        # Update critics
        current_q1 = self.critic_1(obs, action)
        current_q2 = self.critic_2(obs, action)
        
        critic_1_loss = nn.MSELoss()(current_q1, target_q.detach())
        critic_2_loss = nn.MSELoss()(current_q2, target_q.detach())
        
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        
        # Delayed policy updates
        actor_loss = 0
        if self.total_steps % self.params.policy_delay == 0:
            # Update actor
            actor_loss = -self.critic_1(obs, self.actor(obs)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target networks
            self._soft_update_target()
        
        # Update training info
        self.total_steps += 1
        self.training_info['actor_loss'].append(actor_loss.item())
        self.training_info['critic_loss'].append(
            (critic_1_loss.item() + critic_2_loss.item()) / 2
        )
        
        return {
            'critic_1_loss': critic_1_loss.item(),
            'critic_2_loss': critic_2_loss.item(),
            'actor_loss': actor_loss if isinstance(actor_loss, float) else actor_loss.item()
        }
        
    def _soft_update_target(self) -> None:
        """Soft update target networks"""
        for param, target_param in zip(self.actor.parameters(), 
                                     self.actor_target.parameters()):
            target_param.data.copy_(
                self.params.tau * param.data + 
                (1 - self.params.tau) * target_param.data
            )
            
        for param, target_param in zip(self.critic_1.parameters(), 
                                     self.critic_1_target.parameters()):
            target_param.data.copy_(
                self.params.tau * param.data + 
                (1 - self.params.tau) * target_param.data
            )
            
        for param, target_param in zip(self.critic_2.parameters(), 
                                     self.critic_2_target.parameters()):
            target_param.data.copy_(
                self.params.tau * param.data + 
                (1 - self.params.tau) * target_param.data
            )