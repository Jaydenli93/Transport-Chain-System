from typing import Dict, Tuple
import numpy as np

class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, max_size: int = 1_000_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # 预分配内存
        self.observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)

    def add(self, obs: np.ndarray, 
            action: np.ndarray, 
            next_obs: np.ndarray, 
            reward: float, 
            done: bool) -> None:
        """添加一条经验"""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.next_observations[self.ptr] = next_obs
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """采样一个批次的经验"""
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.observations[ind],
            self.actions[ind],
            self.next_observations[ind],
            self.rewards[ind],
            self.dones[ind]
        )

    def __len__(self) -> int:
        return self.size