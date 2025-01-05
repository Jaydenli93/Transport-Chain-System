from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import gym
from gym import spaces
from dataclasses import dataclass

@dataclass
class EnvParameters:
    """Environment parameters"""
    max_steps: int = 1000           # Maximum steps per episode
    reward_scale: float = 1.0       # Reward scaling factor
    time_penalty: float = 0.1       # Penalty for each time step
    disruption_prob: float = 0.05   # Probability of random disruption
    observation_noise: float = 0.01  # Observation noise level

class TransportEnv(gym.Env):
    """Transport chain environment for reinforcement learning"""
    
    def __init__(self, 
                 system_model: Any,         # Transport system model
                 price_model: Any,          # Price calculation model
                 history: Any,             # Transport history manager
                 params: EnvParameters):   # Environment parameters
        """
        Initialize environment
        
        Args:
            system_model: Transport system model
            price_model: Price calculation model
            history: Transport history manager
            params: Environment parameters
        """
        super().__init__()
        self.system_model = system_model
        self.price_model = price_model
        self.history = history
        self.params = params
        
        # Define action space (normalized control inputs)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._get_action_dim(),),
            dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._get_observation_dim(),),
            dtype=np.float32
        )
        
        self.reset()
        
    # 2.Core methods
    # a.Reset environment
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state
        
        Returns:
            Initial observation
        """
        # Reset system model and history
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Get initial state from system model
        self.state = self.system_model.get_initial_state()
        
        # Initialize history with initial state
        self.history.add_record(
            t=0.0,
            flows=self.state['flows'],
            densities=self.state['densities'],
            prices=self.state['prices'],
            costs=self.state['costs']
        )
        
        return self._get_observation()

    # b.Take environment step   
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take environment step
        
        Args:
            action: Normalized action vector
            
        Returns:
            Tuple containing:
            - Next observation
            - Reward
            - Done flag
            - Info dictionary
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError("Invalid action")
            
        # Convert normalized action to actual control inputs
        control_inputs = self._denormalize_action(action)
        
        # Apply random disruptions
        if np.random.random() < self.params.disruption_prob:
            self._apply_disruption()
        
        # Update system state
        try:
            next_state = self.system_model.step(
                self.state,
                control_inputs,
                self.current_step * self.params.time_penalty,
                self.price_model
            )
            
            # Update history
            self.history.add_record(
                t=self.current_step * self.params.time_penalty,
                flows=next_state['flows'],
                densities=next_state['densities'],
                prices=next_state['prices'],
                costs=next_state['costs']
            )
            
            self.state = next_state
            
        except Exception as e:
            print(f"System update failed: {str(e)}")
            return self._get_observation(), -100.0, True, {'error': str(e)}
        
        # Calculate reward
        reward = self._compute_reward(control_inputs)
        
        # Update step counter
        self.current_step += 1
        self.episode_reward += reward
        
        # Check if episode is done
        done = self._is_done()
        
        # Get info dictionary
        info = self._get_info()
        
        return self._get_observation(), reward, done, info

    def _get_action_dim(self) -> int:
        """Get dimension of action space"""
        return len(self.state['flows']['rail']) + len(self.state['flows']['sea'])

    # 4. Helper methods
    # a. Get observation dimension
    def _get_observation_dim(self) -> int:
        """Get dimension of observation space"""
        return (len(self.state['densities']['rail']) + 
                len(self.state['densities']['sea']) + 
                len(self.state['flows']['rail']) + 
                len(self.state['flows']['sea']))

    # b. Get observation
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation with noise
        
        Returns:
            Observation vector
        """
        obs = np.concatenate([
            self.state['densities']['rail'].flatten(),
            self.state['densities']['sea'].flatten(),
            self.state['flows']['rail'].flatten(),
            self.state['flows']['sea'].flatten()
        ])
        
        # Add observation noise
        noise = np.random.normal(
            0, 
            self.params.observation_noise, 
            size=obs.shape
        )
        return obs + noise
        
    def _denormalize_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert normalized actions to control inputs"""
        n_rail = len(self.state['flows']['rail'])
        return {
            'rail': action[:n_rail],
            'sea': action[n_rail:]
        }
    
    # b. Compute reward
    def _compute_reward(self, control_inputs: Dict[str, np.ndarray]) -> float:
        """
        Compute reward for current state and action
        
        Args:
            control_inputs: Applied control inputs
            
        Returns:
            float: Reward value
        """
        # Calculate base reward from system performance
        performance_reward = -np.sum(list(self.state['costs'].values()))
        
        # Add time penalty
        time_penalty = -self.params.time_penalty
        
        # Add control cost
        control_cost = -0.1 * (np.sum(control_inputs['rail']**2) + 
                              np.sum(control_inputs['sea']**2))
        
        # Combine rewards
        total_reward = (
            performance_reward + 
            time_penalty + 
            control_cost
        ) * self.params.reward_scale
        
        return float(total_reward)
        
    def _is_done(self) -> bool:
        """Check if episode is done"""
        # Check step limit
        if self.current_step >= self.params.max_steps:
            return True
            
        # Check system stability
        if not self.system_model.check_stability(self.state):
            return True
            
        return False
        
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information"""
        return {
            'episode_reward': self.episode_reward,
            'current_step': self.current_step,
            'system_state': self.state
        }
        
    def _apply_disruption(self) -> None:
        """Apply random disruption to system"""
        self.system_model.apply_disruption(self.state)