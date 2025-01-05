from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from rich.progress import Progress, track
import matplotlib.pyplot as plt

@dataclass
class BifurcationParameters:
    """Parameters for bifurcation analysis"""
    param_name: str          # Name of bifurcation parameter
    param_range: np.ndarray  # Range of parameter values
    num_points: int         # Number of points in parameter range
    max_iter: int          # Maximum iterations per point
    tolerance: float       # Convergence tolerance
    initial_conditions: np.ndarray  # Initial state

class BifurcationAnalyzer:
    """Class for performing bifurcation analysis"""
    
    def __init__(self, system_model: Any, params: BifurcationParameters):
        self.system_model = system_model
        self.params = params
        self.results = {
            'parameter_values': [],
            'equilibrium_points': [],
            'stability_types': [],
            'eigenvalues': [],
            'bifurcation_points': [],
            'periodic_solutions': []
        }

    def analyze_hopf_bifurcation(self, system_matrix: np.ndarray, parameter_range: np.ndarray) -> List[float]:
        """
        Analyze Hopf bifurcation in the system
        
        Args:
            system_matrix: Jacobian matrix of the system
            parameter_range: Control parameter range
            
        Returns:
            List of bifurcation points
        """
        bifurcation_points = []
        
        for param in parameter_range:
            # Calculate eigenvalues
            eigenvals = np.linalg.eigvals(system_matrix(param))
            
            # Hopf bifurcation conditions
            pure_imaginary_pair = False
            others_negative = True
            
            for eigenval in eigenvals:
                if abs(eigenval.real) < 1e-10 and abs(eigenval.imag) > 1e-10:
                    pure_imaginary_pair = True
                elif eigenval.real > 0:
                    others_negative = False
                    
            if pure_imaginary_pair and others_negative:
                bifurcation_points.append(param)
                
        return bifurcation_points

    def detect_periodic_solution(self, time_series: np.ndarray, time: np.ndarray) -> Tuple[bool, Optional[float]]:
        """
        Detect periodic solutions
        
        Args:
            time_series: Time series data
            time: Time points
            
        Returns:
            Tuple containing:
            - Boolean indicating if periodic solution exists
            - Frequency of oscillation (if periodic)
        """
        peaks, _ = find_peaks(time_series)
        
        if len(peaks) >= 2:
            peak_periods = np.diff(time[peaks])
            
            if np.std(peak_periods) / np.mean(peak_periods) < 0.1:
                frequency = 1.0 / np.mean(peak_periods)
                return True, frequency
        
        return False, None

    def comprehensive_hopf_analysis(self, 
                                  q_rail_history: List[List[float]], 
                                  q_sea_history: List[List[float]], 
                                  simulation_time: np.ndarray, 
                                  control_parameter: float) -> Dict[str, Any]:
        """
        Comprehensive analysis of system Hopf bifurcation
        
        Args:
            q_rail_history: Rail flow history
            q_sea_history: Sea flow history
            simulation_time: Time points
            control_parameter: Control parameter value
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            'has_periodic_solution': False,
            'frequency': None,
            'amplitude_growth': False,
            'center_manifold': False
        }
        
        # Detect periodic solutions
        for chain_idx in range(len(q_rail_history[0])):
            chain_flow = [step[chain_idx][0] for step in q_rail_history]
            has_periodic, freq = self.detect_periodic_solution(chain_flow, simulation_time)
            if has_periodic:
                results['has_periodic_solution'] = True
                results['frequency'] = freq
        
        # Check amplitude growth
        start_amplitude = np.std([step[0][0] for step in q_rail_history[:int(len(simulation_time)/4)]])
        end_amplitude = np.std([step[0][0] for step in q_rail_history[int(3*len(simulation_time)/4):]])
        results['amplitude_growth'] = end_amplitude > start_amplitude * 1.1
        
        # Phase plane analysis
        chain_flow = [step[0][0] for step in q_rail_history]
        phase_space = np.array([chain_flow[:-1], chain_flow[1:]]).T
        
        start_point = phase_space[0]
        end_point = phase_space[-1]
        results['center_manifold'] = np.linalg.norm(end_point - start_point) < 0.1
        
        return results

    def plot_bifurcation_diagram(self, 
                                q_rail_history: List[List[float]], 
                                q_sea_history: List[List[float]], 
                                simulation_time: np.ndarray, 
                                control_parameter_range: np.ndarray) -> None:
        """
        Plot bifurcation diagram
        
        Args:
            q_rail_history: Rail flow history
            q_sea_history: Sea flow history
            simulation_time: Time points
            control_parameter_range: Range of control parameter values
        """
        plt.figure(figsize=(12, 8))
        
        steady_state_start = int(len(simulation_time) * 0.7)
        
        for param_idx, param in enumerate(control_parameter_range):
            for chain_idx in range(len(q_rail_history[0])):
                chain_flow = [step[chain_idx][0] for step in q_rail_history[steady_state_start:]]
                
                maxima = [max(chain_flow)]
                minima = [min(chain_flow)]
                
                plt.plot([param] * len(maxima), maxima, 'b.', markersize=1)
                plt.plot([param] * len(minima), minima, 'r.', markersize=1)
        
        plt.xlabel('Control Parameter')
        plt.ylabel('Flow')
        plt.title('Bifurcation Diagram with Extrema')
        plt.savefig('hopf_bifurcation_diagram.png')
        plt.close()

    def analyze_stability(self, time_series: np.ndarray) -> str:
        """
        Analyze stability of time series
        
        Args:
            time_series: Time series data
            
        Returns:
            Stability type ('stable', 'periodic', or 'chaotic')
        """
        variance = np.var(time_series)
        if variance < 0.01:
            return 'stable'
        else:
            return 'periodic'