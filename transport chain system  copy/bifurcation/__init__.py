"""
Transport Chain Analysis Tools
============================

This package provides tools for analyzing transport chain dynamics:
- Bifurcation analysis
- Hopf bifurcation detection
- Stability analysis
- Periodic solution detection
- System behavior visualization
"""

# Version information
__version__ = '1.0.0'
__author__ = 'Jayden Li'
__email__ = 'jaydenlii93@gmail.com'

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from dataclasses import dataclass
from models import SystemModel

# Default analysis parameters
DEFAULT_PARAMS = {
    'num_points': 100,
    'max_iter': 1000,
    'tolerance': 1e-6,
    'control_param_range': (0.1, 30.0),
    'num_control_points': 50,
    'stability_threshold': 0.01,
    'periodic_threshold': 0.1,
    'initial_conditions': None
}

@dataclass
class BifurcationParameters:
    param_name: str
    param_range: np.ndarray
    num_points: int = 100
    max_iter: int = 1000
    tolerance: float = 1e-6
    initial_conditions: Optional[np.ndarray] = None

class BifurcationAnalyzer:
    def __init__(self, system_model: Any, params: BifurcationParameters):
        self.system_model = system_model
        self.params = params
        
    def analyze_bifurcation(self, *args, **kwargs) -> Tuple[List[float], List[str]]:
        from .bifurcation import analyze_bifurcation
        return analyze_bifurcation(*args, **kwargs)
    
    def analyze_hopf_bifurcation(self, *args, **kwargs) -> Dict[str, Any]:
        from .bifurcation import analyze_hopf_bifurcation
        return analyze_hopf_bifurcation(*args, **kwargs)
    
    def comprehensive_hopf_analysis(self, *args, **kwargs) -> Dict[str, Any]:
        from .bifurcation import comprehensive_hopf_analysis
        return comprehensive_hopf_analysis(*args, **kwargs)
    
    def plot_bifurcation_diagram(self, *args, **kwargs) -> None:
        from .bifurcation import plot_bifurcation_diagram
        plot_bifurcation_diagram(*args, **kwargs)
    
    def analyze_hopf_bifurcation_diagram(self, *args, **kwargs) -> None:
        from .bifurcation import analyze_hopf_bifurcation_diagram
        analyze_hopf_bifurcation_diagram(*args, **kwargs)

__all__ = [
    'BifurcationParameters',
    'BifurcationAnalyzer',
    'DEFAULT_PARAMS'
]