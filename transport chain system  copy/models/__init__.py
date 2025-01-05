"""
Transport Chain System Models
============================

This package contains the core models for simulating and analyzing
transport chain dynamics.

Models:
    - DensityModel: PDE-based flow density evolution
    - PriceModel: ODE-based price dynamics
    - EulerOptimizer: Flow optimization using Euler method
    - SystemModel: Integrated system dynamics
    - JacobianCalculator: Jacobian matrix computation
"""

from typing import Dict, List, Tuple, Any

# Version info
__version__ = '1.0.0'
__author__ = 'Jayden Li'
__email__ = 'jaydenlii93@gmail.com'

# Import all models
from .density_model import (
    DensityModel,
    DisruptionParams
)

from .price_model import (
    PriceModel,
    PriceParameters
)

from .optimization_model import (
    EulerOptimizer,
    OptimizationParameters
)

from .jacobian_matrix import (
    JacobianCalculator
)

from .system_model import (
    SystemModel,
    SystemState,
    SimulationHistory
)

# Define what should be imported with "from models import *"
__all__ = [
    # Core model classes
    'DensityModel',
    'PriceModel',
    'EulerOptimizer',
    'SystemModel',
    'JacobianCalculator',
    
    # Data classes and parameters
    'DisruptionParams',
    'PriceParameters',
    'OptimizationParameters',
    'SystemState',
    'SimulationHistory',
]

# Default parameters
DEFAULT_PARAMS: Dict[str, Any] = {
    'density': {
        'dx': 0.5,
        'dt': 0.2,
        'gamma_rail': 0.3,
        'gamma_sea': 0.5
    },
    'price': {
        'zeta': 20.0,
        'sigma': 0.01
    },
    'optimization': {
        'alpha': 1e-5,
        'epsilon': 1e-8,
        'max_iterations': 10000000,
        'phi_0': 1.0,
        'r_a_rail': 0.1,
        'r_a_sea': 0.15
    },
    'jacobian': {
        'h': 1e-6
    }
}

def create_models(params: Dict[str, Any] = None) -> Tuple[DensityModel, PriceModel, EulerOptimizer, SystemModel, JacobianCalculator]:
    """
    Create all models with given parameters
    
    Args:
        params: Dictionary of model parameters. If None, use default parameters.
        
    Returns:
        Tuple containing:
        - DensityModel
        - PriceModel
        - EulerOptimizer
        - SystemModel
        - JacobianCalculator
    """
    # Use default parameters if none provided
    if params is None:
        params = DEFAULT_PARAMS
        
    # Create individual models
    density_model = DensityModel(
        dx=params['density']['dx'],
        dt=params['density']['dt']
    )
    
    price_model = PriceModel(
        zeta=params['price']['zeta'],
        sigma=params['price']['sigma']
    )
    
    optimizer = EulerOptimizer(
        OptimizationParameters(
            alpha=params['optimization']['alpha'],
            epsilon=params['optimization']['epsilon'],
            max_iterations=params['optimization']['max_iterations'],
            phi_0=params['optimization']['phi_0'],
            r_a_rail=params['optimization']['r_a_rail'],
            r_a_sea=params['optimization']['r_a_sea']
        )
    )

    jacobian_calculator = JacobianCalculator(
        h=params['jacobian']['h']
    )
    
    # Create system model
    system_model = SystemModel(
        density_model=density_model,
        price_model=price_model,
        optimizer=optimizer,
        jacobian_calculator=jacobian_calculator
    )
    
    return density_model, price_model, optimizer, system_model, jacobian_calculator

def get_version() -> str:
    """Get the current version of the models package"""
    return __version__

def get_default_params() -> Dict[str, Any]:
    """Get the default parameters for all models"""
    return DEFAULT_PARAMS.copy()