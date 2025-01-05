"""
Transport Chain Components
=========================

This package provides the core components for transport chain simulation:
- Company: Individual transport company representation
- TransportChain: Chain of connected transport companies
- TransportHistory: Historical data management for transport chains
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

# Version information
__version__ = '1.0.0'
__author__ = 'Jayden Li'
__email__ = 'jaydenlii93@gmail.com'

# Import components
from .company import Company, CompanyParameters
from .transport_chain import TransportChain
from .history import TransportHistory

# Define what should be imported with "from components import *"
__all__ = [
    'Company',
    'CompanyParameters',
    'TransportChain',
    'TransportHistory',
    'create_transport_system',
    'TRANSPORT_CONFIG',
    'ADJUSTMENT_FACTORS'
]

# Define adjustment factors
ADJUSTMENT_FACTORS = {
    'rail': {
        'variable_operational_cost': 1,
        'fixed_operational_cost': 1,
        'variable_transport_cost': 1,
        'fixed_transport_cost': 1,
        'BAF_cost': 0,
        'tariff_cost': 0,
        'subsidy_cost': 0
    },
    'sea': {
        'variable_operational_cost': 1,
        'fixed_operational_cost': 1,
        'variable_transport_cost': 1,
        'fixed_transport_cost': 1,
        'BAF_cost': 0,
        'tariff_cost': 0,
        'subsidy_cost': 0
    }
}

# Complete transport system configuration
TRANSPORT_CONFIG = {
    'rail_chains': {
        'Rail-CA-EU': {
            'companies': [
                {'index': 0, 'params': [0.15, 8.0, 0.4, 8.0, 0.08, 3.9, 3.0]},
                {'index': 1, 'params': [0.15, 7.5, 0.4, 7.5, 0.08, 3.9, 3.0]},
                {'index': 2, 'params': [0.15, 8.5, 0.4, 8.5, 0.08, 3.9, 3.0]},
                {'index': 3, 'params': [0.15, 7.8, 0.4, 7.8, 0.08, 3.9, 3.0]}
            ],
            'initial_flow': [8, 8.5, 7.8, 9]
        },
        'Rail-RU-EU': {
            'companies': [
                {'index': 0, 'params': [0.15, 9.0, 0.45, 9.0, 0.08, 3.65, 3.0]},
                {'index': 1, 'params': [0.15, 8.7, 0.45, 8.7, 0.08, 3.65, 3.0]},
                {'index': 2, 'params': [0.15, 8.9, 0.45, 8.9, 0.08, 3.65, 3.0]},
                {'index': 3, 'params': [0.15, 7.7, 0.45, 8.5, 0.08, 3.65, 3.0]}
            ],
            'initial_flow': [13, 15, 17, 11]
        },
        'Rail-EEU': {
            'companies': [
                {'index': 0, 'params': [0.15, 10.0, 0.5, 10.0, 0.08, 3.7, 3.0]},
                {'index': 1, 'params': [0.15, 9.5, 0.5, 9.5, 0.08, 3.7, 3.0]},
                {'index': 2, 'params': [0.15, 9.8, 0.5, 9.8, 0.08, 3.7, 3.0]},
                {'index': 3, 'params': [0.15, 9.0, 0.5, 9.0, 0.08, 3.7, 3.0]}
            ],
            'initial_flow': [13, 12, 11, 17]
        },
        'Rail-WEU': {
            'companies': [
                {'index': 0, 'params': [0.15, 10.2, 0.6, 10.1, 0.08, 3.8, 3.0]},
                {'index': 1, 'params': [0.15, 10.0, 0.6, 9.7, 0.08, 3.8, 3.0]},
                {'index': 2, 'params': [0.15, 10.0, 0.6, 10.0, 0.08, 3.8, 3.0]},
                {'index': 3, 'params': [0.15, 9.5, 0.6, 9.7, 0.08, 3.8, 3.0]}
            ],
            'initial_flow': [11, 14, 15, 12]
        },
        'Rail-SEU': {
            'companies': [
                {'index': 0, 'params': [0.15, 10.5, 0.65, 10.5, 0.08, 3.9, 3.0]},
                {'index': 1, 'params': [0.15, 10.0, 0.65, 10.0, 0.08, 3.9, 3.0]},
                {'index': 2, 'params': [0.15, 10.3, 0.65, 10.0, 0.08, 3.9, 3.0]},
                {'index': 3, 'params': [0.15, 9.5, 0.65, 9.7, 0.08, 3.9, 3.0]}
            ],
            'initial_flow': [7, 9, 6, 8]
        }
    },
    'sea_chains': {
        'Maritime-NEU': {
            'companies': [
                {'index': 0, 'params': [0.25, 22.0, 0.44, 22.0, 0.3, 3.0, 0.25]},
                {'index': 1, 'params': [0.25, 21.5, 0.44, 21.5, 0.3, 3.0, 0.25]},
                {'index': 2, 'params': [0.25, 22.5, 0.44, 22.5, 0.3, 3.0, 0.25]},
                {'index': 3, 'params': [0.25, 21.8, 0.44, 21.8, 0.3, 3.0, 0.25]}
            ],
            'initial_flow': [10, 9, 12, 8]
        },
        'Maritime-WEU': {
            'companies': [
                {'index': 0, 'params': [0.25, 20.0, 0.42, 22.0, 0.25, 3.0, 0.25]},
                {'index': 1, 'params': [0.25, 21.0, 0.42, 21.5, 0.25, 3.0, 0.25]},
                {'index': 2, 'params': [0.25, 20.0, 0.42, 22.5, 0.25, 3.0, 0.25]},
                {'index': 3, 'params': [0.25, 19.5, 0.42, 21.8, 0.25, 3.0, 0.25]}
            ],
            'initial_flow': [36, 33, 40, 35]
        },
        'Maritime-SEU': {
            'companies': [
                {'index': 0, 'params': [0.25, 27.0, 0.56, 27.0, 0.2, 3.0, 0.2]},
                {'index': 1, 'params': [0.25, 26.5, 0.56, 26.5, 0.2, 3.0, 0.2]},
                {'index': 2, 'params': [0.25, 27.5, 0.56, 27.5, 0.2, 3.0, 0.2]},
                {'index': 3, 'params': [0.25, 26.8, 0.56, 26.8, 0.2, 3.0, 0.2]}
            ],
            'initial_flow': [18, 20, 21, 23]
        },
        'Maritime-MED': {
            'companies': [
                {'index': 0, 'params': [0.25, 26.0, 0.52, 26.0, 0.25, 3.0, 0.3]},
                {'index': 1, 'params': [0.25, 25.5, 0.52, 25.5, 0.25, 3.0, 0.3]},
                {'index': 2, 'params': [0.25, 26.5, 0.52, 26.5, 0.25, 3.0, 0.3]},
                {'index': 3, 'params': [0.25, 25.8, 0.52, 25.8, 0.25, 3.0, 0.3]}
            ],
            'initial_flow': [36, 33, 40, 35]
        },
        'Maritime-BLK': {
            'companies': [
                {'index': 0, 'params': [0.25, 24.0, 0.48, 24.0, 0.25, 3.0, 0.25]},
                {'index': 1, 'params': [0.25, 23.5, 0.48, 23.5, 0.25, 3.0, 0.25]},
                {'index': 2, 'params': [0.25, 24.5, 0.48, 24.5, 0.25, 3.0, 0.25]},
                {'index': 3, 'params': [0.25, 23.8, 0.48, 23.8, 0.25, 3.0, 0.25]}
            ],
            'initial_flow': [12, 10, 15, 9]
        },
        'Maritime-BLT': {
            'companies': [
                {'index': 0, 'params': [0.25, 29.0, 0.6, 29.0, 0.3, 3.0, 0.3]},
                {'index': 1, 'params': [0.25, 28.0, 0.6, 28.0, 0.3, 3.0, 0.3]},
                {'index': 2, 'params': [0.25, 29.5, 0.6, 29.5, 0.3, 3.0, 0.3]},
                {'index': 3, 'params': [0.25, 28.8, 0.6, 28.8, 0.3, 3.0, 0.3]}
            ],
            'initial_flow': [20, 21, 23, 8]
        }
    }
}

# Parameters description
PARAM_DESCRIPTION = {
    'params': [
        'variable_operational_cost',  # [0]
        'fixed_operational_cost',     # [1]
        'variable_transport_cost',    # [2]
        'fixed_transport_cost',       # [3]
        'tariff_cost',               # [4]
        'BAF_cost',                  # [5]
        'subsidy_cost'               # [6]
    ]
}

# Route description
ROUTE_DESCRIPTION = {
    'rail_chains': {
        'Rail-CA-EU': 'Central Asia to European Union',
        'Rail-RU-EU': 'Russia to European Union',
        'Rail-EEU': 'Eastern Europe',
        'Rail-WEU': 'Western Europe',
        'Rail-SEU': 'Southern Europe'
    },
    'sea_chains': {
        'Maritime-NEU': 'Northern European Union',
        'Maritime-WEU': 'Western European Union',
        'Maritime-SEU': 'Southern European Union',
        'Maritime-MED': 'Mediterranean',
        'Maritime-BLK': 'Black Sea',
        'Maritime-BLT': 'Baltic Sea'
    }
}

def apply_factors(params: List[float], factors: Dict[str, float]) -> List[float]:
    """
    Apply adjustment factors to parameters
    
    Args:
        params: List of parameter values
        factors: Dictionary of adjustment factors
        
    Returns:
        List of adjusted parameter values
    """
    param_keys = [
        'variable_operational_cost',
        'fixed_operational_cost',
        'variable_transport_cost',
        'fixed_transport_cost',
        'tariff_cost',
        'BAF_cost',
        'subsidy_cost'
    ]
    return [param * factors[key] for param, key in zip(params, param_keys)]

def create_company(company_config: Dict[str, Any], 
                  factors: Optional[Dict[str, float]] = None) -> Company:
    """
    Create a company with specified configuration
    
    Args:
        company_config: Company configuration dictionary
        factors: Optional adjustment factors
        
    Returns:
        Company: Configured company instance
    """
    params = company_config['params']
    if factors:
        params = apply_factors(params, factors)
        
    return Company(
        index=company_config['index'],
        params=CompanyParameters(
            variable_operational_cost=params[0],
            fixed_operational_cost=params[1],
            variable_transport_cost=params[2],
            fixed_transport_cost=params[3],
            tariff_cost=params[4],
            BAF_cost=params[5],
            subsidy_cost=params[6]
        )
    )

def create_transport_system(
    config: Dict[str, Any] = None,
    apply_factors: bool = True
) -> tuple[List[TransportChain], List[TransportChain], TransportHistory]:
    """
    Create a complete transport system
    
    Args:
        config: Optional custom configuration
        apply_factors: Whether to apply adjustment factors
        
    Returns:
        Tuple containing:
        - List of rail transport chains
        - List of sea transport chains
        - Transport history object
    """
    if config is None:
        config = TRANSPORT_CONFIG
        
    # Create rail chains
    rail_chains = []
    rail_flows = []
    for chain_id, chain_config in config['rail_chains'].items():
        companies = [
            create_company(
                comp_config,
                ADJUSTMENT_FACTORS['rail'] if apply_factors else None
            )
            for comp_config in chain_config['companies']
        ]
        chain = TransportChain(
            chain_id=chain_id,
            companies=companies,
            chain_type='rail'
        )
        rail_chains.append(chain)
        rail_flows.append(chain_config['initial_flow'])
        
    # Create sea chains
    sea_chains = []
    sea_flows = []
    for chain_id, chain_config in config['sea_chains'].items():
        companies = [
            create_company(
                comp_config,
                ADJUSTMENT_FACTORS['sea'] if apply_factors else None
            )
            for comp_config in chain_config['companies']
        ]
        chain = TransportChain(
            chain_id=chain_id,
            companies=companies,
            chain_type='sea'
        )
        sea_chains.append(chain)
        sea_flows.append(chain_config['initial_flow'])
        
    # Create history object with initial flows
    history = TransportHistory(
        max_history_length=1000,
        initial_q_rail=rail_flows,
        initial_q_sea=sea_flows
    )
    
    return rail_chains, sea_chains, history

def get_initial_flows() -> tuple[np.ndarray, np.ndarray]:
    """
    Get initial flows for all chains
    
    Returns:
        Tuple containing:
        - Array of initial rail flows
        - Array of initial sea flows
    """
    rail_flows = [
        chain_config['initial_flow']
        for chain_config in TRANSPORT_CONFIG['rail_chains'].values()
    ]
    sea_flows = [
        chain_config['initial_flow']
        for chain_config in TRANSPORT_CONFIG['sea_chains'].values()
    ]
    return np.array(rail_flows), np.array(sea_flows)