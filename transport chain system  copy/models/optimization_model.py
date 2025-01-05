from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass
from rich.progress import Progress, TaskID

# Transport chain capacity settings
RAIL_CAPACITY = {
    'Rail-CA-EU': 40,   # Central Asia-EU route: facility constraints
    'Rail-RU-EU': 65,   # Russia-EU route: major transport corridor
    'Rail-EEU': 64,     # Eastern Europe route: well-developed network
    'Rail-WEU': 55,     # Western Europe route: complete infrastructure
    'Rail-SEU': 35      # Southern Europe route: terrain restrictions
}

SEA_CAPACITY = {
    'Maritime-NEU': 80,   # Northern Europe route: limited by port size
    'Maritime-WEU': 150,  # Western Europe route: major shipping channel
    'Maritime-SEU': 90,   # Southern Europe route: medium-sized ports
    'Maritime-MED': 120,  # Mediterranean route: important shipping channel
    'Maritime-BLK': 70,   # Black Sea route: geographical limitations
    'Maritime-BLT': 100   # Baltic Sea route: regional shipping center
}

# Flow adjustment parameters
FLOW_PARAMS = {
    'rail': {
        'gamma_ab': 1,
        'delta_as': 1
    },
    'sea': {
        'gamma_ab': 1,
        'delta_as': 1
    }
}

# Disruption scenarios
DISRUPTION_SCENARIOS = {
    'red_sea_crisis': {
        'description': 'Red Sea Crisis Impact on Transport Chains',
        'impacts': {
            'rail': {
                'Rail-CA-EU': (0, 0.10),  # 0-10% disruption
                'Rail-EEU': (0, 0.10),
                'Rail-WEU': (0, 0.10),
                'Rail-SEU': (0, 0.10),
                'Rail-RU-EU': (0, 0.10)
            },
            'sea': {
                'Maritime-BLK': (0.20, 0.40),  # 20-40% disruption
                'Maritime-BLT': (0.20, 0.40),
                'Maritime-MED': (0.50, 0.80),  # 50-80% disruption
                'Maritime-WEU': (0.30, 0.60),
                'Maritime-NEU': (0.30, 0.60),
                'Maritime-SEU': (0.30, 0.60)
            }
        }
    },
    'russo_ukrainian_war': {
        'description': 'Russo-Ukrainian War Impact on Transport Chains',
        'impacts': {
            'rail': {
                'Rail-RU-EU': (0.50, 1.00),  # 50-100% disruption
                'Rail-CA-EU': (0.30, 0.60),
                'Rail-EEU': (0.20, 0.40),
                'Rail-WEU': (0, 0.20),
                'Rail-SEU': (0, 0.20)
            },
            'sea': {
                'Maritime-BLK': (0.50, 1.00),  # 50-100% disruption
                'Maritime-BLT': (0.20, 0.50),
                'Maritime-MED': (0, 0.30),
                'Maritime-WEU': (0, 0.30),
                'Maritime-NEU': (0, 0.30),
                'Maritime-SEU': (0, 0.30)
            }
        }
    }
}

# Default disruption settings
DEFAULT_DISRUPTION = {
    'factors': {
        'rail': {
            0: 0,  # Rail-CA-EU
            1: 0,  # Rail-RU-EU 
            2: 0,  # Rail-EEU 
            3: 0,  # Rail-WEU 
            4: 0   # Rail-SEU 
        },
        'sea': {        
            0: 0,  # Maritime-NEU 
            1: 0,  # Maritime-WEU 
            2: 0,  # Maritime-SEU 
            3: 0,  # Maritime-MED 
            4: 0,  # Maritime-BLK 
            5: 0   # Maritime-BLT 
        }
    },
    'types': ['rail', 'sea'],
    'indices': {
        'rail': [0, 1, 2, 3, 4],
        'sea': [0, 1, 2, 3, 4, 5]
    }
}

class DisruptionManager:
    """Manager for handling transport chain disruptions"""
    
    @staticmethod
    def get_scenario_disruption(scenario_name: str, 
                              severity: float = 0.5) -> Dict[str, Any]:
        """
        Get disruption settings for a specific scenario.
        
        Args:
            scenario_name: Name of the disruption scenario
            severity: Severity level (0-1) to interpolate between min and max impacts
            
        Returns:
            Dictionary containing disruption settings
        """
        if scenario_name not in DISRUPTION_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
            
        scenario = DISRUPTION_SCENARIOS[scenario_name]
        disruption = {
            'factors': {
                'rail': {},
                'sea': {}
            },
            'types': ['rail', 'sea'],
            'indices': DEFAULT_DISRUPTION['indices'].copy()
        }
        
        # Convert named impacts to indexed factors
        chain_indices = {
            'rail': {name: idx for idx, name in enumerate(RAIL_CAPACITY.keys())},
            'sea': {name: idx for idx, name in enumerate(SEA_CAPACITY.keys())}
        }
        
        for transport_type in ['rail', 'sea']:
            for chain_name, (min_impact, max_impact) in scenario['impacts'][transport_type].items():
                idx = chain_indices[transport_type][chain_name]
                impact = min_impact + severity * (max_impact - min_impact)
                disruption['factors'][transport_type][idx] = impact
                
        return disruption

# Price elasticity parameters
class PriceElasticityParameters:
    """Price elasticity parameter settings for transport chains"""
    
    def __init__(self, index: int = 4):
        """
        Initialize price elasticity parameters.
        
        Args:
            index: Index for parameter combination (default: 4)
        """
        # Define parameter ranges
        self.alpha_rail_values = np.linspace(1.5, 2.5, 10)    # Base demand elasticity
        self.alpha_ship_values = np.linspace(2.0, 3.0, 10)
        self.beta_rail_values = np.linspace(-1.2, -0.8, 10)   # Price sensitivity
        self.beta_ship_values = np.linspace(-1.8, -1.7, 10)
        self.lambda_rail_values = np.linspace(0.5, 0.8, 10)   # Cross-price elasticity
        self.lambda_ship_values = np.linspace(0.3, 0.5, 10)
        
        # Set current parameters
        self.update_parameters(index)
    
    def update_parameters(self, index: int) -> None:
        """
        Update parameters based on index.
        
        Args:
            index: Index for parameter combination
        """
        self.alpha_rail = self.alpha_rail_values[index]
        self.beta_rail = self.beta_rail_values[index]
        self.lambda_rail = self.lambda_rail_values[index]
        self.alpha_ship = self.alpha_ship_values[index]
        self.beta_ship = self.beta_ship_values[index]
        self.lambda_ship = self.lambda_ship_values[index]
    
    def get_parameters(self) -> Dict[str, float]:
        """
        Get current parameter values.
        
        Returns:
            Dictionary containing current parameter values
        """
        return {
            'alpha_rail': self.alpha_rail,
            'beta_rail': self.beta_rail,
            'lambda_rail': self.lambda_rail,
            'alpha_ship': self.alpha_ship,
            'beta_ship': self.beta_ship,
            'lambda_ship': self.lambda_ship
        }
    
    @property
    def alternative_settings(self) -> List[Dict[str, List[float]]]:
        """
        Get alternative parameter settings.
        
        Returns:
            List of dictionaries containing alternative parameter ranges
        """
        return [
            {
                # Current settings
                'alpha_rail_values': np.linspace(1.5, 2.5, 10),
                'beta_rail_values': np.linspace(-1.2, -0.8, 10),
                'lambda_rail_values': np.linspace(0.5, 0.8, 10),
                'alpha_ship_values': np.linspace(2.0, 3.0, 10),
                'beta_ship_values': np.linspace(-1.8, -1.7, 10),
                'lambda_ship_values': np.linspace(0.3, 0.5, 10)
            },
            {
                # Alternative setting 1
                'alpha_rail_values': np.linspace(1.8, 2.2, 10),
                'beta_rail_values': np.linspace(-1.5, -1.2, 10),
                'lambda_rail_values': np.linspace(0.5, 2.5, 10),
                'alpha_ship_values': np.linspace(2.0, 3.0, 10),
                'beta_ship_values': np.linspace(-1.8, -1.7, 10),
                'lambda_ship_values': np.linspace(0.3, 2.3, 10)
            },
            {
                # Alternative setting 2 (stable demand, varying sensitivity)
                'alpha_rail_values': np.linspace(1.8, 2.2, 10),
                'beta_rail_values': np.linspace(-0.5, -0.3, 10),
                'lambda_rail_values': np.linspace(0.5, 1.0, 10),
                'alpha_ship_values': np.linspace(2.0, 3.0, 10),
                'beta_ship_values': np.linspace(-1.8, -1.5, 10),
                'lambda_ship_values': np.linspace(1.5, 2.5, 10)
            }
        ]

@dataclass
class OptimizationParameters:
    """Optimization model parameters"""
    r_a_rail: float      # Rail resistance parameter
    r_a_sea: float       # Sea resistance parameter
    alpha: float          # Step size
    epsilon: float        # Convergence tolerance
    max_iterations: int   # Maximum number of iterations
    phi_0: float         # Base cost parameter
    theta: float = 0.5     # Price adjustment parameter
    zeta: float = 0.3    # Lagged price weight
    phi_0: float = 34    # Balance factor
    a: float = 1350    # Container capacity
    M: float = 2       # Number of competitors
    eta: float = 15    # Competition elasticity

class CostFunctions:
    """Cost calculation functions for transport chains"""
    
    @staticmethod
    def rail_chain_cost(q_rail_list: List[np.ndarray], rail_chains: List[Any]) -> float:
        """Compute total cost for all rail chains at the company level."""
        total_cost = 0
        for chain, q_rail_chain in zip(rail_chains, q_rail_list):
            for company, q_rail in zip(chain.companies, q_rail_chain):
                total_cost += company.compute_cost(q_rail=q_rail)
        return total_cost

    @staticmethod
    def sea_chain_cost(q_sea_list: List[np.ndarray], sea_chains: List[Any]) -> float:
        """Compute total cost for all sea chains at the company level."""
        total_cost = 0
        for chain, q_sea_chain in zip(sea_chains, q_sea_list):
            for company, q_sea in zip(chain.companies, q_sea_chain):
                total_cost += company.compute_cost(q_sea=q_sea)
        return total_cost

    @staticmethod
    def tariff_cost(q_rail_list: List[np.ndarray], q_sea_list: List[np.ndarray],
                    rail_chains: List[Any], sea_chains: List[Any]) -> float:
        """Compute total tariff cost for all rail and sea chains."""
        total_cost = 0
        for chain, q_rail_chain in zip(rail_chains, q_rail_list):
            for company, q_rail in zip(chain.companies, q_rail_chain):
                total_cost += company.compute_tariff_cost(q_rail=q_rail)
        for chain, q_sea_chain in zip(sea_chains, q_sea_list):
            for company, q_sea in zip(chain.companies, q_sea_chain):
                total_cost += company.compute_tariff_cost(q_sea=q_sea)
        return total_cost

    @staticmethod
    def BAF_cost(q_sea_list: List[np.ndarray], sea_chains: List[Any]) -> float:
        """Compute total BAF cost for all sea chains."""
        total_cost = 0
        for chain, q_sea_chain in zip(sea_chains, q_sea_list):
            for company, q_sea in zip(chain.companies, q_sea_chain):
                total_cost += company.compute_BAF_cost(q_sea=q_sea)
        return total_cost

    @staticmethod
    def subsidy_cost(q_rail_list: List[np.ndarray], q_sea_list: List[np.ndarray],
                     rail_chains: List[Any], sea_chains: List[Any]) -> float:
        """Compute total subsidy cost for all rail and sea chains."""
        total_cost = 0
        for chain, q_rail_chain in zip(rail_chains, q_rail_list):
            for company, q_rail in zip(chain.companies, q_rail_chain):
                total_cost += company.compute_subsidy_cost(q_rail=q_rail)
        for chain, q_sea_chain in zip(sea_chains, q_sea_list):
            for company, q_sea in zip(chain.companies, q_sea_chain):
                total_cost += company.compute_subsidy_cost(q_sea=q_sea)
        return total_cost

class PriceFunctions:
    """Price calculation functions for transport chains"""
    
    @staticmethod
    def price_rail(q_rail: float, alpha_rail: float, beta_rail: float, 
                  lambda_rail: float, d_rail_ij: float, d_ship_ij: float,
                  eta: float) -> float:
        """Inverse Demand function for rail transport"""
        if not all(isinstance(x, (int, float, np.number)) 
                  for x in [q_rail, alpha_rail, beta_rail, lambda_rail, d_rail_ij, d_ship_ij]):
            raise ValueError("All inputs must be numeric")
        
        if q_rail < 0:
            raise ValueError("Rail flow quantity must be non-negative")
            
        price = (alpha_rail - beta_rail * d_rail_ij + lambda_rail * d_ship_ij - 
                (eta / (1 + eta)) * d_rail_ij)
        return max(0, price)

    @staticmethod
    def price_sea(q_sea: float, alpha_ship: float, beta_ship: float,
                 lambda_ship: float, d_rail_ij: float, d_ship_ij: float,
                 eta: float) -> float:
        """Inverse Demand function for sea transport"""
        if not all(isinstance(x, (int, float, np.number))
                  for x in [q_sea, alpha_ship, beta_ship, lambda_ship, d_rail_ij, d_ship_ij]):
            raise ValueError("All inputs must be numeric")
        
        if q_sea < 0:
            raise ValueError("Sea flow quantity must be non-negative")
            
        price = (alpha_ship - beta_ship * d_ship_ij + lambda_ship * d_rail_ij -
                (eta / (1 + eta)) * d_ship_ij)
        return max(0, price)

    @classmethod
    def compute_price_for_all_companies(cls, 
                                    q_rail_list: List[np.ndarray],
                                    q_sea_list: List[np.ndarray],
                                    params: Dict[str, Any],
                                    t: float,
                                    price_history: Any) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute price functions for all companies in rail and sea chains.
        
        Args:
            q_rail_list: List of rail chain flows
            q_sea_list: List of sea chain flows
            params: Dictionary containing model parameters:
                - alpha_rail: Rail price elasticity parameter
                - beta_rail: Rail price sensitivity parameter
                - lambda_rail: Rail cross-price elasticity parameter
                - alpha_ship: Sea price elasticity parameter
                - beta_ship: Sea price sensitivity parameter
                - lambda_ship: Sea cross-price elasticity parameter
                - d_rail_ij: Rail demand values
                - d_ship_ij: Sea demand values
                - eta: Price adjustment parameter
                - zeta: Lagged price weight
                - rail_chains: List of rail transport chains
                - sea_chains: List of sea transport chains
            t: Current time
            price_history: Object storing historical prices
            
        Returns:
            tuple: Lists of rail and sea prices for each company in each chain
            
        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If there's an error during price computation
        """
        # Input validation
        if not all(isinstance(x, (list, np.ndarray)) for x in [q_rail_list, q_sea_list]):
            raise ValueError("Flow inputs must be lists or numpy arrays")
            
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary")
            
        required_params = {
            'alpha_rail', 'beta_rail', 'lambda_rail',
            'alpha_ship', 'beta_ship', 'lambda_ship',
            'd_rail_ij', 'd_ship_ij', 'eta', 'zeta',
            'rail_chains', 'sea_chains'
        }
        if not all(key in params for key in required_params):
            raise ValueError(f"Missing required parameters. Required: {required_params}")
            
        if t < 0:
            raise ValueError("Time must be non-negative")

        try:
            # Convert lists to numpy arrays for better performance
            d_rail_ij_list = np.array(params['d_rail_ij'])
            d_ship_ij_list = np.array(params['d_ship_ij'])

            price_rail_values = []
            price_sea_values = []

            # Process rail chains
            for chain_index, (chain, q_rail_chain) in enumerate(zip(params['rail_chains'], q_rail_list)):
                price_rail_chain = []
                for company_index, q_rail in enumerate(q_rail_chain):
                    # Validate company indices
                    if company_index >= len(chain.companies):
                        raise IndexError(f"Invalid company index {company_index} for rail chain {chain.name}")

                    # Extract demands for specific company
                    try:
                        d_rail_company = d_rail_ij_list[chain_index][company_index]
                        d_ship_company = d_ship_ij_list[chain_index][company_index]
                    except IndexError:
                        raise IndexError(f"Demand data missing for company {company_index} in rail chain {chain.name}")

                    # Get lagged price with error handling
                    try:
                        p_lagged = price_history.get_lagged_price(t, chain_index, company_index, 'rail')
                    except Exception as e:
                        raise RuntimeError(f"Error getting lagged price for rail chain {chain.name}: {str(e)}")

                    # Compute current price
                    current_price = cls.price_rail(
                        q_rail=q_rail,
                        alpha_rail=params['alpha_rail'],
                        beta_rail=params['beta_rail'],
                        lambda_rail=params['lambda_rail'],
                        d_rail_ij=d_rail_company,
                        d_ship_ij=d_ship_company,
                        theta=params['theta']
                    )
                    
                    # Apply price adjustment with zeta factor
                    adjusted_price = (1 - params['zeta']) * current_price + params['zeta'] * p_lagged
                    
                    # Ensure price is non-negative
                    adjusted_price = max(0, adjusted_price)
                    
                    price_rail_chain.append(adjusted_price)

                price_rail_values.append(price_rail_chain)

            # Process sea chains
            for chain_index, (chain, q_sea_chain) in enumerate(zip(params['sea_chains'], q_sea_list)):
                price_sea_chain = []
                for company_index, q_sea in enumerate(q_sea_chain):
                    # Validate company indices
                    if company_index >= len(chain.companies):
                        raise IndexError(f"Invalid company index {company_index} for sea chain {chain.name}")

                    # Extract demands for specific company
                    try:
                        d_rail_company = d_rail_ij_list[chain_index % len(d_rail_ij_list)][company_index]
                        d_ship_company = d_ship_ij_list[chain_index][company_index]
                    except IndexError:
                        raise IndexError(f"Demand data missing for company {company_index} in sea chain {chain.name}")

                    # Get lagged price with error handling
                    try:
                        p_lagged = price_history.get_lagged_price(t, chain_index, company_index, 'sea')
                    except Exception as e:
                        raise RuntimeError(f"Error getting lagged price for sea chain {chain.name}: {str(e)}")

                    # Compute current price
                    current_price = cls.price_sea(
                        q_sea=q_sea,
                        alpha_ship=params['alpha_ship'],
                        beta_ship=params['beta_ship'],
                        lambda_ship=params['lambda_ship'],
                        d_rail_ij=d_rail_company,
                        d_ship_ij=d_ship_company,
                        theta=params['theta']
                    )
                    
                    # Apply price adjustment with zeta factor
                    adjusted_price = (1 - params['zeta']) * current_price + params['zeta'] * p_lagged
                    
                    # Ensure price is non-negative
                    adjusted_price = max(0, adjusted_price)
                    
                    price_sea_chain.append(adjusted_price)

                price_sea_values.append(price_sea_chain)

            # Validate output dimensions
            if len(price_rail_values) != len(params['rail_chains']):
                raise RuntimeError("Mismatch in rail price dimensions")
            if len(price_sea_values) != len(params['sea_chains']):
                raise RuntimeError("Mismatch in sea price dimensions")

            return price_rail_values, price_sea_values

        except Exception as e:
            raise RuntimeError(f"Error computing prices: {str(e)}")

class ObjectiveFunctions:
    """Objective functions for optimization"""
    
    @staticmethod
    def profit(q_rail_list: List[np.ndarray], 
              q_sea_list: List[np.ndarray],
              params: Dict[str, Any],
              price_history: Any) -> float:
        """
        Calculate total profit for all companies in rail and sea chains.
        
        Args:
            q_rail_list: List of rail flows
            q_sea_list: List of sea flows
            params: Dictionary containing model parameters
            price_history: Object storing price history
            
        Returns:
            float: Total profit
        """
        total_profit = 0
        
        # Compute prices for all companies
        price_rail_values, price_sea_values = PriceFunctions.compute_price_for_all_companies(
            q_rail_list=q_rail_list,
            q_sea_list=q_sea_list,
            params=params,
            t=params.get('current_time', 0),
            price_history=price_history
        )
        
        # Calculate profit for rail chains
        for chain_idx, (chain, q_rail_chain, price_rail_chain) in enumerate(
            zip(params['rail_chains'], q_rail_list, price_rail_values)):
            for company_idx, (company, q_rail, price) in enumerate(
                zip(chain.companies, q_rail_chain, price_rail_chain)):
                company_profit = (
                    price * q_rail 
                    - company.compute_cost(q_rail=q_rail)
                    - company.compute_tariff_cost(q_rail=q_rail)
                    - company.compute_BAF_cost(q_rail=q_rail)
                    + company.compute_subsidy_cost(q_rail=q_rail)
                )
                total_profit += company_profit
        
        # Calculate profit for sea chains
        for chain_idx, (chain, q_sea_chain, price_sea_chain) in enumerate(
            zip(params['sea_chains'], q_sea_list, price_sea_values)):
            for company_idx, (company, q_sea, price) in enumerate(
                zip(chain.companies, q_sea_chain, price_sea_chain)):
                company_profit = (
                    price * q_sea 
                    - company.compute_cost(q_sea=q_sea)
                    - company.compute_tariff_cost(q_sea=q_sea)
                    - company.compute_BAF_cost(q_sea=q_sea)
                    + company.compute_subsidy_cost(q_sea=q_sea)
                )
                total_profit += company_profit
                
        return total_profit

class ConstraintFunctions:
    """Constraint functions for optimization"""
    
    @staticmethod
    def dynamic_tau_global(phi_0: float, q_rail_list: List[np.ndarray],
                          q_sea_list: List[np.ndarray],
                          rho_rail_values: List[float],
                          rho_sea_values: List[float]) -> float:
        """Calculate global dynamic tau for all transport chains."""
        return phi_0
    
    @staticmethod
    def rail_capacity_constraint(q_rail_list: List[np.ndarray],
                               r_a_rail: List[float]) -> bool:
        """
        Ensure rail flows don't exceed capacity constraints.
        
        Args:
            q_rail_list: List of rail flows
            r_a_rail: List of rail capacity limits
            
        Returns:
            bool: True if constraints are satisfied
        """
        for chain_idx, q_rail_chain in enumerate(q_rail_list):
            if sum(q_rail_chain) > r_a_rail[chain_idx]:
                return False
        return True
    
    @staticmethod
    def sea_capacity_constraint(q_sea_list: List[np.ndarray],
                              r_a_sea: List[float]) -> bool:
        """
        Ensure sea flows don't exceed capacity constraints.
        
        Args:
            q_sea_list: List of sea flows
            r_a_sea: List of sea capacity limits
            
        Returns:
            bool: True if constraints are satisfied
        """
        for chain_idx, q_sea_chain in enumerate(q_sea_list):
            if sum(q_sea_chain) > r_a_sea[chain_idx]:
                return False
        return True
    
    @staticmethod
    def balance_constraint(q_rail_list: List[np.ndarray],
                         q_sea_list: List[np.ndarray],
                         tau: float) -> float:
        """
        Calculate flow balance between rail and sea transport.
        
        Args:
            q_rail_list: List of rail flows
            q_sea_list: List of sea flows
            tau: Balance factor
            
        Returns:
            float: Balance value
        """
        total_q_rail = sum(sum(q_rail_chain) for q_rail_chain in q_rail_list)
        total_q_sea = sum(sum(q_sea_chain) for q_sea_chain in q_sea_list)
        return total_q_sea - tau * total_q_rail
    
    @staticmethod
    def flow_balance_constraint(q_rail_list: List[np.ndarray],
                              q_sea_list: List[np.ndarray],
                              price_rail_values: List[List[float]],
                              price_sea_values: List[List[float]],
                              a: float,
                              eta: float,
                              M: int) -> bool:
        """
        Check if flow balance constraint is satisfied.
        
        Args:
            q_rail_list: List of rail flows
            q_sea_list: List of sea flows
            price_rail_values: Rail prices
            price_sea_values: Sea prices
            a: Container capacity
            eta: Competition elasticity
            M: Number of competitors
            
        Returns:
            bool: True if constraint is satisfied
        """
        total_flow = (sum(sum(q_rail_chain) for q_rail_chain in q_rail_list) +
                     sum(sum(q_sea_chain) for q_sea_chain in q_sea_list))
        
        total_price_rail = sum(sum(chain) for chain in price_rail_values)
        total_price_sea = sum(sum(chain) for chain in price_sea_values)
        
        expected_flow = a - (1 + eta) * total_price_rail + (eta / (M - 1)) * total_price_sea
        
        return abs(total_flow - expected_flow) < 1e-6  # Using small tolerance for float comparison

    @staticmethod
    def non_negative_flow_constraint(q_rail_list: List[np.ndarray],
                                   q_sea_list: List[np.ndarray]) -> bool:
        """
        Ensure all flows are non-negative.
        
        Args:
            q_rail_list: List of rail flows
            q_sea_list: List of sea flows
            
        Returns:
            bool: True if all flows are non-negative
        """
        return (all(q_rail >= 0 for q_rail_chain in q_rail_list for q_rail in q_rail_chain) and
                all(q_sea >= 0 for q_sea_chain in q_sea_list for q_sea in q_sea_chain))

class EulerOptimizer:
    """Euler optimization algorithm implementation"""
    
    def __init__(self, params: OptimizationParameters):
        self.params = params
        
    def optimize(self,
                q_rail_list: List[np.ndarray],
                q_sea_list: List[np.ndarray],
                params: Dict[str, Any],
                disruption_params: Optional[Dict[str, Any]] = None,
                progress: Optional[Progress] = None,
                task_id: Optional[TaskID] = None,
                t: float = 0,
                price_history: Any = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Optimize flows using Euler method with all constraints.
        
        Args:
            q_rail_list: Initial rail flows
            q_sea_list: Initial sea flows
            params: Model parameters
            disruption_params: Disruption settings (optional)
            progress: Progress bar object (optional)
            task_id: Task ID for progress bar (optional)
            t: Current time
            price_history: Price history object
        """
        prev_q_rail_list = np.array(q_rail_list)
        prev_q_sea_list = np.array(q_sea_list)
        k = 0
        
        while True:
            if progress and task_id:
                progress.update(task_id, completed=k)
                
            # Compute prices
            price_rail_values, price_sea_values = PriceFunctions.compute_price_for_all_companies(
                q_rail_list, q_sea_list, params, t, price_history
            )
            
            # Calculate gradients
            grad_profit_rail_list = []
            grad_profit_sea_list = []
            grad_cost_rail_list = []
            grad_cost_sea_list = []
            
            # Compute gradients for rail chains
            for chain_idx, (chain, q_rail_chain, price_rail_chain) in enumerate(
                zip(params['rail_chains'], q_rail_list, price_rail_values)):
                
                total_profit_gradient_rail = sum(
                    price * q for price, q in zip(price_rail_chain, q_rail_chain)
                )
                
                total_cost_gradient_rail = sum(
                    company.compute_cost(q_rail=q) +
                    company.compute_tariff_cost(q_rail=q) +
                    company.compute_BAF_cost(q_rail=q) -
                    company.compute_subsidy_cost(q_rail=q)
                    for company, q in zip(chain.companies, q_rail_chain)
                )
                
                grad_profit_rail_list.append(total_profit_gradient_rail)
                grad_cost_rail_list.append(total_cost_gradient_rail)
            
            # Compute gradients for sea chains
            for chain_idx, (chain, q_sea_chain, price_sea_chain) in enumerate(
                zip(params['sea_chains'], q_sea_list, price_sea_values)):
                
                total_profit_gradient_sea = sum(
                    price * q for price, q in zip(price_sea_chain, q_sea_chain)
                )
                
                total_cost_gradient_sea = sum(
                    company.compute_cost(q_sea=q) +
                    company.compute_tariff_cost(q_sea=q) +
                    company.compute_BAF_cost(q_sea=q) -
                    company.compute_subsidy_cost(q_sea=q)
                    for company, q in zip(chain.companies, q_sea_chain)
                )
                
                grad_profit_sea_list.append(total_profit_gradient_sea)
                grad_cost_sea_list.append(total_cost_gradient_sea)
            
            # Update flows with Euler method
            for i in range(len(params['rail_chains'])):
                total_gradient = grad_profit_rail_list[i] - grad_cost_rail_list[i]
                q_rail_list[i] = [max(q - self.params.alpha * total_gradient, 0) 
                                 for q in q_rail_list[i]]
                
                # Apply disruption if specified
                if disruption_params and 'rail' in disruption_params['types'] and \
                   i in disruption_params['indices']['rail']:
                    q_rail_list[i] = [q * (1 - disruption_params['factors']['rail'][i]) 
                                     for q in q_rail_list[i]]
            
            for i in range(len(params['sea_chains'])):
                total_gradient = grad_profit_sea_list[i] - grad_cost_sea_list[i]
                q_sea_list[i] = [max(q - self.params.alpha * total_gradient, 0) 
                                for q in q_sea_list[i]]
                
                # Apply disruption if specified
                if disruption_params and 'sea' in disruption_params['types'] and \
                   i in disruption_params['indices']['sea']:
                    q_sea_list[i] = [q * (1 - disruption_params['factors']['sea'][i]) 
                                    for q in q_sea_list[i]]
            
            # Apply constraints
            q_rail_list, q_sea_list = self._apply_constraints(
                q_rail_list, q_sea_list,
                price_rail_values, price_sea_values,
                params
            )
            
            # Check convergence
            if self._check_convergence(q_rail_list, q_sea_list,
                                     prev_q_rail_list, prev_q_sea_list):
                if progress and task_id:
                    progress.update(task_id, completed=self.params.max_iterations)
                break
            
            prev_q_rail_list = np.array(q_rail_list)
            prev_q_sea_list = np.array(q_sea_list)
            k += 1
            
            if k >= self.params.max_iterations:
                if progress and task_id:
                    progress.update(task_id, completed=self.params.max_iterations)
                print("Warning: Maximum iterations reached without convergence")
                break
        
        return q_rail_list, q_sea_list
    
    def _apply_constraints(self,
                         q_rail_list: List[np.ndarray],
                         q_sea_list: List[np.ndarray],
                         price_rail_values: List[List[float]],
                         price_sea_values: List[List[float]],
                         params: Dict[str, Any]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Apply all constraints to the flows"""
        
        # 1. Non-negative flow constraint
        q_rail_list = [np.maximum(q_rail, 0) for q_rail in q_rail_list]
        q_sea_list = [np.maximum(q_sea, 0) for q_sea in q_sea_list]
        
        # 2. Capacity constraints
        for i, q_rail_chain in enumerate(q_rail_list):
            total_rail_flow = sum(q_rail_chain)
            if total_rail_flow > RAIL_CAPACITY[params['rail_chains'][i].name]:
                scaling_factor = RAIL_CAPACITY[params['rail_chains'][i].name] / total_rail_flow
                q_rail_list[i] = [q_rail * scaling_factor for q_rail in q_rail_chain]
        
        for i, q_sea_chain in enumerate(q_sea_list):
            total_sea_flow = sum(q_sea_chain)
            if total_sea_flow > SEA_CAPACITY[params['sea_chains'][i].name]:
                scaling_factor = SEA_CAPACITY[params['sea_chains'][i].name] / total_sea_flow
                q_sea_list[i] = [q_sea * scaling_factor for q_sea in q_sea_chain]
        
        # 3. Balance constraint with tau
        total_q_rail = sum(sum(q_rail_chain) for q_rail_chain in q_rail_list)
        total_q_sea = sum(sum(q_sea_chain) for q_sea_chain in q_sea_list)
        
        if total_q_sea != self.params.phi_0 * total_q_rail:
            desired_total_sea = self.params.phi_0 * total_q_rail
            current_total_sea = total_q_sea
            max_possible_sea = sum(SEA_CAPACITY.values())
            
            adjustment_factor = min(desired_total_sea, max_possible_sea) / current_total_sea
            
            new_q_sea_list = []
            for i, q_sea_chain in enumerate(q_sea_list):
                adjusted_chain = [q_sea * adjustment_factor for q_sea in q_sea_chain]
                chain_total = sum(adjusted_chain)
                
                if chain_total > SEA_CAPACITY[params['sea_chains'][i].name]:
                    scaling_factor = SEA_CAPACITY[params['sea_chains'][i].name] / chain_total
                    adjusted_chain = [q_sea * scaling_factor for q_sea in adjusted_chain]
                
                new_q_sea_list.append(adjusted_chain)
            
            q_sea_list = new_q_sea_list
        
        # 4. Flow balance constraint
        if not ConstraintFunctions.flow_balance_constraint(
            q_rail_list, q_sea_list, price_rail_values, price_sea_values,
            params['a'], params['eta'], params['M']):
            
            total_flow = (sum(sum(q_rail_chain) for q_rail_chain in q_rail_list) +
                         sum(sum(q_sea_chain) for q_sea_chain in q_sea_list))
            total_price_rail = sum(sum(chain) for chain in price_rail_values)
            total_price_sea = sum(sum(chain) for chain in price_sea_values)
            
            required_flow = (params['a'] - (1 + params['eta']) * total_price_rail +
                           (params['eta'] / (params['M'] - 1)) * total_price_sea)
            
            if total_flow > 0:
                scaling_factor = required_flow / total_flow
                q_rail_list = [[q_rail * scaling_factor for q_rail in q_rail_chain]
                              for q_rail_chain in q_rail_list]
                q_sea_list = [[q_sea * scaling_factor for q_sea in q_sea_chain]
                             for q_sea_chain in q_sea_list]
        
        return q_rail_list, q_sea_list