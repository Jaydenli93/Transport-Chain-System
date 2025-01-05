from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass

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

@dataclass
class PriceParameters:
    """Price model parameters"""
    alpha_rail: float  # Rail price elasticity parameter
    beta_rail: float   # Rail price sensitivity parameter
    lambda_rail: float # Rail cross-price elasticity parameter
    alpha_ship: float  # Sea price elasticity parameter
    beta_ship: float   # Sea price sensitivity parameter
    lambda_ship: float # Sea cross-price elasticity parameter
    eta: float        # Market competition parameter
    gamma_rail: float = 0.3  # Rail price adjustment rate
    gamma_sea: float = 0.5   # Sea price adjustment rate

class PriceModel:
    """ODE model for price dynamics"""
    
    def __init__(self, zeta: float, sigma: float):
        """
        Initialize price model
        
        Args:
            zeta: Price adjustment rate
            sigma: Time lag parameter
        """
        self.zeta = zeta
        self.sigma = sigma
        # Density influence parameters
        self.lambda_density = 1.0  # Density influence coefficient
        self.alpha_density = 0.5   # Density influence exponent
        self._validate_parameters()
        
    def _validate_parameters(self) -> None:
        """Validate model parameters"""
        if self.zeta <= 0:
            raise ValueError("Price adjustment rate must be positive")
        if self.sigma < 0:
            raise ValueError("Time lag must be non-negative")
            
    def compute_target_price_rail(self, 
                                q_rail: float,
                                d_rail: float,
                                d_ship: float,
                                params: PriceParameters) -> float:
        """
        Compute target rail price using inverse demand function
        
        Args:
            q_rail: Rail flow quantity
            d_rail: Rail demand
            d_ship: Sea demand
            params: Price parameters
            
        Returns:
            float: Target rail price
        """
        price = (params.alpha_rail - 
                params.beta_rail * d_rail + 
                params.lambda_rail * d_ship - 
                (params.eta / (1 + params.eta)) * d_rail)
        return max(0, price)
        
    def compute_target_price_sea(self,
                               q_sea: float,
                               d_rail: float,
                               d_ship: float,
                               params: PriceParameters) -> float:
        """
        Compute target sea price using inverse demand function
        
        Args:
            q_sea: Sea flow quantity
            d_rail: Rail demand
            d_ship: Sea demand
            params: Price parameters
            
        Returns:
            float: Target sea price
        """
        price = (params.alpha_ship - 
                params.beta_ship * d_ship + 
                params.lambda_ship * d_rail - 
                (params.eta / (1 + params.eta)) * d_ship)
        return max(0, price)
        
    def price_ode(self, 
                 y: np.ndarray,
                 t: float,
                 params: Dict[str, Any],
                 price_history: Any) -> np.ndarray:
        """
        ODE system implementing price dynamics with cost structure and density influence
        
        Args:
            y: Current state vector (flattened prices)
            t: Current time
            params: System parameters including:
                - current_densities: Current flow densities
                - rail_chains: Rail transport chains
                - sea_chains: Sea transport chains
            price_history: Historical price data
            
        Returns:
            np.ndarray: Price derivatives
        """
        # Extract parameters
        current_densities = params['current_densities']
        rail_chains = params['rail_chains']
        sea_chains = params['sea_chains']
        
        # Get dimensions
        num_rail_chains = len(rail_chains)
        num_sea_chains = len(sea_chains)
        companies_per_chain = len(rail_chains[0].companies)
        rail_prices_size = num_rail_chains * companies_per_chain
        
        # Reshape price arrays
        current_prices_rail = y[:rail_prices_size].reshape(num_rail_chains, companies_per_chain)
        current_prices_sea = y[rail_prices_size:].reshape(num_sea_chains, companies_per_chain)
        
        # Initialize derivatives
        dp_dt_rail = np.zeros_like(current_prices_rail)
        dp_dt_sea = np.zeros_like(current_prices_sea)
        
        # Calculate derivatives for rail chains
        for chain_idx in range(num_rail_chains):
            rho_max_rail = RAIL_CAPACITY[rail_chains[chain_idx].name] / max(RAIL_CAPACITY.values())
            for company_idx in range(companies_per_chain):
                company = rail_chains[chain_idx].companies[company_idx]
                p_lagged = price_history.get_lagged_price(t, chain_idx, company_idx, 'rail')
                current_density = current_densities['rail'][chain_idx][company_idx]
                
                # Cost-based price calculation
                cost_based_price = (
                    company.fixed_operational_cost + 
                    company.variable_operational_cost * current_density +
                    company.fixed_transport_cost +
                    company.variable_transport_cost * current_density
                )
                
                # Market adjustments
                market_adjustments = (
                    company.tariff_cost * current_density +
                    company.BAF_cost * current_density -
                    company.subsidy_cost * current_density
                )
                
                # Density influence
                density_term = self.lambda_density * (current_density / rho_max_rail) ** self.alpha_density
                
                dp_dt_rail[chain_idx, company_idx] = (
                    -self.zeta * (current_prices_rail[chain_idx, company_idx] - p_lagged) +
                    (cost_based_price + market_adjustments - current_prices_rail[chain_idx, company_idx]) -
                    density_term
                )
        
        # Calculate derivatives for sea chains
        for chain_idx in range(num_sea_chains):
            rho_max_sea = SEA_CAPACITY[sea_chains[chain_idx].name] / max(SEA_CAPACITY.values())
            for company_idx in range(companies_per_chain):
                company = sea_chains[chain_idx].companies[company_idx]
                p_lagged = price_history.get_lagged_price(t, chain_idx, company_idx, 'sea')
                current_density = current_densities['sea'][chain_idx][company_idx]
                
                # Cost-based price calculation
                cost_based_price = (
                    company.fixed_operational_cost + 
                    company.variable_operational_cost * current_density +
                    company.fixed_transport_cost +
                    company.variable_transport_cost * current_density
                )
                
                # Market adjustments
                market_adjustments = (
                    company.tariff_cost * current_density +
                    company.BAF_cost * current_density -
                    company.subsidy_cost * current_density
                )
                
                # Density influence
                density_term = self.lambda_density * (current_density / rho_max_sea) ** self.alpha_density
                
                dp_dt_sea[chain_idx, company_idx] = (
                    -self.zeta * (current_prices_sea[chain_idx, company_idx] - p_lagged) +
                    (cost_based_price + market_adjustments - current_prices_sea[chain_idx, company_idx]) -
                    density_term
                )
        
        return np.concatenate([dp_dt_rail.flatten(), dp_dt_sea.flatten()])
    
    def update_densities(self,
                        current_prices: Dict[str, List[np.ndarray]],
                        current_densities: Dict[str, np.ndarray],
                        params: Dict[str, Any],
                        price_history: Any,
                        t: float) -> Dict[str, np.ndarray]:
        """
        Update flow densities based on price changes
        
        Args:
            current_prices: Current prices for all chains
            current_densities: Current flow densities
            params: System parameters
            price_history: Historical price data
            t: Current time
            
        Returns:
            Dict[str, np.ndarray]: Updated densities
        """
        dt = params.get('dt', 0.1)
        price_params = params['price_params']
        
        new_densities = {
            'rail': np.zeros_like(current_densities['rail']),
            'sea': np.zeros_like(current_densities['sea'])
        }
        
        # Update rail densities
        for i, chain in enumerate(params['rail_chains']):
            rho_max_rail = RAIL_CAPACITY[chain.name] / max(RAIL_CAPACITY.values())
            for j in range(len(chain.companies)):
                current_rho = current_densities['rail'][i,j]
                current_p = current_prices['rail'][i][j]
                p_lagged = price_history.get_lagged_price(t, i, j, 'rail')
                
                price_effect = -price_params.gamma_rail * ((current_p - p_lagged)/p_lagged)
                capacity_constraint = (1 - current_rho/rho_max_rail)
                
                drho = current_rho * price_effect * capacity_constraint
                new_rho = current_rho + drho * dt
                new_densities['rail'][i,j] = np.clip(new_rho, 0, rho_max_rail)
        
        # Update sea densities
        for i, chain in enumerate(params['sea_chains']):
            rho_max_sea = SEA_CAPACITY[chain.name] / max(SEA_CAPACITY.values())
            for j in range(len(chain.companies)):
                current_rho = current_densities['sea'][i,j]
                current_p = current_prices['sea'][i][j]
                p_lagged = price_history.get_lagged_price(t, i, j, 'sea')
                
                price_effect = -price_params.gamma_sea * ((current_p - p_lagged)/p_lagged)
                capacity_constraint = (1 - current_rho/rho_max_sea)
                
                drho = current_rho * price_effect * capacity_constraint
                new_rho = current_rho + drho * dt
                new_densities['sea'][i,j] = np.clip(new_rho, 0, rho_max_sea)
        
        # Maintain total flow constraint
        total_demand = np.sum(current_densities['rail']) + np.sum(current_densities['sea'])
        current_total = np.sum(new_densities['rail']) + np.sum(new_densities['sea'])
        
        if current_total > 0:
            adjustment_factor = total_demand / current_total
            new_densities['rail'] *= adjustment_factor
            new_densities['sea'] *= adjustment_factor
        
        return new_densities
    
    def solve_ode(self,
                 initial_prices: Dict[str, List[np.ndarray]],
                 t: float,
                 dt: float,
                 params: Dict[str, Any],
                 price_history: Any) -> Dict[str, List[np.ndarray]]:
        """
        Solve price ODE system for one time step
        
        Args:
            initial_prices: Initial price distribution
            t: Current time
            dt: Time step size
            params: System parameters
            price_history: Historical price data
            
        Returns:
            Dict[str, List[np.ndarray]]: Updated prices
        """
        # Flatten initial prices
        y0 = np.concatenate([
            np.array(initial_prices['rail']).flatten(),
            np.array(initial_prices['sea']).flatten()
        ])
        
        # Solve ODE
        solution = odeint(
            self.price_ode,
            y0,
            [t, t + dt],
            args=(params, price_history)
        )
        
        # Extract final state
        y_final = solution[-1]
        
        # Reshape back to original structure
        num_rail_chains = len(params['rail_chains'])
        num_sea_chains = len(params['sea_chains'])
        companies_per_chain = len(params['rail_chains'][0].companies)
        rail_prices_size = num_rail_chains * companies_per_chain
        
        new_prices = {
            'rail': list(y_final[:rail_prices_size].reshape(num_rail_chains, companies_per_chain)),
            'sea': list(y_final[rail_prices_size:].reshape(num_sea_chains, companies_per_chain))
        }
        
        return new_prices
    
    def check_price_stability(self, 
                            prices: Dict[str, List[np.ndarray]], 
                            threshold: float = 1e3) -> bool:
        """
        Check if prices are within reasonable bounds
        
        Args:
            prices: Current price distributions
            threshold: Maximum allowed price value
            
        Returns:
            bool: True if prices are stable, False otherwise
        """
        for mode in ['rail', 'sea']:
            for price_chain in prices[mode]:
                if np.any(price_chain < 0) or np.any(price_chain > threshold):
                    return False
        return True