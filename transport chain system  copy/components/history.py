from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from datetime import datetime
import json

class TransportHistory:
    """Class for storing and managing transport chain history"""
    
    def __init__(self, 
                 params: Dict[str, Any],
                 max_history_length: int = 1000,
                 initial_q_rail: Optional[List[List[float]]] = None,
                 initial_q_sea: Optional[List[List[float]]] = None):
        """
        Initialize history storage
        
        Args:
            params: Configuration parameters
            max_history_length: Maximum number of time steps to store
            initial_q_rail: Initial rail transport quantities
            initial_q_sea: Initial sea transport quantities
        """
        self.params = params
        self.max_history_length = max_history_length
        self.sigma = params.get('sigma', 1.0)
        
        # Time history
        self.times: List[float] = []
        
        # Main history storage
        self.flows: Dict[str, List[np.ndarray]] = {'rail': [], 'sea': []}
        self.densities: Dict[str, List[np.ndarray]] = {'rail': [], 'sea': []}
        self.prices: Dict[str, List[np.ndarray]] = {'rail': [], 'sea': []}
        self.costs: Dict[str, List[float]] = {'rail': [], 'sea': []}
        
        # Calculate initial prices if initial quantities are provided
        if initial_q_rail is not None and initial_q_sea is not None:
            self.initial_prices_rail, self.initial_prices_sea = None, None
        else:
            self.initial_prices_rail = None
            self.initial_prices_sea = None

    def compute_initial_prices(self, 
                             q_rail_list: List[List[float]], 
                             q_sea_list: List[List[float]],
                             price_model: Any) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Compute initial prices based on initial flows
        
        Args:
            q_rail_list: Initial rail transport quantities
            q_sea_list: Initial sea transport quantities
            price_model: Instance of PriceModel for calculations
            
        Returns:
            Tuple of initial prices for rail and sea transport
        """
        price_rail_values = []
        price_sea_values = []
        
        # Process rail chain
        for chain_index, q_rail_chain in enumerate(q_rail_list):
            price_rail_chain = []
            for company_index, q_rail in enumerate(q_rail_chain):
                # Extract demand for specific company   
                d_rail_company = self.params['demands']['rail'][chain_index][company_index]
                d_ship_company = self.params['demands']['sea'][chain_index][company_index]
                
                # Compute current price using price model
                current_price = price_model.compute_target_price_rail(
                    q_rail,
                    d_rail_company,
                    d_ship_company,
                    self.params['price_params']
                )
                price_rail_chain.append(current_price)
            price_rail_values.append(price_rail_chain)
        
        # Process sea chain
        for chain_index, q_sea_chain in enumerate(q_sea_list):
            price_sea_chain = []
            for company_index, q_sea in enumerate(q_sea_chain):
                # Extract demand for specific company   
                d_rail_company = self.params['demands']['rail'][chain_index % len(self.params['demands']['rail'])][company_index]
                d_ship_company = self.params['demands']['sea'][chain_index][company_index]
                
                # Compute current price using price model
                current_price = price_model.compute_target_price_sea(
                    q_sea,
                    d_rail_company,
                    d_ship_company,
                    self.params['price_params']
                )
                price_sea_chain.append(current_price)
            price_sea_values.append(price_sea_chain)
        
        return price_rail_values, price_sea_values

    def add_record(self,
                  t: float,
                  flows: Dict[str, np.ndarray],
                  densities: Dict[str, np.ndarray],
                  prices: Dict[str, np.ndarray],
                  costs: Dict[str, float]) -> None:
        """
        Add new record to history
        
        Args:
            t: Current time
            flows: Flow quantities for rail and sea
            densities: Density values for rail and sea
            prices: Prices for rail and sea
            costs: Costs for rail and sea
        """
        # Manage history length
        if len(self.times) >= self.max_history_length:
            self._trim_history()
            
        self.times.append(t)
        for mode in ['rail', 'sea']:
            self.flows[mode].append(flows[mode].copy())
            self.densities[mode].append(densities[mode].copy())
            self.prices[mode].append(prices[mode].copy())
            self.costs[mode].append(costs[mode])

    def get_lagged_price(self, 
                        t: float, 
                        chain_index: int, 
                        company_index: int, 
                        mode: str = 'rail') -> float:
        """
        Get the price at time (t-sigma) for a specific chain and company
        
        Args:
            t: Current timestamp
            chain_index: Index of the transport chain
            company_index: Index of the company within the chain
            mode: Transport mode ('rail' or 'sea')
            
        Returns:
            float: Lagged price value
        """
        if not self.times:
            # Return initial price based on initial flows
            if mode == 'rail' and self.initial_prices_rail is not None:
                return self.initial_prices_rail[chain_index][company_index]
            elif mode == 'sea' and self.initial_prices_sea is not None:
                return self.initial_prices_sea[chain_index][company_index]
            else:
                raise ValueError("No initial prices available")
    
        target_time = t - self.sigma
        time_differences = np.abs(np.array(self.times) - target_time)
        closest_index = np.argmin(time_differences)
        
        try:
            return self.prices[mode][closest_index][chain_index][company_index]
        except IndexError:
            if mode == 'rail':
                return self.initial_prices_rail[chain_index][company_index]
            else:
                return self.initial_prices_sea[chain_index][company_index]

    def get_state_at_time(self, t: float) -> Dict[str, Any]:
        """
        Get system state at specific time
        
        Args:
            t: Time point
            
        Returns:
            Dict containing state information
        """
        if t not in self.times:
            raise ValueError(f"No record found for time {t}")
            
        idx = self.times.index(t)
        return {
            'time': t,
            'flows': {mode: self.flows[mode][idx] for mode in ['rail', 'sea']},
            'densities': {mode: self.densities[mode][idx] for mode in ['rail', 'sea']},
            'prices': {mode: self.prices[mode][idx] for mode in ['rail', 'sea']},
            'costs': {mode: self.costs[mode][idx] for mode in ['rail', 'sea']}
        }

    def _trim_history(self) -> None:
        """Remove oldest records when history length exceeds maximum"""
        self.times.pop(0)
        for mode in ['rail', 'sea']:
            self.flows[mode].pop(0)
            self.densities[mode].pop(0)
            self.prices[mode].pop(0)
            self.costs[mode].pop(0)

    def save_to_file(self, filename: str) -> None:
        """
        Save history to file
        
        Args:
            filename: Output filename
        """
        data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'max_history_length': self.max_history_length,
                'sigma': self.sigma
            },
            'initial_prices': {
                'rail': self.initial_prices_rail,
                'sea': self.initial_prices_sea
            },
            'times': self.times,
            'flows': {mode: [arr.tolist() for arr in self.flows[mode]] 
                     for mode in ['rail', 'sea']},
            'densities': {mode: [arr.tolist() for arr in self.densities[mode]] 
                         for mode in ['rail', 'sea']},
            'prices': {mode: [arr.tolist() for arr in self.prices[mode]] 
                      for mode in ['rail', 'sea']},
            'costs': self.costs
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load_from_file(cls, filename: str, params: Dict[str, Any]) -> 'TransportHistory':
        """
        Load history from file
        
        Args:
            filename: Input filename
            params: Configuration parameters
            
        Returns:
            TransportHistory: Loaded history object
        """
        with open(filename, 'r') as f:
            data = json.load(f)
            
        history = cls(
            params=params,
            max_history_length=data['metadata']['max_history_length']
        )
        
        history.sigma = data['metadata'].get('sigma', 1.0)
        history.initial_prices_rail = data['initial_prices']['rail']
        history.initial_prices_sea = data['initial_prices']['sea']
        history.times = data['times']
        
        for mode in ['rail', 'sea']:
            history.flows[mode] = [np.array(arr) for arr in data['flows'][mode]]
            history.densities[mode] = [np.array(arr) for arr in data['densities'][mode]]
            history.prices[mode] = [np.array(arr) for arr in data['prices'][mode]]
            history.costs[mode] = data['costs'][mode]
            
        return history