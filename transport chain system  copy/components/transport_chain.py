from typing import List, Dict, Any, Optional
from .company import Company
import numpy as np

class TransportChain:
    """Transport chain class representing a sequence of companies"""
    
    def __init__(self, 
                 chain_id: str,
                 companies: List[Company],
                 chain_type: str = 'rail'):
        """
        Initialize transport chain
        
        Args:
            chain_id: Unique identifier for the chain
            companies: List of companies in the chain
            chain_type: Type of chain ('rail' or 'sea')
        """
        self.chain_id = chain_id
        self.companies = companies
        self.chain_type = chain_type
        self._validate_chain()
        
        # Initialize state variables
        self.flows = np.zeros(len(companies))
        self.densities = np.zeros(len(companies))
        self.prices = np.zeros(len(companies))
        
    def _validate_chain(self) -> None:
        """Validate chain configuration"""
        if not self.companies:
            raise ValueError("Transport chain must have at least one company")
        if self.chain_type not in ['rail', 'sea']:
            raise ValueError("Chain type must be 'rail' or 'sea'")
        # Verify company indices are sequential
        indices = [company.index for company in self.companies]
        if sorted(indices) != list(range(len(indices))):
            raise ValueError("Company indices must be sequential")

    def compute_total_cost(self, 
                         q_rail_chain: Optional[np.ndarray] = None, 
                         q_sea_chain: Optional[np.ndarray] = None) -> float:
        """
        Compute total cost for the chain including subsidy deductions
        
        Args:
            q_rail_chain: Rail transport quantities for each company
            q_sea_chain: Sea transport quantities for each company
            
        Returns:
            float: Total chain cost
        """
        total_cost = 0.0
        if q_rail_chain is not None:
            if len(q_rail_chain) != len(self.companies):
                raise ValueError("Flow array length must match number of companies")
            for company, q_rail in zip(self.companies, q_rail_chain):
                total_cost += (
                    company.compute_cost(q_rail=q_rail) -
                    company.compute_subsidy_cost(q_rail=q_rail)
                )
        elif q_sea_chain is not None:
            if len(q_sea_chain) != len(self.companies):
                raise ValueError("Flow array length must match number of companies")
            for company, q_sea in zip(self.companies, q_sea_chain):
                total_cost += (
                    company.compute_cost(q_sea=q_sea) -
                    company.compute_subsidy_cost(q_sea=q_sea)
                )
        else:
            raise ValueError("Either q_rail_chain or q_sea_chain must be provided")
        return total_cost

    def compute_total_tariff_cost(self, 
                                q_rail_chain: Optional[np.ndarray] = None, 
                                q_sea_chain: Optional[np.ndarray] = None) -> float:
        """
        Compute total tariff cost for the chain
        
        Args:
            q_rail_chain: Rail transport quantities for each company
            q_sea_chain: Sea transport quantities for each company
            
        Returns:
            float: Total tariff cost
        """
        if q_rail_chain is not None:
            return sum(
                company.compute_tariff_cost(q_rail=q_rail)
                for company, q_rail in zip(self.companies, q_rail_chain)
            )
        elif q_sea_chain is not None:
            return sum(
                company.compute_tariff_cost(q_sea=q_sea)
                for company, q_sea in zip(self.companies, q_sea_chain)
            )
        raise ValueError("Either q_rail_chain or q_sea_chain must be provided")

    def compute_total_BAF_cost(self, 
                             q_rail_chain: Optional[np.ndarray] = None, 
                             q_sea_chain: Optional[np.ndarray] = None) -> float:
        """
        Compute total BAF cost for the chain
        
        Args:
            q_rail_chain: Rail transport quantities for each company
            q_sea_chain: Sea transport quantities for each company
            
        Returns:
            float: Total BAF cost
        """
        if q_rail_chain is not None:
            return sum(
                company.compute_BAF_cost(q_rail=q_rail)
                for company, q_rail in zip(self.companies, q_rail_chain)
            )
        elif q_sea_chain is not None:
            return sum(
                company.compute_BAF_cost(q_sea=q_sea)
                for company, q_sea in zip(self.companies, q_sea_chain)
            )
        raise ValueError("Either q_rail_chain or q_sea_chain must be provided")

    def compute_total_subsidy_cost(self, 
                                 q_rail_chain: Optional[np.ndarray] = None, 
                                 q_sea_chain: Optional[np.ndarray] = None) -> float:
        """
        Compute total subsidy cost for the chain
        
        Args:
            q_rail_chain: Rail transport quantities for each company
            q_sea_chain: Sea transport quantities for each company
            
        Returns:
            float: Total subsidy cost
        """
        if q_rail_chain is not None:
            return sum(
                company.compute_subsidy_cost(q_rail=q_rail)
                for company, q_rail in zip(self.companies, q_rail_chain)
            )
        elif q_sea_chain is not None:
            return sum(
                company.compute_subsidy_cost(q_sea=q_sea)
                for company, q_sea in zip(self.companies, q_sea_chain)
            )
        raise ValueError("Either q_rail_chain or q_sea_chain must be provided")


    def update_state(self, 
                    q_rail_chain: Optional[np.ndarray] = None,
                    q_sea_chain: Optional[np.ndarray] = None,
                    densities: np.ndarray = None,
                    prices: np.ndarray = None) -> None:
        """
        Update chain state
        
        Args:
            q_rail_chain: Rail transport quantities for each company
            q_sea_chain: Sea transport quantities for each company
            densities: New density values
            prices: New prices
        """
        # Update flows based on transport type
        if self.chain_type == 'rail' and q_rail_chain is not None:
            if len(q_rail_chain) != len(self.companies):
                raise ValueError("Rail flow array length must match number of companies")
            self.flows = q_rail_chain.copy()
        elif self.chain_type == 'sea' and q_sea_chain is not None:
            if len(q_sea_chain) != len(self.companies):
                raise ValueError("Sea flow array length must match number of companies")
            self.flows = q_sea_chain.copy()
        elif q_rail_chain is not None or q_sea_chain is not None:
            raise ValueError(f"Provided flow type does not match chain type: {self.chain_type}")

        # Update densities if provided
        if densities is not None:
            if len(densities) != len(self.companies):
                raise ValueError("Density array length must match number of companies")
            self.densities = densities.copy()

        # Update prices if provided
        if prices is not None:
            if len(prices) != len(self.companies):
                raise ValueError("Price array length must match number of companies")
            self.prices = prices.copy()

    def get_company_costs(self, 
                        q_rail_chain: Optional[np.ndarray] = None,
                        q_sea_chain: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get costs for each company
        
        Args:
            q_rail_chain: Optional rail transport quantities
            q_sea_chain: Optional sea transport quantities
            
        Returns:
            np.ndarray: Array of company costs
        """
        flows = self.flows
        if self.chain_type == 'rail' and q_rail_chain is not None:
            if len(q_rail_chain) != len(self.companies):
                raise ValueError("Rail flow array length must match number of companies")
            flows = q_rail_chain
        elif self.chain_type == 'sea' and q_sea_chain is not None:
            if len(q_sea_chain) != len(self.companies):
                raise ValueError("Sea flow array length must match number of companies")
            flows = q_sea_chain
        elif q_rail_chain is not None or q_sea_chain is not None:
            raise ValueError(f"Provided flow type does not match chain type: {self.chain_type}")
                
        costs = np.zeros(len(self.companies))
        for i, (company, flow) in enumerate(zip(self.companies, flows)):
            if self.chain_type == 'rail':
                costs[i] = company.compute_cost(q_rail=flow)
            else:
                costs[i] = company.compute_cost(q_sea=flow)
        return costs

    def get_current_state(self) -> Dict[str, np.ndarray]:
        """
        Get current state of the transport chain
        
        Returns:
            Dict containing current flows, densities and prices
        """
        return {
            f'{self.chain_type}_flows': self.flows,
            'densities': self.densities,
            'prices': self.prices
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert chain to dictionary"""
        return {
            'chain_id': self.chain_id,
            'chain_type': self.chain_type,
            'companies': [company.to_dict() for company in self.companies],
            'current_state': {
                f'{self.chain_type}_flows': self.flows.tolist(),
                'densities': self.densities.tolist(),
                'prices': self.prices.tolist()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransportChain':
        """Create chain from dictionary"""
        companies = [Company.from_dict(comp_data) for comp_data in data['companies']]
        chain = cls(data['chain_id'], companies, data['chain_type'])
        
        # Restore state if available
        if 'current_state' in data:
            flow_key = f"{data['chain_type']}_flows"
            chain.flows = np.array(data['current_state'][flow_key])
            chain.densities = np.array(data['current_state']['densities'])
            chain.prices = np.array(data['current_state']['prices'])
            
        return chain