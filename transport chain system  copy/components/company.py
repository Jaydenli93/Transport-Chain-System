from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class CompanyParameters:
    """Company operational parameters"""
    fixed_operational_cost: float
    variable_operational_cost: float
    fixed_transport_cost: float
    variable_transport_cost: float
    tariff_cost: float
    BAF_cost: float  # Bunker Adjustment Factor
    subsidy_cost: float

class Company:
    """Company class representing a node in transport chain"""
    
    def __init__(self, 
                 index: int,
                 params: CompanyParameters):
        """
        Initialize company
        
        Args:
            index: Company index in chain
            params: Company operational parameters
        """
        self.index = index
        self.params = params
        self._validate_parameters()
        
    def _validate_parameters(self) -> None:
        """Validate company parameters"""
        if self.params.fixed_operational_cost < 0:
            raise ValueError("Fixed operational cost cannot be negative")
        if self.params.variable_operational_cost < 0:
            raise ValueError("Variable operational cost cannot be negative")
        if self.params.fixed_transport_cost < 0:
            raise ValueError("Fixed transport cost cannot be negative")
        if self.params.variable_transport_cost < 0:
            raise ValueError("Variable transport cost cannot be negative")
            
    def compute_cost(self, 
                    q_rail: Optional[float] = None, 
                    q_sea: Optional[float] = None) -> float:
        """
        Compute total operational cost
        
        Args:
            q_rail: Rail transport quantity
            q_sea: Sea transport quantity
            
        Returns:
            float: Total cost
        """
        if q_rail is not None:
            # Rail transport cost calculation
            base_transport_cost = self.params.variable_transport_cost * q_rail
            adjusted_transport_cost = base_transport_cost * (
                0.75 +  # Non-fuel portion
                0.25 * (1 + self.params.BAF_cost) +  # Fuel portion affected by BAF
                self.params.tariff_cost  # Tariff impact
            ) + self.params.fixed_transport_cost
            
            # Operational costs
            operational_cost = (
                self.params.variable_operational_cost * q_rail + 
                self.params.fixed_operational_cost
            )
            
        elif q_sea is not None:
            # Sea transport cost calculation
            base_transport_cost = self.params.variable_transport_cost * q_sea
            adjusted_transport_cost = base_transport_cost * (
                0.65 +  # Non-fuel portion
                0.35 * (1 + self.params.BAF_cost) +  # Fuel portion affected by BAF
                self.params.tariff_cost  # Tariff impact
            ) + self.params.fixed_transport_cost
            
            # Operational costs
            operational_cost = (
                self.params.variable_operational_cost * q_sea + 
                self.params.fixed_operational_cost
            )
            
        else:
            raise ValueError("Either q_rail or q_sea must be provided")
            
        return adjusted_transport_cost + operational_cost
        
    def compute_tariff_cost(self, 
                          q_rail: Optional[float] = None, 
                          q_sea: Optional[float] = None) -> float:
        """Compute tariff cost"""
        if q_rail is not None:
            return self.params.tariff_cost * q_rail
        elif q_sea is not None:
            return self.params.tariff_cost * q_sea
        else:
            raise ValueError("Either q_rail or q_sea must be provided")
            
    def compute_BAF_cost(self, 
                        q_rail: Optional[float] = None, 
                        q_sea: Optional[float] = None) -> float:
        """Compute Bunker Adjustment Factor cost"""
        if q_rail is not None:
            return self.params.BAF_cost * q_rail
        elif q_sea is not None:
            return self.params.BAF_cost * q_sea
        else:
            raise ValueError("Either q_rail or q_sea must be provided")
            
    def compute_subsidy_cost(self, 
                           q_rail: Optional[float] = None, 
                           q_sea: Optional[float] = None) -> float:
        """Compute subsidy cost"""
        if q_rail is not None:
            return self.params.subsidy_cost * q_rail
        elif q_sea is not None:
            return self.params.subsidy_cost * q_sea
        else:
            raise ValueError("Either q_rail or q_sea must be provided")
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert company to dictionary"""
        return {
            'index': self.index,
            'parameters': self.params.__dict__
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Company':
        """Create company from dictionary"""
        params = CompanyParameters(**data['parameters'])
        return cls(data['index'], params)