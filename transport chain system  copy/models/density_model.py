from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class DisruptionParams:
    """Disruption parameters data class"""
    disruption_factor: Dict[str, Dict[int, float]]
    disrupted_chain_indices: Dict[str, List[int]]
    disrupted_chain_types: List[str]

class DensityModel:
    """PDE model for flow density dynamics"""
    
    def __init__(self, dx: float, dt: float):
        """Initialize density model"""
        self.dx = dx
        self.dt = dt
        self._validate_parameters()
        
    def _validate_parameters(self) -> None:
        """Validate model parameters"""
        if self.dx <= 0 or self.dt <= 0:
            raise ValueError("Step sizes must be positive")
        if self.dt/self.dx > 1:
            raise ValueError("CFL condition not satisfied")
    
    def compute_external_source(self,
                              t: float,
                              chain_type: str,
                              chain_index: int,
                              current_flow: float,
                              disruption_params: DisruptionParams) -> float:
        """Calculate external disturbance source"""
        if not isinstance(chain_type, str) or chain_type not in ['rail', 'sea']:
            raise ValueError("Chain type must be 'rail' or 'sea'")
            
        if not isinstance(chain_index, int) or chain_index < 0:
            raise ValueError("Chain index must be a non-negative integer")
            
        if not isinstance(current_flow, (int, float, np.number)) or current_flow < 0:
            raise ValueError("Current flow must be a non-negative number")
            
        try:
            if (chain_type in disruption_params.disrupted_chain_types and 
                chain_index in disruption_params.disrupted_chain_indices[chain_type]):
                
                reduction_percentage = disruption_params.disruption_factor[chain_type][chain_index]
                
                if not 0 <= reduction_percentage <= 1:
                    raise ValueError("Disruption factor must be between 0 and 1")
                    
                return -reduction_percentage * current_flow
                
            return 0.0
            
        except Exception as e:
            raise RuntimeError(f"Error computing external source: {str(e)}")

    def update_link_density(self,
                          current_density: float,
                          current_flow: float,
                          previous_flow: Optional[float],
                          external_source: float,
                          rho_max: float) -> float:
        """Update link density using nonlinear PDE"""
        if not all(isinstance(x, (int, float, np.number)) for x in 
                  [current_density, current_flow, self.dt, self.dx, external_source, rho_max]):
            raise ValueError("All inputs must be numeric")
            
        if previous_flow is not None and not isinstance(previous_flow, (int, float, np.number)):
            raise ValueError("Previous flow must be numeric or None")
            
        if current_density < 0 or current_density > rho_max:
            raise ValueError("Current density must be between 0 and maximum density")
            
        try:
            # Calculate density constraint factor
            density_constraint = 1 - current_density/rho_max
            
            # Calculate nonlinear flow term
            constrained_flow = current_flow * density_constraint
            
            # Calculate spatial gradient
            if previous_flow is None:
                flow_gradient = 0
            else:
                flow_gradient = (constrained_flow - previous_flow * (1 - previous_flow/rho_max)) / self.dx
            
            # Update density using forward Euler method
            new_density = current_density + self.dt * (-flow_gradient + external_source)
            
            # Ensure density is within physical constraints
            new_density = max(0.0, min(new_density, rho_max))
            
            return new_density
            
        except Exception as e:
            raise RuntimeError(f"Error updating link density: {str(e)}")

    def compute_flows_from_density(self,
                                 densities: List[List[float]],
                                 r_as: List[float],
                                 transport_type: str) -> List[List[float]]:
        """Compute flows from densities using f = rho * r"""
        if not isinstance(densities, (list, np.ndarray)):
            raise ValueError("Densities must be a list or numpy array")
            
        if not isinstance(r_as, (list, np.ndarray)):
            raise ValueError("Capacity values must be a list or numpy array")
            
        if transport_type not in ['rail', 'sea']:
            raise ValueError("Transport type must be 'rail' or 'sea'")

        try:
            flows = []
            for chain_index, chain_densities in enumerate(densities):
                if any(d < 0 or d > 1 for d in chain_densities):
                    raise ValueError(f"Density values must be between 0 and 1 for chain {chain_index}")
                    
                capacity = r_as[chain_index]
                chain_flows = [density * capacity for density in chain_densities]
                flows.append(chain_flows)
                
            return flows
            
        except Exception as e:
            raise RuntimeError(f"Error computing flows from densities: {str(e)}")

    def system_pde(self,
                  densities: Dict[str, List[List[float]]],
                  current_flows: Dict[str, List[List[float]]],
                  previous_flows: Dict[str, List[List[float]]],
                  t: float,
                  params: Dict[str, Any]) -> Tuple[List[List[float]], List[List[float]], 
                                                 List[List[float]], List[List[float]]]:
        """PDE system implementing equations with density constraints and direct flow reduction"""
        
        disruption_params = DisruptionParams(
            disruption_factor=params['disruption_factor'],
            disrupted_chain_indices=params['disrupted_chain_indices'],
            disrupted_chain_types=params['disrupted_chain_types']
        )

        new_densities_rail = []
        new_densities_sea = []
        
        # Set maximum density for each rail and sea chain
        rho_max_rail = {i: r/max(params['r_a_rail']) for i, r in enumerate(params['r_a_rail'])}
        rho_max_sea = {i: r/max(params['r_a_sea']) for i, r in enumerate(params['r_a_sea'])}
        
        # Update rail and sea chain densities
        for chain_type in ['rail', 'sea']:
            chains = params[f'{chain_type}_chains']
            rho_max = rho_max_rail if chain_type == 'rail' else rho_max_sea
            new_densities = []
            
            for chain_index, chain in enumerate(chains):
                chain_densities = []
                for company_index in range(len(chain.companies)):
                    E_as = self.compute_external_source(
                        t, chain_type, chain_index,
                        current_flows[chain_type][chain_index][company_index],
                        disruption_params
                    )
                    
                    new_density = self.update_link_density(
                        densities[chain_type][chain_index][company_index],
                        current_flows[chain_type][chain_index][company_index],
                        previous_flows[chain_type][chain_index][company_index],
                        E_as,
                        rho_max[chain_index]
                    )
                    chain_densities.append(new_density)
                new_densities.append(chain_densities)
                
            if chain_type == 'rail':
                new_densities_rail = new_densities
            else:
                new_densities_sea = new_densities
        
        # Compute new flows and apply disruptions
        new_flows_rail = self.compute_flows_from_density(new_densities_rail, params['r_a_rail'], 'rail')
        new_flows_sea = self.compute_flows_from_density(new_densities_sea, params['r_a_sea'], 'sea')
        
        # Apply disruption factors and capacity constraints
        for chain_type in ['rail', 'sea']:
            flows = new_flows_rail if chain_type == 'rail' else new_flows_sea
            capacities = params['r_a_rail'] if chain_type == 'rail' else params['r_a_sea']
            
            if chain_type in disruption_params.disrupted_chain_types:
                for chain_index in disruption_params.disrupted_chain_indices[chain_type]:
                    reduction = disruption_params.disruption_factor[chain_type][chain_index]
                    flows[chain_index] = [flow * (1 - reduction) for flow in flows[chain_index]]
            
            # Apply capacity constraints
            for i, chain_flows in enumerate(flows):
                total_flow = sum(chain_flows)
                if total_flow > capacities[i]:
                    scaling_factor = capacities[i] / total_flow
                    flows[i] = [flow * scaling_factor for flow in chain_flows]
        
        return new_densities_rail, new_densities_sea, new_flows_rail, new_flows_sea