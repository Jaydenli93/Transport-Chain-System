from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from scipy.integrate import odeint

from .density_model import DensityModel, DisruptionParams
from .price_model import PriceModel, PriceParameters
from .optimization_model import EulerOptimizer, OptimizationParameters
from .jacobian_matrix import JacobianCalculator

@dataclass
class SystemState:
    """System state container"""
    time: float
    densities: Dict[str, List[np.ndarray]]
    flows: Dict[str, List[np.ndarray]]
    prices: Dict[str, List[np.ndarray]]
    costs: Dict[str, float]
    jacobian: Optional[Dict[str, np.ndarray]] = None

@dataclass
class SimulationHistory:
    """Container for simulation history"""
    times: List[float]
    densities: Dict[str, List[List[np.ndarray]]]
    flows: Dict[str, List[List[np.ndarray]]]
    prices: Dict[str, List[List[np.ndarray]]]
    costs: Dict[str, List[float]]
    jacobians: List[Dict[str, np.ndarray]]

class SystemModel:
    """Integrated system dynamics model"""
    
    def __init__(self,
                 density_model: DensityModel,
                 price_model: PriceModel,
                 optimizer: EulerOptimizer,
                 jacobian_calculator: JacobianCalculator):
        """
        Initialize system model
        
        Args:
            density_model: PDE model for density evolution
            price_model: ODE model for price dynamics
            optimizer: Euler optimization model
            jacobian_calculator: Calculator for Jacobian matrices
        """
        self.density_model = density_model
        self.price_model = price_model
        self.optimizer = optimizer
        self.jacobian_calculator = jacobian_calculator
        
    def _initialize_history(self, initial_state: SystemState) -> SimulationHistory:
        """
        Initialize simulation history
        
        Args:
            initial_state: Initial system state
            
        Returns:
            SimulationHistory: Initialized history container
        """
        return SimulationHistory(
            times=[initial_state.time],
            densities={
                'rail': [initial_state.densities['rail']],
                'sea': [initial_state.densities['sea']]
            },
            flows={
                'rail': [initial_state.flows['rail']],
                'sea': [initial_state.flows['sea']]
            },
            prices={
                'rail': [initial_state.prices['rail']],
                'sea': [initial_state.prices['sea']]
            },
            costs={
                'rail': [initial_state.costs['rail']],
                'sea': [initial_state.costs['sea']]
            },
            jacobians=[initial_state.jacobian] if initial_state.jacobian else []
        )
        
    def _update_history(self,
                       history: SimulationHistory,
                       current_state: SystemState) -> None:
        """
        Update simulation history with current state
        
        Args:
            history: Current simulation history
            current_state: Current system state
        """
        history.times.append(current_state.time)
        history.densities['rail'].append(current_state.densities['rail'])
        history.densities['sea'].append(current_state.densities['sea'])
        history.flows['rail'].append(current_state.flows['rail'])
        history.flows['sea'].append(current_state.flows['sea'])
        history.prices['rail'].append(current_state.prices['rail'])
        history.prices['sea'].append(current_state.prices['sea'])
        history.costs['rail'].append(current_state.costs['rail'])
        history.costs['sea'].append(current_state.costs['sea'])
        if current_state.jacobian:
            history.jacobians.append(current_state.jacobian)

    def _flatten_prices(self, prices: Dict[str, List[np.ndarray]]) -> np.ndarray:
        """
        Flatten price dictionary into single array
        
        Args:
            prices: Dictionary of prices for rail and sea
            
        Returns:
            numpy.ndarray: Flattened price array
        """
        return np.concatenate([
            np.array(prices['rail']).flatten(),
            np.array(prices['sea']).flatten()
        ])

    def _calculate_costs(self,
                        flows: Dict[str, List[np.ndarray]],
                        densities: Dict[str, List[np.ndarray]],
                        params: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate system costs
        
        Args:
            flows: Current flows
            densities: Current densities
            params: System parameters
            
        Returns:
            Dict[str, float]: Calculated costs for rail and sea
        """
        costs = {'rail': 0.0, 'sea': 0.0}
        
        for mode in ['rail', 'sea']:
            for chain_idx, chain_flows in enumerate(flows[mode]):
                chain = params[f'{mode}_chains'][chain_idx]
                for company_idx, flow in enumerate(chain_flows):
                    company = chain.companies[company_idx]
                    density = densities[mode][chain_idx][company_idx]
                    
                    # Calculate operational costs
                    costs[mode] += (company.fixed_operational_cost +
                                  company.variable_operational_cost * flow)
                    
                    # Calculate transport costs
                    costs[mode] += (company.fixed_transport_cost +
                                  company.variable_transport_cost * flow)
                    
                    # Add additional costs
                    costs[mode] += (company.tariff_cost * flow +
                                  company.BAF_cost * flow -
                                  company.subsidy_cost * flow)
        
        return costs

    def _update_system_state(self,
                           current_state: SystemState,
                           t: float,
                           dt: float,
                           params: Dict[str, Any],
                           progress: Optional[Progress] = None) -> SystemState:
        """
        Update system state for one time step
        
        Args:
            current_state: Current system state
            t: Current time
            dt: Time step size
            params: System parameters
            progress: Progress bar object
            
        Returns:
            SystemState: Updated system state
        """
        try:
            # Calculate Jacobian matrices
            ode_jacobian = self.jacobian_calculator.calculate_ode_jacobian(
                y=self._flatten_prices(current_state.prices),
                t=t,
                params=params,
                price_history=params['price_history'],
                system_ode=self.price_model.price_ode
            )
            
            pde_jacobian = self.jacobian_calculator.calculate_pde_jacobian(
                densities=current_state.densities,
                flows=current_state.flows,
                t=t,
                dx=params['dx'],
                params=params
            )
            
            # Update prices using ODE
            new_prices = self.price_model.solve_ode(
                current_state.prices,
                t,
                dt,
                params,
                params['price_history']
            )
            
            # Update densities using PDE
            new_densities, new_flows = self.density_model.system_pde(
                current_state.densities,
                current_state.flows,
                t,
                params
            )
            
            # Optimize flows
            optimized_flows = self.optimizer.optimize(
                new_flows['rail'],
                new_flows['sea'],
                params,
                progress
            )
            
            # Calculate new costs
            new_costs = self._calculate_costs(optimized_flows, new_densities, params)
            
            return SystemState(
                time=t + dt,
                densities=new_densities,
                flows=optimized_flows,
                prices=new_prices,
                costs=new_costs,
                jacobian={'ode': ode_jacobian, 'pde': pde_jacobian}
            )
            
        except Exception as e:
            raise RuntimeError(f"Error updating system state: {str(e)}")

    def _check_system_stability(self, state: SystemState) -> bool:
        """
        Check if system state is stable
        
        Args:
            state: Current system state
            
        Returns:
            bool: True if system is stable, False otherwise
        """
        # Check price stability
        if not self.price_model.check_price_stability(state.prices):
            return False
            
        # Check density bounds
        for mode in ['rail', 'sea']:
            for chain_densities in state.densities[mode]:
                if np.any(chain_densities < 0) or np.any(chain_densities > 1):
                    return False
                    
        # Check flow positivity
        for mode in ['rail', 'sea']:
            for chain_flows in state.flows[mode]:
                if np.any(np.array(chain_flows) < 0):
                    return False
                    
        return True

    def simulate(self,
                initial_state: SystemState,
                T: float,
                dt: float,
                params: Dict[str, Any],
                show_progress: bool = True) -> SimulationHistory:
        """
        Run full system simulation
        
        Args:
            initial_state: Initial state of the system
            T: Total simulation time
            dt: Time step size
            params: System parameters
            show_progress: Whether to show progress bar
            
        Returns:
            SimulationHistory: Complete simulation history
        """
        time_steps = int(T/dt)
        history = self._initialize_history(initial_state)
        current_state = initial_state
        
        # Setup progress bar
        progress = None
        if show_progress:
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn(),
                TextColumn("Step {task.completed}/{task.total}")
            )
            
        try:
            if progress:
                progress.start()
                task_id = progress.add_task("[cyan]Simulating...", total=time_steps)
                
            for step in range(time_steps):
                t = step * dt
                
                # Update state
                current_state = self._update_system_state(
                    current_state,
                    t,
                    dt,
                    params,
                    progress
                )
                
                # Update history
                self._update_history(history, current_state)
                
                # Update progress
                if progress:
                    progress.update(task_id, advance=1)
                    
                # Check for instabilities
                if not self._check_system_stability(current_state):
                    print(f"\nWarning: System instability detected at t={t}")
                    break
                    
        finally:
            if progress:
                progress.stop()
                
        return history