from typing import Dict, List, Tuple, Optional, Any
import numpy as np

class JacobianCalculator:
    """Calculator for ODE and PDE system Jacobian matrices"""
    
    def __init__(self, h: float = 1e-6):
        """
        Initialize Jacobian calculator
        
        Args:
            h: Finite difference step size
        """
        self.h = h
        
    def calculate_ode_jacobian(self,
                             y: np.ndarray,
                             t: float,
                             params: Dict[str, Any],
                             price_history: Any,
                             system_ode: callable) -> np.ndarray:
        """
        Calculate the Jacobian matrix of the ODE system (price dynamics)
        
        Args:
            y: Current state vector (contains all prices)
            t: Current time
            params: System parameters
            price_history: Price history object
            system_ode: ODE system function
            
        Returns:
            numpy.ndarray: Jacobian matrix of the ODE system
        """
        num_rail_chains = len(params['rail_chains'])
        num_sea_chains = len(params['sea_chains'])
        companies_per_chain = len(params['rail_chains'][0].companies)
        
        total_size = (num_rail_chains + num_sea_chains) * companies_per_chain
        jacobian = np.zeros((total_size, total_size))
        
        # Baseline function value
        f0 = system_ode(y, t, params, price_history)
        
        # Calculate partial derivatives
        for i in range(total_size):
            y_perturbed = y.copy()
            y_perturbed[i] += self.h
            f1 = system_ode(y_perturbed, t, params, price_history)
            jacobian[:, i] = (f1 - f0) / self.h
            
        return jacobian

    def calculate_pde_jacobian(self,
                             densities: Dict[str, List[List[float]]],
                             flows: Dict[str, List[List[float]]],
                             t: float,
                             dx: float,
                             params: Dict[str, Any]) -> np.ndarray:
        """
        Calculate the Jacobian matrix of the PDE system
        
        Args:
            densities: Density dictionary {'rail': [...], 'sea': [...]}
            flows: Flow dictionary {'rail': [...], 'sea': [...]}
            t: Current time
            dx: Spatial step size
            params: System parameters
        """
        num_rail_chains = len(params['rail_chains'])
        num_sea_chains = len(params['sea_chains'])
        companies_per_chain = len(params['rail_chains'][0].companies)
        
        total_size = (num_rail_chains + num_sea_chains) * companies_per_chain
        jacobian = np.zeros((total_size, total_size))
        
        # Flatten density data
        flat_densities = np.concatenate([
            np.array(densities['rail']).flatten(),
            np.array(densities['sea']).flatten()
        ])
        
        # Calculate partial derivatives for each variable
        for i in range(total_size):
            perturbed = flat_densities.copy()
            perturbed[i] += self.h
            
            # Reconstruct density array
            rail_size = num_rail_chains * companies_per_chain
            perturbed_rail = perturbed[:rail_size].reshape(num_rail_chains, -1)
            perturbed_sea = perturbed[rail_size:].reshape(num_sea_chains, -1)
            
            # Calculate velocity field
            v_rail = [flows['rail'][j]/max(1e-10, np.mean(densities['rail'][j])) 
                     for j in range(num_rail_chains)]
            v_sea = [flows['sea'][j]/max(1e-10, np.mean(densities['sea'][j])) 
                    for j in range(num_sea_chains)]
            
            # Calculate derivatives for each variable
            derivatives_rail = []
            for j, (chain, v) in enumerate(zip(perturbed_rail, v_rail)):
                diffusion = self._calculate_second_derivatives(chain, dx)
                advection = self._calculate_mixed_derivatives(chain, v, dx)
                derivatives_rail.append(params['gamma_rail'] * diffusion - advection)
                
            derivatives_sea = []
            for j, (chain, v) in enumerate(zip(perturbed_sea, v_sea)):
                diffusion = self._calculate_second_derivatives(chain, dx)
                advection = self._calculate_mixed_derivatives(chain, v, dx)
                derivatives_sea.append(params['gamma_sea'] * diffusion - advection)
            
            # Combine all derivatives
            derivatives = np.concatenate([
                np.array(derivatives_rail).flatten(),
                np.array(derivatives_sea).flatten()
            ])
            
            # Calculate one column of the Jacobian matrix
            jacobian[:, i] = (derivatives - flat_densities) / self.h
        
        return jacobian
    
    @staticmethod
    def _calculate_spatial_derivatives(rho: np.ndarray, dx: float) -> np.ndarray:
        """
        Calculate spatial derivatives using central difference
        
        Args:
            rho: Density array
            dx: Spatial step size
            
        Returns:
            numpy.ndarray: Spatial derivatives
        """
        n = len(rho)
        d_rho = np.zeros_like(rho)
        
        # Internal points use central difference
        d_rho[1:-1] = (rho[2:] - rho[:-2]) / (2*dx)
        
        # Boundary points use second-order one-sided difference
        d_rho[0] = (-3*rho[0] + 4*rho[1] - rho[2]) / (2*dx)
        d_rho[-1] = (3*rho[-1] - 4*rho[-2] + rho[-3]) / (2*dx)
        
        return d_rho
    
    @staticmethod
    def _calculate_second_derivatives(rho: np.ndarray, dx: float) -> np.ndarray:
        """
        Calculate second derivatives using central difference
        
        Args:
            rho: Density array
            dx: Spatial step size
            
        Returns:
            numpy.ndarray: Second derivatives
        """
        n = len(rho)
        d2_rho = np.zeros_like(rho)
        
        # Internal points use central difference
        d2_rho[1:-1] = (rho[2:] - 2*rho[1:-1] + rho[:-2]) / (dx**2)
        
        # Boundary points use second-order difference
        d2_rho[0] = (2*rho[0] - 5*rho[1] + 4*rho[2] - rho[3]) / (dx**2)
        d2_rho[-1] = (2*rho[-1] - 5*rho[-2] + 4*rho[-3] - rho[-4]) / (dx**2)
        
        return d2_rho
    
    def _calculate_mixed_derivatives(self,
                                  rho: np.ndarray,
                                  v: np.ndarray,
                                  dx: float) -> np.ndarray:
        """
        Calculate mixed derivatives (advection term)
        
        Args:
            rho: Density array
            v: Velocity array
            dx: Spatial step size
            
        Returns:
            numpy.ndarray: Mixed derivatives
        """
        n = len(rho)
        mixed = np.zeros_like(rho)
        
        # Internal points use central difference
        drho = self._calculate_spatial_derivatives(rho, dx)
        dv = self._calculate_spatial_derivatives(v, dx)
        
        # Calculate v*∂ρ/∂x + ρ*∂v/∂x
        mixed[1:-1] = v[1:-1] * drho[1:-1] + rho[1:-1] * dv[1:-1]
        
        # Boundary points use one-sided difference
        mixed[0] = v[0] * drho[0] + rho[0] * dv[0]
        mixed[-1] = v[-1] * drho[-1] + rho[-1] * dv[-1]
        
        return mixed