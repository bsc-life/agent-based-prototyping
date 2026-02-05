"""
Boundary condition implementations for diffusion equations.

This module provides various types of boundary conditions that can be applied
to the diffusion equation in 1D, 2D, and 3D domains.
"""

from abc import ABC, abstractmethod
from typing import Union, Tuple, Callable
import numpy as np


class BoundaryCondition(ABC):
    """
    Abstract base class for boundary conditions.
    
    Boundary conditions are applied to the edges/faces of the computational
    domain to ensure the PDE is well-posed.
    """
    
    @abstractmethod
    def apply(
        self,
        state: np.ndarray,
        dx: Tuple[float, ...],
        t: float
    ) -> np.ndarray:
        """
        Apply the boundary condition to the state array.
        
        Parameters
        ----------
        state : np.ndarray
            Current state array.
        dx : Tuple[float, ...]
            Grid spacing in each dimension.
        t : float
            Current time.
            
        Returns
        -------
        np.ndarray
            State with boundary conditions applied.
        """
        pass


class DirichletBC(BoundaryCondition):
    """
    Dirichlet (fixed value) boundary condition.
    
    Sets the value at the boundary to a specified value:
        u(boundary) = value
    
    Parameters
    ----------
    value : Union[float, Callable]
        Boundary value. Can be a constant or a function of time.
    """
    
    def __init__(self, value: Union[float, Callable] = 0.0):
        """Initialize Dirichlet boundary condition."""
        self.value = value
        
    def _get_value(self, t: float) -> float:
        """Get the boundary value at time t."""
        if callable(self.value):
            return self.value(t)
        return self.value
    
    def apply(
        self,
        state: np.ndarray,
        dx: Tuple[float, ...],
        t: float
    ) -> np.ndarray:
        """Apply Dirichlet boundary conditions to all boundaries."""
        result = state.copy()
        val = self._get_value(t)
        
        ndim = len(state.shape)
        
        if ndim == 1:
            result[0] = val
            result[-1] = val
        elif ndim == 2:
            result[0, :] = val   # Left
            result[-1, :] = val  # Right
            result[:, 0] = val   # Bottom
            result[:, -1] = val  # Top
        elif ndim == 3:
            result[0, :, :] = val   # Left face
            result[-1, :, :] = val  # Right face
            result[:, 0, :] = val   # Front face
            result[:, -1, :] = val  # Back face
            result[:, :, 0] = val   # Bottom face
            result[:, :, -1] = val  # Top face
        else:
            raise ValueError(f"Unsupported number of dimensions: {ndim}")
        
        return result


class NeumannBC(BoundaryCondition):
    """
    Neumann (fixed derivative/flux) boundary condition.
    
    Sets the normal derivative at the boundary:
        ∂u/∂n(boundary) = flux
    
    A zero Neumann BC (flux=0) represents an insulating/no-flux boundary.
    
    Parameters
    ----------
    flux : Union[float, Callable]
        Boundary flux. Can be a constant or a function of time.
        Positive flux means flow out of the domain.
    """
    
    def __init__(self, flux: Union[float, Callable] = 0.0):
        """Initialize Neumann boundary condition."""
        self.flux = flux
        
    def _get_flux(self, t: float) -> float:
        """Get the boundary flux at time t."""
        if callable(self.flux):
            return self.flux(t)
        return self.flux
    
    def apply(
        self,
        state: np.ndarray,
        dx: Tuple[float, ...],
        t: float
    ) -> np.ndarray:
        """Apply Neumann boundary conditions using ghost points."""
        result = state.copy()
        flux_val = self._get_flux(t)
        
        ndim = len(state.shape)
        
        if ndim == 1:
            # Left boundary: u[-1] = u[1] - 2*dx*flux
            result[0] = result[1] - 2 * dx[0] * flux_val
            # Right boundary: u[N] = u[N-2] + 2*dx*flux
            result[-1] = result[-2] + 2 * dx[0] * flux_val
            
        elif ndim == 2:
            # Left and right boundaries (x-direction)
            result[0, :] = result[1, :] - 2 * dx[0] * flux_val
            result[-1, :] = result[-2, :] + 2 * dx[0] * flux_val
            # Bottom and top boundaries (y-direction)
            result[:, 0] = result[:, 1] - 2 * dx[1] * flux_val
            result[:, -1] = result[:, -2] + 2 * dx[1] * flux_val
            
        elif ndim == 3:
            # x-direction boundaries
            result[0, :, :] = result[1, :, :] - 2 * dx[0] * flux_val
            result[-1, :, :] = result[-2, :, :] + 2 * dx[0] * flux_val
            # y-direction boundaries
            result[:, 0, :] = result[:, 1, :] - 2 * dx[1] * flux_val
            result[:, -1, :] = result[:, -2, :] + 2 * dx[1] * flux_val
            # z-direction boundaries
            result[:, :, 0] = result[:, :, 1] - 2 * dx[2] * flux_val
            result[:, :, -1] = result[:, :, -2] + 2 * dx[2] * flux_val
            
        else:
            raise ValueError(f"Unsupported number of dimensions: {ndim}")
        
        return result


class PeriodicBC(BoundaryCondition):
    """
    Periodic boundary conditions.
    
    Connects opposite boundaries so that the domain wraps around:
        u(left) = u(right)
        u(bottom) = u(top)
        etc.
    """
    
    def apply(
        self,
        state: np.ndarray,
        dx: Tuple[float, ...],
        t: float
    ) -> np.ndarray:
        """Apply periodic boundary conditions."""
        result = state.copy()
        
        ndim = len(state.shape)
        
        if ndim == 1:
            # Copy values from opposite ends
            result[0] = result[-2]
            result[-1] = result[1]
            
        elif ndim == 2:
            # x-direction periodicity
            result[0, :] = result[-2, :]
            result[-1, :] = result[1, :]
            # y-direction periodicity
            result[:, 0] = result[:, -2]
            result[:, -1] = result[:, 1]
            
        elif ndim == 3:
            # x-direction periodicity
            result[0, :, :] = result[-2, :, :]
            result[-1, :, :] = result[1, :, :]
            # y-direction periodicity
            result[:, 0, :] = result[:, -2, :]
            result[:, -1, :] = result[:, 1, :]
            # z-direction periodicity
            result[:, :, 0] = result[:, :, -2]
            result[:, :, -1] = result[:, :, 1]
            
        else:
            raise ValueError(f"Unsupported number of dimensions: {ndim}")
        
        return result


class RobinBC(BoundaryCondition):
    """
    Robin (mixed) boundary condition.
    
    Linear combination of Dirichlet and Neumann conditions:
        α·u + β·∂u/∂n = γ
    
    Parameters
    ----------
    alpha : float
        Coefficient for the value term.
    beta : float
        Coefficient for the derivative term.
    gamma : Union[float, Callable]
        Right-hand side value. Can be constant or function of time.
        
    Notes
    -----
    Special cases:
        - α=1, β=0: Dirichlet BC (u = γ)
        - α=0, β=1: Neumann BC (∂u/∂n = γ)
        - α=1, β=h: Convective/radiation BC
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: Union[float, Callable] = 0.0
    ):
        """Initialize Robin boundary condition."""
        if alpha == 0 and beta == 0:
            raise ValueError("Both alpha and beta cannot be zero")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def _get_gamma(self, t: float) -> float:
        """Get the gamma value at time t."""
        if callable(self.gamma):
            return self.gamma(t)
        return self.gamma
    
    def apply(
        self,
        state: np.ndarray,
        dx: Tuple[float, ...],
        t: float
    ) -> np.ndarray:
        """Apply Robin boundary conditions."""
        result = state.copy()
        gamma_val = self._get_gamma(t)
        
        ndim = len(state.shape)
        
        if ndim == 1:
            # Left boundary: alpha*u[0] + beta*(u[1]-u[-1])/(2*dx) = gamma
            # Solving for u[0]: u[0] = (gamma - beta*(u[1]-u[-1])/(2*dx)) / alpha
            if self.alpha != 0:
                result[0] = (gamma_val - self.beta * (result[1] - result[0]) / dx[0]) / self.alpha
                result[-1] = (gamma_val - self.beta * (result[-1] - result[-2]) / dx[0]) / self.alpha
            else:
                # Pure Neumann case
                result[0] = result[1] - dx[0] * gamma_val / self.beta
                result[-1] = result[-2] + dx[0] * gamma_val / self.beta
                
        elif ndim == 2:
            # Apply to all four boundaries (simplified version)
            if self.alpha != 0:
                # x-boundaries
                result[0, :] = (gamma_val - self.beta * (result[1, :] - result[0, :]) / dx[0]) / self.alpha
                result[-1, :] = (gamma_val - self.beta * (result[-1, :] - result[-2, :]) / dx[0]) / self.alpha
                # y-boundaries
                result[:, 0] = (gamma_val - self.beta * (result[:, 1] - result[:, 0]) / dx[1]) / self.alpha
                result[:, -1] = (gamma_val - self.beta * (result[:, -1] - result[:, -2]) / dx[1]) / self.alpha
            else:
                result[0, :] = result[1, :] - dx[0] * gamma_val / self.beta
                result[-1, :] = result[-2, :] + dx[0] * gamma_val / self.beta
                result[:, 0] = result[:, 1] - dx[1] * gamma_val / self.beta
                result[:, -1] = result[:, -2] + dx[1] * gamma_val / self.beta
                
        elif ndim == 3:
            if self.alpha != 0:
                # x-boundaries
                result[0, :, :] = (gamma_val - self.beta * (result[1, :, :] - result[0, :, :]) / dx[0]) / self.alpha
                result[-1, :, :] = (gamma_val - self.beta * (result[-1, :, :] - result[-2, :, :]) / dx[0]) / self.alpha
                # y-boundaries
                result[:, 0, :] = (gamma_val - self.beta * (result[:, 1, :] - result[:, 0, :]) / dx[1]) / self.alpha
                result[:, -1, :] = (gamma_val - self.beta * (result[:, -1, :] - result[:, -2, :]) / dx[1]) / self.alpha
                # z-boundaries
                result[:, :, 0] = (gamma_val - self.beta * (result[:, :, 1] - result[:, :, 0]) / dx[2]) / self.alpha
                result[:, :, -1] = (gamma_val - self.beta * (result[:, :, -1] - result[:, :, -2]) / dx[2]) / self.alpha
            else:
                result[0, :, :] = result[1, :, :] - dx[0] * gamma_val / self.beta
                result[-1, :, :] = result[-2, :, :] + dx[0] * gamma_val / self.beta
                result[:, 0, :] = result[:, 1, :] - dx[1] * gamma_val / self.beta
                result[:, -1, :] = result[:, -2, :] + dx[1] * gamma_val / self.beta
                result[:, :, 0] = result[:, :, 1] - dx[2] * gamma_val / self.beta
                result[:, :, -1] = result[:, :, -2] + dx[2] * gamma_val / self.beta
        else:
            raise ValueError(f"Unsupported number of dimensions: {ndim}")
        
        return result
