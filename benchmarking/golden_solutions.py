"""
Analytical (golden) solutions for validating numerical diffusion schemas.

This module provides analytical solutions to diffusion equations that can be used
as reference solutions for testing numerical methods.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Union, Dict, Any


class GoldenSolution(ABC):
    """Base class for analytical solutions."""
    
    @abstractmethod
    def evaluate(self, coordinates: Union[np.ndarray, Tuple[np.ndarray, ...]], t: float) -> np.ndarray:
        """
        Evaluate the analytical solution at given coordinates and time.
        
        Parameters
        ----------
        coordinates : np.ndarray or tuple of np.ndarray
            Spatial coordinates. For 1D: single array. For 2D/3D: tuple of meshgrids.
        t : float
            Time at which to evaluate the solution.
            
        Returns
        -------
        np.ndarray
            Solution values at the given points and time.
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return a description of the analytical solution."""
        pass


class GaussianDiffusion1D(GoldenSolution):
    """
    Fundamental solution for 1D diffusion equation with Gaussian initial condition.
    
    Solves: ∂u/∂t = D∇²u
    Initial condition: u(x,0) = A·exp(-(x-x0)²/(2σ²))
    Solution: u(x,t) = A·√(σ²/(σ² + 2Dt))·exp(-(x-x0)²/(2(σ² + 2Dt)))
    """
    
    def __init__(self, center: float = 0.5, amplitude: float = 1.0, 
                 initial_width: float = 0.1, diffusion_coefficient: float = 1.0):
        """
        Initialize 1D Gaussian diffusion solution.
        
        Parameters
        ----------
        center : float
            Center position of initial Gaussian pulse.
        amplitude : float
            Amplitude of initial Gaussian pulse.
        initial_width : float
            Standard deviation of initial Gaussian pulse.
        diffusion_coefficient : float
            Diffusion coefficient D.
        """
        self.center = center
        self.amplitude = amplitude
        self.sigma0 = initial_width
        self.D = diffusion_coefficient
        
    def evaluate(self, coordinates: Union[np.ndarray, Tuple[np.ndarray, ...]], t: float) -> np.ndarray:
        """Evaluate 1D Gaussian diffusion solution."""
        # Handle both list and direct array input
        if isinstance(coordinates, (list, tuple)):
            x = coordinates[0]
        else:
            x = coordinates
            
        sigma_squared = self.sigma0**2 + 2 * self.D * t
        
        # Amplitude decreases as pulse spreads to conserve mass
        amplitude_factor = np.sqrt(self.sigma0**2 / sigma_squared)
        
        return (self.amplitude * amplitude_factor * 
                np.exp(-(x - self.center)**2 / (2 * sigma_squared)))
    
    def get_description(self) -> str:
        return f"1D Gaussian diffusion (D={self.D}, x0={self.center}, σ0={self.sigma0})"


class GaussianDiffusion2D(GoldenSolution):
    """
    Fundamental solution for 2D diffusion equation with Gaussian initial condition.
    
    Solves: ∂u/∂t = D∇²u
    Initial condition: u(x,y,0) = A·exp(-r²/(2σ²)) where r² = (x-x0)² + (y-y0)²
    Solution: u(x,y,t) = A·σ²/(σ² + 2Dt)·exp(-r²/(2(σ² + 2Dt)))
    """
    
    def __init__(self, center: Tuple[float, float] = (0.5, 0.5), 
                 amplitude: float = 1.0, initial_width: float = 0.1, 
                 diffusion_coefficient: float = 1.0):
        """
        Initialize 2D Gaussian diffusion solution.
        
        Parameters
        ----------
        center : tuple of float
            (x0, y0) center position of initial Gaussian pulse.
        amplitude : float
            Amplitude of initial Gaussian pulse.
        initial_width : float
            Standard deviation of initial Gaussian pulse.
        diffusion_coefficient : float
            Diffusion coefficient D.
        """
        self.center = center
        self.amplitude = amplitude
        self.sigma0 = initial_width
        self.D = diffusion_coefficient
        
    def evaluate(self, coordinates: Tuple[np.ndarray, np.ndarray], t: float) -> np.ndarray:
        """Evaluate 2D Gaussian diffusion solution."""
        x, y = coordinates
        x0, y0 = self.center
        sigma_squared = self.sigma0**2 + 2 * self.D * t
        
        # Amplitude decreases as pulse spreads to conserve mass (2D case)
        amplitude_factor = self.sigma0**2 / sigma_squared
        
        r_squared = (x - x0)**2 + (y - y0)**2
        
        return (self.amplitude * amplitude_factor * 
                np.exp(-r_squared / (2 * sigma_squared)))
    
    def get_description(self) -> str:
        return f"2D Gaussian diffusion (D={self.D}, center={self.center}, σ0={self.sigma0})"


class GaussianDiffusion3D(GoldenSolution):
    """
    Fundamental solution for 3D diffusion equation with Gaussian initial condition.
    
    Solves: ∂u/∂t = D∇²u
    Initial condition: u(x,y,z,0) = A·exp(-r²/(2σ²)) where r² = (x-x0)² + (y-y0)² + (z-z0)²
    Solution: u(x,y,z,t) = A·(σ²/(σ² + 2Dt))^(3/2)·exp(-r²/(2(σ² + 2Dt)))
    """
    
    def __init__(self, center: Tuple[float, float, float] = (0.5, 0.5, 0.5), 
                 amplitude: float = 1.0, initial_width: float = 0.1, 
                 diffusion_coefficient: float = 1.0):
        """
        Initialize 3D Gaussian diffusion solution.
        
        Parameters
        ----------
        center : tuple of float
            (x0, y0, z0) center position of initial Gaussian pulse.
        amplitude : float
            Amplitude of initial Gaussian pulse.
        initial_width : float
            Standard deviation of initial Gaussian pulse.
        diffusion_coefficient : float
            Diffusion coefficient D.
        """
        self.center = center
        self.amplitude = amplitude
        self.sigma0 = initial_width
        self.D = diffusion_coefficient
        
    def evaluate(self, coordinates: Tuple[np.ndarray, np.ndarray, np.ndarray], t: float) -> np.ndarray:
        """Evaluate 3D Gaussian diffusion solution."""
        x, y, z = coordinates
        x0, y0, z0 = self.center
        sigma_squared = self.sigma0**2 + 2 * self.D * t
        
        # Amplitude decreases as pulse spreads to conserve mass (3D case)
        amplitude_factor = (self.sigma0**2 / sigma_squared)**(3/2)
        
        r_squared = (x - x0)**2 + (y - y0)**2 + (z - z0)**2
        
        return (self.amplitude * amplitude_factor * 
                np.exp(-r_squared / (2 * sigma_squared)))
    
    def get_description(self) -> str:
        return f"3D Gaussian diffusion (D={self.D}, center={self.center}, σ0={self.sigma0})"


class ExponentialDecay(GoldenSolution):
    """
    Solution for pure exponential decay (no diffusion).
    
    Solves: ∂u/∂t = -λu
    Solution: u(x,t) = u(x,0)·exp(-λt)
    """
    
    def __init__(self, initial_condition: Callable, decay_rate: float):
        """
        Initialize exponential decay solution.
        
        Parameters
        ----------
        initial_condition : callable
            Function that returns initial condition at given coordinates.
        decay_rate : float
            Decay rate λ.
        """
        self.initial_condition = initial_condition
        self.decay_rate = decay_rate
        
    def evaluate(self, coordinates: Union[np.ndarray, Tuple[np.ndarray, ...]], t: float) -> np.ndarray:
        """Evaluate exponential decay solution."""
        u0 = self.initial_condition(coordinates)
        return u0 * np.exp(-self.decay_rate * t)
    
    def get_description(self) -> str:
        return f"Exponential decay (λ={self.decay_rate})"


class SteadyStateAgentDiffusion(GoldenSolution):
    """
    Steady-state solution for constant point source with diffusion and decay.
    
    Solves: D∇²u - λu + S·δ(x-x0) = 0
    
    For 1D: u(x) = (S/(2D√(λ/D)))·exp(-√(λ/D)|x-x0|)
    For 2D: u(r) = (S/(2πD))·K0(r√(λ/D)) where K0 is modified Bessel function
    For 3D: u(r) = (S/(4πDr))·exp(-r√(λ/D))
    """
    
    def __init__(self, source_position: Tuple[float, ...], source_strength: float,
                 diffusion_coefficient: float, decay_rate: float, ndim: int):
        """
        Initialize steady-state agent diffusion solution.
        
        Parameters
        ----------
        source_position : tuple
            Position of point source.
        source_strength : float
            Source strength S.
        diffusion_coefficient : float
            Diffusion coefficient D.
        decay_rate : float
            Decay rate λ.
        ndim : int
            Number of spatial dimensions (1, 2, or 3).
        """
        self.source_position = source_position
        self.S = source_strength
        self.D = diffusion_coefficient
        self.lambda_ = decay_rate
        self.ndim = ndim
        
        if decay_rate <= 0:
            raise ValueError("Decay rate must be positive for steady-state solution")
        
    def evaluate(self, coordinates: Union[np.ndarray, Tuple[np.ndarray, ...]], t: float) -> np.ndarray:
        """Evaluate steady-state solution (independent of time)."""
        kappa = np.sqrt(self.lambda_ / self.D)
        
        if self.ndim == 1:
            x = coordinates
            x0 = self.source_position[0]
            r = np.abs(x - x0)
            return (self.S / (2 * self.D * kappa)) * np.exp(-kappa * r)
            
        elif self.ndim == 2:
            from scipy.special import k0  # Modified Bessel function of 2nd kind, order 0
            x, y = coordinates
            x0, y0 = self.source_position
            r = np.sqrt((x - x0)**2 + (y - y0)**2)
            # Avoid singularity at source
            r = np.maximum(r, 1e-10)
            return (self.S / (2 * np.pi * self.D)) * k0(kappa * r)
            
        elif self.ndim == 3:
            x, y, z = coordinates
            x0, y0, z0 = self.source_position
            r = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
            # Avoid singularity at source
            r = np.maximum(r, 1e-10)
            return (self.S / (4 * np.pi * self.D * r)) * np.exp(-kappa * r)
        else:
            raise ValueError(f"Unsupported dimensionality: {self.ndim}")
    
    def get_description(self) -> str:
        return f"{self.ndim}D steady-state agent diffusion (D={self.D}, λ={self.lambda_})"


def create_golden_solution_from_dict(spec: Dict[str, Any]) -> GoldenSolution:
    """
    Factory function to create GoldenSolution from dictionary specification.
    
    Parameters
    ----------
    spec : dict
        Specification dictionary with 'type' key and type-specific parameters.
        
        Supported types:
        - 'gaussian_1d': center, amplitude, initial_width, diffusion_coefficient
        - 'gaussian_2d': center (tuple), amplitude, initial_width, diffusion_coefficient
        - 'gaussian_3d': center (tuple), amplitude, initial_width, diffusion_coefficient
        - 'exponential_decay': initial_condition (callable), decay_rate
        - 'steady_state_agent': source_position, source_strength, diffusion_coefficient, decay_rate, ndim
        
    Returns
    -------
    GoldenSolution
        Instantiated golden solution object.
    """
    solution_type = spec['type']
    
    if solution_type == 'gaussian_1d':
        return GaussianDiffusion1D(
            center=spec.get('center', 0.5),
            amplitude=spec.get('amplitude', 1.0),
            initial_width=spec.get('initial_width', 0.1),
            diffusion_coefficient=spec.get('diffusion_coefficient', 1.0)
        )
    
    elif solution_type == 'gaussian_2d':
        return GaussianDiffusion2D(
            center=spec.get('center', (0.5, 0.5)),
            amplitude=spec.get('amplitude', 1.0),
            initial_width=spec.get('initial_width', 0.1),
            diffusion_coefficient=spec.get('diffusion_coefficient', 1.0)
        )
    
    elif solution_type == 'gaussian_3d':
        return GaussianDiffusion3D(
            center=spec.get('center', (0.5, 0.5, 0.5)),
            amplitude=spec.get('amplitude', 1.0),
            initial_width=spec.get('initial_width', 0.1),
            diffusion_coefficient=spec.get('diffusion_coefficient', 1.0)
        )
    
    elif solution_type == 'exponential_decay':
        return ExponentialDecay(
            initial_condition=spec['initial_condition'],
            decay_rate=spec['decay_rate']
        )
    
    elif solution_type == 'steady_state_agent':
        return SteadyStateAgentDiffusion(
            source_position=spec['source_position'],
            source_strength=spec['source_strength'],
            diffusion_coefficient=spec['diffusion_coefficient'],
            decay_rate=spec['decay_rate'],
            ndim=spec['ndim']
        )
    
    else:
        raise ValueError(f"Unknown golden solution type: {solution_type}")
