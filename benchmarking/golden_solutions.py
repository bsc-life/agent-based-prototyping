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

class StepFunctionDiffusion1D(GoldenSolution):
    """
    Solution for 1D diffusion with step function initial condition and zero-flux boundaries.
    
    Solves: ∂u/∂t = D∇²u
    Initial condition: u(x,0) = A for x in [x1, x2], else 0
    Boundary conditions: ∂u/∂x = 0 at x=0 and x=L
    """
    def __init__(self, 
                domain_length: float = 1.0, 
                position: float = 0.5, 
                value_left: float = 1.0, 
                value_right: float = 0.0, 
                axis: int = 0,
                diffusion_coefficient: float = 1.0, 
                n_terms: int = 1000):
        self.L = domain_length
        self.x0 = position
        self.val_l = value_left
        self.val_r = value_right
        self.axis = axis
        self.D = diffusion_coefficient
        self.n_terms = n_terms
        
        # --- Precompute Fourier Coefficients ---
        
        # 1. Steady State (A0): The weighted average (Conservation of Mass)
        # Total Mass = (val_l * x0) + (val_r * (L - x0))
        # Average = Total Mass / L
        self.a0 = (self.val_l * self.x0 + self.val_r * (self.L - self.x0)) / self.L
        
        # 2. Harmonics (An): 
        # An = (2/L) * ∫[0,L] f(x) cos(nπx/L) dx
        # Integration splits into [0, x0] and [x0, L].
        # Result simplifies to: (2 * (val_l - val_r) / nπ) * sin(nπx0/L)
        
        self.coeffs = []
        self.wave_numbers = [] # k = nπ/L
        
        delta_v = self.val_l - self.val_r
        
        for n in range(1, self.n_terms + 1):
            k = (n * np.pi) / self.L
            
            # An coefficient derivation:
            # ∫[0,x0] val_l*cos(kx)dx = (val_l/k) * sin(kx0)
            # ∫[x0,L] val_r*cos(kx)dx = (val_r/k) * (sin(kL) - sin(kx0))
            # Since sin(kL) = sin(nπ) = 0, the second term is -(val_r/k)*sin(kx0)
            # Sum = ((val_l - val_r) / k) * sin(kx0)
            # Multiply by normalizing factor (2/L)
            
            an = (2.0 / self.L) * (delta_v / k) * np.sin(k * self.x0)
            
            self.coeffs.append(an)
            self.wave_numbers.append(k)
            
        self.coeffs = np.array(self.coeffs)
        self.wave_numbers = np.array(self.wave_numbers)

    def evaluate(self, coordinates: Union[np.ndarray, Tuple[np.ndarray, ...]], t: float) -> np.ndarray:
        """Evaluate the Fourier series solution at time t."""
        
        # 1. Extract the relevant coordinate array based on axis
        if isinstance(coordinates, (list, tuple)):
            # If we have meshgrids (X, Y), pick the one for self.axis
            if self.axis < len(coordinates):
                x = coordinates[self.axis]
            else:
                raise IndexError(f"Axis {self.axis} out of range for provided coordinates.")
        else:
            # Single array provided (1D simulation)
            x = coordinates
            
        # 2. Initialize with steady state value
        u = np.full_like(x, self.a0, dtype=float)
        
        # 3. Sum Fourier terms
        # u(x,t) = A0 + Σ An * cos(kx) * exp(-D*k²*t)
        
        # Calculate decay for all terms at current time t
        # Shape: (n_terms,)
        decay = np.exp(-self.D * (self.wave_numbers**2) * t)
        
        # Effective coefficients for this time step
        # Shape: (n_terms,)
        time_coeffs = self.coeffs * decay
        
        # Vectorized summation
        # Compute cos(kx) for all k and all x
        # Outer product: (n_terms, x_shape)
        # Note: flatten x to ensure dot product works, then reshape back
        x_flat = x.flatten()
        kx = np.outer(self.wave_numbers, x_flat)
        cos_terms = np.cos(kx)
        
        # Dot product sums over n_terms
        # (n_terms,) dot (n_terms, x_flat_size) -> (x_flat_size,)
        summation = np.dot(time_coeffs, cos_terms)
        
        return u + summation.reshape(x.shape)

    def get_description(self) -> str:
        return (f"1D Step Diffusion (L={self.L}, Step at {self.x0}, "
                f"Values {self.val_l}|{self.val_r}, Axis={self.axis})")

class StepFunctionDiffusion2D(GoldenSolution):
    """
    Make use of the fact that a 2D step function is the product of 2 1D step functions.
    u(x,y,t) = u_x(x,t) * u_y(y,t)
    
    :var spec: Description
    :vartype spec: dict
    :var types: Description
    """
    def __init__(self, 
                 domain_size: Tuple[float, float] = (1.0, 1.0), 
                 split_point: Tuple[float, float] = (0.5, 0.5),
                 val_in: float = 1.0,    # Value inside the corner (e.g., Bottom-Left)
                 val_out: float = 0.0,   # Value everywhere else
                 diffusion_coefficient: float = 1.0, 
                 n_terms: int = 100):
        
        self.D = diffusion_coefficient
        
        # We construct the 2D solution by multiplying two 1D solutions.
        # Logic: 
        # u_x is 1.0 for x < split_x, else 0.0
        # u_y is val_in for y < split_y, else 0.0 (We put the amplitude here)
        # Result: val_in only where BOTH are true.
        
        # Solution for X direction (Normalized to 0-1)
        self.sol_x = StepFunctionDiffusion1D(
            domain_length=domain_size[0],
            position=split_point[0],
            value_left=1.0,  # "Active" region
            value_right=0.0,
            diffusion_coefficient=diffusion_coefficient,
            n_terms=n_terms,
            axis=0
        )
        
        # Solution for Y direction (Carries the actual amplitude val_in)
        self.sol_y = StepFunctionDiffusion1D(
            domain_length=domain_size[1],
            position=split_point[1],
            value_left=val_in, # "Active" region
            value_right=val_out, # Assuming val_out is 0 for simple corner
            diffusion_coefficient=diffusion_coefficient,
            n_terms=n_terms,
            axis=1
        )

    def evaluate(self, coordinates: Tuple[np.ndarray, np.ndarray], t: float) -> np.ndarray:
        """
        Evaluate u(x,y,t) = u_x(x,t) * u_y(y,t)
        """
        X, Y = coordinates
        
        # Calculate 1D profiles independently
        # Note: We pass the full 2D meshgrids; the 1D class handles them fine via axis logic
        u_x = self.sol_x.evaluate(X, t)
        u_y = self.sol_y.evaluate(Y, t)
        
        return u_x * u_y

    def get_description(self) -> str:
        return "2D Corner Step Diffusion (Product Solution)"

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
    
    elif solution_type == 'step_function_1d':
        return StepFunctionDiffusion1D(
            domain_length=spec.get('domain_length', 1.0),
            position=spec.get('position', 0.5),
            value_left=spec.get('value_left', 1.0),
            value_right=spec.get('value_right', 0.0),
            axis=spec.get('axis', 0),
            diffusion_coefficient=spec.get('diffusion_coefficient', 1.0),
            n_terms=spec.get('n_terms', 200)
        )
    
    elif solution_type == 'step_function_2d':
        return StepFunctionDiffusion2D(
            domain_size=spec.get('domain_size', (1.0, 1.0)),
            split_point=spec.get('split_point', (0.5, 0.5)),
            val_in=spec.get('val_in', 1.0),
            diffusion_coefficient=spec.get('diffusion_coefficient', 1.0),
            n_terms=spec.get('n_terms', 100)
        )

    else:
        raise ValueError(f"Unknown golden solution type: {solution_type}")
