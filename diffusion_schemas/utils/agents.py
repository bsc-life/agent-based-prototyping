"""
Agent system for substrate-secreting sources.

This module implements agents that can secrete substrates into the diffusion
field at specified locations.
"""

from typing import Tuple, Callable, Optional, List
import numpy as np


class Agent:
    """
    Substrate-secreting agent with position and secretion rate.
    
    Agents produce a source term in the diffusion equation, representing
    cells, particles, or other entities that release chemical substances.
    
    Parameters
    ----------
    position : Tuple[float, ...]
        Position of the agent in the domain (x,) for 1D, (x, y) for 2D,
        or (x, y, z) for 3D.
    secretion_rate : Union[float, Callable]
        Rate of substrate secretion. Can be a constant or a function
        of time: secretion_rate(t) -> float.
    kernel_width : float, optional
        Width of the Gaussian kernel used to distribute the source.
        If None, uses a point source approximation. Default is None.
    name : str, optional
        Optional name for the agent.
        
    Attributes
    ----------
    position : Tuple[float, ...]
        Agent position.
    secretion_rate : Union[float, Callable]
        Secretion rate.
    kernel_width : Optional[float]
        Kernel width for source distribution.
    name : str
        Agent name.
    """
    
    def __init__(
        self,
        position: Tuple[float, ...],
        secretion_rate: float = 1.0,
        kernel_width: Optional[float] = None,
        name: str = ""
    ):
        """Initialize the agent."""
        self.position = tuple(position)
        self.secretion_rate = secretion_rate
        self.kernel_width = kernel_width
        self.name = name or f"Agent_{id(self)}"
        
    def get_secretion_rate(self, t: float) -> float:
        """
        Get the secretion rate at time t.
        
        Parameters
        ----------
        t : float
            Current time.
            
        Returns
        -------
        float
            Secretion rate at time t.
        """
        if callable(self.secretion_rate):
            return self.secretion_rate(t)
        return self.secretion_rate
    
    def compute_source(
        self,
        coords: List[np.ndarray],
        dx: Tuple[float, ...],
        t: float
    ) -> np.ndarray:
        """
        Compute the source term contribution from this agent.
        
        Parameters
        ----------
        coords : List[np.ndarray]
            Coordinate grids for each dimension.
        dx : Tuple[float, ...]
            Grid spacing in each dimension.
        t : float
            Current time.
            
        Returns
        -------
        np.ndarray
            Source term array with same shape as coordinate grids.
        """
        rate = self.get_secretion_rate(t)
        
        if rate == 0:
            # No secretion
            if len(coords) == 1:
                return np.zeros_like(coords[0])
            else:
                return np.zeros_like(coords[0])
        
        ndim = len(self.position)
        
        if self.kernel_width is None:
            # Point source approximation using delta function
            return self._point_source(coords, dx, rate)
        else:
            # Smooth Gaussian source
            return self._gaussian_source(coords, rate)
    
    def _point_source(
        self,
        coords: List[np.ndarray],
        dx: Tuple[float, ...],
        rate: float
    ) -> np.ndarray:
        """
        Create a point source at the agent position.
        
        Approximates a delta function by placing the source at the
        nearest grid point, normalized by grid volume.
        """
        ndim = len(self.position)
        
        if ndim == 1:
            x = coords[0] if isinstance(coords[0], np.ndarray) else coords[0]
            # Find nearest grid point
            idx = np.argmin(np.abs(x - self.position[0]))
            source = np.zeros_like(x)
            # Normalize by grid spacing (1D volume element)
            source[idx] = rate / dx[0]
            return source
            
        else:
            # Multi-dimensional case
            # Find nearest grid point for each dimension
            indices = []
            for i, (coord, pos) in enumerate(zip(coords, self.position)):
                # For meshgrid, take one slice to get 1D array
                if ndim == 2:
                    coord_1d = coord[:, 0] if i == 0 else coord[0, :]
                elif ndim == 3:
                    if i == 0:
                        coord_1d = coord[:, 0, 0]
                    elif i == 1:
                        coord_1d = coord[0, :, 0]
                    else:
                        coord_1d = coord[0, 0, :]
                else:
                    coord_1d = coord
                    
                idx = np.argmin(np.abs(coord_1d - pos))
                indices.append(idx)
            
            # Create source array
            source = np.zeros_like(coords[0])
            
            # Place point source at nearest grid point
            # Normalize by grid volume
            grid_volume = np.prod(dx)
            if ndim == 2:
                source[indices[0], indices[1]] = rate / grid_volume
            elif ndim == 3:
                source[indices[0], indices[1], indices[2]] = rate / grid_volume
                
            return source
    
    def _gaussian_source(
        self,
        coords: List[np.ndarray],
        rate: float
    ) -> np.ndarray:
        """
        Create a Gaussian-distributed source centered at the agent position.
        
        The Gaussian kernel smoothly distributes the source over nearby grid
        points, avoiding numerical singularities.
        """
        ndim = len(self.position)
        sigma = self.kernel_width
        
        # Compute squared distance from agent position
        if ndim == 1:
            x = coords[0] if isinstance(coords[0], np.ndarray) else coords[0]
            r_squared = (x - self.position[0]) ** 2
        else:
            r_squared = np.zeros_like(coords[0])
            for i, (coord, pos) in enumerate(zip(coords, self.position)):
                r_squared += (coord - pos) ** 2
        
        # Gaussian kernel
        # Normalize so that integral over all space equals rate
        normalization = 1.0 / ((2 * np.pi * sigma**2) ** (ndim / 2))
        source = rate * normalization * np.exp(-r_squared / (2 * sigma**2))
        
        return source
    
    def set_position(self, position: Tuple[float, ...]) -> None:
        """
        Update the agent position.
        
        Parameters
        ----------
        position : Tuple[float, ...]
            New position.
        """
        if len(position) != len(self.position):
            raise ValueError(
                f"Position dimension mismatch: expected {len(self.position)}, "
                f"got {len(position)}"
            )
        self.position = tuple(position)
    
    def set_secretion_rate(self, rate: float) -> None:
        """
        Update the secretion rate.
        
        Parameters
        ----------
        rate : float
            New secretion rate (must be non-negative).
        """
        if rate < 0:
            raise ValueError("Secretion rate must be non-negative")
        self.secretion_rate = rate
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        rate = self.secretion_rate if not callable(self.secretion_rate) else "f(t)"
        return f"Agent(name='{self.name}', position={self.position}, rate={rate})"
