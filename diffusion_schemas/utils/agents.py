"""
Agent system for substrate-secreting sources.

This module implements agents that can secrete substrates into the diffusion
field at specified locations.
"""

from typing import Tuple, Union, Callable, Optional, List
import numpy as np


class Agent:
    """
    Agent with position and net rate source term.
    
    Agents produce a source term in the diffusion equation, representing
    cells, particles, or other entities that release or consume chemical substances.
    
    Parameters
    ----------
    position : Tuple[float, ...]
        Position of the agent in the domain (x,) for 1D, (x, y) for 2D,
        or (x, y, z) for 3D.
    net_rate : Union[float, Callable]
        Net rate of source/sink. Can be a constant or a function
        of time: net_rate(t) -> float.
    kernel_width : float, optional
        Width of the Gaussian kernel used to distribute the source.
        If None, uses a point source approximation. Default is None.
    name : str, optional
        Optional name for the agent.
        
    Attributes
    ----------
    position : Tuple[float, ...]
        Agent position.
    net_rate : Union[float, Callable]
        Net source/sink rate.
    kernel_width : Optional[float]
        Kernel width for source distribution.
    name : str
        Agent name.
    """
    
    def __init__(
        self,
        position: Tuple[float, ...],
        net_rate: float = 1.0,
        kernel_width: Optional[float] = None,
        name: str = ""
    ):
        """Initialize the agent."""
        self.position = tuple(position)
        self.net_rate = net_rate
        self.kernel_width = kernel_width
        self.name = name or f"Agent_{id(self)}"
        
    def get_net_rate(self, t: float) -> float:
        """
        Get the net rate at time t.
        
        Parameters
        ----------
        t : float
            Current time.
            
        Returns
        -------
        float
            Net rate at time t.
        """
        if callable(self.net_rate):
            return self.net_rate(t)
        return self.net_rate
    
    def _sample_field(
        self, 
        field: np.ndarray, 
        coords: List[np.ndarray]
    ) -> float:
        """
        Sample the substrate density (rho) at the agent's current position.
        Uses nearest-neighbor interpolation for simplicity.
        """
        indices = []
        for coord_grid, pos in zip(coords, self.position):
            # Assuming uniform grid, find index of nearest coordinate
            # Handle both 1D arrays and meshgrids
            if coord_grid.ndim > 1:
                # Extract 1D axis from meshgrid
                grid_1d = np.unique(coord_grid)
            else:
                grid_1d = coord_grid
                
            idx = (np.abs(grid_1d - pos)).argmin()
            indices.append(idx)
            
        return field[tuple(indices)]

    def compute_source(
        self,
        field: np.ndarray,
        coords: List[np.ndarray],
        dx: Tuple[float, ...],
        dt: float,
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
        rate = self.get_net_rate(t)
        
        if rate == 0:
            # No source/sink
            if len(coords) == 1:
                return np.zeros_like(coords[0])
            else:
                return np.zeros_like(coords[0])
        
        # Negative fields problem (revisit this logic)
        rho_local = self._sample_field(field, coords)
        if rate < 0 and rate*dt < -rho_local:
            # If local density is negative (should not happen), return zero source
            rate = -rho_local / dt

        ndim = len(self.position)
        
        if self.kernel_width is None or self.kernel_width < 1e-6:
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
                
            # print(f"Agent '{self.name}' at time {t:.3f}: Placing point source at indices {indices} with rate {rate:.3e}")
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
    
    def set_net_rate(self, rate: float) -> None:
        """
        Update the net rate.
        
        Parameters
        ----------
        rate : float
            New net rate.
        """
        self.net_rate = rate
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        rate = self.net_rate if not callable(self.net_rate) else "f(t)"
        return f"Agent(name='{self.name}', position={self.position}, net_rate={rate})"

class CompleteAgent(Agent):
    """
    Substrate-interacting agent with secretion (supply) and uptake capabilities.
    
    Implements the cell-based net source term:
    Net Rate = Supply - Uptake
             = S_k * (rho_star - rho) - U_k * rho
    
    Parameters
    ----------
    position : Tuple[float, ...]
        Position of the agent (x, y) or (x, y, z).
    secretion_rate : Union[float, Callable], optional
        Rate constant for secretion (Sk). Target density approach.
        Default is 0.0.
    uptake_rate : Union[float, Callable], optional
        Rate constant for uptake (Uk). Proportional to density.
        Default is 0.0.
    saturation_density : float, optional
        Target saturation density (rho*). The density at which secretion stops.
        Default is 0.0.
    kernel_width : float, optional
        Width of the Gaussian kernel. If None, uses point source.
    name : str, optional
        ID/Name for the agent.
    """
    
    def __init__(
        self,
        position: Tuple[float, ...],
        secretion_rate: Union[float, Callable] = 0.0,
        uptake_rate: Union[float, Callable] = 0.0,
        saturation_density: float = 0.0,
        kernel_width: Optional[float] = None,
        name: str = ""
    ):
        super().__init__(
            position=position,
            net_rate=0.0,
            kernel_width=kernel_width,
            name=name,
        )
        self.secretion_rate = secretion_rate
        self.uptake_rate = uptake_rate
        self.saturation_density = saturation_density
        
    def get_rates(self, t: float) -> Tuple[float, float]:
        """Get current Sk and Uk values at time t."""
        S = self.secretion_rate(t) if callable(self.secretion_rate) else self.secretion_rate
        U = self.uptake_rate(t) if callable(self.uptake_rate) else self.uptake_rate
        # print(f"Agent '{self.name}' at time {t:.3f}: S_k={S}, U_k={U}")
        return S, U

    def compute_source(
        self,
        field: np.ndarray,
        coords: List[np.ndarray],
        dx: Tuple[float, ...],
        dt: float,
        t: float
    ) -> np.ndarray:
        """
        Compute the net source term contribution from this agent.
        
        New Formula: Rate = Sk * (rho* - rho) - Uk * rho
        
        Parameters
        ----------
        field : np.ndarray
            Current substrate density field (rho).
        coords : List[np.ndarray]
            Coordinate grids.
        dx : Tuple[float, ...]
            Grid spacing.
        t : float
            Current time.
        """
        # 1. Get current parameters
        S_k, U_k = self.get_rates(t)
        rho_star = self.saturation_density
        
        # 2. Sample local density (rho) at agent position
        rho_local = self._sample_field(field, coords)

        # Negative fields problem (revisit this logic)
        # If local density is negative (should not happen)
        # Agent does not contribute to source term
        # if rho_local < 0:
        #     return np.zeros_like(field)
        
        # 3. Calculate Net Rate
        # Term 1: Supply (stops if rho_local = rho_star)
        # A supply term cannot be negative, as it would be uptaking
        supply_term = max(0.0, S_k * (rho_star - rho_local))
        
        # Term 2: Uptake (proportional to available density)
        # Uptake term contribution to net_rate would be positive if rho_local was negative
        # A return is made before this happens
        if rho_local < 0:
            uptake_term = 0.0
        else:
            uptake_term = U_k * rho_local
        
        net_rate = supply_term - uptake_term
        
        # If rate is effectively zero, return empty grid
        if abs(net_rate) < 1e-15:
            return np.zeros_like(field)
        elif net_rate < 0 and net_rate * dt < -rho_local:
            net_rate = -rho_local / dt

        # 4. Distribute this net rate spatially (Point or Gaussian)
        if self.kernel_width is None or self.kernel_width < 1e-6:
            return self._point_source(coords, dx, net_rate)
        else:
            return self._gaussian_source(coords, net_rate)

    def set_secretion_rate(self, rate: Union[float, Callable]) -> None:
        """Update secretion rate."""
        if not callable(rate) and rate < 0:
            raise ValueError("Secretion rate must be non-negative")
        self.secretion_rate = rate

    def set_uptake_rate(self, rate: Union[float, Callable]) -> None:
        """Update uptake rate."""
        if not callable(rate) and rate < 0:
            raise ValueError("Uptake rate must be non-negative")
        self.uptake_rate = rate

    def set_saturation_density(self, density: float) -> None:
        """Update saturation density."""
        self.saturation_density = density