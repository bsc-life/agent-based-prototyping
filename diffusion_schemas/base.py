"""
Abstract base class for diffusion schemas.

This module defines the Schema interface that all concrete numerical methods
must implement for solving the diffusion equation with agent-based sources.
"""

from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Optional, Callable
import numpy as np
import warnings
from diffusion_schemas.utils.agents import CompleteAgent

from tqdm import tqdm


class Schema(ABC):
    """
    Abstract base class for diffusion equation solvers.
    
    This class provides the interface for solving the diffusion/heat equation:
        ∂u/∂t = D∇²u - λu + S(x, t)
    
    where:
        - u is the concentration/temperature field
        - D is the diffusion coefficient
        - λ is the decay rate
        - S(x, t) is the source term from agents
    
    Parameters
    ----------
    domain_size : Union[float, Tuple[float, ...]]
        Size of the domain in each dimension. Single float for 1D,
        tuple of (Lx, Ly) for 2D, or (Lx, Ly, Lz) for 3D.
    grid_points : Union[int, Tuple[int, ...]]
        Number of grid points in each dimension. Single int for 1D,
        tuple of (Nx, Ny) for 2D, or (Nx, Ny, Nz) for 3D.
    dt : float
        Time step size.
    diffusion_coefficient : float, optional
        Diffusion coefficient D. Default is 1.0.
    decay_rate : float, optional
        Decay rate λ. Default is 0.0 (no decay).
        
    Attributes
    ----------
    ndim : int
        Number of spatial dimensions (1, 2, or 3).
    domain_size : Tuple[float, ...]
        Domain size in each dimension.
    grid_points : Tuple[int, ...]
        Number of grid points in each dimension.
    dx : Tuple[float, ...]
        Grid spacing in each dimension.
    dt : float
        Time step size.
    t : float
        Current simulation time.
    state : np.ndarray
        Current concentration field.
    """
    
    def __init__(
        self,
        domain_size: Union[float, Tuple[float, ...]],
        grid_points: Union[int, Tuple[int, ...]],
        dt: float,
        diffusion_coefficient: float = 1.0,
        decay_rate: float = 0.0,
    ):
        """Initialize the diffusion schema."""
        # Convert scalar inputs to tuples
        if isinstance(domain_size, (int, float)):
            domain_size = (float(domain_size),)
        if isinstance(grid_points, int):
            grid_points = (grid_points,)
            
        self.domain_size = tuple(domain_size)
        self.grid_points = tuple(grid_points)
        self.ndim = len(self.domain_size)
        
        if len(self.grid_points) != self.ndim:
            raise ValueError(
                f"Mismatch: domain_size has {self.ndim} dimensions but "
                f"grid_points has {len(self.grid_points)} dimensions"
            )
        
        # Calculate grid spacing
        self.dx = tuple(L / (N - 1) for L, N in zip(self.domain_size, self.grid_points))
        
        # Time parameters
        self.dt = dt
        self.t = 0.0
        
        # Physical parameters
        self._diffusion_coefficient = diffusion_coefficient
        self._decay_rate = decay_rate
        
        # Initialize state (will be set by initial condition)
        self.state = np.zeros(self.grid_points)
        
        # Boundary conditions (default is Neumann with zero flux)
        self._boundary_conditions = None
        
        # Agents
        self._agents = []

        # Bulk regions
        self._bulk = None
        
    @property
    def diffusion_coefficient(self) -> float:
        """Get the diffusion coefficient."""
        return self._diffusion_coefficient
    
    def set_diffusion_coefficient(self, value: float) -> None:
        """
        Set the diffusion coefficient.
        
        Parameters
        ----------
        value : float
            Diffusion coefficient D (must be non-negative).
        """
        if value < 0:
            raise ValueError("Diffusion coefficient must be non-negative")
        self._diffusion_coefficient = value
        
    @property
    def decay_rate(self) -> float:
        """Get the decay rate."""
        return self._decay_rate
    
    def set_decay_rate(self, value: float) -> None:
        """
        Set the decay rate.
        
        Parameters
        ----------
        value : float
            Decay rate λ (must be non-negative).
        """
        if value < 0:
            raise ValueError("Decay rate must be non-negative")
        self._decay_rate = value
        
    def set_initial_condition(
        self,
        initial_condition: Union[np.ndarray, Callable, float]
    ) -> None:
        """
        Set the initial condition for the concentration field.
        
        Parameters
        ----------
        initial_condition : Union[np.ndarray, Callable, float]
            Initial condition. Can be:
            - numpy array with shape matching grid_points
            - callable that takes coordinate arrays and returns values
            - scalar float for uniform initial condition
        """
        if isinstance(initial_condition, np.ndarray):
            if initial_condition.shape != self.grid_points:
                raise ValueError(
                    f"Initial condition shape {initial_condition.shape} "
                    f"does not match grid shape {self.grid_points}"
                )
            self.state = initial_condition.copy()
        elif callable(initial_condition):
            # Create coordinate grids
            coords = self._create_coordinate_grids()
            self.state = initial_condition(*coords)
        elif isinstance(initial_condition, (int, float)):
            self.state = np.full(self.grid_points, float(initial_condition))
        else:
            raise TypeError(
                "Initial condition must be ndarray, callable, or scalar"
            )
        
        # Reset time
        self.t = 0.0
        
    def _create_coordinate_grids(self) -> List[np.ndarray]:
        """
        Create coordinate grids for the domain.
        
        Returns
        -------
        List[np.ndarray]
            List of coordinate arrays, one for each dimension.
        """
        # Create 1D coordinate arrays
        coords_1d = [
            np.linspace(0, L, N) for L, N in zip(self.domain_size, self.grid_points)
        ]
        
        # Create meshgrid
        if self.ndim == 1:
            return coords_1d
        else:
            return np.meshgrid(*coords_1d, indexing='ij')
    
    def set_boundary_conditions(self, boundary_conditions) -> None:
        """
        Set the boundary conditions.
        
        Parameters
        ----------
        boundary_conditions : BoundaryCondition or dict
            Boundary condition object or dictionary mapping boundary
            names to BoundaryCondition objects.
        """
        self._boundary_conditions = boundary_conditions
        
    def add_agent(self, agent) -> None:
        """
        Add a substrate-secreting agent.
        
        Parameters
        ----------
        agent : Agent
            Agent object with position and secretion rate.
        """
        # Validate agent position is within domain
        for i, (pos, L) in enumerate(zip(agent.position, self.domain_size)):
            if not (0 <= pos <= L):
                warnings.warn(
                    f"Agent position {agent.position} is outside domain bounds. "
                    f"Dimension {i}: position={pos}, domain=[0, {L}]"
                )
        
        self._agents.append(agent)
        
    def clear_agents(self) -> None:
        """Remove all agents."""
        self._agents = []
        
    def set_bulk(self, bulk) -> None:
        """
        Set the bulk region.
        
        Parameters
        ----------
        bulk : Bulk
            Bulk region object with compute_source method.
        """

        # Maybe adding a warning like in the agents?
        # In practice, no erorrs arise as overlap matrix just returns 0 for all voxels

        self._bulk = bulk
    
    def clear_bulk(self) -> None:
        """Remove bulk region."""
        self._bulk = None

    def get_state(self) -> np.ndarray:
        """
        Get the current concentration field.
        
        Returns
        -------
        np.ndarray
            Copy of the current state.
        """
        return self.state.copy()
    
    def reset(self) -> None:
        """Reset the simulation to initial state (zeros) and time zero."""
        self.state = np.zeros(self.grid_points)
        self.t = 0.0
        
    @abstractmethod
    def step(self) -> None:
        """
        Perform a single time step.
        
        This method must be implemented by concrete subclasses.
        """
        pass
    
    def solve(self, t_final: float, store_history: bool = False, progress: bool = False) -> Tuple[List[np.ndarray], List[float]]:
        """
        Solve the diffusion equation up to a final time.
        
        Parameters
        ----------
        t_final : float
            Final time to integrate to.
        store_history : bool, optional
            If True, store and return the state at each time step.
            Default is False.
        progress : bool, optional
            If True, display a progress bar (requires tqdm).
            Default is False.
            
        Returns
        -------
        Tuple[List[np.ndarray], List[float]]
            If store_history is True, returns list of states at each step and corresponding times.
            Otherwise returns list with a single element (final state) and corresponding time.
        """
        if t_final <= self.t:
            raise ValueError(f"t_final ({t_final}) must be greater than current time ({self.t})")
        
        history = []
        times = []

        if store_history:
            history.append(self.state.copy())
            times.append(self.t)
        
        pbar = None
        if progress:
            # Properly calculate number of steps to display in progress bar
            # accounting for floating point rounding issues
            n_steps = 0
            temp_t = self.t
            while temp_t < t_final:
                temp_t += self.dt
                n_steps += 1
            pbar = tqdm(total=n_steps, desc="Solving", unit="step", dynamic_ncols=True)

        while self.t < t_final:
            # # Adjust last step to hit t_final exactly if needed
            # if self.t + self.dt > t_final:
            #     old_dt = self.dt
            #     self.dt = t_final - self.t

            #     # Solve BC integration into matrices problem
            #     if hasattr(self, '_build_system_matrix'):
            #         self._build_system_matrix()
            #     elif hasattr(self, '_build_system_matrices'):
            #         self._build_system_matrices()

            #     self.step()
            #     self.dt = old_dt
            # else:
            #     self.step()
            self.step()
            
            if store_history:
                history.append(self.state.copy())
                times.append(self.t)
            # At each step, update the progress bar if enabled
            if pbar is not None:
                pbar.update(1)
        
        # Close progress bar if it was used
        if pbar is not None:
            pbar.close()

        # If history was not stored, return the final state as a single-element list for consistency
        if not store_history:
            history.append(self.state.copy())
            times.append(self.t)

        return history, times
    
    def _compute_source_term(self) -> np.ndarray:
        """
        Compute the source term from all agents and bulk regions.
        
        Returns
        -------
        np.ndarray
            Source term array with same shape as state.
        """
        if not self._agents and not self._bulk:
            return np.zeros_like(self.state)
        
        source = np.zeros_like(self.state)
        coords = self._create_coordinate_grids()
        
        for agent in self._agents:
            # if isinstance(agent, CompleteAgent):
            #     source += agent.compute_source(self.state, coords, self.dx, self.t)
            # else:
            #     source += agent.compute_source(coords, self.dx, self.t)
            source += agent.compute_source(self.state, coords, self.dx, self.dt, self.t)
        
        if self._bulk is not None:
            # there is no need to iterate here, as self._bulk is an object itself
            # and self._bulk.compute_source already computes the contribution from the whole bulk region
            # by iterating internally over its region list (or bulk in self._bulk)
            # pass the self.state in order to avoid going below 0
            source += self._bulk.compute_source(self.state, coords, self.dx, self.dt, self.t)

        return source
    
    def _apply_boundary_conditions(self, state: np.ndarray) -> np.ndarray:
        """
        Apply boundary conditions to the state.
        
        Parameters
        ----------
        state : np.ndarray
            State array to apply boundary conditions to.
            
        Returns
        -------
        np.ndarray
            State with boundary conditions applied.
        """
        if self._boundary_conditions is None:
            # Default: Neumann (zero flux) - do nothing (natural BC)
            return state
        
        # Apply boundary conditions
        return self._boundary_conditions.apply(state, self.dx, self.t)
