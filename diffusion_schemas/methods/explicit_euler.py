"""
Explicit Euler (Forward-Time Central-Space) method for diffusion equation.

This module implements the explicit finite difference scheme for solving
the diffusion equation. The method is conditionally stable.
"""

import numpy as np
import warnings
from diffusion_schemas.base import Schema


class ExplicitEulerSchema(Schema):
    """
    Explicit Euler method for the diffusion equation.
    
    Implements the Forward-Time Central-Space (FTCS) finite difference scheme:
        u^(n+1) = u^n + dt * (D * ∇²u^n - λ * u^n + S^n)
    
    The method is conditionally stable. For stability in d dimensions:
        dt ≤ (dx²) / (2 * d * D)
    
    The implementation automatically checks stability and warns if violated.
    
    Parameters
    ----------
    domain_size : Union[float, Tuple[float, ...]]
        Size of the domain in each dimension.
    grid_points : Union[int, Tuple[int, ...]]
        Number of grid points in each dimension.
    dt : float
        Time step size.
    diffusion_coefficient : float, optional
        Diffusion coefficient D. Default is 1.0.
    decay_rate : float, optional
        Decay rate λ. Default is 0.0.
    check_stability : bool, optional
        Whether to check and warn about stability violations. Default is True.
    spatial_discretization : 'centered' (default), 'forward_1', or 'backward_1'.
    """
    
    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0,
        check_stability=True,
        spatial_discretization='centered'
    ):
        """Initialize the explicit Euler schema."""
        super().__init__(domain_size, grid_points, dt, diffusion_coefficient, decay_rate)
        self.spatial_discretization = spatial_discretization
        self.check_stability = check_stability
        
        if self.check_stability:
            self._verify_stability()
    
    def _verify_stability(self) -> None:
        """Check if the time step satisfies the stability condition."""
        D = self.diffusion_coefficient
        
        # For each dimension, compute CFL number
        r_values = []
        for dx_i in self.dx:
            r = D * self.dt / (dx_i ** 2)
            r_values.append(r)
        
        # Total CFL number
        r_total = sum(r_values)
        
        # Stability condition: r_total <= 0.5 (for d dimensions)
        max_stable = 0.5
        
        if r_total > max_stable:
            # Compute maximum stable time step
            min_dx_squared = min(dx_i ** 2 for dx_i in self.dx)
            dt_max = min_dx_squared / (2 * self.ndim * D) if D > 0 else float('inf')
            
            warnings.warn(
                f"\nStability condition violated! "
                f"\nCFL number = {r_total:.4f} > {max_stable}. "
                f"For stability in {self.ndim}D, use dt ≤ {dt_max:.6e}. "
                f"Current dt = {self.dt:.6e}.",
                UserWarning,
                stacklevel=3
            )
    
    def step(self) -> None:
        """Perform one explicit Euler time step."""
        # Compute Laplacian (∇²u)
        laplacian = self._compute_laplacian(self.state)
        
        # Compute source term from agents
        source = self._compute_source_term()
        
        # Explicit Euler update
        # u^(n+1) = u^n + dt * (D * ∇²u - λ * u + S)
        self.state = (
            self.state
            + self.dt * (
                self.diffusion_coefficient * laplacian
                - self.decay_rate * self.state
                + source
            )
        )
        
        # Apply boundary conditions
        if self._boundary_conditions is not None:
            self.state = self._apply_boundary_conditions(self.state)
        
        # Update time
        self.t += self.dt
    
    def _compute_laplacian(self, u: np.ndarray) -> np.ndarray:
        """
        Compute the Laplacian using central finite differences.
        
        Parameters
        ----------
        u : np.ndarray
            Field to compute Laplacian of.
            
        Returns
        -------
        np.ndarray
            Laplacian of u.
        """
        if self.ndim == 1:
            return self._laplacian_1d(u)
        elif self.ndim == 2:
            return self._laplacian_2d(u)
        elif self.ndim == 3:
            return self._laplacian_3d(u)
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.ndim}")
    
    def _laplacian_1d(self, u: np.ndarray) -> np.ndarray:
        """Compute 1D Laplacian: d²u/dx²."""
        laplacian = np.zeros_like(u)
        dx = self.dx[0]
        
        # could be optimized by using method _compute_derivative_along_axis with axis=0 instead of explicit loops

        if self.spatial_discretization == 'centered':

            # Interior points: central difference
            laplacian[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
            
            # Boundary points (will be overwritten by BC if needed)
            # Enforcing Neumann BC (zero flux) at boundaries for stability
            laplacian[0] = (u[1] - 2*u[0] + u[1]) / (dx**2)  # One-sided
            laplacian[-1] = (u[-2] - 2*u[-1] + u[-2]) / (dx**2)

        elif self.spatial_discretization == 'forward_1':
            # u[i+2] - 2u[i+1] + u[i]

            # Interior points
            laplacian[:-2] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
        
            # Boundary points N-2 and N-1
            # instead of forward discretization, we can use backward discretization at the end
            laplacian[-2] = (u[-2] - 2*u[-3] + u[-4]) / (dx**2)
            laplacian[-1] = (u[-1] - 2*u[-2] + u[-3]) / (dx**2)

        elif self.spatial_discretization == 'backward_1':
            # u[i] - 2u[i-1] + u[i-2]

            # Interior points
            laplacian[2:] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
        
            # Boundary points 0 and 1
            # instead of backward discretization, we can use forward discretization at the beginning
            laplacian[0] = (u[0] - 2*u[1] + u[2]) / (dx**2)
            laplacian[1] = (u[1] - 2*u[0] + u[1]) / (dx**2)

        return laplacian
    
    def _laplacian_2d(self, u: np.ndarray) -> np.ndarray:
        """Compute 2D Laplacian: d²u/dx² + d²u/dy²."""
        laplacian = np.zeros_like(u)
        dx, dy = self.dx
        
        if self.spatial_discretization == 'centered': 
            # could be optimized by applying _compute_derivative_along_axis for each axis instead of full 2D loops

            # Interior points
            laplacian[1:-1, 1:-1] = (
                (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx**2)
                + (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy**2)
            )
            
            # Boundaries (simplified - will be overwritten by BC if needed)
            # Left and right edges
            laplacian[0, 1:-1] = (
                (u[1, 1:-1] - 2*u[0, 1:-1] + u[1, 1:-1]) / (dx**2)
                + (u[0, 2:] - 2*u[0, 1:-1] + u[0, :-2]) / (dy**2)
            )
            laplacian[-1, 1:-1] = (
                (u[-2, 1:-1] - 2*u[-1, 1:-1] + u[-2, 1:-1]) / (dx**2)
                + (u[-1, 2:] - 2*u[-1, 1:-1] + u[-1, :-2]) / (dy**2)
            )
            
            # Bottom and top edges
            laplacian[1:-1, 0] = (
                (u[2:, 0] - 2*u[1:-1, 0] + u[:-2, 0]) / (dx**2)
                + (u[1:-1, 1] - 2*u[1:-1, 0] + u[1:-1, 1]) / (dy**2)
            )
            laplacian[1:-1, -1] = (
                (u[2:, -1] - 2*u[1:-1, -1] + u[:-2, -1]) / (dx**2)
                + (u[1:-1, -2] - 2*u[1:-1, -1] + u[1:-1, -2]) / (dy**2)
            )
            
            # Corners
            laplacian[0, 0] = (u[1, 0] - 2*u[0, 0] + u[1, 0]) / (dx**2) + (u[0, 1] - 2*u[0, 0] + u[0, 1]) / (dy**2)
            laplacian[0, -1] = (u[1, -1] - 2*u[0, -1] + u[1, -1]) / (dx**2) + (u[0, -2] - 2*u[0, -1] + u[0, -2]) / (dy**2)
            laplacian[-1, 0] = (u[-2, 0] - 2*u[-1, 0] + u[-2, 0]) / (dx**2) + (u[-1, 1] - 2*u[-1, 0] + u[-1, 1]) / (dy**2)
            laplacian[-1, -1] = (u[-2, -1] - 2*u[-1, -1] + u[-2, -1]) / (dx**2) + (u[-1, -2] - 2*u[-1, -1] + u[-1, -2]) / (dy**2)

        if self.spatial_discretization in ['forward_1', 'backward_1']:
            # Apply 1D Laplacian along x-axis
            laplacian += self._compute_derivative_along_axis(u, axis=0, h=dx)
            # Apply 1D Laplacian along y-axis
            laplacian += self._compute_derivative_along_axis(u, axis=1, h=dy)
        
        return laplacian
    
    def _laplacian_3d(self, u: np.ndarray) -> np.ndarray:
        """Compute 3D Laplacian: d²u/dx² + d²u/dy² + d²u/dz²."""
        laplacian = np.zeros_like(u)
        dx, dy, dz = self.dx
        
        if self.spatial_discretization == 'centered': 
            # could be optimized by applying _compute_derivative_along_axis for each axis instead of full 3D loops

            # Interior points
            laplacian[1:-1, 1:-1, 1:-1] = (
                (u[2:, 1:-1, 1:-1] - 2*u[1:-1, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]) / (dx**2)
                + (u[1:-1, 2:, 1:-1] - 2*u[1:-1, 1:-1, 1:-1] + u[1:-1, :-2, 1:-1]) / (dy**2)
                + (u[1:-1, 1:-1, 2:] - 2*u[1:-1, 1:-1, 1:-1] + u[1:-1, 1:-1, :-2]) / (dz**2)
            )
            
            # Note: Boundary handling for 3D is simplified
            # In practice, boundaries will be handled by BC application
        
        if self.spatial_discretization in ['forward_1', 'backward_1']:
            # Apply 1D Laplacian along x-axis
            laplacian += self._compute_derivative_along_axis(u, axis=0, h=dx)
            # Apply 1D Laplacian along y-axis
            laplacian += self._compute_derivative_along_axis(u, axis=1, h=dy)
            # Apply 1D Laplacian along z-axis
            laplacian += self._compute_derivative_along_axis(u, axis=2, h=dz)

        return laplacian

    def _compute_derivative_along_axis(self, u: np.ndarray, axis: int, h: float) -> np.ndarray:
        """
        Apply 1D Laplacian stencil along a specific axis (0 or 1), 
        handling boundaries for that entire edge.

        Boundary handling is done by applying:
        - Opposite method for forward or backward discretization at the edges
        - No-flux (Neumann) conditions at boundaries for centered differences
        """
        # Move the target axis to position 0 so we can treat it like a 1D problem
        # u_swapped shape becomes (N_target, N_other)
        u_swapped = np.moveaxis(u, axis, 0)
        d2u = np.zeros_like(u_swapped)
        
        if self.spatial_discretization == 'forward_1':
            # Main Body: u[i+2] - 2u[i+1] + u[i]
            d2u[:-2] = (u_swapped[2:] - 2*u_swapped[1:-1] + u_swapped[:-2]) / h**2
            
            # Patch Right Edge (Rows N-2 and N-1) using Backward Difference
            d2u[-2] = (u_swapped[-2] - 2*u_swapped[-3] + u_swapped[-4]) / h**2
            d2u[-1] = (u_swapped[-1] - 2*u_swapped[-2] + u_swapped[-3]) / h**2

        elif self.spatial_discretization == 'backward_1':
            # Main Body: u[i] - 2u[i-1] + u[i-2]
            d2u[2:] = (u_swapped[2:] - 2*u_swapped[1:-1] + u_swapped[:-2]) / h**2
            
            # Patch Left Edge (Rows 0 and 1) using Forward Difference
            d2u[1] = (u_swapped[3] - 2*u_swapped[2] + u_swapped[1]) / h**2
            d2u[0] = (u_swapped[2] - 2*u_swapped[1] + u_swapped[0]) / h**2

        elif self.spatial_discretization == 'centered':
            # Main Body
            d2u[1:-1] = (u_swapped[2:] - 2*u_swapped[1:-1] + u_swapped[:-2]) / h**2
            
            # Patch Boundaries with Forward (Left) and Backward (Right)
            # d2u[0]  = (u_swapped[2] - 2*u_swapped[1] + u_swapped[0]) / h**2
            # d2u[-1] = (u_swapped[-1] - 2*u_swapped[-2] + u_swapped[-3]) / h**2

            # Path Boundaries with Neumann (zero flux)
            d2u[0] = (u_swapped[1] - 2*u_swapped[0] + u_swapped[1]) / h**2
            d2u[-1] = (u_swapped[-2] - 2*u_swapped[-1] + u_swapped[-2]) / h**2

        # Move the axis back to its original position
        return np.moveaxis(d2u, 0, axis)