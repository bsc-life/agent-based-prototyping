"""
Crank-Nicolson method for diffusion equation.

This module implements the Crank-Nicolson finite difference scheme for solving
the diffusion equation. The method is unconditionally stable and second-order
accurate in time.
"""

import numpy as np
from scipy.sparse import diags, kron, eye, csr_matrix
from scipy.sparse.linalg import spsolve
from diffusion_schemas.base import Schema


class CrankNicolsonSchema(Schema):
    """
    Crank-Nicolson method for the diffusion equation.
    
    Implements the Crank-Nicolson (θ-method with θ=0.5) finite difference scheme:
        u^(n+1) = u^n + (dt/2) * (D*∇²u^n - λ*u^n + D*∇²u^(n+1) - λ*u^(n+1) + S^n + S^(n+1))
    
    Rearranging:
        (I - dt*D*L/2 + dt*λ*I/2) * u^(n+1) = (I + dt*D*L/2 - dt*λ*I/2) * u^n + dt*(S^n + S^(n+1))/2
    
    where L is the discrete Laplacian operator.
    
    The method is:
        - Unconditionally stable
        - Second-order accurate in time (vs. first-order for explicit/implicit Euler)
        - Requires solving a sparse linear system at each step
    
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
    theta : float, optional
        Weighting parameter (0 = explicit, 0.5 = Crank-Nicolson, 1 = implicit).
        Default is 0.5.
    """
    
    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0,
        theta=0.6
    ):
        """Initialize the Crank-Nicolson schema."""
        super().__init__(domain_size, grid_points, dt, diffusion_coefficient, decay_rate)
        
        if not 0 <= theta <= 1:
            raise ValueError("theta must be in [0, 1]")
        self.theta = theta
        
        # Build the system matrices
        self._build_system_matrices()
    
    def _build_system_matrices(self) -> None:
        """Build the sparse system matrices for the Crank-Nicolson scheme."""
        if self.ndim == 1:
            self.A_impl, self.A_expl = self._build_matrices_1d()
        elif self.ndim == 2:
            self.A_impl, self.A_expl = self._build_matrices_2d()
        elif self.ndim == 3:
            self.A_impl, self.A_expl = self._build_matrices_3d()
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.ndim}")
    
    def _build_matrices_1d(self):
        """Build the 1D system matrices."""
        N = self.grid_points[0]
        dx = self.dx[0]
        
        # Discrete Laplacian in 1D
        diag_main = -2 * np.ones(N) / (dx**2)
        diag_off = np.ones(N-1) / (dx**2)
        
        L = diags([diag_off, diag_main, diag_off], [-1, 0, 1], shape=(N, N), format='csr')
        
        # Identity matrix
        I = eye(N, format='csr')
        
        # Implicit side: I - θ*dt*D*L + θ*dt*λ*I
        A_impl = I - self.theta * self.dt * self.diffusion_coefficient * L + \
                 self.theta * self.dt * self.decay_rate * I
        
        # Explicit side: I + (1-θ)*dt*D*L - (1-θ)*dt*λ*I
        A_expl = I + (1 - self.theta) * self.dt * self.diffusion_coefficient * L - \
                 (1 - self.theta) * self.dt * self.decay_rate * I
        
        return A_impl, A_expl
    
    def _build_matrices_2d(self):
        """Build the 2D system matrices using Kronecker products."""
        Nx, Ny = self.grid_points
        dx, dy = self.dx
        
        # 1D Laplacian operators
        diag_main_x = -2 * np.ones(Nx) / (dx**2)
        diag_off_x = np.ones(Nx-1) / (dx**2)
        Lx = diags([diag_off_x, diag_main_x, diag_off_x], [-1, 0, 1], shape=(Nx, Nx), format='csr')
        
        diag_main_y = -2 * np.ones(Ny) / (dy**2)
        diag_off_y = np.ones(Ny-1) / (dy**2)
        Ly = diags([diag_off_y, diag_main_y, diag_off_y], [-1, 0, 1], shape=(Ny, Ny), format='csr')
        
        # # Applying non-permeability BC (remove or comment later)
        # Lx[0, 0] = -1.0 / (dx**2)
        # Lx[0, 1] = 1.0 / (dx**2)
        # Lx[-1, -1] = -1.0 / (dx**2)
        # Lx[-1, -2] = 1.0 / (dx**2)
        # Ly[0, 0] = -1.0 / (dy**2)
        # Ly[0, 1] = 1.0 / (dy**2)
        # Ly[-1, -1] = -1.0 / (dy**2)
        # Ly[-1, -2] = 1.0 / (dy**2)

        # 2D Laplacian
        Ix = eye(Nx, format='csr')
        Iy = eye(Ny, format='csr')
        
        L = kron(Lx, Iy) + kron(Ix, Ly)
        
        # Identity matrix
        I = eye(Nx * Ny, format='csr')
        
        # Implicit and explicit matrices
        A_impl = I - self.theta * self.dt * self.diffusion_coefficient * L + \
                 self.theta * self.dt * self.decay_rate * I
        
        A_expl = I + (1 - self.theta) * self.dt * self.diffusion_coefficient * L - \
                 (1 - self.theta) * self.dt * self.decay_rate * I
        
        return A_impl, A_expl
    
    def _build_matrices_3d(self):
        """Build the 3D system matrices using Kronecker products."""
        Nx, Ny, Nz = self.grid_points
        dx, dy, dz = self.dx
        
        # 1D Laplacian operators
        diag_main_x = -2 * np.ones(Nx) / (dx**2)
        diag_off_x = np.ones(Nx-1) / (dx**2)
        Lx = diags([diag_off_x, diag_main_x, diag_off_x], [-1, 0, 1], shape=(Nx, Nx), format='csr')
        
        diag_main_y = -2 * np.ones(Ny) / (dy**2)
        diag_off_y = np.ones(Ny-1) / (dy**2)
        Ly = diags([diag_off_y, diag_main_y, diag_off_y], [-1, 0, 1], shape=(Ny, Ny), format='csr')
        
        diag_main_z = -2 * np.ones(Nz) / (dz**2)
        diag_off_z = np.ones(Nz-1) / (dz**2)
        Lz = diags([diag_off_z, diag_main_z, diag_off_z], [-1, 0, 1], shape=(Nz, Nz), format='csr')
        
        # 3D Laplacian
        Ix = eye(Nx, format='csr')
        Iy = eye(Ny, format='csr')
        Iz = eye(Nz, format='csr')
        
        L = kron(kron(Lx, Iy), Iz) + kron(kron(Ix, Ly), Iz) + kron(kron(Ix, Iy), Lz)
        
        # Identity matrix
        I = eye(Nx * Ny * Nz, format='csr')
        
        # Implicit and explicit matrices
        A_impl = I - self.theta * self.dt * self.diffusion_coefficient * L + \
                 self.theta * self.dt * self.decay_rate * I
        
        A_expl = I + (1 - self.theta) * self.dt * self.diffusion_coefficient * L - \
                 (1 - self.theta) * self.dt * self.decay_rate * I
        
        return A_impl, A_expl
    
    def step(self) -> None:
        """Perform one Crank-Nicolson time step."""
        # Compute source term at current time
        source_n = self._compute_source_term()
        
        # For Crank-Nicolson, we approximate S^(n+1) ≈ S^n
        # (true CN would need to evaluate at n+1, but agents are at fixed positions)
        source_np1 = source_n  # Approximation
        
        # Right-hand side: A_expl * u^n + dt * θ * S^(n+1) + dt * (1-θ) * S^n
        rhs = self.A_expl.dot(self.state.flatten()) + \
              self.dt * self.theta * source_np1.flatten() + \
              self.dt * (1 - self.theta) * source_n.flatten()
        
        # Solve the linear system: A_impl * u^(n+1) = rhs
        u_new_flat = spsolve(self.A_impl, rhs)
        
        # Reshape to grid
        self.state = u_new_flat.reshape(self.grid_points)
        
        # Apply boundary conditions
        if self._boundary_conditions is not None:
            self.state = self._apply_boundary_conditions(self.state)
        
        # Update time
        self.t += self.dt
    
    def set_diffusion_coefficient(self, value: float) -> None:
        """
        Set the diffusion coefficient and rebuild system matrices.
        
        Parameters
        ----------
        value : float
            Diffusion coefficient D (must be non-negative).
        """
        super().set_diffusion_coefficient(value)
        self._build_system_matrices()
    
    def set_decay_rate(self, value: float) -> None:
        """
        Set the decay rate and rebuild system matrices.
        
        Parameters
        ----------
        value : float
            Decay rate λ (must be non-negative).
        """
        super().set_decay_rate(value)
        self._build_system_matrices()
