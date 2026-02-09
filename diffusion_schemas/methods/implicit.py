"""
Implicit Euler (Backward Euler) method for diffusion equation.

This module implements an implicit finite difference scheme for solving
the diffusion equation. The method is unconditionally stable but requires
solving a linear system at each time step.
"""

import numpy as np
from scipy.sparse import diags, kron, eye, csr_matrix
from scipy.sparse.linalg import spsolve
from diffusion_schemas.base import Schema


class ImplicitEulerSchema(Schema):
    """
    Implicit Euler method for the diffusion equation.
    
    Implements the Backward Euler finite difference scheme:
        u^(n+1) = u^n + dt * (D * ∇²u^(n+1) - λ * u^(n+1) + S^(n+1))
    
    Rearranging:
        (I - dt*D*L + dt*λ*I) * u^(n+1) = u^n + dt*S^(n+1)
    
    where L is the discrete Laplacian operator.
    
    The method is unconditionally stable, allowing larger time steps than
    explicit methods, but requires solving a sparse linear system at each step.
    
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
    """
    
    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0
    ):
        """Initialize the implicit Euler schema."""
        super().__init__(domain_size, grid_points, dt, diffusion_coefficient, decay_rate)
        
        # Build the system matrix
        self._build_system_matrix()
    
    def _build_system_matrix(self) -> None:
        """Build the sparse system matrix for the implicit scheme."""
        if self.ndim == 1:
            self.system_matrix = self._build_matrix_1d()
        elif self.ndim == 2:
            self.system_matrix = self._build_matrix_2d()
        elif self.ndim == 3:
            self.system_matrix = self._build_matrix_3d()
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.ndim}")
    
    def _build_matrix_1d(self) -> csr_matrix:
        """Build the 1D system matrix (I - dt*D*L + dt*λ*I)."""
        N = self.grid_points[0]
        dx = self.dx[0]
        
        # Discrete Laplacian in 1D: [1, -2, 1] / dx²
        diag_main = -2 * np.ones(N) / (dx**2)
        diag_off = np.ones(N-1) / (dx**2)
        
        L = diags([diag_off, diag_main, diag_off], [-1, 0, 1], shape=(N, N), format='csr')
        
        # Identity matrix
        I = eye(N, format='csr')
        
        # System matrix: I - dt*D*L + dt*λ*I
        A = I - self.dt * self.diffusion_coefficient * L + self.dt * self.decay_rate * I
        
        return A
    
    def _build_matrix_2d(self) -> csr_matrix:
        """Build the 2D system matrix using Kronecker products."""
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

        # 2D Laplacian: Lx ⊗ I + I ⊗ Ly
        Ix = eye(Nx, format='csr')
        Iy = eye(Ny, format='csr')
        
        L = kron(Lx, Iy) + kron(Ix, Ly)
        
        # Identity matrix
        I = eye(Nx * Ny, format='csr')


        # System matrix
        A = I - self.dt * self.diffusion_coefficient * L + self.dt * self.decay_rate * I
        
        return A
    
    def _build_matrix_3d(self) -> csr_matrix:
        """Build the 3D system matrix using Kronecker products."""
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
        
        # 3D Laplacian: Lx ⊗ I ⊗ I + I ⊗ Ly ⊗ I + I ⊗ I ⊗ Lz
        Ix = eye(Nx, format='csr')
        Iy = eye(Ny, format='csr')
        Iz = eye(Nz, format='csr')
        
        L = kron(kron(Lx, Iy), Iz) + kron(kron(Ix, Ly), Iz) + kron(kron(Ix, Iy), Lz)
        
        # Identity matrix
        I = eye(Nx * Ny * Nz, format='csr')
        
        # System matrix
        A = I - self.dt * self.diffusion_coefficient * L + self.dt * self.decay_rate * I
        
        return A
    
    def step(self) -> None:
        """Perform one implicit Euler time step."""
        # Compute source term
        source = self._compute_source_term()
        
        # Right-hand side: u^n + dt*S
        rhs = self.state.flatten() + self.dt * source.flatten()
        
        # Solve the linear system: A * u^(n+1) = rhs
        u_new_flat = spsolve(self.system_matrix, rhs)
        
        # Reshape to grid
        self.state = u_new_flat.reshape(self.grid_points)
        
        # Apply boundary conditions
        if self._boundary_conditions is not None:
            self.state = self._apply_boundary_conditions(self.state)
        
        # Update time
        self.t += self.dt
    
    def set_diffusion_coefficient(self, value: float) -> None:
        """
        Set the diffusion coefficient and rebuild system matrix.
        
        Parameters
        ----------
        value : float
            Diffusion coefficient D (must be non-negative).
        """
        super().set_diffusion_coefficient(value)
        self._build_system_matrix()
    
    def set_decay_rate(self, value: float) -> None:
        """
        Set the decay rate and rebuild system matrix.
        
        Parameters
        ----------
        value : float
            Decay rate λ (must be non-negative).
        """
        super().set_decay_rate(value)
        self._build_system_matrix()
