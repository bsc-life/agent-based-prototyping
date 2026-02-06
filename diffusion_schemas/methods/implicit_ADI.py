"""
Implicit Euler (Backward Euler) method for diffusion equation.

This module implements an implicit finite difference scheme for solving
the diffusion equation. The method is unconditionally stable but requires
solving a linear system at each time step.
"""

import numpy as np
from scipy.sparse import diags, eye, csr_matrix
from scipy.sparse.linalg import spsolve
from diffusion_schemas.base import Schema


class ADISchema(Schema):
    """
    Alternating Direction Implicit (ADI) method for the diffusion equation.
    
    Implements the Implicit Euler finite difference scheme:
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
        factor = 1 / 2
        
        # 1D Laplacian operators
        diag_main_x = -2 * np.ones(Nx) / (dx**2)
        diag_off_x = np.ones(Nx-1) / (dx**2)
        Lx = diags([diag_off_x, diag_main_x, diag_off_x], [-1, 0, 1], shape=(Nx, Nx), format='csr')
        
        diag_main_y = -2 * np.ones(Ny) / (dy**2)
        diag_off_y = np.ones(Ny-1) / (dy**2)
        Ly = diags([diag_off_y, diag_main_y, diag_off_y], [-1, 0, 1], shape=(Ny, Ny), format='csr')
        
        # 2D Laplacian: Lx ⊗ I + I ⊗ Ly
        # For ADI, we will split the operator into two parts
        Ix = eye(Nx, format='csr')
        Iy = eye(Ny, format='csr')
        
        # System matrices
        Ax = Ix - self.dt * self.diffusion_coefficient * Lx + factor * self.dt * self.decay_rate * Ix
        Ay = Iy - self.dt * self.diffusion_coefficient * Ly + factor * self.dt * self.decay_rate * Iy


        return Ax, Ay
    
    def _build_matrix_3d(self) -> csr_matrix:
        """Build the 3D system matrix using Kronecker products."""
        Nx, Ny, Nz = self.grid_points
        dx, dy, dz = self.dx
        factor = 1 / 3
        
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
        
        # System matrices
        Ax = Ix - self.dt * self.diffusion_coefficient * Lx + factor * self.dt * self.decay_rate * Ix
        Ay = Iy - self.dt * self.diffusion_coefficient * Ly + factor * self.dt * self.decay_rate * Iy
        Az = Iz - self.dt * self.diffusion_coefficient * Lz + factor * self.dt * self.decay_rate * Iz

        return Ax, Ay, Az
    
    def step(self) -> None:
        """Perform one implicit Euler time step."""
        # Compute source term
        source = self._compute_source_term()
        
        # Right-hand side: u^n + dt*S
        # Note: The diffusion and decay terms are included in the system matrix, so they are not part of the RHS
        # Additionally, no flattening is included now so as to maintain problem dimensions
        rhs = self.state + self.dt * source
        
        # Case distinguish based on dimensionality of the problem
        # ADI implementation

        if self.ndim == 1:
            # Solve the linear system A * u^(n+1) = rhs
            Ax = self.system_matrix
            self.state = spsolve(Ax, rhs)

        elif self.ndim == 2:
            Ax, Ay = self.system_matrix
            # Step 1: Solve (Ax) * u* = rhs
            u_star = spsolve(Ax, rhs)
            # Intermediate step: Apply boundary conditions 
            if self._boundary_conditions is not None: u_star = self._apply_boundary_conditions(u_star)
            # Step 2: Solve (Ay) * u^(n+1) = u*
            self.state = spsolve(Ay, u_star.T)
            # Transpose back to original shape
            self.state = self.state.T 

        elif self.ndim == 3:
            Ax, Ay, Az = self.system_matrix
            Nx, Ny, Nz = self.grid_points

            # Reshape to (Nx, Ny*Nz) to feed in spsolve
            rhs_x = rhs.reshape(Nx, Ny * Nz)
            # Step 1: Solve (Ax) * u* = rhs
            u_star = spsolve(Ax, rhs_x)
            # Reshape back to 3D before applying BC
            u_star = u_star.reshape(Nx, Ny, Nz)
            # Intermediate step: Apply boundary conditions 
            if self._boundary_conditions is not None: u_star = self._apply_boundary_conditions(u_star)

            # Transpose and reshape to (Ny, Nz, N) to feed in spsolve
            rhs_y = u_star.transpose(1,0,2).reshape(Ny, Nx * Nz)
            # Step 2: Solve (Ay) * u** = u*
            u_star_star = spsolve(Ay, rhs_y)
            # Transpose back to original shape
            u_star_star = u_star_star.reshape(Ny, Nx, Nz).transpose(1,0,2)
            # Intermediate step: Apply boundary conditions 
            if self._boundary_conditions is not None: u_star_star = self._apply_boundary_conditions(u_star_star)

            # Reshape to (Nz, Nx*Ny) to feed in spsolve
            rhs_z = u_star_star.transpose(2,0,1).reshape(Nz, Nx * Ny)
            # Step 3: Solve (Az) * u^(n+1) = u**
            self.state = spsolve(Az, rhs_z)
            # Reshape back to 3D and transpose back to original shape
            self.state = self.state.reshape(Nz, Nx, Ny).transpose(1,2,0)

        # Final boundary condition imposition 
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
