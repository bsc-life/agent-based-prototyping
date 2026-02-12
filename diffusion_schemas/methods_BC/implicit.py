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
from diffusion_schemas.utils.boundary import DirichletBC, NeumannBC


class ImplicitEulerBCSchema(Schema):
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
        
        # Apply boundary conditions directly to the system if needed
        if isinstance(self._boundary_conditions, DirichletBC):
            rhs = self._apply_dirichlet_bc(rhs) # mofidies rhs and system matrix
        elif isinstance(self._boundary_conditions, NeumannBC):
            rhs = self._apply_neumann_bc(rhs)

        # Solve the linear system: A * u^(n+1) = rhs
        u_new_flat = spsolve(self.system_matrix, rhs)
        
        # Reshape to grid
        self.state = u_new_flat.reshape(self.grid_points)
        
        # Update time
        self.t += self.dt
    
    def _apply_dirichlet_bc(self, rhs):

        value = self._boundary_conditions._get_value(self.t)

        if self.ndim == 1:
            N = self.grid_points
            
            # Left boundary (i=0)
            self.system_matrix[0, :] = 0
            self.system_matrix[0, 0] = 1
            rhs[0] = value
            
            # Right boundary (i = N-1)
            self.system_matrix[-1, :] = 0
            self.system_matrix[-1, -1] = 1
            rhs[-1] = value

        if self.ndim == 2:
            Nx, Ny = self.grid_points
            
            # Left boundary (i=0) and right boundary (i=Nx-1)
            for j in range(Ny):
                idx = j * Nx  # index of (0, j)
                self.system_matrix[idx, :] = 0
                self.system_matrix[idx, idx] = 1
                rhs[idx] = value
            
                idx2 = j * Nx + (Nx - 1)  # index of (Nx-1, j)
                self.system_matrix[idx2, :] = 0
                self.system_matrix[idx2, idx2] = 1
                rhs[idx2] = value

            # Bottom boundary (j=0) and top boundary (j=Ny-1)
            for i in range(Nx):
                idx = i  # index of (i, 0)
                self.system_matrix[idx, :] = 0
                self.system_matrix[idx, idx] = 1
                rhs[idx] = value
            
                idx2 = (Ny - 1) * Nx + i  # index of (i, Ny-1)
                self.system_matrix[idx2, :] = 0
                self.system_matrix[idx2, idx2] = 1
                rhs[idx2] = value

        if self.ndim == 3:
            Nx, Ny, Nz = self.grid_points

            # X-direction boundaries
            for j in range(Ny):
                for k in range(Nz):
                    idx_left = 0 * Ny * Nz + j * Nz + k
                    idx_right = (Nx-1) * Ny * Nz + j * Nz + k
                    
                    self.system_matrix[idx_left, :] = 0
                    self.system_matrix[idx_left, idx_left] = 1
                    rhs[idx_left] = value
                    
                    self.system_matrix[idx_right, :] = 0
                    self.system_matrix[idx_right, idx_right] = 1
                    rhs[idx_right] = value
            
            # Y-direction boundaries
            for i in range(Nx):
                for k in range(Nz):
                    idx_bottom = i * Ny * Nz + 0 * Nz + k
                    idx_top = i * Ny * Nz + (Ny-1) * Nz + k
                    
                    self.system_matrix[idx_bottom, :] = 0
                    self.system_matrix[idx_bottom, idx_bottom] = 1
                    rhs[idx_bottom] = value
                    
                    self.system_matrix[idx_top, :] = 0
                    self.system_matrix[idx_top, idx_top] = 1
                    rhs[idx_top] = value
            
            # Z-direction boundaries
            for i in range(Nx):
                for j in range(Ny):
                    idx_front = i * Ny * Nz + j * Nz + 0
                    idx_back = i * Ny * Nz + j * Nz + (Nz-1)
                    
                    self.system_matrix[idx_front, :] = 0
                    self.system_matrix[idx_front, idx_front] = 1
                    rhs[idx_front] = value
                    
                    self.system_matrix[idx_back, :] = 0
                    self.system_matrix[idx_back, idx_back] = 1
                    rhs[idx_back] = value
        
        return rhs

        """
        if self.ndim == 1:
            N = self.grid_points
            dx = self.dx
            
            # Left boundary (i=0): one-sided stencil (u[1] - u[0]) / dx
            self.system_matrix[0, :] = 0
            self.system_matrix[0, 0] = -1 / (dx**2)
            self.system_matrix[0, 1] = 1 / (dx**2)
            rhs[0] = -2 * flux / dx
            
            # Right boundary (i=N-1): one-sided stencil (u[N-2] - u[N-1]) / dx
            self.system_matrix[-1, :] = 0
            self.system_matrix[-1, -2] = 1 / (dx**2)
            self.system_matrix[-1, -1] = -1 / (dx**2)
            rhs[-1] = 2 * flux / dx

        if self.ndim == 2:
            Nx, Ny = self.grid_points
            dx, dy = self.dx
            
            # Left boundary (i=0) and right boundary (i=Nx-1)
            for j in range(Ny):
                idx_left = j * Nx  # index of (0, j)
                idx_right = j * Nx + (Nx - 1)  # index of (Nx-1, j)

                self.system_matrix[idx_left, :] = 0
                self.system_matrix[idx_left, idx_left] = -1 / (dx**2)
                self.system_matrix[idx_left, idx_left + Ny] = 1 / (dx**2)
                self.system_matrix[idx_right, :] = 0
                self.system_matrix[idx_right, idx_right - Ny] = 1 / (dx**2)
                self.system_matrix[idx_right, idx_right] = -1 / (dx**2)

                rhs[idx_left] += self.dt * self.diffusion_coefficient * flux / dx
                rhs[idx_right] -= self.dt * self.diffusion_coefficient * flux / dx

            # Bottom boundary (j=0) and top boundary (j=Ny-1)
            for i in range(Nx):
                idx_bottom = i  # index of (i, 0)
                idx_top = (Ny - 1) * Nx + i  # index of (i, Ny-1)

                self.system_matrix[idx_bottom, :] = 0
                self.system_matrix[idx_bottom, idx_bottom] = -1 / (dy**2)
                self.system_matrix[idx_bottom, idx_bottom + Nx] = 1 / (dy**2)
                self.system_matrix[idx_top, :] = 0
                self.system_matrix[idx_top, idx_top - Nx] = 1 / (dy**2)
                self.system_matrix[idx_top, idx_top] = -1 / (dy**2)

                rhs[idx_bottom] += self.dt * self.diffusion_coefficient * flux / dy
                rhs[idx_top] -= self.dt * self.diffusion_coefficient * flux / dy

        if self.ndim == 3:
            Nx, Ny, Nz = self.grid_points
            dx, dy, dz = self.dx

            ???
        """

    def _apply_neumann_bc(self, rhs):
        flux = self._boundary_conditions._get_flux(self.t + self.dt)
        D = self.diffusion_coefficient
        dt = self.dt

        # --- 1D CASE ---
        if self.ndim == 1:
            N = self.grid_points[0]
            dx = self.dx[0]
            alpha = (dt * D) / (dx**2)
            forcing = (2 * dt * D * flux) / dx

            # Left (i=0): The neighbor (i=1) weight doubles
            self.system_matrix[0, 1] = -2 * alpha
            rhs[0] -= forcing 

            # Right (i=N-1): The neighbor (i=N-2) weight doubles
            self.system_matrix[-1, -2] = -2 * alpha
            rhs[-1] += forcing

        # --- 2D CASE ---
        elif self.ndim == 2:
            Nx, Ny = self.grid_points
            dx, dy = self.dx
            alpha_x = (dt * D) / (dx**2)
            alpha_y = (dt * D) / (dy**2)
            force_x = (2 * dt * D * flux) / dx
            force_y = (2 * dt * D * flux) / dy

            # Strides: X is slow (Ny), Y is fast (1)
            # Left (x=0) / Right (x=Nx-1)
            idx_l = np.arange(Ny)
            idx_r = np.arange((Nx - 1) * Ny, Nx * Ny)
            self.system_matrix[idx_l, idx_l + Ny] = -2 * alpha_x
            self.system_matrix[idx_r, idx_r - Ny] = -2 * alpha_x
            rhs[idx_l] -= force_x
            rhs[idx_r] += force_x

            # Bottom (y=0) / Top (y=Ny-1)
            idx_b = np.arange(0, Nx * Ny, Ny)
            idx_t = np.arange(Ny - 1, Nx * Ny, Ny)
            self.system_matrix[idx_b, idx_b + 1] = -2 * alpha_y
            self.system_matrix[idx_t, idx_t - 1] = -2 * alpha_y
            rhs[idx_b] -= force_y
            rhs[idx_t] += force_y

        # --- 3D CASE ---
        elif self.ndim == 3:
            Nx, Ny, Nz = self.grid_points
            dx, dy, dz = self.dx
            sx, sy, sz = Ny * Nz, Nz, 1
            alpha_x, alpha_y, alpha_z = (dt*D)/dx**2, (dt*D)/dy**2, (dt*D)/dz**2
            fx, fy, fz = (2*dt*D*flux)/dx, (2*dt*D*flux)/dy, (2*dt*D*flux)/dz

            # X-planes (Left/Right)
            idx_l, idx_r = np.arange(sx), np.arange((Nx-1)*sx, Nx*sx)
            self.system_matrix[idx_l, idx_l + sx] = -2 * alpha_x
            self.system_matrix[idx_r, idx_r - sx] = -2 * alpha_x
            rhs[idx_l] -= fx
            rhs[idx_r] += fx

            # Y-planes (Front/Back)
            base_y = np.arange(Nz)
            idx_f = np.concatenate([base_y + i*sx for i in range(Nx)])
            idx_bk = idx_f + (Ny-1)*sy
            self.system_matrix[idx_f, idx_f + sy] = -2 * alpha_y
            self.system_matrix[idx_bk, idx_bk - sy] = -2 * alpha_y
            rhs[idx_f] -= fy
            rhs[idx_bk] += fy

            # Z-planes (Bottom/Top)
            idx_bt = np.arange(0, Nx*Ny*Nz, Nz)
            idx_tp = np.arange(Nz-1, Nx*Ny*Nz, Nz)
            self.system_matrix[idx_bt, idx_bt + 1] = -2 * alpha_z
            self.system_matrix[idx_tp, idx_tp - 1] = -2 * alpha_z
            rhs[idx_bt] -= fz
            rhs[idx_tp] += fz

        return rhs
    
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
