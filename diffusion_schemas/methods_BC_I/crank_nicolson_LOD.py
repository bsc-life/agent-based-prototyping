"""
Crank-Nicolson method for diffusion equation using LOD splitting.

This module implements the Crank-Nicolson finite difference scheme with
Alternating Direction Implicit (ADI) splitting. It integrates Neumann and 
Dirichlet boundary conditions directly into the matrix operators and explicit
stencils, avoiding Operator Splitting errors.
"""

import numpy as np
from scipy.sparse import diags, eye, csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from diffusion_schemas.base import Schema
from diffusion_schemas.utils.boundary import DirichletBC, NeumannBC


class CrankNicolsonLODBCISchema(Schema):
    """
    Crank-Nicolson method for the diffusion equation using ADI.
    
    Implements the Crank-Nicolson (θ-method with θ=0.5) finite difference scheme.
    
    - Explicit Part (RHS): Evaluated using modified finite difference stencils 
      to account for BCs at time n.
    - Implicit Part (LHS): Solved using LOD splitting, with BCs injected 
      directly into the tridiagonal systems at time n+1.
    
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
        Weighting parameter. Default is 0.5 (Crank-Nicolson).
    """
    
    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0,
        theta=0.5
    ):
        """Initialize the Crank-Nicolson schema."""
        super().__init__(domain_size, grid_points, dt, diffusion_coefficient, decay_rate)
        
        if not 0 <= theta <= 1:
            raise ValueError("theta must be in [0, 1]")
        self.theta = theta
        
        # Build the system matrices
        self._build_system_matrices()
        self._boundary_mask = self._compute_boundary_indices()

    def _compute_boundary_indices(self) -> np.ndarray:
        """Precompute flattened boundary indices for the current grid shape."""
        mask = np.zeros(self.grid_points, dtype=bool)

        # First/last plane in x
        mask[0, ...] = True
        mask[-1, ...] = True

        if self.ndim >= 2:
            # First/last plane in y
            mask[:, 0, ...] = True
            mask[:, -1, ...] = True

        if self.ndim == 3:
            # First/last plane in z
            mask[:, :, 0] = True
            mask[:, :, -1] = True

        # ravel makes a 1D view 
        # flatnonzero returns integer positions where value is True
        # this can be used to index into flattened arrays 
        # return np.flatnonzero(mask.ravel())
        return mask
    
    def _build_system_matrices(self) -> None:
        """Build the sparse system matrices for the Crank-Nicolson scheme."""
        if self.ndim == 1:
            self.A_impl_x = self._build_matrices_1d()
        elif self.ndim == 2:
            self.A_impl_x, self.A_impl_y = self._build_matrices_2d()
        elif self.ndim == 3:
            self.A_impl_x, self.A_impl_y, self.A_impl_z = self._build_matrices_3d()
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.ndim}")
    
    def _build_matrices_1d(self):
        """Build the 1D system matrices."""
        N = self.grid_points[0]
        dx = self.dx[0]
        
        # Standard Laplacian for Matrix construction
        diag_main = -2 * np.ones(N) / (dx**2)
        diag_off = np.ones(N-1) / (dx**2)
        
        L = diags([diag_off, diag_main, diag_off], [-1, 0, 1], shape=(N, N), format='csr')
        I = eye(N, format='csr')
        
        # Implicit side: I - θ*dt*D*L + θ*dt*λ*I
        A_impl = I - self.theta * self.dt * self.diffusion_coefficient * L + \
                 self.theta * self.dt * self.decay_rate * I
        
        return A_impl
    
    def _build_matrices_2d(self):
        """Build the 2D system matrices using Kronecker products."""
        Nx, Ny = self.grid_points
        dx, dy = self.dx
        factor = 1 / 2 # LOD factor for 2D splitting
        
        # 1D Laplacian operators
        diag_main_x = -2 * np.ones(Nx) / (dx**2)
        diag_off_x = np.ones(Nx-1) / (dx**2)
        Lx = diags([diag_off_x, diag_main_x, diag_off_x], [-1, 0, 1], shape=(Nx, Nx), format='csr')
        
        diag_main_y = -2 * np.ones(Ny) / (dy**2)
        diag_off_y = np.ones(Ny-1) / (dy**2)
        Ly = diags([diag_off_y, diag_main_y, diag_off_y], [-1, 0, 1], shape=(Ny, Ny), format='csr')
        
        Ix = eye(Nx, format='csr')
        Iy = eye(Ny, format='csr')
        
        # Implicit matrices (Ax, Ay)
        A_impl_x = Ix - self.theta * self.dt * self.diffusion_coefficient * Lx + \
                   self.theta * factor * self.dt * self.decay_rate * Ix
        A_impl_y = Iy - self.theta * self.dt * self.diffusion_coefficient * Ly + \
                   self.theta * factor * self.dt * self.decay_rate * Iy
        
        return A_impl_x, A_impl_y
    
    def _build_matrices_3d(self):
        """Build the 3D system matrices using Kronecker products."""
        Nx, Ny, Nz = self.grid_points
        dx, dy, dz = self.dx
        factor = 1 / 3 # LOD factor for 3D splitting
        
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
        
        Ix = eye(Nx, format='csr')
        Iy = eye(Ny, format='csr')
        Iz = eye(Nz, format='csr')
        
        # Implicit matrices (Ax, Ay, Az)
        A_impl_x = Ix - self.theta * self.dt * self.diffusion_coefficient * Lx + \
                   self.theta * factor * self.dt * self.decay_rate * Ix
        A_impl_y = Iy - self.theta * self.dt * self.diffusion_coefficient * Ly + \
                   self.theta * factor * self.dt * self.decay_rate * Iy
        A_impl_z = Iz - self.theta * self.dt * self.diffusion_coefficient * Lz + \
                   self.theta * factor * self.dt * self.decay_rate * Iz
        
        return A_impl_x, A_impl_y, A_impl_z
    
    def step(self) -> None:
        """Perform one Crank-Nicolson time step with Integrated BCs."""
        # Compute source term at current time
        source_n = self._compute_source_term() # both agents and bulk at time n (explicit)

        # Compute source term at next time for implicit contribution
        t_next = self.t + self.dt      
        agent_source_np1 = self._compute_source_term(implicit=True, t=t_next) 
        bulk_rhs_np1 = np.zeros_like(self.state)
        bulk_lhs_np1 = np.zeros_like(self.state)
        if self._bulk is not None:
            bulk_rhs_np1 = self._bulk.rhs_contribution
            bulk_lhs_np1 = self._bulk.lhs_contribution.copy()
            # Preserve BC-imposed rows by not changing boundary diagonal entries.
            bulk_lhs_np1[self._boundary_mask] = 0.0

        # 1. Compute Explicit Part (Right-Hand Side)
        # We integrate the Explicit BC logic here using modified stencils
        # Build full RHS: u^n + explicit_term + source terms

        laplacian_n = self._compute_laplacian(self.state)

        rhs_grid = self.state + \
              (1 - self.theta) * self.dt * (
                  self.diffusion_coefficient * laplacian_n - 
                  self.decay_rate * self.state + 
                  source_n
              ) + \
              self.theta * self.dt * (bulk_rhs_np1 + agent_source_np1) 

        # 2. Solve the Implicit System (ADI)
        # We inject the Implicit BC logic here into the matrix solvers
        # Pass also to function the still unused bulk_lhs_np1 for matrix modification
        u_new_grid = self.step_lod(rhs_grid, bulk_lhs_np1) 
        
        # Reshape to grid
        self.state = u_new_grid

        # Update time
        self.t += self.dt

    def step_lod(self, rhs, bulk_lhs_np1):
        """
        Solve the implicit system using LOD splitting.
        Integrates Implicit BCs into each sweep using the LOD logic.
        """

        if self.ndim == 1:
            Ax = self.A_impl_x.copy().tolil()
            Nx = self.grid_points[0]
            rhs_x = rhs.reshape(Nx, 1)

            source_diag = diags([self.theta * self.dt * bulk_lhs_np1.reshape(Nx)], [0], shape=(Nx, Nx), format='csr')
            Ax_eff = (Ax + source_diag).tolil()

            rhs_x = self._apply_bc_to_sweep(Ax_eff, rhs_x, self.dx[0])
            self.state = spsolve(Ax_eff.tocsr(), rhs_x).reshape(self.grid_points)

        elif self.ndim == 2:
            Ax = self.A_impl_x.copy().tolil()  # LIL for efficient BC modification
            Ay = self.A_impl_y.copy().tolil()
            Nx, Ny = self.grid_points
            rhs = rhs.reshape(Nx, Ny)

            u_star = np.zeros((Nx, Ny))
            for j in range(Ny):
                source_diag = diags([self.theta * self.dt * bulk_lhs_np1[:, j]], [0], shape=(Nx, Nx), format='csr')
                Ax_j = (Ax + source_diag).tolil()

                rhs_j = rhs[:, j].reshape(Nx, 1)
                rhs_j = self._apply_bc_to_sweep(Ax_j, rhs_j, self.dx[0])
                u_star[:, j] = spsolve(Ax_j.tocsr(), rhs_j).flatten()

            u_new = np.zeros((Nx, Ny))
            for i in range(Nx):
                source_diag = diags([self.theta * self.dt * bulk_lhs_np1[i, :]], [0], shape=(Ny, Ny), format='csr')
                Ay_i = (Ay + source_diag).tolil()

                rhs_i = u_star[i, :].reshape(Ny, 1)
                rhs_i = self._apply_bc_to_sweep(Ay_i, rhs_i, self.dx[1])
                u_new[i, :] = spsolve(Ay_i.tocsr(), rhs_i).flatten()

            self.state = u_new

        elif self.ndim == 3:
            Ax = self.A_impl_x.copy().tolil()
            Ay = self.A_impl_y.copy().tolil()
            Az = self.A_impl_z.copy().tolil()
            Nx, Ny, Nz = self.grid_points
            rhs = rhs.reshape(Nx, Ny, Nz)

            u_star = np.zeros((Nx, Ny, Nz))
            for j in range(Ny):
                for k in range(Nz):
                    source_diag = diags([self.theta * self.dt * bulk_lhs_np1[:, j, k]], [0], shape=(Nx, Nx), format='csr')
                    Ax_jk = (Ax + source_diag).tolil()

                    rhs_jk = rhs[:, j, k].reshape(Nx, 1)
                    rhs_jk = self._apply_bc_to_sweep(Ax_jk, rhs_jk, self.dx[0])
                    u_star[:, j, k] = spsolve(Ax_jk.tocsr(), rhs_jk).flatten()

            u_star_star = np.zeros((Nx, Ny, Nz))
            for i in range(Nx):
                for k in range(Nz):
                    source_diag = diags([self.theta * self.dt * bulk_lhs_np1[i, :, k]], [0], shape=(Ny, Ny), format='csr')
                    Ay_ik = (Ay + source_diag).tolil()

                    rhs_ik = u_star[i, :, k].reshape(Ny, 1)
                    rhs_ik = self._apply_bc_to_sweep(Ay_ik, rhs_ik, self.dx[1])
                    u_star_star[i, :, k] = spsolve(Ay_ik.tocsr(), rhs_ik).flatten()

            u_new = np.zeros((Nx, Ny, Nz))
            for i in range(Nx):
                for j in range(Ny):
                    source_diag = diags([self.theta * self.dt * bulk_lhs_np1[i, j, :]], [0], shape=(Nz, Nz), format='csr')
                    Az_ij = (Az + source_diag).tolil()

                    rhs_ij = u_star_star[i, j, :].reshape(Nz, 1)
                    rhs_ij = self._apply_bc_to_sweep(Az_ij, rhs_ij, self.dx[2])
                    u_new[i, j, :] = spsolve(Az_ij.tocsr(), rhs_ij).flatten()

            self.state = u_new

        # Fix return type in step_lod
        return self.state 
    
    # =========================================================================
    # HELPER: IMPLICIT BC LOGIC (From LOD Schema)
    # =========================================================================
    def _apply_bc_to_sweep(self, matrix, rhs_array, h):
        """
        Apply boundary conditions to the 1D sweep system (Matrix and RHS).
        """
        if self._boundary_conditions is None:
            return rhs_array

        D = self.diffusion_coefficient
        dt = self.dt
        theta = self.theta
        
        if isinstance(self._boundary_conditions, NeumannBC):
            # Flux at n+1 (Implicit)
            flux = self._boundary_conditions._get_flux(self.t + self.dt)
            
            # Calculate alpha (off-diagonal weight in the matrix)
            # Matrix is I - theta*dt*D*L. L has 1/h^2.
            # So off-diagonal is: - (theta * dt * D) / h^2
            alpha = (theta * dt * D) / (h**2)
            
            # Forcing term from ghost point logic
            forcing = (2 * theta * dt * D * flux) / h
            
            # Left Boundary (i=0)
            # Modify neighbor weight (0,1) to be -2*alpha
            matrix[0, 1] = -2 * alpha
            # Modify RHS (subtract forcing based on sign convention)
            rhs_array[0, :] -= forcing
            
            # Right Boundary (i=N-1)
            matrix[-1, -2] = -2 * alpha
            rhs_array[-1, :] += forcing

        elif isinstance(self._boundary_conditions, DirichletBC):
            val = self._boundary_conditions._get_value(self.t + self.dt)
            
            # Zero out rows and set diagonal to 1
            matrix[0, :] = 0; matrix[0, 0] = 1; rhs_array[0, :] = val
            matrix[-1, :] = 0; matrix[-1, -1] = 1; rhs_array[-1, :] = val
            
        return rhs_array

    # =========================================================================
    # HELPER: EXPLICIT LAPLACIAN LOGIC (From ExplicitEuler Schema)
    # =========================================================================
    def _compute_laplacian(self, u: np.ndarray) -> np.ndarray:
        if self.ndim == 1: return self._laplacian_1d(u)
        elif self.ndim == 2: return self._laplacian_2d(u)
        elif self.ndim == 3: return self._laplacian_3d(u)
        else: raise ValueError(f"Unsupported number of dimensions: {self.ndim}")

    def _laplacian_1d(self, u: np.ndarray) -> np.ndarray:
        laplacian = np.zeros_like(u)
        dx = self.dx[0]
        laplacian[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
        
        if isinstance(self._boundary_conditions, NeumannBC):
            g = self._boundary_conditions._get_flux(self.t)
            laplacian[0] = 2 * (u[1] - u[0] - g * dx) / (dx**2)
            laplacian[-1] = 2 * (u[-2] - u[-1] + g * dx) / (dx**2)
        elif isinstance(self._boundary_conditions, DirichletBC):
            laplacian[0] = 0; laplacian[-1] = 0
        else:
            laplacian[0] = (u[1] - 2*u[0] + u[1]) / (dx**2)
            laplacian[-1] = (u[-2] - 2*u[-1] + u[-2]) / (dx**2)
        return laplacian

    def _laplacian_2d(self, u: np.ndarray) -> np.ndarray:
        laplacian = np.zeros_like(u)
        dx, dy = self.dx
        laplacian[1:-1, 1:-1] = (
            (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx**2)
            + (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy**2)
        )

        if isinstance(self._boundary_conditions, NeumannBC):
            flux = self._boundary_conditions._get_flux(self.t)
            # Left/Right
            laplacian[0, 1:-1] = (2 * (u[1, 1:-1] - u[0, 1:-1]) / (dx**2) - 2 * flux / dx) + \
                                 (u[0, 2:] - 2*u[0, 1:-1] + u[0, :-2]) / (dy**2)
            laplacian[-1, 1:-1] = (2 * (u[-2, 1:-1] - u[-1, 1:-1]) / (dx**2) + 2 * flux / dx) + \
                                  (u[-1, 2:] - 2*u[-1, 1:-1] + u[-1, :-2]) / (dy**2)
            # Bottom/Top
            laplacian[1:-1, 0] = (u[2:, 0] - 2*u[1:-1, 0] + u[:-2, 0]) / (dx**2) + \
                                 (2 * (u[1:-1, 1] - u[1:-1, 0]) / (dy**2) - 2 * flux / dy)
            laplacian[1:-1, -1] = (u[2:, -1] - 2*u[1:-1, -1] + u[:-2, -1]) / (dx**2) + \
                                  (2 * (u[1:-1, -2] - u[1:-1, -1]) / (dy**2) + 2 * flux / dy)
            # Corners
            laplacian[0, 0] = (2*(u[1,0]-u[0,0])/dx**2 - 2*flux/dx) + (2*(u[0,1]-u[0,0])/dy**2 - 2*flux/dy)
            laplacian[0, -1] = (2*(u[1,-1]-u[0,-1])/dx**2 - 2*flux/dx) + (2*(u[0,-2]-u[0,-1])/dy**2 + 2*flux/dy)
            laplacian[-1, 0] = (2*(u[-2,0]-u[-1,0])/dx**2 + 2*flux/dx) + (2*(u[-1,1]-u[-1,0])/dy**2 - 2*flux/dy)
            laplacian[-1, -1] = (2*(u[-2,-1]-u[-1,-1])/dx**2 + 2*flux/dx) + (2*(u[-1,-2]-u[-1,-1])/dy**2 + 2*flux/dy)
        
        elif isinstance(self._boundary_conditions, DirichletBC):
            laplacian[0, :] = 0; laplacian[-1, :] = 0; laplacian[:, 0] = 0; laplacian[:, -1] = 0
        
        else:
            # Default zero flux
            laplacian[0, 1:-1] = (2*(u[1, 1:-1] - u[0, 1:-1])/dx**2) + (u[0, 2:] - 2*u[0, 1:-1] + u[0, :-2])/dy**2
            laplacian[-1, 1:-1] = (2*(u[-2, 1:-1] - u[-1, 1:-1])/dx**2) + (u[-1, 2:] - 2*u[-1, 1:-1] + u[-1, :-2])/dy**2
            laplacian[1:-1, 0] = (u[2:, 0] - 2*u[1:-1, 0] + u[:-2, 0])/dx**2 + (2*(u[1:-1, 1] - u[1:-1, 0])/dy**2)
            laplacian[1:-1, -1] = (u[2:, -1] - 2*u[1:-1, -1] + u[:-2, -1])/dx**2 + (2*(u[1:-1, -2] - u[1:-1, -1])/dy**2)
            laplacian[0, 0] = 2*(u[1,0]-u[0,0])/dx**2 + 2*(u[0,1]-u[0,0])/dy**2
            laplacian[0, -1] = 2*(u[1,-1]-u[0,-1])/dx**2 + 2*(u[0,-2]-u[0,-1])/dy**2
            laplacian[-1, 0] = 2*(u[-2,0]-u[-1,0])/dx**2 + 2*(u[-1,1]-u[-1,0])/dy**2
            laplacian[-1, -1] = 2*(u[-2,-1]-u[-1,-1])/dx**2 + 2*(u[-1,-2]-u[-1,-1])/dy**2
        
        return laplacian

    def _laplacian_3d(self, u: np.ndarray) -> np.ndarray:
        laplacian = np.zeros_like(u)
        dx, dy, dz = self.dx
        
        # Interior
        laplacian[1:-1, 1:-1, 1:-1] = (
            (u[2:, 1:-1, 1:-1] - 2*u[1:-1, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]) / (dx**2)
            + (u[1:-1, 2:, 1:-1] - 2*u[1:-1, 1:-1, 1:-1] + u[1:-1, :-2, 1:-1]) / (dy**2)
            + (u[1:-1, 1:-1, 2:] - 2*u[1:-1, 1:-1, 1:-1] + u[1:-1, 1:-1, :-2]) / (dz**2)
        )
        
        if isinstance(self._boundary_conditions, NeumannBC):
            flux = self._boundary_conditions._get_flux(self.t)
            # X-direction boundaries
            laplacian[0, 1:-1, 1:-1] = (2 * (u[1, 1:-1, 1:-1] - u[0, 1:-1, 1:-1]) / (dx**2) - 2 * flux / dx) + \
                                       (u[0, 2:, 1:-1] - 2*u[0, 1:-1, 1:-1] + u[0, :-2, 1:-1]) / (dy**2) + \
                                       (u[0, 1:-1, 2:] - 2*u[0, 1:-1, 1:-1] + u[0, 1:-1, :-2]) / (dz**2)
            laplacian[-1, 1:-1, 1:-1] = (2 * (u[-2, 1:-1, 1:-1] - u[-1, 1:-1, 1:-1]) / (dx**2) + 2 * flux / dx) + \
                                        (u[-1, 2:, 1:-1] - 2*u[-1, 1:-1, 1:-1] + u[-1, :-2, 1:-1]) / (dy**2) + \
                                        (u[-1, 1:-1, 2:] - 2*u[-1, 1:-1, 1:-1] + u[-1, 1:-1, :-2]) / (dz**2)
            # Y-direction boundaries
            laplacian[1:-1, 0, 1:-1] = (u[2:, 0, 1:-1] - 2*u[1:-1, 0, 1:-1] + u[:-2, 0, 1:-1]) / (dx**2) + \
                                       (2 * (u[1:-1, 1, 1:-1] - u[1:-1, 0, 1:-1]) / (dy**2) - 2 * flux / dy) + \
                                       (u[1:-1, 0, 2:] - 2*u[1:-1, 0, 1:-1] + u[1:-1, 0, :-2]) / (dz**2)
            laplacian[1:-1, -1, 1:-1] = (u[2:, -1, 1:-1] - 2*u[1:-1, -1, 1:-1] + u[:-2, -1, 1:-1]) / (dx**2) + \
                                        (2 * (u[1:-1, -2, 1:-1] - u[1:-1, -1, 1:-1]) / (dy**2) + 2 * flux / dy) + \
                                        (u[1:-1, -1, 2:] - 2*u[1:-1, -1, 1:-1] + u[1:-1, -1, :-2]) / (dz**2)
            # Z-direction boundaries
            laplacian[1:-1, 1:-1, 0] = (u[2:, 1:-1, 0] - 2*u[1:-1, 1:-1, 0] + u[:-2, 1:-1, 0]) / (dx**2) + \
                                       (u[1:-1, 2:, 0] - 2*u[1:-1, 1:-1, 0] + u[1:-1, :-2, 0]) / (dy**2) + \
                                       (2 * (u[1:-1, 1:-1, 1] - u[1:-1, 1:-1, 0]) / (dz**2) - 2 * flux / dz)
            laplacian[1:-1, 1:-1, -1] = (u[2:, 1:-1, -1] - 2*u[1:-1, 1:-1, -1] + u[:-2, 1:-1, -1]) / (dx**2) + \
                                        (u[1:-1, 2:, -1] - 2*u[1:-1, 1:-1, -1] + u[1:-1, :-2, -1]) / (dy**2) + \
                                        (2 * (u[1:-1, 1:-1, -2] - u[1:-1, 1:-1, -1]) / (dz**2) + 2 * flux / dz)
            
            # EDGES
            # 1. Front-Bottom Edge (y=0, z=0)
            laplacian[1:-1, 0, 0] = (u[2:,0,0]-2*u[1:-1,0,0]+u[:-2,0,0])/dx**2 + \
                                    (2*(u[1:-1,1,0]-u[1:-1,0,0])/dy**2 - 2*flux/dy) + \
                                    (2*(u[1:-1,0,1]-u[1:-1,0,0])/dz**2 - 2*flux/dz)
            # 2. Front-Top Edge (y=0, z=-1)
            laplacian[1:-1, 0, -1] = (u[2:,0,-1]-2*u[1:-1,0,-1]+u[:-2,0,-1])/dx**2 + \
                                     (2*(u[1:-1,1,-1]-u[1:-1,0,-1])/dy**2 - 2*flux/dy) + \
                                     (2*(u[1:-1,0,-2]-u[1:-1,0,-1])/dz**2 + 2*flux/dz)
            # 3. Back-Bottom Edge (y=-1, z=0)
            laplacian[1:-1, -1, 0] = (u[2:,-1,0]-2*u[1:-1,-1,0]+u[:-2,-1,0])/dx**2 + \
                                     (2*(u[1:-1,-2,0]-u[1:-1,-1,0])/dy**2 + 2*flux/dy) + \
                                     (2*(u[1:-1,-1,1]-u[1:-1,-1,0])/dz**2 - 2*flux/dz)
            # 4. Back-Top Edge (y=-1, z=-1)
            laplacian[1:-1, -1, -1] = (u[2:,-1,-1]-2*u[1:-1,-1,-1]+u[:-2,-1,-1])/dx**2 + \
                                      (2*(u[1:-1,-2,-1]-u[1:-1,-1,-1])/dy**2 + 2*flux/dy) + \
                                      (2*(u[1:-1,-1,-2]-u[1:-1,-1,-1])/dz**2 + 2*flux/dz)

            # Edges parallel to Y-axis (x=0, x=-1, z=0, z=-1)
            # 5. Left-Bottom Edge (x=0, z=0)
            laplacian[0, 1:-1, 0] = (2*(u[1,1:-1,0]-u[0,1:-1,0])/dx**2 - 2*flux/dx) + \
                                    (u[0,2:,0]-2*u[0,1:-1,0]+u[0,:-2,0])/dy**2 + \
                                    (2*(u[0,1:-1,1]-u[0,1:-1,0])/dz**2 - 2*flux/dz)
            # 6. Left-Top Edge (x=0, z=-1)
            laplacian[0, 1:-1, -1] = (2*(u[1,1:-1,-1]-u[0,1:-1,-1])/dx**2 - 2*flux/dx) + \
                                     (u[0,2:,-1]-2*u[0,1:-1,-1]+u[0,:-2,-1])/dy**2 + \
                                     (2*(u[0,1:-1,-2]-u[0,1:-1,-1])/dz**2 + 2*flux/dz)
            # 7. Right-Bottom Edge (x=-1, z=0)
            laplacian[-1, 1:-1, 0] = (2*(u[-2,1:-1,0]-u[-1,1:-1,0])/dx**2 + 2*flux/dx) + \
                                     (u[-1,2:,0]-2*u[-1,1:-1,0]+u[-1,:-2,0])/dy**2 + \
                                     (2*(u[-1,1:-1,1]-u[-1,1:-1,0])/dz**2 - 2*flux/dz)
            # 8. Right-Top Edge (x=-1, z=-1)
            laplacian[-1, 1:-1, -1] = (2*(u[-2,1:-1,-1]-u[-1,1:-1,-1])/dx**2 + 2*flux/dx) + \
                                      (u[-1,2:,-1]-2*u[-1,1:-1,-1]+u[-1,:-2,-1])/dy**2 + \
                                      (2*(u[-1,1:-1,-2]-u[-1,1:-1,-1])/dz**2 + 2*flux/dz)

            # Edges parallel to Z-axis (x=0, x=-1, y=0, y=-1)
            # 9. Left-Front Edge (x=0, y=0)
            laplacian[0, 0, 1:-1] = (2*(u[1,0,1:-1]-u[0,0,1:-1])/dx**2 - 2*flux/dx) + \
                                    (2*(u[0,1,1:-1]-u[0,0,1:-1])/dy**2 - 2*flux/dy) + \
                                    (u[0,0,2:]-2*u[0,0,1:-1]+u[0,0,:-2])/dz**2
            # 10. Left-Back Edge (x=0, y=-1)
            laplacian[0, -1, 1:-1] = (2*(u[1,-1,1:-1]-u[0,-1,1:-1])/dx**2 - 2*flux/dx) + \
                                     (2*(u[0,-2,1:-1]-u[0,-1,1:-1])/dy**2 + 2*flux/dy) + \
                                     (u[0,-1,2:]-2*u[0,-1,1:-1]+u[0,-1,:-2])/dz**2
            # 11. Right-Front Edge (x=-1, y=0)
            laplacian[-1, 0, 1:-1] = (2*(u[-2,0,1:-1]-u[-1,0,1:-1])/dx**2 + 2*flux/dx) + \
                                     (2*(u[-1,1,1:-1]-u[-1,0,1:-1])/dy**2 - 2*flux/dy) + \
                                     (u[-1,0,2:]-2*u[-1,0,1:-1]+u[-1,0,:-2])/dz**2
            # 12. Right-Back Edge (x=-1, y=-1)
            laplacian[-1, -1, 1:-1] = (2*(u[-2,-1,1:-1]-u[-1,-1,1:-1])/dx**2 + 2*flux/dx) + \
                                      (2*(u[-1,-2,1:-1]-u[-1,-1,1:-1])/dy**2 + 2*flux/dy) + \
                                      (u[-1,-1,2:]-2*u[-1,-1,1:-1]+u[-1,-1,:-2])/dz**2

            # CORNERS
            # 1. Left-Front-Bottom (0,0,0)
            laplacian[0,0,0] = (2*(u[1,0,0]-u[0,0,0])/dx**2 - 2*flux/dx) + \
                               (2*(u[0,1,0]-u[0,0,0])/dy**2 - 2*flux/dy) + \
                               (2*(u[0,0,1]-u[0,0,0])/dz**2 - 2*flux/dz)
            # 2. Right-Front-Bottom (-1,0,0)
            laplacian[-1,0,0] = (2*(u[-2,0,0]-u[-1,0,0])/dx**2 + 2*flux/dx) + \
                                (2*(u[-1,1,0]-u[-1,0,0])/dy**2 - 2*flux/dy) + \
                                (2*(u[-1,0,1]-u[-1,0,0])/dz**2 - 2*flux/dz)
            # 3. Left-Back-Bottom (0,-1,0)
            laplacian[0,-1,0] = (2*(u[1,-1,0]-u[0,-1,0])/dx**2 - 2*flux/dx) + \
                                (2*(u[0,-2,0]-u[0,-1,0])/dy**2 + 2*flux/dy) + \
                                (2*(u[0,-1,1]-u[0,-1,0])/dz**2 - 2*flux/dz)
            # 4. Right-Back-Bottom (-1,-1,0)
            laplacian[-1,-1,0] = (2*(u[-2,-1,0]-u[-1,-1,0])/dx**2 + 2*flux/dx) + \
                                 (2*(u[-1,-2,0]-u[-1,-1,0])/dy**2 + 2*flux/dy) + \
                                 (2*(u[-1,-1,1]-u[-1,-1,0])/dz**2 - 2*flux/dz)
            # 5. Left-Front-Top (0,0,-1)
            laplacian[0,0,-1] = (2*(u[1,0,-1]-u[0,0,-1])/dx**2 - 2*flux/dx) + \
                                (2*(u[0,1,-1]-u[0,0,-1])/dy**2 - 2*flux/dy) + \
                                (2*(u[0,0,-2]-u[0,0,-1])/dz**2 + 2*flux/dz)
            # 6. Right-Front-Top (-1,0,-1)
            laplacian[-1,0,-1] = (2*(u[-2,0,-1]-u[-1,0,-1])/dx**2 + 2*flux/dx) + \
                                 (2*(u[-1,1,-1]-u[-1,0,-1])/dy**2 - 2*flux/dy) + \
                                 (2*(u[-1,0,-2]-u[-1,0,-1])/dz**2 + 2*flux/dz)
            # 7. Left-Back-Top (0,-1,-1)
            laplacian[0,-1,-1] = (2*(u[1,-1,-1]-u[0,-1,-1])/dx**2 - 2*flux/dx) + \
                                 (2*(u[0,-2,-1]-u[0,-1,-1])/dy**2 + 2*flux/dy) + \
                                 (2*(u[0,-1,-2]-u[0,-1,-1])/dz**2 + 2*flux/dz)
            # 8. Right-Back-Top (-1,-1,-1)
            laplacian[-1,-1,-1] = (2*(u[-2,-1,-1]-u[-1,-1,-1])/dx**2 + 2*flux/dx) + \
                                  (2*(u[-1,-2,-1]-u[-1,-1,-1])/dy**2 + 2*flux/dy) + \
                                  (2*(u[-1,-1,-2]-u[-1,-1,-1])/dz**2 + 2*flux/dz)

        elif isinstance(self._boundary_conditions, DirichletBC):
            laplacian[0, :, :] = 0; laplacian[-1, :, :] = 0
            laplacian[:, 0, :] = 0; laplacian[:, -1, :] = 0
            laplacian[:, :, 0] = 0; laplacian[:, :, -1] = 0
            
        return laplacian

    def set_diffusion_coefficient(self, value: float) -> None:
        super().set_diffusion_coefficient(value)
        self._build_system_matrices()
    
    def set_decay_rate(self, value: float) -> None:
        super().set_decay_rate(value)
        self._build_system_matrices()