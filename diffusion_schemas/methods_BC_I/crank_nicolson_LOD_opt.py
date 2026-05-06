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
from scipy.linalg import solve_banded
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
            # Only Dirichlet rows are identity-constrained; 
            # keep Neumann boundary source contributions active
            if isinstance(self._boundary_conditions, DirichletBC):
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
        Uses cached SuperLU factorizations when possible, falling back
        to solve_banded for complex/dynamic conditions.
        """

        has_per_node_source = np.any(bulk_lhs_np1)
        has_dirichlet = isinstance(self._boundary_conditions, DirichletBC)
        is_neumann = isinstance(self._boundary_conditions, NeumannBC)
        
        t_eval = self.t + self.dt

        # ==========================================================
        # 1D CASE
        # ==========================================================
        if self.ndim == 1:
            Nx = self.grid_points[0]
            rhs_x = rhs.reshape(Nx, 1)

            if (not has_per_node_source) and (not has_dirichlet) and getattr(self, '_lu_x', None) is not None:
                self.state = self._lu_x.solve(rhs_x).reshape(self.grid_points)
            else:
                alpha_x = self.theta * self.dt * self.diffusion_coefficient / (self.dx[0]**2)
                decay_term = self.theta * self.dt * self.decay_rate
                
                ab_x = np.zeros((3, Nx))
                ab_x[0, 1:] = -alpha_x
                ab_x[2, :-1] = -alpha_x
                ab_x[1, :] = 1.0 + 2*alpha_x + decay_term + (self.theta * self.dt) * bulk_lhs_np1.reshape(Nx)
                
                rhs_x_flat = rhs_x.flatten()
                
                if is_neumann:
                    bc_val_x = self._boundary_conditions._get_flux(t_eval)
                    forcing_x = (2 * self.theta * self.dt * self.diffusion_coefficient * bc_val_x) / self.dx[0]
                    ab_x[0, 1] = -2 * alpha_x
                    ab_x[2, -2] = -2 * alpha_x
                    rhs_x_flat[0] -= forcing_x
                    rhs_x_flat[-1] += forcing_x
                elif has_dirichlet:
                    bc_val_x = self._boundary_conditions._get_value(t_eval)
                    ab_x[1, 0] = 1.0; ab_x[0, 1] = 0.0; rhs_x_flat[0] = bc_val_x
                    ab_x[1, -1] = 1.0; ab_x[2, -2] = 0.0; rhs_x_flat[-1] = bc_val_x
                    
                self.state = solve_banded((1, 1), ab_x, rhs_x_flat).reshape(self.grid_points)

        # ==========================================================
        # 2D CASE
        # ==========================================================
        elif self.ndim == 2:
            Nx, Ny = self.grid_points
            rhs_2d = rhs.reshape(Nx, Ny)
            
            # --- SWEEP 1: X-Direction ---
            if (not has_per_node_source) and (not has_dirichlet) and getattr(self, '_lu_x', None) is not None:
                u_star = self._lu_x.solve(rhs_2d)
            else:
                u_star = np.zeros((Nx, Ny))
                alpha_x = self.theta * self.dt * self.diffusion_coefficient / (self.dx[0]**2)
                decay_term = self.theta * self.dt * self.decay_rate / 2.0
                
                if is_neumann:
                    bc_val_x = self._boundary_conditions._get_flux(t_eval)
                    forcing_x = (2 * self.theta * self.dt * self.diffusion_coefficient * bc_val_x) / self.dx[0]
                elif has_dirichlet:
                    bc_val_x = self._boundary_conditions._get_value(t_eval)

                for j in range(Ny):
                    ab_x = np.zeros((3, Nx))
                    ab_x[0, 1:] = -alpha_x
                    ab_x[2, :-1] = -alpha_x
                    ab_x[1, :] = 1.0 + 2*alpha_x + decay_term + (self.theta * self.dt / 2.0) * bulk_lhs_np1[:, j]
                    
                    rhs_j = rhs_2d[:, j].copy()
                    
                    if is_neumann:
                        ab_x[0, 1] = -2 * alpha_x
                        ab_x[2, -2] = -2 * alpha_x
                        rhs_j[0] -= forcing_x
                        rhs_j[-1] += forcing_x
                    elif has_dirichlet:
                        ab_x[1, 0] = 1.0; ab_x[0, 1] = 0.0; rhs_j[0] = bc_val_x
                        ab_x[1, -1] = 1.0; ab_x[2, -2] = 0.0; rhs_j[-1] = bc_val_x
                        
                    u_star[:, j] = solve_banded((1, 1), ab_x, rhs_j)

            # --- SWEEP 2: Y-Direction ---
            if (not has_per_node_source) and (not has_dirichlet) and getattr(self, '_lu_y', None) is not None:
                u_new_T = self._lu_y.solve(u_star.T)
                self.state = u_new_T.T
            else:
                u_new = np.zeros((Nx, Ny))
                alpha_y = self.theta * self.dt * self.diffusion_coefficient / (self.dx[1]**2)
                decay_term = self.theta * self.dt * self.decay_rate / 2.0
                
                if is_neumann:
                    bc_val_y = self._boundary_conditions._get_flux(t_eval)
                    forcing_y = (2 * self.theta * self.dt * self.diffusion_coefficient * bc_val_y) / self.dx[1]
                elif has_dirichlet:
                    bc_val_y = self._boundary_conditions._get_value(t_eval)

                for i in range(Nx):
                    ab_y = np.zeros((3, Ny))
                    ab_y[0, 1:] = -alpha_y
                    ab_y[2, :-1] = -alpha_y
                    ab_y[1, :] = 1.0 + 2*alpha_y + decay_term + (self.theta * self.dt / 2.0) * bulk_lhs_np1[i, :]
                    
                    rhs_i = u_star[i, :].copy()
                    
                    if is_neumann:
                        ab_y[0, 1] = -2 * alpha_y
                        ab_y[2, -2] = -2 * alpha_y
                        rhs_i[0] -= forcing_y
                        rhs_i[-1] += forcing_y
                    elif has_dirichlet:
                        ab_y[1, 0] = 1.0; ab_y[0, 1] = 0.0; rhs_i[0] = bc_val_y
                        ab_y[1, -1] = 1.0; ab_y[2, -2] = 0.0; rhs_i[-1] = bc_val_y
                        
                    u_new[i, :] = solve_banded((1, 1), ab_y, rhs_i)
                self.state = u_new

        # ==========================================================
        # 3D CASE
        # ==========================================================
        elif self.ndim == 3:
            Nx, Ny, Nz = self.grid_points
            rhs_3d = rhs.reshape(Nx, Ny, Nz)
            
            # --- SWEEP 1: X-Direction ---
            if (not has_per_node_source) and (not has_dirichlet) and getattr(self, '_lu_x', None) is not None:
                rhs_x_flat = rhs_3d.reshape(Nx, Ny * Nz)
                u_star_flat = self._lu_x.solve(rhs_x_flat)
                u_star = u_star_flat.reshape(Nx, Ny, Nz)
            else:
                u_star = np.zeros((Nx, Ny, Nz))
                alpha_x = self.theta * self.dt * self.diffusion_coefficient / (self.dx[0]**2)
                decay_term = self.theta * self.dt * self.decay_rate / 3.0
                
                if is_neumann:
                    bc_val_x = self._boundary_conditions._get_flux(t_eval)
                    forcing_x = (2 * self.theta * self.dt * self.diffusion_coefficient * bc_val_x) / self.dx[0]
                elif has_dirichlet:
                    bc_val_x = self._boundary_conditions._get_value(t_eval)

                for j in range(Ny):
                    for k in range(Nz):
                        ab_x = np.zeros((3, Nx))
                        ab_x[0, 1:] = -alpha_x
                        ab_x[2, :-1] = -alpha_x
                        ab_x[1, :] = 1.0 + 2*alpha_x + decay_term + (self.theta * self.dt / 3.0) * bulk_lhs_np1[:, j, k]
                        
                        rhs_jk = rhs_3d[:, j, k].copy()
                        
                        if is_neumann:
                            ab_x[0, 1] = -2 * alpha_x
                            ab_x[2, -2] = -2 * alpha_x
                            rhs_jk[0] -= forcing_x
                            rhs_jk[-1] += forcing_x
                        elif has_dirichlet:
                            ab_x[1, 0] = 1.0; ab_x[0, 1] = 0.0; rhs_jk[0] = bc_val_x
                            ab_x[1, -1] = 1.0; ab_x[2, -2] = 0.0; rhs_jk[-1] = bc_val_x
                            
                        u_star[:, j, k] = solve_banded((1, 1), ab_x, rhs_jk)

            # --- SWEEP 2: Y-Direction ---
            if (not has_per_node_source) and (not has_dirichlet) and getattr(self, '_lu_y', None) is not None:
                u_star_T = u_star.transpose(1, 0, 2).reshape(Ny, Nx * Nz)
                u_star_star_flat = self._lu_y.solve(u_star_T)
                u_star_star = u_star_star_flat.reshape(Ny, Nx, Nz).transpose(1, 0, 2)
            else:
                u_star_star = np.zeros((Nx, Ny, Nz))
                alpha_y = self.theta * self.dt * self.diffusion_coefficient / (self.dx[1]**2)
                decay_term = self.theta * self.dt * self.decay_rate / 3.0
                
                if is_neumann:
                    bc_val_y = self._boundary_conditions._get_flux(t_eval)
                    forcing_y = (2 * self.theta * self.dt * self.diffusion_coefficient * bc_val_y) / self.dx[1]
                elif has_dirichlet:
                    bc_val_y = self._boundary_conditions._get_value(t_eval)

                for i in range(Nx):
                    for k in range(Nz):
                        ab_y = np.zeros((3, Ny))
                        ab_y[0, 1:] = -alpha_y
                        ab_y[2, :-1] = -alpha_y
                        ab_y[1, :] = 1.0 + 2*alpha_y + decay_term + (self.theta * self.dt / 3.0) * bulk_lhs_np1[i, :, k]
                        
                        rhs_ik = u_star[i, :, k].copy()
                        
                        if is_neumann:
                            ab_y[0, 1] = -2 * alpha_y
                            ab_y[2, -2] = -2 * alpha_y
                            rhs_ik[0] -= forcing_y
                            rhs_ik[-1] += forcing_y
                        elif has_dirichlet:
                            ab_y[1, 0] = 1.0; ab_y[0, 1] = 0.0; rhs_ik[0] = bc_val_y
                            ab_y[1, -1] = 1.0; ab_y[2, -2] = 0.0; rhs_ik[-1] = bc_val_y
                            
                        u_star_star[i, :, k] = solve_banded((1, 1), ab_y, rhs_ik)

            # --- SWEEP 3: Z-Direction ---
            if (not has_per_node_source) and (not has_dirichlet) and getattr(self, '_lu_z', None) is not None:
                u_star_star_T = u_star_star.transpose(2, 0, 1).reshape(Nz, Nx * Ny)
                u_final_flat = self._lu_z.solve(u_star_star_T)
                self.state = u_final_flat.reshape(Nz, Nx, Ny).transpose(1, 2, 0)
            else:
                u_new = np.zeros((Nx, Ny, Nz))
                alpha_z = self.theta * self.dt * self.diffusion_coefficient / (self.dx[2]**2)
                decay_term = self.theta * self.dt * self.decay_rate / 3.0
                
                if is_neumann:
                    bc_val_z = self._boundary_conditions._get_flux(t_eval)
                    forcing_z = (2 * self.theta * self.dt * self.diffusion_coefficient * bc_val_z) / self.dx[2]
                elif has_dirichlet:
                    bc_val_z = self._boundary_conditions._get_value(t_eval)

                for i in range(Nx):
                    for j in range(Ny):
                        ab_z = np.zeros((3, Nz))
                        ab_z[0, 1:] = -alpha_z
                        ab_z[2, :-1] = -alpha_z
                        ab_z[1, :] = 1.0 + 2*alpha_z + decay_term + (self.theta * self.dt / 3.0) * bulk_lhs_np1[i, j, :]
                        
                        rhs_ij = u_star_star[i, j, :].copy()
                        
                        if is_neumann:
                            ab_z[0, 1] = -2 * alpha_z
                            ab_z[2, -2] = -2 * alpha_z
                            rhs_ij[0] -= forcing_z
                            rhs_ij[-1] += forcing_z
                        elif has_dirichlet:
                            ab_z[1, 0] = 1.0; ab_z[0, 1] = 0.0; rhs_ij[0] = bc_val_z
                            ab_z[1, -1] = 1.0; ab_z[2, -2] = 0.0; rhs_ij[-1] = bc_val_z
                            
                        u_new[i, j, :] = solve_banded((1, 1), ab_z, rhs_ij)
                self.state = u_new

        return self.state
    

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