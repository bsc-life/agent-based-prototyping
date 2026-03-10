"""
Implicit Euler with Alternating Direction Implicit (ADI) method.

This module implements the ADI splitting scheme. The 2D case uses the 
second-order Peaceman-Rachford formulation.
"""

import numpy as np
from scipy.sparse import diags, eye, csr_matrix
from scipy.sparse.linalg import spsolve
from diffusion_schemas.base import Schema
from diffusion_schemas.utils.boundary import DirichletBC, NeumannBC


class ADISchema(Schema):
    """
    Alternating Direction Implicit (ADI) method for the diffusion equation.
    
    2D uses Peaceman-Rachford (O(dt^2)).
    1D/3D use fractional-step implicit Euler (O(dt)).
    """
    
    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0
    ):
        super().__init__(domain_size, grid_points, dt, diffusion_coefficient, decay_rate)
        self._build_system_matrix()
    
    def _build_system_matrix(self) -> None:
        if self.ndim == 1:
            self.system_matrix = self._build_matrix_1d()
        elif self.ndim == 2:
            self.system_matrix = self._build_matrix_2d()
        elif self.ndim == 3:
            self.system_matrix = self._build_matrix_3d()
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.ndim}")
    
    def _build_matrix_1d(self) -> csr_matrix:
        N = self.grid_points[0]
        dx = self.dx[0]
        
        diag_main = -2 * np.ones(N) / (dx**2)
        diag_off = np.ones(N-1) / (dx**2)
        L = diags([diag_off, diag_main, diag_off], [-1, 0, 1], shape=(N, N), format='csr')
        I = eye(N, format='csr')
        
        return I - self.dt * self.diffusion_coefficient * L + self.dt * self.decay_rate * I
    
    def _build_matrix_2d(self):
        """Build 2D splitting matrices for Peaceman-Rachford (LHS and RHS)."""
        Nx, Ny = self.grid_points
        dx, dy = self.dx
        
        # CHANGE: INSTEAD OF FULL DT, ONLY DT/2
        dt_half = self.dt / 2.0
        decay_term = (self.decay_rate / 2.0) * self.dt
        
        # X-Operators
        diag_main_x = -2 * np.ones(Nx) / (dx**2)
        diag_off_x = np.ones(Nx-1) / (dx**2)
        Lx = diags([diag_off_x, diag_main_x, diag_off_x], [-1, 0, 1], shape=(Nx, Nx), format='csr')
        Ix = eye(Nx, format='csr')
        
        LHS_x = Ix - dt_half * self.diffusion_coefficient * Lx + decay_term * Ix
        RHS_x = Ix + dt_half * self.diffusion_coefficient * Lx - decay_term * Ix
        
        # Y-Operators
        diag_main_y = -2 * np.ones(Ny) / (dy**2)
        diag_off_y = np.ones(Ny-1) / (dy**2)
        Ly = diags([diag_off_y, diag_main_y, diag_off_y], [-1, 0, 1], shape=(Ny, Ny), format='csr')
        Iy = eye(Ny, format='csr')
        
        LHS_y = Iy - dt_half * self.diffusion_coefficient * Ly + decay_term * Iy
        RHS_y = Iy + dt_half * self.diffusion_coefficient * Ly - decay_term * Iy

        return LHS_x, RHS_x, LHS_y, RHS_y
    
    def _build_matrix_3d(self):
        Nx, Ny, Nz = self.grid_points
        dx, dy, dz = self.dx
        factor = 1 / 3
        
        dt_third = self.dt / 3.0
        decay_term = (self.decay_rate / 3.0) * self.dt

        # X-Operators    
        Lx = diags([np.ones(Nx-1)/dx**2, -2*np.ones(Nx)/dx**2, np.ones(Nx-1)/dx**2], [-1, 0, 1], shape=(Nx, Nx), format='csr')
        Ix = eye(Nx, format='csr')

        LHS_x = Ix - dt_third * self.diffusion_coefficient * Lx + decay_term * Ix
        RHS_x = Ix + dt_third * self.diffusion_coefficient * Lx - decay_term * Ix

        # Y-Operators
        Ly = diags([np.ones(Ny-1)/dy**2, -2*np.ones(Ny)/dy**2, np.ones(Ny-1)/dy**2], [-1, 0, 1], shape=(Ny, Ny), format='csr')
        Iy = eye(Ny, format='csr')

        LHS_y = Iy - dt_third * self.diffusion_coefficient * Ly + decay_term * Iy
        RHS_y = Iy + dt_third * self.diffusion_coefficient * Ly - decay_term * Iy

        # Z-Operators
        Lz = diags([np.ones(Nz-1)/dz**2, -2*np.ones(Nz)/dz**2, np.ones(Nz-1)/dz**2], [-1, 0, 1], shape=(Nz, Nz), format='csr')
        Iz = eye(Nz, format='csr')

        LHS_z = Iz - dt_third * self.diffusion_coefficient * Lz + decay_term * Iz
        RHS_z = Iz + dt_third * self.diffusion_coefficient * Lz - decay_term * Iz

        return LHS_x, RHS_x, LHS_y, RHS_y, LHS_z, RHS_z


    def step(self) -> None:
        if self.ndim == 3:
            raise NotImplementedError("3D ADI is not implemented yet.")

        source = self._compute_source_term()

        # --------------------- 1D CASE ---------------------
        if self.ndim == 1:
            rhs = self.state + self.dt * source if source is not None else self.state
            Ax = self.system_matrix.copy().tolil()
            rhs = rhs.reshape(self.grid_points[0], 1)
            # No BCs applied to Ax or rhs here
            self.state = spsolve(Ax.tocsr(), rhs).reshape(self.grid_points)
            # Apply BCs after solve
            if self._boundary_conditions is not None:
                self.state = self._apply_boundary_conditions(self.state)

        # HUGE CHANGE: 2D ADI SPLIT INTO TWO SWEEPS WITH HALF SOURCE TERM IN EACH
        # --------------------- 2D CASE (Peaceman-Rachford) ---------------------
        elif self.ndim == 2:
            dt_half = self.dt / 2.0

            LHS_x, RHS_x, LHS_y, RHS_y = self.system_matrix
            LHS_x_lil = LHS_x.copy().tolil()
            LHS_y_lil = LHS_y.copy().tolil()
            
            # CHANGE: SPLIT SOURCE TERM IN HALF FOR EACH SWEEP
            half_source = (dt_half) * source if source is not None else 0.0
            
            # --- SWEEP 1: Implicit X, Explicit Y ---
            rhs_1 = (RHS_y @ self.state.T).T + half_source
            # No BCs applied to LHS_x_lil or rhs_1 here
            u_star = spsolve(LHS_x_lil.tocsr(), rhs_1)
            # Apply BCs after solve
            if self._boundary_conditions is not None:
                u_star = self._apply_boundary_conditions(u_star)

            # --- SWEEP 2: Explicit X, Implicit Y ---
            # RHS_x evaluates along X (axis 0).
            rhs_2 = (RHS_x @ u_star) + half_source
            rhs_2_T = rhs_2.T
            # No BCs applied to LHS_y_lil or rhs_2_T here
            u_new_T = spsolve(LHS_y_lil.tocsr(), rhs_2_T)

            self.state = u_new_T.T          

            # Apply BCs after solve
            if self._boundary_conditions is not None:
                self.state = self._apply_boundary_conditions(self.state)

        # --------------------- 3D CASE ---------------------
        elif self.ndim == 3:
            dt_third = self.dt / 3.0                
            Nx, Ny, Nz = self.grid_points
            LHS_x, RHS_x, LHS_y, RHS_y, LHS_z, RHS_z = self.system_matrix
            LHS_x_lil = LHS_x.copy().tolil()
            LHS_y_lil = LHS_y.copy().tolil()
            LHS_z_lil = LHS_z.copy().tolil()

            state_flat = self.state.flatten()
            third_source = dt_third * source if source is not None else 0.0

            # --- SWEEP 1: Implicit X, Explicit Y & Z ---
            # Apply RHS_y (axis 1) and RHS_z (axis 2)
            u_y_expl = (RHS_y @ self.state.transpose(1, 0, 2).reshape(Ny, Nx * Nz)).reshape(Ny, Nx, Nz).transpose(1, 0, 2)
            u_z_expl = (RHS_z @ self.state.transpose(2, 0, 1).reshape(Nz, Nx * Ny)).reshape(Nz, Nx, Ny).transpose(1, 2, 0)
            
            # The "Identity" is already in the RHS matrices, so we don't add 'state' again.
            # We subtract one 'state' because (I + Ay) + (I + Az) = 2I + Ay + Az. We only want 1I.
            rhs_1 = u_y_expl + u_z_expl - self.state + third_source
            
            rhs_1_x = rhs_1.reshape(Nx, Ny * Nz)
            u_star = spsolve(LHS_x.tocsr(), rhs_1_x).reshape(Nx, Ny, Nz)

            # Apply BCs after solve
            if self._boundary_conditions is not None:
                u_star = self._apply_boundary_conditions(u_star)

            # --- SWEEP 2: Implicit Y, Explicit X & Z ---
            u_x_expl = (RHS_x @ u_star).reshape(Nx, Ny, Nz)
            u_z_expl = (RHS_z @ u_star.transpose(2, 0, 1).reshape(Nz, Nx * Ny)).reshape(Nz, Nx, Ny).transpose(1, 2, 0)
            
            rhs_2 = u_x_expl + u_z_expl - u_star + third_source
            
            rhs_2_y = rhs_2.transpose(1, 0, 2).reshape(Ny, Nx * Nz)
            u_star_star = spsolve(LHS_y.tocsr(), rhs_2_y).reshape(Ny, Nx, Nz).transpose(1, 0, 2)

            self.state = u_new_T.T
            # Apply BCs after solve
            if self._boundary_conditions is not None:
                u_star_star = self._apply_boundary_conditions(u_star_star)

            # --- SWEEP 3: Implicit Z, Explicit X & Y ---
            u_x_expl = (RHS_x @ u_star_star).reshape(Nx, Ny, Nz)
            u_y_expl = (RHS_y @ u_star_star.transpose(1, 0, 2).reshape(Ny, Nx * Nz)).reshape(Ny, Nx, Nz).transpose(1, 0, 2)
            
            rhs_3 = u_x_expl + u_y_expl - u_star_star + third_source
            
            rhs_3_z = rhs_3.transpose(2, 0, 1).reshape(Nz, Nx * Ny)
            u_final_z = spsolve(LHS_z.tocsr(), rhs_3_z).reshape(Nz, Nx, Ny).transpose(1, 2, 0)
            
            if self._boundary_conditions is not None:
                u_final_z = self._apply_boundary_conditions(u_final_z)

            self.state = u_final_z
        
        self.t += self.dt

        def set_diffusion_coefficient(self, value: float) -> None:
        super().set_diffusion_coefficient(value)
        self._build_system_matrix()
    
    def set_decay_rate(self, value: float) -> None:
        super().set_decay_rate(value)
        self._build_system_matrix()