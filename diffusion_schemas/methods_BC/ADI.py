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


class ADIBCSchema(Schema):
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
        
        dt_half = self.dt / 2.0
        # Each half-step carries half the decay: (λ/2) * (dt/2) = λ·dt/4
        decay_term = (self.decay_rate / 2.0) * dt_half
        
        # X-Operators
        diag_main_x = -2 * np.ones(Nx) / (dx**2)
        diag_off_x = np.ones(Nx-1) / (dx**2)
        Lx = diags([diag_off_x, diag_main_x, diag_off_x], [-1, 0, 1], shape=(Nx, Nx), format='csr').tolil()
        Ix = eye(Nx, format='csr')

        # if isinstance(self._boundary_conditions, DirichletBC):
        #     Lx[0, :], Lx[-1, :] = 0, 0
        if isinstance(self._boundary_conditions, NeumannBC):
            Lx[0, 0], Lx[0, 1] = -2 / (dx**2), 2 / (dx**2)
            Lx[-1, -1], Lx[-1, -2] = -2 / (dx**2), 2 / (dx**2)

        Lx, Ix = Lx.tocsr(), Ix.tocsr()
        # Note decay term appears twice at each step, thus why it is previously divided by 4
        LHS_x = (Ix - dt_half * self.diffusion_coefficient * Lx + decay_term * Ix).tolil()
        RHS_x = (Ix + dt_half * self.diffusion_coefficient * Lx - decay_term * Ix).tolil()

        if isinstance(self._boundary_conditions, DirichletBC):
            for row in [0, -1]:
                LHS_x[row, :] = 0; LHS_x[row, row] = 1
                RHS_x[row, :] = 0; RHS_x[row, row] = 1
        
        # Y-Operators
        diag_main_y = -2 * np.ones(Ny) / (dy**2)
        diag_off_y = np.ones(Ny-1) / (dy**2)
        Ly = diags([diag_off_y, diag_main_y, diag_off_y], [-1, 0, 1], shape=(Ny, Ny), format='csr').tolil()
        Iy = eye(Ny, format='csr')

        # if isinstance(self._boundary_conditions, DirichletBC):
        #     Ly[0, :], Ly[-1, :] = 0, 0
        if isinstance(self._boundary_conditions, NeumannBC):
            Ly[0, 0], Ly[0, 1] = -2 / (dy**2), 2 / (dy**2)
            Ly[-1, -1], Ly[-1, -2] = -2 / (dy**2), 2 / (dy**2)
        
        Ly, Iy = Ly.tocsr(), Iy.tocsr()
        LHS_y = (Iy - dt_half * self.diffusion_coefficient * Ly + decay_term * Iy).tolil()
        RHS_y = (Iy + dt_half * self.diffusion_coefficient * Ly - decay_term * Iy).tolil()

        if isinstance(self._boundary_conditions, DirichletBC):
            for row in [0, -1]:
                LHS_y[row, :] = 0; LHS_y[row, row] = 1
                RHS_y[row, :] = 0; RHS_y[row, row] = 1

        return LHS_x, RHS_x, LHS_y, RHS_y
    
    def _build_matrix_3d(self):
        # Just copied the logic above, but still has to be tested and debugged

        Nx, Ny, Nz = self.grid_points
        dx, dy, dz = self.dx
        
        dt_third = self.dt / 3.0
        # Each third-step carries a third of the decay: (λ/3) * (dt/3) = λ·dt/9
        decay_term = (self.decay_rate / 3.0) * dt_third

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

        if self.ndim == 3: raise NotImplementedError("3D ADI is not implemented yet") # just in case

        source = self._compute_source_term()
        
        # --------------------- 1D CASE ---------------------
        if self.ndim == 1:
            rhs = self.state + self.dt * source
            Ax = self.system_matrix.copy().tolil()
            rhs = rhs.reshape(self.grid_points[0], 1)
            rhs = self._apply_bc_to_sweep(Ax, rhs, self.dx[0], self.dt)
            self.state = spsolve(Ax.tocsr(), rhs).reshape(self.grid_points)

        # HUGE CHANGE: 2D ADI SPLIT INTO TWO SWEEPS WITH HALF SOURCE TERM IN EACH
        # --------------------- 2D CASE (Peaceman-Rachford) ---------------------
        elif self.ndim == 2:
            dt_half = self.dt / 2.0
            D = self.diffusion_coefficient
            dx, dy = self.dx

            # Mask source at Dirichlet boundaries (those nodes are pinned)
            if isinstance(self._boundary_conditions, DirichletBC):
                source = source.copy()
                source[0, :] = 0; source[-1, :] = 0
                source[:, 0] = 0; source[:, -1] = 0
            half_source = dt_half * source

            LHS_x, RHS_x, LHS_y, RHS_y = self.system_matrix

            # --- SWEEP 1: Implicit X, Explicit Y ---
            rhs_1 = (RHS_y @ self.state.T).T + half_source

            # Add explicit transverse Neumann forcing (y-direction)
            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(self.t)
                explicit_y_forcing = dt_half * D * 2 * flux / dy
                rhs_1[:, 0]  -= explicit_y_forcing
                rhs_1[:, -1] += explicit_y_forcing

            LHS_x_lil = LHS_x.copy().tolil()
            rhs_1 = self._apply_bc_to_sweep(LHS_x_lil, rhs_1, dx, dt_half)
            u_star = spsolve(LHS_x_lil.tocsr(), rhs_1)

            # Enforce all Dirichlet boundaries on intermediate solution
            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(self.t + dt_half)
                u_star[0, :] = val; u_star[-1, :] = val
                u_star[:, 0] = val; u_star[:, -1] = val

            # --- SWEEP 2: Explicit X, Implicit Y ---
            rhs_2 = (RHS_x @ u_star) + half_source

            # Add explicit transverse Neumann forcing (x-direction)
            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(self.t + dt_half)
                explicit_x_forcing = dt_half * D * 2 * flux / dx
                rhs_2[0, :]  -= explicit_x_forcing
                rhs_2[-1, :] += explicit_x_forcing

            rhs_2_T = rhs_2.T
            LHS_y_lil = LHS_y.copy().tolil()
            rhs_2_T = self._apply_bc_to_sweep(LHS_y_lil, rhs_2_T, dy, self.dt)
            u_new_T = spsolve(LHS_y_lil.tocsr(), rhs_2_T)

            # Enforce all Dirichlet boundaries on final solution
            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(self.t + self.dt)
                u_new = u_new_T.T
                u_new[0, :] = val; u_new[-1, :] = val
                u_new[:, 0] = val; u_new[:, -1] = val
                self.state = u_new
            else:
                self.state = u_new_T.T

        # --------------------- 3D CASE ---------------------
        elif self.ndim == 3:
            dt_third = self.dt / 3.0                
            Nx, Ny, Nz = self.grid_points
            LHS_x, RHS_x, LHS_y, RHS_y, LHS_z, RHS_z = self.system_matrix
            LHS_x_lil = LHS_x.copy().tolil()
            LHS_y_lil = LHS_y.copy().tolil()
            LHS_z_lil = LHS_z.copy().tolil()

            third_source = dt_third * source

            # --- SWEEP 1: Implicit X, Explicit Y & Z ---
            # Apply RHS_y (axis 1) and RHS_z (axis 2)
            u_y_expl = (RHS_y @ self.state.transpose(1, 0, 2).reshape(Ny, Nx * Nz)).reshape(Ny, Nx, Nz).transpose(1, 0, 2)
            u_z_expl = (RHS_z @ self.state.transpose(2, 0, 1).reshape(Nz, Nx * Ny)).reshape(Nz, Nx, Ny).transpose(1, 2, 0)
            
            # The "Identity" is already in the RHS matrices, so we don't add 'state' again.
            # We subtract one 'state' because (I + Ay) + (I + Az) = 2I + Ay + Az. We only want 1I.
            rhs_1 = u_y_expl + u_z_expl - self.state + third_source
            
            rhs_1_x = rhs_1.reshape(Nx, Ny * Nz)
            rhs_1_x = self._apply_bc_to_sweep(LHS_x.copy().tolil(), rhs_1_x, self.dx[0], dt_third)
            u_star = spsolve(LHS_x.tocsr(), rhs_1_x).reshape(Nx, Ny, Nz)

            # --- SWEEP 2: Implicit Y, Explicit X & Z ---
            u_x_expl = (RHS_x @ u_star).reshape(Nx, Ny, Nz)
            u_z_expl = (RHS_z @ u_star.transpose(2, 0, 1).reshape(Nz, Nx * Ny)).reshape(Nz, Nx, Ny).transpose(1, 2, 0)
            
            rhs_2 = u_x_expl + u_z_expl - u_star + third_source
            
            rhs_2_y = rhs_2.transpose(1, 0, 2).reshape(Ny, Nx * Nz)
            rhs_2_y = self._apply_bc_to_sweep(LHS_y.copy().tolil(), rhs_2_y, self.dx[1], dt_third)
            u_star_star = spsolve(LHS_y.tocsr(), rhs_2_y).reshape(Ny, Nx, Nz).transpose(1, 0, 2)

            # --- SWEEP 3: Implicit Z, Explicit X & Y ---
            u_x_expl = (RHS_x @ u_star_star).reshape(Nx, Ny, Nz)
            u_y_expl = (RHS_y @ u_star_star.transpose(1, 0, 2).reshape(Ny, Nx * Nz)).reshape(Ny, Nx, Nz).transpose(1, 0, 2)
            
            rhs_3 = u_x_expl + u_y_expl - u_star_star + third_source
            
            rhs_3_z = rhs_3.transpose(2, 0, 1).reshape(Nz, Nx * Ny)
            rhs_3_z = self._apply_bc_to_sweep(LHS_z.copy().tolil(), rhs_3_z, self.dx[2], dt_third)
            u_final_z = spsolve(LHS_z.tocsr(), rhs_3_z).reshape(Nz, Nx, Ny).transpose(1, 2, 0)
            
            self.state = u_final_z
        
        self.t += self.dt

    def _apply_bc_to_sweep(self, matrix, rhs_array: np.ndarray, h: float, dt_sweep: float) -> np.ndarray:
        if self._boundary_conditions is None:
            return rhs_array

        D = self.diffusion_coefficient
        
        if isinstance(self._boundary_conditions, NeumannBC):
            flux = self._boundary_conditions._get_flux(self.t + dt_sweep)
            
            alpha = (dt_sweep * D) / (h**2)
            forcing = (2 * dt_sweep * D * flux) / h
            
            matrix[0, 1] = -2 * alpha
            rhs_array[0, :] -= forcing
            
            matrix[-1, -2] = -2 * alpha
            rhs_array[-1, :] += forcing

        elif isinstance(self._boundary_conditions, DirichletBC):
            val = self._boundary_conditions._get_value(self.t + dt_sweep)
            
            matrix[0, :] = 0
            matrix[0, 0] = 1
            rhs_array[0, :] = val
            
            matrix[-1, :] = 0
            matrix[-1, -1] = 1
            rhs_array[-1, :] = val

            # ADI BC implementation fix
            # matrix[:, 0] = 0
            # matrix[:, -1] = 0
            # rhs_array[:, 0] = val
            # rhs_array[:, -1] = val
            
        return rhs_array

    def set_diffusion_coefficient(self, value: float) -> None:
        super().set_diffusion_coefficient(value)
        self._build_system_matrix()
    
    def set_decay_rate(self, value: float) -> None:
        super().set_decay_rate(value)
        self._build_system_matrix()