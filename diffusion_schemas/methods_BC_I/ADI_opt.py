"""
Implicit Euler with Alternating Direction Implicit (ADI) method.

This module implements the ADI splitting scheme. The 2D case uses the 
second-order Peaceman-Rachford formulation.
"""

import numpy as np
from scipy.sparse import diags, eye, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded
from diffusion_schemas.base import Schema
from diffusion_schemas.utils.boundary import DirichletBC, NeumannBC


class ADIBCISchema(Schema):
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
    
    # def _build_matrix_3d(self):
    #     Nx, Ny, Nz = self.grid_points
    #     dx, dy, dz = self.dx
        
    #     dt_third = self.dt / 3.0
    #     # Each third-step carries a third of the decay: (λ/3) * (dt/3) = λ·dt/9
    #     decay_term = (self.decay_rate / 3.0) * dt_third

    #     # X-Operators
    #     Lx = diags([np.ones(Nx-1)/dx**2, -2*np.ones(Nx)/dx**2, np.ones(Nx-1)/dx**2], [-1, 0, 1], shape=(Nx, Nx), format='csr').tolil()
    #     Ix = eye(Nx, format='csr')

    #     if isinstance(self._boundary_conditions, NeumannBC):
    #         Lx[0, 0], Lx[0, 1] = -2 / (dx**2), 2 / (dx**2)
    #         Lx[-1, -1], Lx[-1, -2] = -2 / (dx**2), 2 / (dx**2)

    #     Lx, Ix = Lx.tocsr(), Ix.tocsr()
    #     LHS_x = (Ix - dt_third * self.diffusion_coefficient * Lx + decay_term * Ix).tolil()
    #     RHS_x = (Ix + dt_third * self.diffusion_coefficient * Lx - decay_term * Ix).tolil()

    #     if isinstance(self._boundary_conditions, DirichletBC):
    #         for row in [0, -1]:
    #             LHS_x[row, :] = 0; LHS_x[row, row] = 1
    #             RHS_x[row, :] = 0; RHS_x[row, row] = 1

    #     # Y-Operators
    #     Ly = diags([np.ones(Ny-1)/dy**2, -2*np.ones(Ny)/dy**2, np.ones(Ny-1)/dy**2], [-1, 0, 1], shape=(Ny, Ny), format='csr').tolil()
    #     Iy = eye(Ny, format='csr')

    #     if isinstance(self._boundary_conditions, NeumannBC):
    #         Ly[0, 0], Ly[0, 1] = -2 / (dy**2), 2 / (dy**2)
    #         Ly[-1, -1], Ly[-1, -2] = -2 / (dy**2), 2 / (dy**2)

    #     Ly, Iy = Ly.tocsr(), Iy.tocsr()
    #     LHS_y = (Iy - dt_third * self.diffusion_coefficient * Ly + decay_term * Iy).tolil()
    #     RHS_y = (Iy + dt_third * self.diffusion_coefficient * Ly - decay_term * Iy).tolil()

    #     if isinstance(self._boundary_conditions, DirichletBC):
    #         for row in [0, -1]:
    #             LHS_y[row, :] = 0; LHS_y[row, row] = 1
    #             RHS_y[row, :] = 0; RHS_y[row, row] = 1

    #     # Z-Operators
    #     Lz = diags([np.ones(Nz-1)/dz**2, -2*np.ones(Nz)/dz**2, np.ones(Nz-1)/dz**2], [-1, 0, 1], shape=(Nz, Nz), format='csr').tolil()
    #     Iz = eye(Nz, format='csr')

    #     if isinstance(self._boundary_conditions, NeumannBC):
    #         Lz[0, 0], Lz[0, 1] = -2 / (dz**2), 2 / (dz**2)
    #         Lz[-1, -1], Lz[-1, -2] = -2 / (dz**2), 2 / (dz**2)

    #     Lz, Iz = Lz.tocsr(), Iz.tocsr()
    #     LHS_z = (Iz - dt_third * self.diffusion_coefficient * Lz + decay_term * Iz).tolil()
    #     RHS_z = (Iz + dt_third * self.diffusion_coefficient * Lz - decay_term * Iz).tolil()

    #     if isinstance(self._boundary_conditions, DirichletBC):
    #         for row in [0, -1]:
    #             LHS_z[row, :] = 0; LHS_z[row, row] = 1
    #             RHS_z[row, :] = 0; RHS_z[row, row] = 1

    #     return LHS_x, RHS_x, LHS_y, RHS_y, LHS_z, RHS_z

    def _build_matrix_3d(self):
        Nx, Ny, Nz = self.grid_points
        dx, dy, dz = self.dx
        
        # Split decay rate symmetrically across the 3 operators
        decay_term = (self.decay_rate / 3.0) * self.dt

        # X-Operators
        Lx = diags([np.ones(Nx-1)/dx**2, -2*np.ones(Nx)/dx**2, np.ones(Nx-1)/dx**2], [-1, 0, 1], shape=(Nx, Nx), format='csr').tolil()
        Ix = eye(Nx, format='csr')

        if isinstance(self._boundary_conditions, NeumannBC):
            Lx[0, 0], Lx[0, 1] = -2 / (dx**2), 2 / (dx**2)
            Lx[-1, -1], Lx[-1, -2] = -2 / (dx**2), 2 / (dx**2)

        Lx, Ix = Lx.tocsr(), Ix.tocsr()
        LHS_x = None  # Not needed since we build banded arrays dynamically
        A_x = (self.dt * self.diffusion_coefficient * Lx - decay_term * Ix).tolil()

        if isinstance(self._boundary_conditions, DirichletBC):
            for row in [0, -1]:
                A_x[row, :] = 0; A_x[row, row] = 0

        # Y-Operators
        Ly = diags([np.ones(Ny-1)/dy**2, -2*np.ones(Ny)/dy**2, np.ones(Ny-1)/dy**2], [-1, 0, 1], shape=(Ny, Ny), format='csr').tolil()
        Iy = eye(Ny, format='csr')

        if isinstance(self._boundary_conditions, NeumannBC):
            Ly[0, 0], Ly[0, 1] = -2 / (dy**2), 2 / (dy**2)
            Ly[-1, -1], Ly[-1, -2] = -2 / (dy**2), 2 / (dy**2)

        Ly, Iy = Ly.tocsr(), Iy.tocsr()
        LHS_y = None
        A_y = (self.dt * self.diffusion_coefficient * Ly - decay_term * Iy).tolil()

        if isinstance(self._boundary_conditions, DirichletBC):
            for row in [0, -1]:
                A_y[row, :] = 0; A_y[row, row] = 0

        # Z-Operators
        Lz = diags([np.ones(Nz-1)/dz**2, -2*np.ones(Nz)/dz**2, np.ones(Nz-1)/dz**2], [-1, 0, 1], shape=(Nz, Nz), format='csr').tolil()
        Iz = eye(Nz, format='csr')

        if isinstance(self._boundary_conditions, NeumannBC):
            Lz[0, 0], Lz[0, 1] = -2 / (dz**2), 2 / (dz**2)
            Lz[-1, -1], Lz[-1, -2] = -2 / (dz**2), 2 / (dz**2)

        Lz, Iz = Lz.tocsr(), Iz.tocsr()
        LHS_z = None
        A_z = (self.dt * self.diffusion_coefficient * Lz - decay_term * Iz).tolil()

        if isinstance(self._boundary_conditions, DirichletBC):
            for row in [0, -1]:
                A_z[row, :] = 0; A_z[row, row] = 0

        return LHS_x, A_x, LHS_y, A_y, LHS_z, A_z

    def step(self) -> None:
        from scipy.linalg import solve_banded
        
        if self.ndim not in (1, 2, 3):
            raise NotImplementedError(f"{self.ndim}D ADI is not implemented yet")

        # --------------------- 1D CASE ---------------------
        if self.ndim == 1:
            t_next = self.t + self.dt
            N = self.grid_points[0]

            source_explicit = self._compute_source_term(implicit=True, t=t_next)
            source_rhs = np.zeros_like(self.state)
            source_lhs = np.zeros_like(self.state)
            
            if self._bulk is not None:
                source_rhs = self._bulk.rhs_contribution
                source_lhs = self._bulk.lhs_contribution
                if isinstance(self._boundary_conditions, DirichletBC):
                    source_lhs[self._boundary_mask] = 0.0

            rhs = self.state + self.dt * (source_rhs + source_explicit)
            rhs_flat = rhs.flatten()

            alpha = self.dt * self.diffusion_coefficient / (self.dx[0]**2)
            decay_term = self.dt * self.decay_rate

            ab_x = np.zeros((3, N))
            ab_x[0, 1:] = -alpha
            ab_x[2, :-1] = -alpha
            ab_x[1, :] = 1.0 + 2*alpha + decay_term + self.dt * source_lhs.flatten()

            self._apply_bc_to_banded(ab_x, rhs_flat, self.dx[0], self.dt, t_next)
            self.state = solve_banded((1, 1), ab_x, rhs_flat).reshape(self.grid_points)

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(t_next)
                self.state[0] = val
                self.state[-1] = val

        # --------------------- 2D CASE ---------------------
        elif self.ndim == 2:
            dt_half = self.dt / 2.0
            t_mid = self.t + dt_half
            
            D = self.diffusion_coefficient
            dx, dy = self.dx
            Nx, Ny = self.grid_points

            # Unpack RHS matrices (Ignore LHS, we build banded matrices on the fly)
            _, RHS_x, _, RHS_y = self.system_matrix

            alpha_x = dt_half * D / (dx**2)
            alpha_y = dt_half * D / (dy**2)
            decay_term = (self.decay_rate / 2.0) * dt_half

            # --- SWEEP 1: Implicit X, Explicit Y ---
            source_explicit = self._compute_source_term(implicit=True, t=t_mid)
            source_rhs = np.zeros_like(self.state)
            source_lhs = np.zeros_like(self.state)
            
            if self._bulk is not None:  
                source_rhs = self._bulk.rhs_contribution
                source_lhs = self._bulk.lhs_contribution
                if isinstance(self._boundary_conditions, DirichletBC):
                    source_lhs[self._boundary_mask] = 0.0

            rhs_1 = (RHS_y @ self.state.T).T + dt_half * (source_rhs + source_explicit)

            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(t_mid)
                explicit_y_forcing = dt_half * D * 2 * flux / dy
                rhs_1[:, 0]  -= explicit_y_forcing
                rhs_1[:, -1] += explicit_y_forcing

            u_star = np.zeros((Nx, Ny))
            for j in range(Ny):
                ab_x = np.zeros((3, Nx))
                ab_x[0, 1:] = -alpha_x
                ab_x[2, :-1] = -alpha_x
                ab_x[1, :] = 1.0 + 2*alpha_x + decay_term + dt_half * source_lhs[:, j]
                
                rhs_1_j = rhs_1[:, j].copy()
                self._apply_bc_to_banded(ab_x, rhs_1_j, dx, dt_half, t_eval=t_mid)
                u_star[:, j] = solve_banded((1, 1), ab_x, rhs_1_j)

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(t_mid)
                u_star[0, :] = val; u_star[-1, :] = val
                u_star[:, 0] = val; u_star[:, -1] = val

            # --- SWEEP 2: Explicit X, Implicit Y ---
            source_explicit = self._compute_source_term(state=u_star, implicit=True, t=self.t + self.dt)
            source_rhs = np.zeros_like(self.state)
            source_lhs = np.zeros_like(self.state)
            
            if self._bulk is not None:  
                source_rhs = self._bulk.rhs_contribution
                source_lhs = self._bulk.lhs_contribution
                if isinstance(self._boundary_conditions, DirichletBC):
                    source_lhs[self._boundary_mask] = 0.0

            rhs_2 = (RHS_x @ u_star) + dt_half * (source_rhs + source_explicit)

            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(self.t + self.dt)
                explicit_x_forcing = dt_half * D * 2 * flux / dx
                rhs_2[0, :]  -= explicit_x_forcing
                rhs_2[-1, :] += explicit_x_forcing

            u_new = np.zeros((Nx, Ny))
            for i in range(Nx):
                ab_y = np.zeros((3, Ny))
                ab_y[0, 1:] = -alpha_y
                ab_y[2, :-1] = -alpha_y
                ab_y[1, :] = 1.0 + 2*alpha_y + decay_term + dt_half * source_lhs[i, :]
                
                rhs_2_i = rhs_2[i, :].copy()
                self._apply_bc_to_banded(ab_y, rhs_2_i, dy, dt_half, t_eval=self.t + self.dt)
                u_new[i, :] = solve_banded((1, 1), ab_y, rhs_2_i)

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(self.t + self.dt)
                u_new[0, :] = val; u_new[-1, :] = val
                u_new[:, 0] = val; u_new[:, -1] = val

            self.state = u_new

        # --------------------- 3D CASE (Peaceman-Rachford formulation) ---------------------
        # elif self.ndim == 3:
        #     dt_third = self.dt / 3.0
        #     t_1 = self.t + dt_third
        #     t_2 = self.t + 2 * dt_third
        #     t_3 = self.t + self.dt

        #     D = self.diffusion_coefficient
        #     dx, dy, dz = self.dx
        #     Nx, Ny, Nz = self.grid_points

        #     # Unpack RHS matrices
        #     _, RHS_x, _, RHS_y, _, RHS_z = self.system_matrix
            
        #     alpha_x = dt_third * D / (dx**2)
        #     alpha_y = dt_third * D / (dy**2)
        #     alpha_z = dt_third * D / (dz**2)
        #     decay_term = (self.decay_rate / 3.0) * dt_third

        #     # --- SWEEP 1: Implicit X, Explicit Y & Z ---
        #     source_explicit = self._compute_source_term(implicit=True, t=t_1)
        #     source_rhs = np.zeros_like(self.state)
        #     source_lhs = np.zeros_like(self.state)
            
        #     if self._bulk is not None:
        #         source_rhs = self._bulk.rhs_contribution
        #         source_lhs = self._bulk.lhs_contribution
        #         if isinstance(self._boundary_conditions, DirichletBC):
        #             source_lhs[self._boundary_mask] = 0.0

        #     rhs_y = (RHS_y @ self.state.transpose(1, 0, 2).reshape(Ny, Nx * Nz)).reshape(Ny, Nx, Nz).transpose(1, 0, 2)
        #     rhs_z = (RHS_z @ self.state.transpose(2, 0, 1).reshape(Nz, Nx * Ny)).reshape(Nz, Nx, Ny).transpose(1, 2, 0)
        #     rhs_1 = rhs_y + rhs_z - self.state + dt_third * (source_rhs + source_explicit)

        #     if isinstance(self._boundary_conditions, NeumannBC):
        #         flux = self._boundary_conditions._get_flux(t_1)
        #         rhs_1[:, 0, :]  -= dt_third * D * 2 * flux / dy
        #         rhs_1[:, -1, :] += dt_third * D * 2 * flux / dy
        #         rhs_1[:, :, 0]  -= dt_third * D * 2 * flux / dz
        #         rhs_1[:, :, -1] += dt_third * D * 2 * flux / dz

        #     u_star = np.zeros((Nx, Ny, Nz))
        #     for j in range(Ny):
        #         for k in range(Nz):
        #             ab_x = np.zeros((3, Nx))
        #             ab_x[0, 1:] = -alpha_x
        #             ab_x[2, :-1] = -alpha_x
        #             ab_x[1, :] = 1.0 + 2*alpha_x + decay_term + dt_third * source_lhs[:, j, k]
                    
        #             rhs_1_jk = rhs_1[:, j, k].copy()
        #             self._apply_bc_to_banded(ab_x, rhs_1_jk, dx, dt_third, t_eval=t_1)
        #             u_star[:, j, k] = solve_banded((1, 1), ab_x, rhs_1_jk)

        #     if isinstance(self._boundary_conditions, DirichletBC):
        #         val = self._boundary_conditions._get_value(t_1)
        #         u_star[0, :, :] = val; u_star[-1, :, :] = val
        #         u_star[:, 0, :] = val; u_star[:, -1, :] = val
        #         u_star[:, :, 0] = val; u_star[:, :, -1] = val

        #     # --- SWEEP 2: Implicit Y, Explicit X & Z ---
        #     source_explicit = self._compute_source_term(state=u_star, implicit=True, t=t_2)
        #     source_rhs = np.zeros_like(self.state)
        #     source_lhs = np.zeros_like(self.state)
            
        #     if self._bulk is not None:
        #         source_rhs = self._bulk.rhs_contribution
        #         source_lhs = self._bulk.lhs_contribution
        #         if isinstance(self._boundary_conditions, DirichletBC):
        #             source_lhs[self._boundary_mask] = 0.0

        #     rhs_x = (RHS_x @ u_star.reshape(Nx, Ny * Nz)).reshape(Nx, Ny, Nz)
        #     rhs_z = (RHS_z @ u_star.transpose(2, 0, 1).reshape(Nz, Nx * Ny)).reshape(Nz, Nx, Ny).transpose(1, 2, 0)
        #     rhs_2 = rhs_x + rhs_z - u_star + dt_third * (source_rhs + source_explicit)

        #     if isinstance(self._boundary_conditions, NeumannBC):
        #         flux = self._boundary_conditions._get_flux(t_2)
        #         rhs_2[0, :, :]  -= dt_third * D * 2 * flux / dx
        #         rhs_2[-1, :, :] += dt_third * D * 2 * flux / dx
        #         rhs_2[:, :, 0]  -= dt_third * D * 2 * flux / dz
        #         rhs_2[:, :, -1] += dt_third * D * 2 * flux / dz

        #     u_star_star = np.zeros((Nx, Ny, Nz))
        #     for i in range(Nx):
        #         for k in range(Nz):
        #             ab_y = np.zeros((3, Ny))
        #             ab_y[0, 1:] = -alpha_y
        #             ab_y[2, :-1] = -alpha_y
        #             ab_y[1, :] = 1.0 + 2*alpha_y + decay_term + dt_third * source_lhs[i, :, k]
                    
        #             rhs_2_ik = rhs_2[i, :, k].copy()
        #             self._apply_bc_to_banded(ab_y, rhs_2_ik, dy, dt_third, t_eval=t_2)
        #             u_star_star[i, :, k] = solve_banded((1, 1), ab_y, rhs_2_ik)

        #     if isinstance(self._boundary_conditions, DirichletBC):
        #         val = self._boundary_conditions._get_value(t_2)
        #         u_star_star[0, :, :] = val; u_star_star[-1, :, :] = val
        #         u_star_star[:, 0, :] = val; u_star_star[:, -1, :] = val
        #         u_star_star[:, :, 0] = val; u_star_star[:, :, -1] = val

        #     # --- SWEEP 3: Implicit Z, Explicit X & Y ---
        #     source_explicit = self._compute_source_term(state=u_star_star, implicit=True, t=t_3)
        #     source_rhs = np.zeros_like(self.state)
        #     source_lhs = np.zeros_like(self.state)
            
        #     if self._bulk is not None:
        #         source_rhs = self._bulk.rhs_contribution
        #         source_lhs = self._bulk.lhs_contribution
        #         if isinstance(self._boundary_conditions, DirichletBC):
        #             source_lhs[self._boundary_mask] = 0.0

        #     rhs_x = (RHS_x @ u_star_star.reshape(Nx, Ny * Nz)).reshape(Nx, Ny, Nz)
        #     rhs_y = (RHS_y @ u_star_star.transpose(1, 0, 2).reshape(Ny, Nx * Nz)).reshape(Ny, Nx, Nz).transpose(1, 0, 2)
        #     rhs_3 = rhs_x + rhs_y - u_star_star + dt_third * (source_rhs + source_explicit)

        #     if isinstance(self._boundary_conditions, NeumannBC):
        #         flux = self._boundary_conditions._get_flux(t_3)
        #         rhs_3[0, :, :]  -= dt_third * D * 2 * flux / dx
        #         rhs_3[-1, :, :] += dt_third * D * 2 * flux / dx
        #         rhs_3[:, 0, :]  -= dt_third * D * 2 * flux / dy
        #         rhs_3[:, -1, :] += dt_third * D * 2 * flux / dy

        #     u_new = np.zeros((Nx, Ny, Nz))
        #     for i in range(Nx):
        #         for j in range(Ny):
        #             ab_z = np.zeros((3, Nz))
        #             ab_z[0, 1:] = -alpha_z
        #             ab_z[2, :-1] = -alpha_z
        #             ab_z[1, :] = 1.0 + 2*alpha_z + decay_term + dt_third * source_lhs[i, j, :]
                    
        #             rhs_3_ij = rhs_3[i, j, :].copy()
        #             self._apply_bc_to_banded(ab_z, rhs_3_ij, dz, dt_third, t_eval=t_3)
        #             u_new[i, j, :] = solve_banded((1, 1), ab_z, rhs_3_ij)

        #     if isinstance(self._boundary_conditions, DirichletBC):
        #         val = self._boundary_conditions._get_value(t_3)
        #         u_new[0, :, :] = val; u_new[-1, :, :] = val
        #         u_new[:, 0, :] = val; u_new[:, -1, :] = val
        #         u_new[:, :, 0] = val; u_new[:, :, -1] = val

        #     self.state = u_new

        # --------------------- 3D CASE (Douglas-Gunn) ---------------------
        elif self.ndim == 3:
            t_next = self.t + self.dt

            D = self.diffusion_coefficient
            dx, dy, dz = self.dx
            Nx, Ny, Nz = self.grid_points

            # Unpack full-step explicit matrices
            _, A_x, _, A_y, _, A_z = self.system_matrix
            
            alpha_x = self.dt * D / (dx**2)
            alpha_y = self.dt * D / (dy**2)
            alpha_z = self.dt * D / (dz**2)
            decay_term = (self.decay_rate / 3.0) * self.dt
            
            un = self.state

            # --- Sources ---
            source_explicit = self._compute_source_term(implicit=True, t=t_next)
            source_rhs = np.zeros_like(self.state)
            source_lhs = np.zeros_like(self.state)
            
            if self._bulk is not None:
                source_rhs = self._bulk.rhs_contribution
                source_lhs = self._bulk.lhs_contribution
                if isinstance(self._boundary_conditions, DirichletBC):
                    source_lhs[self._boundary_mask] = 0.0

            # Pre-calculate explicit operations on u^n
            A_y_un = (A_y @ un.transpose(1, 0, 2).reshape(Ny, Nx * Nz)).reshape(Ny, Nx, Nz).transpose(1, 0, 2)
            A_z_un = (A_z @ un.transpose(2, 0, 1).reshape(Nz, Nx * Ny)).reshape(Nz, Nx, Ny).transpose(1, 2, 0)

            # --- SWEEP 1: X-direction ---
            rhs_1 = un + A_y_un + A_z_un + self.dt * (source_rhs + source_explicit)

            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(t_next)
                rhs_1[:, 0, :]  -= self.dt * D * 2 * flux / dy
                rhs_1[:, -1, :] += self.dt * D * 2 * flux / dy
                rhs_1[:, :, 0]  -= self.dt * D * 2 * flux / dz
                rhs_1[:, :, -1] += self.dt * D * 2 * flux / dz

            u_star = np.zeros((Nx, Ny, Nz))
            for j in range(Ny):
                for k in range(Nz):
                    ab_x = np.zeros((3, Nx))
                    ab_x[0, 1:] = -alpha_x
                    ab_x[2, :-1] = -alpha_x
                    ab_x[1, :] = 1.0 + 2*alpha_x + decay_term + (self.dt * source_lhs[:, j, k] / 3.0)
                    
                    rhs_1_jk = rhs_1[:, j, k].copy()
                    self._apply_bc_to_banded(ab_x, rhs_1_jk, dx, self.dt, t_eval=t_next)
                    u_star[:, j, k] = solve_banded((1, 1), ab_x, rhs_1_jk)

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(t_next)
                u_star[0, :, :] = val; u_star[-1, :, :] = val
                u_star[:, 0, :] = val; u_star[:, -1, :] = val
                u_star[:, :, 0] = val; u_star[:, :, -1] = val

            # --- SWEEP 2: Y-direction ---
            rhs_2 = u_star - A_y_un

            u_star_star = np.zeros((Nx, Ny, Nz))
            for i in range(Nx):
                for k in range(Nz):
                    ab_y = np.zeros((3, Ny))
                    ab_y[0, 1:] = -alpha_y
                    ab_y[2, :-1] = -alpha_y
                    ab_y[1, :] = 1.0 + 2*alpha_y + decay_term + (self.dt * source_lhs[i, :, k] / 3.0)
                    
                    rhs_2_ik = rhs_2[i, :, k].copy()
                    self._apply_bc_to_banded(ab_y, rhs_2_ik, dy, self.dt, t_eval=t_next)
                    u_star_star[i, :, k] = solve_banded((1, 1), ab_y, rhs_2_ik)

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(t_next)
                u_star_star[0, :, :] = val; u_star_star[-1, :, :] = val
                u_star_star[:, 0, :] = val; u_star_star[:, -1, :] = val
                u_star_star[:, :, 0] = val; u_star_star[:, :, -1] = val

            # --- SWEEP 3: Z-direction ---
            rhs_3 = u_star_star - A_z_un

            u_new = np.zeros((Nx, Ny, Nz))
            for i in range(Nx):
                for j in range(Ny):
                    ab_z = np.zeros((3, Nz))
                    ab_z[0, 1:] = -alpha_z
                    ab_z[2, :-1] = -alpha_z
                    ab_z[1, :] = 1.0 + 2*alpha_z + decay_term + (self.dt * source_lhs[i, j, :] / 3.0)
                    
                    rhs_3_ij = rhs_3[i, j, :].copy()
                    self._apply_bc_to_banded(ab_z, rhs_3_ij, dz, self.dt, t_eval=t_next)
                    u_new[i, j, :] = solve_banded((1, 1), ab_z, rhs_3_ij)

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(t_next)
                u_new[0, :, :] = val; u_new[-1, :, :] = val
                u_new[:, 0, :] = val; u_new[:, -1, :] = val
                u_new[:, :, 0] = val; u_new[:, :, -1] = val

            self.state = u_new

        self.t += self.dt


    def _apply_bc_to_banded(self, ab: np.ndarray, rhs_array: np.ndarray, h: float, dt_sweep: float, t_eval: float) -> None:
        """Apply boundary conditions to the banded matrix and 1D RHS array in-place."""
        if self._boundary_conditions is None:
            return

        alpha = (dt_sweep * self.diffusion_coefficient) / (h**2)

        if isinstance(self._boundary_conditions, NeumannBC):
            flux = self._boundary_conditions._get_flux(t_eval)
            forcing = (2 * dt_sweep * self.diffusion_coefficient * flux) / h
            
            # Left Boundary
            ab[0, 1] = -2 * alpha
            rhs_array[0] -= forcing
            
            # Right Boundary
            ab[2, -2] = -2 * alpha
            rhs_array[-1] += forcing

        elif isinstance(self._boundary_conditions, DirichletBC):
            val = self._boundary_conditions._get_value(t_eval)
            
            # Left Boundary
            ab[1, 0] = 1.0
            ab[0, 1] = 0.0
            rhs_array[0] = val
            
            # Right Boundary
            ab[1, -1] = 1.0
            ab[2, -2] = 0.0
            rhs_array[-1] = val

    def _apply_bc_to_sweep(self, matrix, rhs_array: np.ndarray, h: float, dt_sweep: float, t_eval: float = None) -> np.ndarray:
        if self._boundary_conditions is None:
            return rhs_array

        D = self.diffusion_coefficient
        bc_time = self.t + dt_sweep if t_eval is None else t_eval
        
        if isinstance(self._boundary_conditions, NeumannBC):
            flux = self._boundary_conditions._get_flux(bc_time)
            
            alpha = (dt_sweep * D) / (h**2)
            forcing = (2 * dt_sweep * D * flux) / h
            
            matrix[0, 1] = -2 * alpha
            rhs_array[0, :] -= forcing
            
            matrix[-1, -2] = -2 * alpha
            rhs_array[-1, :] += forcing
                
        elif isinstance(self._boundary_conditions, DirichletBC):
            val = self._boundary_conditions._get_value(bc_time)
                        
            matrix[0, :] = 0
            matrix[0, 0] = 1
            rhs_array[0, :] = val
            
            matrix[-1, :] = 0
            matrix[-1, -1] = 1
            rhs_array[-1, :] = val            
        return rhs_array

    def set_boundary_conditions(self, boundary_conditions) -> None:
        super().set_boundary_conditions(boundary_conditions)
        # ADI operators include BC-dependent stencil terms and must be rebuilt.
        self._build_system_matrix()

    def set_diffusion_coefficient(self, value: float) -> None:
        super().set_diffusion_coefficient(value)
        self._build_system_matrix()
    
    def set_decay_rate(self, value: float) -> None:
        super().set_decay_rate(value)
        self._build_system_matrix()