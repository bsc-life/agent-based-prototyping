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
    
    def _build_matrix_3d(self):
        Nx, Ny, Nz = self.grid_points
        dx, dy, dz = self.dx
        
        dt_third = self.dt / 3.0
        # Each third-step carries a third of the decay: (λ/3) * (dt/3) = λ·dt/9
        decay_term = (self.decay_rate / 3.0) * dt_third

        # X-Operators
        Lx = diags([np.ones(Nx-1)/dx**2, -2*np.ones(Nx)/dx**2, np.ones(Nx-1)/dx**2], [-1, 0, 1], shape=(Nx, Nx), format='csr').tolil()
        Ix = eye(Nx, format='csr')

        if isinstance(self._boundary_conditions, NeumannBC):
            Lx[0, 0], Lx[0, 1] = -2 / (dx**2), 2 / (dx**2)
            Lx[-1, -1], Lx[-1, -2] = -2 / (dx**2), 2 / (dx**2)

        Lx, Ix = Lx.tocsr(), Ix.tocsr()
        LHS_x = (Ix - dt_third * self.diffusion_coefficient * Lx + decay_term * Ix).tolil()
        RHS_x = (Ix + dt_third * self.diffusion_coefficient * Lx - decay_term * Ix).tolil()

        if isinstance(self._boundary_conditions, DirichletBC):
            for row in [0, -1]:
                LHS_x[row, :] = 0; LHS_x[row, row] = 1
                RHS_x[row, :] = 0; RHS_x[row, row] = 1

        # Y-Operators
        Ly = diags([np.ones(Ny-1)/dy**2, -2*np.ones(Ny)/dy**2, np.ones(Ny-1)/dy**2], [-1, 0, 1], shape=(Ny, Ny), format='csr').tolil()
        Iy = eye(Ny, format='csr')

        if isinstance(self._boundary_conditions, NeumannBC):
            Ly[0, 0], Ly[0, 1] = -2 / (dy**2), 2 / (dy**2)
            Ly[-1, -1], Ly[-1, -2] = -2 / (dy**2), 2 / (dy**2)

        Ly, Iy = Ly.tocsr(), Iy.tocsr()
        LHS_y = (Iy - dt_third * self.diffusion_coefficient * Ly + decay_term * Iy).tolil()
        RHS_y = (Iy + dt_third * self.diffusion_coefficient * Ly - decay_term * Iy).tolil()

        if isinstance(self._boundary_conditions, DirichletBC):
            for row in [0, -1]:
                LHS_y[row, :] = 0; LHS_y[row, row] = 1
                RHS_y[row, :] = 0; RHS_y[row, row] = 1

        # Z-Operators
        Lz = diags([np.ones(Nz-1)/dz**2, -2*np.ones(Nz)/dz**2, np.ones(Nz-1)/dz**2], [-1, 0, 1], shape=(Nz, Nz), format='csr').tolil()
        Iz = eye(Nz, format='csr')

        if isinstance(self._boundary_conditions, NeumannBC):
            Lz[0, 0], Lz[0, 1] = -2 / (dz**2), 2 / (dz**2)
            Lz[-1, -1], Lz[-1, -2] = -2 / (dz**2), 2 / (dz**2)

        Lz, Iz = Lz.tocsr(), Iz.tocsr()
        LHS_z = (Iz - dt_third * self.diffusion_coefficient * Lz + decay_term * Iz).tolil()
        RHS_z = (Iz + dt_third * self.diffusion_coefficient * Lz - decay_term * Iz).tolil()

        if isinstance(self._boundary_conditions, DirichletBC):
            for row in [0, -1]:
                LHS_z[row, :] = 0; LHS_z[row, row] = 1
                RHS_z[row, :] = 0; RHS_z[row, row] = 1

        return LHS_x, RHS_x, LHS_y, RHS_y, LHS_z, RHS_z

    def step(self) -> None:
        if self.ndim not in (1, 2, 3):
            raise NotImplementedError(f"{self.ndim}D ADI is not implemented yet")

        # --------------------- 1D CASE (Implicit source) ---------------------
        if self.ndim == 1:
            t_next = self.t + self.dt
            N = self.grid_points[0]

            source_explicit = self._compute_source_term(implicit=True, t=t_next)
            source_rhs = np.zeros_like(self.state)
            source_lhs = np.zeros_like(self.state)
            if self._bulk is not None:
                source_rhs = self._bulk.rhs_contribution
                source_lhs = self._bulk.lhs_contribution
                source_lhs[self._boundary_mask] = 0.0

            rhs = self.state + self.dt * (source_rhs + source_explicit)

            source_diag = diags([self.dt * source_lhs], [0], shape=(N, N), format='csr')
            Ax = (self.system_matrix + source_diag).tolil()
            rhs = rhs.reshape(N, 1)
            rhs = self._apply_bc_to_sweep(Ax, rhs, self.dx[0], self.dt)
            self.state = spsolve(Ax.tocsr(), rhs).reshape(self.grid_points)

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(t_next)
                self.state[0] = val
                self.state[-1] = val

        # --------------------- 2D CASE (Peaceman-Rachford with IMPLICIT source) ---------------------
        elif self.ndim == 2:
            dt_half = self.dt / 2.0
            t_mid = self.t + dt_half
            
            D = self.diffusion_coefficient
            dx, dy = self.dx
            Nx, Ny = self.grid_points

            LHS_x, RHS_x, LHS_y, RHS_y = self.system_matrix

            # --- SWEEP 1: Implicit X, Explicit Y (with implicit source) ---

            # Returns explicit agent source; bulk implicit split is cached in self._bulk.
            source_explicit = self._compute_source_term(implicit = True, t = t_mid)
            source_rhs = np.zeros_like(self.state)
            source_lhs = np.zeros_like(self.state)
            if self._bulk is not None:  
                source_rhs = self._bulk.rhs_contribution
                source_lhs = self._bulk.lhs_contribution

                # Ignore boundary values for the implicit source term since they will be overwritten by BC enforcement
                source_lhs[self._boundary_mask] = 0.0

            rhs_1 = (RHS_y @ self.state.T).T + dt_half * (source_rhs + source_explicit)

            # Add explicit transverse Neumann forcing (y-direction)
            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(t_mid)
                explicit_y_forcing = dt_half * D * 2 * flux / dy
                rhs_1[:, 0]  -= explicit_y_forcing
                rhs_1[:, -1] += explicit_y_forcing

            # Solve row-by-row with implicit source term
            u_star = np.zeros((Nx, Ny))
            for j in range(Ny):
                # Create modified LHS with diagonal source term
                source_diag = diags([dt_half * source_lhs[:, j]], [0], shape=(Nx, Nx), format='csr')
                LHS_x_j = (LHS_x + source_diag).tolil()
                
                # Extract RHS for this column
                rhs_1_j = rhs_1[:, j].reshape(Nx, 1)
                
                # Apply boundary conditions
                rhs_1_j = self._apply_bc_to_sweep(LHS_x_j, rhs_1_j, dx, dt_half)
                
                # Solve and store
                u_star[:, j] = spsolve(LHS_x_j.tocsr(), rhs_1_j).flatten()

            # Enforce all Dirichlet boundaries on intermediate solution
            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(self.t + dt_half)
                u_star[0, :] = val; u_star[-1, :] = val
                u_star[:, 0] = val; u_star[:, -1] = val

            # --- SWEEP 2: Explicit X, Implicit Y (with implicit source) ---
            source_explicit = self._compute_source_term(state = u_star, implicit = True, t = self.t + self.dt)
            source_rhs = np.zeros_like(self.state)
            source_lhs = np.zeros_like(self.state)
            if self._bulk is not None:  
                source_rhs = self._bulk.rhs_contribution
                source_lhs = self._bulk.lhs_contribution
                source_lhs[self._boundary_mask] = 0.0
            rhs_2 = (RHS_x @ u_star) + dt_half * (source_rhs + source_explicit)

            # Add explicit transverse Neumann forcing (x-direction)
            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(self.t + self.dt)
                explicit_x_forcing = dt_half * D * 2 * flux / dx
                rhs_2[0, :]  -= explicit_x_forcing
                rhs_2[-1, :] += explicit_x_forcing

            # Solve column-by-column with implicit source term
            u_new = np.zeros((Nx, Ny))
            for i in range(Nx):
                # Create modified LHS with diagonal source term
                source_diag = diags([dt_half * source_lhs[i, :]], [0], shape=(Ny, Ny), format='csr')
                LHS_y_i = (LHS_y + source_diag).tolil()
                
                # Extract RHS for this row (as column vector)
                rhs_2_i = rhs_2[i, :].reshape(Ny, 1)
                
                # Apply boundary conditions
                rhs_2_i = self._apply_bc_to_sweep(LHS_y_i, rhs_2_i, dy, dt_half)
                
                # Solve and store
                u_new[i, :] = spsolve(LHS_y_i.tocsr(), rhs_2_i).flatten()

            # Enforce all Dirichlet boundaries on final solution
            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(self.t + self.dt)
                u_new[0, :] = val; u_new[-1, :] = val
                u_new[:, 0] = val; u_new[:, -1] = val

            self.state = u_new

        # --------------------- 3D CASE (Fractional-step with IMPLICIT source) ---------------------
        elif self.ndim == 3:
            dt_third = self.dt / 3.0
            t_1 = self.t + dt_third
            t_2 = self.t + 2 * dt_third
            t_3 = self.t + self.dt

            D = self.diffusion_coefficient
            dx, dy, dz = self.dx
            Nx, Ny, Nz = self.grid_points

            LHS_x, RHS_x, LHS_y, RHS_y, LHS_z, RHS_z = self.system_matrix

            # --- SWEEP 1: Implicit X, Explicit Y & Z (with implicit source) ---
            source_explicit = self._compute_source_term(implicit=True, t=t_1)
            source_rhs = np.zeros_like(self.state)
            source_lhs = np.zeros_like(self.state)
            if self._bulk is not None:
                source_rhs = self._bulk.rhs_contribution
                source_lhs = self._bulk.lhs_contribution
                source_lhs[self._boundary_mask] = 0.0

            rhs_y = (RHS_y @ self.state.transpose(1, 0, 2).reshape(Ny, Nx * Nz)).reshape(Ny, Nx, Nz).transpose(1, 0, 2)
            rhs_z = (RHS_z @ self.state.transpose(2, 0, 1).reshape(Nz, Nx * Ny)).reshape(Nz, Nx, Ny).transpose(1, 2, 0)
            rhs_1 = rhs_y + rhs_z - self.state + dt_third * (source_rhs + source_explicit)

            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(t_1)
                rhs_1[:, 0, :]  -= dt_third * D * 2 * flux / dy
                rhs_1[:, -1, :] += dt_third * D * 2 * flux / dy
                rhs_1[:, :, 0]  -= dt_third * D * 2 * flux / dz
                rhs_1[:, :, -1] += dt_third * D * 2 * flux / dz

            u_star = np.zeros((Nx, Ny, Nz))
            for j in range(Ny):
                for k in range(Nz):
                    source_diag = diags([dt_third * source_lhs[:, j, k]], [0], shape=(Nx, Nx), format='csr')
                    LHS_x_jk = (LHS_x + source_diag).tolil()
                    rhs_1_jk = rhs_1[:, j, k].reshape(Nx, 1)
                    rhs_1_jk = self._apply_bc_to_sweep(LHS_x_jk, rhs_1_jk, dx, dt_third)
                    u_star[:, j, k] = spsolve(LHS_x_jk.tocsr(), rhs_1_jk).flatten()

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(t_1)
                u_star[0, :, :] = val; u_star[-1, :, :] = val
                u_star[:, 0, :] = val; u_star[:, -1, :] = val
                u_star[:, :, 0] = val; u_star[:, :, -1] = val

            # --- SWEEP 2: Implicit Y, Explicit X & Z (with implicit source) ---
            source_explicit = self._compute_source_term(state=u_star, implicit=True, t=t_2)
            source_rhs = np.zeros_like(self.state)
            source_lhs = np.zeros_like(self.state)
            if self._bulk is not None:
                source_rhs = self._bulk.rhs_contribution
                source_lhs = self._bulk.lhs_contribution
                source_lhs[self._boundary_mask] = 0.0

            rhs_x = (RHS_x @ u_star).reshape(Nx, Ny, Nz)
            rhs_z = (RHS_z @ u_star.transpose(2, 0, 1).reshape(Nz, Nx * Ny)).reshape(Nz, Nx, Ny).transpose(1, 2, 0)
            rhs_2 = rhs_x + rhs_z - u_star + dt_third * (source_rhs + source_explicit)

            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(t_2)
                rhs_2[0, :, :]  -= dt_third * D * 2 * flux / dx
                rhs_2[-1, :, :] += dt_third * D * 2 * flux / dx
                rhs_2[:, :, 0]  -= dt_third * D * 2 * flux / dz
                rhs_2[:, :, -1] += dt_third * D * 2 * flux / dz

            u_star_star = np.zeros((Nx, Ny, Nz))
            for i in range(Nx):
                for k in range(Nz):
                    source_diag = diags([dt_third * source_lhs[i, :, k]], [0], shape=(Ny, Ny), format='csr')
                    LHS_y_ik = (LHS_y + source_diag).tolil()
                    rhs_2_ik = rhs_2[i, :, k].reshape(Ny, 1)
                    rhs_2_ik = self._apply_bc_to_sweep(LHS_y_ik, rhs_2_ik, dy, dt_third)
                    u_star_star[i, :, k] = spsolve(LHS_y_ik.tocsr(), rhs_2_ik).flatten()

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(t_2)
                u_star_star[0, :, :] = val; u_star_star[-1, :, :] = val
                u_star_star[:, 0, :] = val; u_star_star[:, -1, :] = val
                u_star_star[:, :, 0] = val; u_star_star[:, :, -1] = val

            # --- SWEEP 3: Implicit Z, Explicit X & Y (with implicit source) ---
            source_explicit = self._compute_source_term(state=u_star_star, implicit=True, t=t_3)
            source_rhs = np.zeros_like(self.state)
            source_lhs = np.zeros_like(self.state)
            if self._bulk is not None:
                source_rhs = self._bulk.rhs_contribution
                source_lhs = self._bulk.lhs_contribution
                source_lhs[self._boundary_mask] = 0.0

            rhs_x = (RHS_x @ u_star_star).reshape(Nx, Ny, Nz)
            rhs_y = (RHS_y @ u_star_star.transpose(1, 0, 2).reshape(Ny, Nx * Nz)).reshape(Ny, Nx, Nz).transpose(1, 0, 2)
            rhs_3 = rhs_x + rhs_y - u_star_star + dt_third * (source_rhs + source_explicit)

            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(t_3)
                rhs_3[0, :, :]  -= dt_third * D * 2 * flux / dx
                rhs_3[-1, :, :] += dt_third * D * 2 * flux / dx
                rhs_3[:, 0, :]  -= dt_third * D * 2 * flux / dy
                rhs_3[:, -1, :] += dt_third * D * 2 * flux / dy

            u_new = np.zeros((Nx, Ny, Nz))
            for i in range(Nx):
                for j in range(Ny):
                    source_diag = diags([dt_third * source_lhs[i, j, :]], [0], shape=(Nz, Nz), format='csr')
                    LHS_z_ij = (LHS_z + source_diag).tolil()
                    rhs_3_ij = rhs_3[i, j, :].reshape(Nz, 1)
                    rhs_3_ij = self._apply_bc_to_sweep(LHS_z_ij, rhs_3_ij, dz, dt_third)
                    u_new[i, j, :] = spsolve(LHS_z_ij.tocsr(), rhs_3_ij).flatten()

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(t_3)
                u_new[0, :, :] = val; u_new[-1, :, :] = val
                u_new[:, 0, :] = val; u_new[:, -1, :] = val
                u_new[:, :, 0] = val; u_new[:, :, -1] = val

            self.state = u_new

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
        return rhs_array

    def set_diffusion_coefficient(self, value: float) -> None:
        super().set_diffusion_coefficient(value)
        self._build_system_matrix()
    
    def set_decay_rate(self, value: float) -> None:
        super().set_decay_rate(value)
        self._build_system_matrix()