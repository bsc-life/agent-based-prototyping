"""
Alternating Direction Implicit (ADI) method with operator splitting.

This module implements ADI sweeps for diffusion/decay and splits bulk and
agent source terms into separate substeps.
"""

import numpy as np
from scipy.sparse import diags, eye, csr_matrix
from scipy.sparse.linalg import spsolve
from diffusion_schemas.base import Schema
from diffusion_schemas.utils.boundary import DirichletBC, NeumannBC


class ADIBCOSSchema(Schema):
    """
    ADI schema with operator splitting for source terms.

    2D uses Peaceman-Rachford (second-order in time for the diffusive part).
    1D/3D use fractional-step implicit Euler for diffusion/decay.
    """

    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0,
    ):
        super().__init__(domain_size, grid_points, dt, diffusion_coefficient, decay_rate)
        self._build_system_matrix()
        self._boundary_mask = self._compute_boundary_indices()

    def _compute_boundary_indices(self) -> np.ndarray:
        """Precompute boundary mask for the current grid shape."""
        mask = np.zeros(self.grid_points, dtype=bool)

        mask[0, ...] = True
        mask[-1, ...] = True

        if self.ndim >= 2:
            mask[:, 0, ...] = True
            mask[:, -1, ...] = True

        if self.ndim == 3:
            mask[:, :, 0] = True
            mask[:, :, -1] = True

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
        n = self.grid_points[0]
        dx = self.dx[0]

        diag_main = -2 * np.ones(n) / (dx**2)
        diag_off = np.ones(n - 1) / (dx**2)
        l = diags([diag_off, diag_main, diag_off], [-1, 0, 1], shape=(n, n), format="csr")
        i = eye(n, format="csr")

        return i - self.dt * self.diffusion_coefficient * l + self.dt * self.decay_rate * i

    def _build_matrix_2d(self):
        """Build 2D ADI operators for Peaceman-Rachford."""
        nx, ny = self.grid_points
        dx, dy = self.dx

        dt_half = self.dt / 2.0
        decay_term = (self.decay_rate / 2.0) * dt_half

        diag_main_x = -2 * np.ones(nx) / (dx**2)
        diag_off_x = np.ones(nx - 1) / (dx**2)
        lx = diags([diag_off_x, diag_main_x, diag_off_x], [-1, 0, 1], shape=(nx, nx), format="csr").tolil()
        ix = eye(nx, format="csr")

        if isinstance(self._boundary_conditions, NeumannBC):
            lx[0, 0], lx[0, 1] = -2 / (dx**2), 2 / (dx**2)
            lx[-1, -1], lx[-1, -2] = -2 / (dx**2), 2 / (dx**2)

        lx, ix = lx.tocsr(), ix.tocsr()
        lhs_x = (ix - dt_half * self.diffusion_coefficient * lx + decay_term * ix).tolil()
        rhs_x = (ix + dt_half * self.diffusion_coefficient * lx - decay_term * ix).tolil()

        if isinstance(self._boundary_conditions, DirichletBC):
            for row in [0, -1]:
                lhs_x[row, :] = 0
                lhs_x[row, row] = 1
                rhs_x[row, :] = 0
                rhs_x[row, row] = 1

        diag_main_y = -2 * np.ones(ny) / (dy**2)
        diag_off_y = np.ones(ny - 1) / (dy**2)
        ly = diags([diag_off_y, diag_main_y, diag_off_y], [-1, 0, 1], shape=(ny, ny), format="csr").tolil()
        iy = eye(ny, format="csr")

        if isinstance(self._boundary_conditions, NeumannBC):
            ly[0, 0], ly[0, 1] = -2 / (dy**2), 2 / (dy**2)
            ly[-1, -1], ly[-1, -2] = -2 / (dy**2), 2 / (dy**2)

        ly, iy = ly.tocsr(), iy.tocsr()
        lhs_y = (iy - dt_half * self.diffusion_coefficient * ly + decay_term * iy).tolil()
        rhs_y = (iy + dt_half * self.diffusion_coefficient * ly - decay_term * iy).tolil()

        if isinstance(self._boundary_conditions, DirichletBC):
            for row in [0, -1]:
                lhs_y[row, :] = 0
                lhs_y[row, row] = 1
                rhs_y[row, :] = 0
                rhs_y[row, row] = 1

        return lhs_x, rhs_x, lhs_y, rhs_y

    def _build_matrix_3d(self):
        """Build 3D ADI operators for fractional-step implicit Euler."""
        nx, ny, nz = self.grid_points
        dx, dy, dz = self.dx

        dt_third = self.dt / 3.0
        decay_term = (self.decay_rate / 3.0) * dt_third

        lx = diags([np.ones(nx - 1) / dx**2, -2 * np.ones(nx) / dx**2, np.ones(nx - 1) / dx**2], [-1, 0, 1], shape=(nx, nx), format="csr").tolil()
        ix = eye(nx, format="csr")
        if isinstance(self._boundary_conditions, NeumannBC):
            lx[0, 0], lx[0, 1] = -2 / (dx**2), 2 / (dx**2)
            lx[-1, -1], lx[-1, -2] = -2 / (dx**2), 2 / (dx**2)
        lx, ix = lx.tocsr(), ix.tocsr()
        lhs_x = (ix - dt_third * self.diffusion_coefficient * lx + decay_term * ix).tolil()
        rhs_x = (ix + dt_third * self.diffusion_coefficient * lx - decay_term * ix).tolil()

        ly = diags([np.ones(ny - 1) / dy**2, -2 * np.ones(ny) / dy**2, np.ones(ny - 1) / dy**2], [-1, 0, 1], shape=(ny, ny), format="csr").tolil()
        iy = eye(ny, format="csr")
        if isinstance(self._boundary_conditions, NeumannBC):
            ly[0, 0], ly[0, 1] = -2 / (dy**2), 2 / (dy**2)
            ly[-1, -1], ly[-1, -2] = -2 / (dy**2), 2 / (dy**2)
        ly, iy = ly.tocsr(), iy.tocsr()
        lhs_y = (iy - dt_third * self.diffusion_coefficient * ly + decay_term * iy).tolil()
        rhs_y = (iy + dt_third * self.diffusion_coefficient * ly - decay_term * iy).tolil()

        lz = diags([np.ones(nz - 1) / dz**2, -2 * np.ones(nz) / dz**2, np.ones(nz - 1) / dz**2], [-1, 0, 1], shape=(nz, nz), format="csr").tolil()
        iz = eye(nz, format="csr")
        if isinstance(self._boundary_conditions, NeumannBC):
            lz[0, 0], lz[0, 1] = -2 / (dz**2), 2 / (dz**2)
            lz[-1, -1], lz[-1, -2] = -2 / (dz**2), 2 / (dz**2)
        lz, iz = lz.tocsr(), iz.tocsr()
        lhs_z = (iz - dt_third * self.diffusion_coefficient * lz + decay_term * iz).tolil()
        rhs_z = (iz + dt_third * self.diffusion_coefficient * lz - decay_term * iz).tolil()

        if isinstance(self._boundary_conditions, DirichletBC):
            for op_lhs, op_rhs in [(lhs_x, rhs_x), (lhs_y, rhs_y), (lhs_z, rhs_z)]:
                for row in [0, -1]:
                    op_lhs[row, :] = 0
                    op_lhs[row, row] = 1
                    op_rhs[row, :] = 0
                    op_rhs[row, row] = 1

        return lhs_x, rhs_x, lhs_y, rhs_y, lhs_z, rhs_z

    def step(self) -> None:
        """Perform one time step using operator splitting."""
        self._step_diffusion_decay()

        t_next = self.t + self.dt
        self.agents_rhs_contribution = self._compute_source_term(implicit=True, t=t_next)

        if self._bulk is not None:
            self._step_bulk_sources()

        if self._agents:
            self._step_agent_sources()

        if isinstance(self._boundary_conditions, DirichletBC):
            value = self._boundary_conditions._get_value(t_next)
            self.state[self._boundary_mask] = value

        self.t += self.dt

    def _step_diffusion_decay(self) -> None:
        """Solve only diffusion/decay with ADI sweeps."""
        rhs = self.state.copy()

        if self.ndim == 1:
            n = self.grid_points[0]
            ax = self.system_matrix.copy().tolil()
            rhs_1d = rhs.reshape(n, 1)
            rhs_1d = self._apply_bc_to_sweep(ax, rhs_1d, self.dx[0], self.dt, t_eval=self.t + self.dt)
            self.state = spsolve(ax.tocsr(), rhs_1d).reshape(self.grid_points)

        elif self.ndim == 2:
            dt_half = self.dt / 2.0
            t_mid = self.t + dt_half

            d = self.diffusion_coefficient
            dx, dy = self.dx
            nx, ny = self.grid_points

            lhs_x, rhs_x_op, lhs_y, rhs_y_op = self.system_matrix

            rhs_1 = (rhs_y_op @ rhs.T).T
            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(t_mid)
                explicit_y_forcing = dt_half * d * 2 * flux / dy
                rhs_1[:, 0] -= explicit_y_forcing
                rhs_1[:, -1] += explicit_y_forcing

            lhs_x_eff = lhs_x.copy().tolil()
            rhs_1 = self._apply_bc_to_sweep(lhs_x_eff, rhs_1, dx, dt_half, t_eval=t_mid)
            u_star = spsolve(lhs_x_eff.tocsr(), rhs_1)

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(t_mid)
                u_star[0, :] = val
                u_star[-1, :] = val
                u_star[:, 0] = val
                u_star[:, -1] = val

            rhs_2 = rhs_x_op @ u_star
            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(self.t + self.dt)
                explicit_x_forcing = dt_half * d * 2 * flux / dx
                rhs_2[0, :] -= explicit_x_forcing
                rhs_2[-1, :] += explicit_x_forcing

            lhs_y_eff = lhs_y.copy().tolil()
            rhs_2_t = rhs_2.T
            rhs_2_t = self._apply_bc_to_sweep(lhs_y_eff, rhs_2_t, dy, dt_half, t_eval=self.t + self.dt)
            u_new = spsolve(lhs_y_eff.tocsr(), rhs_2_t).T

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(self.t + self.dt)
                u_new[0, :] = val
                u_new[-1, :] = val
                u_new[:, 0] = val
                u_new[:, -1] = val

            self.state = u_new

        elif self.ndim == 3:
            dt_third = self.dt / 3.0
            t_1 = self.t + dt_third
            t_2 = self.t + 2 * dt_third
            t_3 = self.t + self.dt

            d = self.diffusion_coefficient
            dx, dy, dz = self.dx
            nx, ny, nz = self.grid_points

            lhs_x, rhs_x_op, lhs_y, rhs_y_op, lhs_z, rhs_z_op = self.system_matrix

            rhs_y = (rhs_y_op @ rhs.transpose(1, 0, 2).reshape(ny, nx * nz)).reshape(ny, nx, nz).transpose(1, 0, 2)
            rhs_z = (rhs_z_op @ rhs.transpose(2, 0, 1).reshape(nz, nx * ny)).reshape(nz, nx, ny).transpose(1, 2, 0)
            rhs_1 = rhs_y + rhs_z - rhs

            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(t_1)
                rhs_1[:, 0, :] -= dt_third * d * 2 * flux / dy
                rhs_1[:, -1, :] += dt_third * d * 2 * flux / dy
                rhs_1[:, :, 0] -= dt_third * d * 2 * flux / dz
                rhs_1[:, :, -1] += dt_third * d * 2 * flux / dz

            lhs_x_eff = lhs_x.copy().tolil()
            rhs_1_x = rhs_1.reshape(nx, ny * nz)
            rhs_1_x = self._apply_bc_to_sweep(lhs_x_eff, rhs_1_x, dx, dt_third, t_eval=t_1)
            u_star = spsolve(lhs_x_eff.tocsr(), rhs_1_x).reshape(nx, ny, nz)

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(t_1)
                u_star[0, :, :] = val
                u_star[-1, :, :] = val
                u_star[:, 0, :] = val
                u_star[:, -1, :] = val
                u_star[:, :, 0] = val
                u_star[:, :, -1] = val

            rhs_x = (rhs_x_op @ u_star).reshape(nx, ny, nz)
            rhs_z = (rhs_z_op @ u_star.transpose(2, 0, 1).reshape(nz, nx * ny)).reshape(nz, nx, ny).transpose(1, 2, 0)
            rhs_2 = rhs_x + rhs_z - u_star

            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(t_2)
                rhs_2[0, :, :] -= dt_third * d * 2 * flux / dx
                rhs_2[-1, :, :] += dt_third * d * 2 * flux / dx
                rhs_2[:, :, 0] -= dt_third * d * 2 * flux / dz
                rhs_2[:, :, -1] += dt_third * d * 2 * flux / dz

            lhs_y_eff = lhs_y.copy().tolil()
            rhs_2_y = rhs_2.transpose(1, 0, 2).reshape(ny, nx * nz)
            rhs_2_y = self._apply_bc_to_sweep(lhs_y_eff, rhs_2_y, dy, dt_third, t_eval=t_2)
            u_star_star = spsolve(lhs_y_eff.tocsr(), rhs_2_y).reshape(ny, nx, nz).transpose(1, 0, 2)

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(t_2)
                u_star_star[0, :, :] = val
                u_star_star[-1, :, :] = val
                u_star_star[:, 0, :] = val
                u_star_star[:, -1, :] = val
                u_star_star[:, :, 0] = val
                u_star_star[:, :, -1] = val

            rhs_x = (rhs_x_op @ u_star_star).reshape(nx, ny, nz)
            rhs_y = (rhs_y_op @ u_star_star.transpose(1, 0, 2).reshape(ny, nx * nz)).reshape(ny, nx, nz).transpose(1, 0, 2)
            rhs_3 = rhs_x + rhs_y - u_star_star

            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(t_3)
                rhs_3[0, :, :] -= dt_third * d * 2 * flux / dx
                rhs_3[-1, :, :] += dt_third * d * 2 * flux / dx
                rhs_3[:, 0, :] -= dt_third * d * 2 * flux / dy
                rhs_3[:, -1, :] += dt_third * d * 2 * flux / dy

            lhs_z_eff = lhs_z.copy().tolil()
            rhs_3_z = rhs_3.transpose(2, 0, 1).reshape(nz, nx * ny)
            rhs_3_z = self._apply_bc_to_sweep(lhs_z_eff, rhs_3_z, dz, dt_third, t_eval=t_3)
            u_new = spsolve(lhs_z_eff.tocsr(), rhs_3_z).reshape(nz, nx, ny).transpose(1, 2, 0)

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(t_3)
                u_new[0, :, :] = val
                u_new[-1, :, :] = val
                u_new[:, 0, :] = val
                u_new[:, -1, :] = val
                u_new[:, :, 0] = val
                u_new[:, :, -1] = val

            self.state = u_new

        else:
            raise NotImplementedError(f"{self.ndim}D ADI is not implemented yet")

    def _step_bulk_sources(self) -> None:
        """Solve bulk source split step: (sigma* - sigma)/dt = S_rhs - S_lhs*sigma*."""
        s_rhs = self._bulk.rhs_contribution.copy()
        s_lhs = self._bulk.lhs_contribution.copy()
        self.state = (self.state + self.dt * s_rhs) / (1.0 + self.dt * s_lhs)

    def _step_agent_sources(self) -> None:
        """Solve agent split step with explicit additive update."""
        self.state += self.dt * self.agents_rhs_contribution

    def _apply_bc_to_sweep(self, matrix, rhs_array: np.ndarray, h: float, dt_sweep: float, t_eval: float = None) -> np.ndarray:
        """Apply BCs to one implicit 1D sweep system."""
        if self._boundary_conditions is None:
            return rhs_array

        d = self.diffusion_coefficient
        bc_time = self.t + dt_sweep if t_eval is None else t_eval

        if isinstance(self._boundary_conditions, NeumannBC):
            flux = self._boundary_conditions._get_flux(bc_time)
            alpha = (dt_sweep * d) / (h**2)
            forcing = (2 * dt_sweep * d * flux) / h

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
        self._build_system_matrix()

    def set_diffusion_coefficient(self, value: float) -> None:
        super().set_diffusion_coefficient(value)
        self._build_system_matrix()

    def set_decay_rate(self, value: float) -> None:
        super().set_decay_rate(value)
        self._build_system_matrix()
