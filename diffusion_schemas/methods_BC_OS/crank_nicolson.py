"""
Crank-Nicolson method with operator splitting.

This module implements Crank-Nicolson for diffusion/decay and applies
bulk/agent sources in split substeps.
"""

import numpy as np
from scipy.sparse import diags, kron, eye, csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from diffusion_schemas.base import Schema
from diffusion_schemas.utils.boundary import DirichletBC, NeumannBC


class CrankNicolsonBCOSSchema(Schema):
    """
    Crank-Nicolson method for diffusion/decay with operator-split sources.

    Diffusion/decay is solved with the theta-method (default theta=0.5), then
    bulk and agent sources are applied in separate substeps.
    """

    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0,
        theta=0.5,
    ):
        super().__init__(domain_size, grid_points, dt, diffusion_coefficient, decay_rate)

        if not 0 <= theta <= 1:
            raise ValueError("theta must be in [0, 1]")
        self.theta = theta

        self._build_system_matrices()
        self._boundary_mask = self._compute_boundary_indices()

    def _compute_boundary_indices(self) -> np.ndarray:
        """Precompute boundary mask for current grid shape."""
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

    def _build_system_matrices(self) -> None:
        if self.ndim == 1:
            self.A_impl, self.A_expl = self._build_matrices_1d()
        elif self.ndim == 2:
            self.A_impl, self.A_expl = self._build_matrices_2d()
        elif self.ndim == 3:
            self.A_impl, self.A_expl = self._build_matrices_3d()
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.ndim}")

    def _build_matrices_1d(self):
        n = self.grid_points[0]
        dx = self.dx[0]

        diag_main = -2 * np.ones(n) / (dx**2)
        diag_off = np.ones(n - 1) / (dx**2)

        l = diags([diag_off, diag_main, diag_off], [-1, 0, 1], shape=(n, n), format="csr")
        i = eye(n, format="csr")

        a_impl = i - self.theta * self.dt * self.diffusion_coefficient * l + self.theta * self.dt * self.decay_rate * i
        a_expl = i + (1 - self.theta) * self.dt * self.diffusion_coefficient * l - (1 - self.theta) * self.dt * self.decay_rate * i

        return a_impl, a_expl

    def _build_matrices_2d(self):
        nx, ny = self.grid_points
        dx, dy = self.dx

        diag_main_x = -2 * np.ones(nx) / (dx**2)
        diag_off_x = np.ones(nx - 1) / (dx**2)
        lx = diags([diag_off_x, diag_main_x, diag_off_x], [-1, 0, 1], shape=(nx, nx), format="csr")

        diag_main_y = -2 * np.ones(ny) / (dy**2)
        diag_off_y = np.ones(ny - 1) / (dy**2)
        ly = diags([diag_off_y, diag_main_y, diag_off_y], [-1, 0, 1], shape=(ny, ny), format="csr")

        ix = eye(nx, format="csr")
        iy = eye(ny, format="csr")

        l = kron(lx, iy) + kron(ix, ly)
        i = eye(nx * ny, format="csr")

        a_impl = i - self.theta * self.dt * self.diffusion_coefficient * l + self.theta * self.dt * self.decay_rate * i
        a_expl = i + (1 - self.theta) * self.dt * self.diffusion_coefficient * l - (1 - self.theta) * self.dt * self.decay_rate * i

        return a_impl, a_expl

    def _build_matrices_3d(self):
        nx, ny, nz = self.grid_points
        dx, dy, dz = self.dx

        diag_main_x = -2 * np.ones(nx) / (dx**2)
        diag_off_x = np.ones(nx - 1) / (dx**2)
        lx = diags([diag_off_x, diag_main_x, diag_off_x], [-1, 0, 1], shape=(nx, nx), format="csr")

        diag_main_y = -2 * np.ones(ny) / (dy**2)
        diag_off_y = np.ones(ny - 1) / (dy**2)
        ly = diags([diag_off_y, diag_main_y, diag_off_y], [-1, 0, 1], shape=(ny, ny), format="csr")

        diag_main_z = -2 * np.ones(nz) / (dz**2)
        diag_off_z = np.ones(nz - 1) / (dz**2)
        lz = diags([diag_off_z, diag_main_z, diag_off_z], [-1, 0, 1], shape=(nz, nz), format="csr")

        ix = eye(nx, format="csr")
        iy = eye(ny, format="csr")
        iz = eye(nz, format="csr")

        l = kron(kron(lx, iy), iz) + kron(kron(ix, ly), iz) + kron(kron(ix, iy), lz)
        i = eye(nx * ny * nz, format="csr")

        a_impl = i - self.theta * self.dt * self.diffusion_coefficient * l + self.theta * self.dt * self.decay_rate * i
        a_expl = i + (1 - self.theta) * self.dt * self.diffusion_coefficient * l - (1 - self.theta) * self.dt * self.decay_rate * i

        return a_impl, a_expl

    def step(self) -> None:
        """Perform one time step with operator splitting."""
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
        """Solve Crank-Nicolson diffusion/decay step without source terms."""
        laplacian_n = self._compute_laplacian(self.state)
        rhs = self.state + (1 - self.theta) * self.dt * (
            self.diffusion_coefficient * laplacian_n - self.decay_rate * self.state
        )
        rhs = rhs.flatten()

        if self._boundary_conditions is not None:
            if isinstance(self._boundary_conditions, DirichletBC):
                rhs = self._apply_dirichlet_bc(rhs)
            elif isinstance(self._boundary_conditions, NeumannBC):
                rhs = self._apply_neumann_bc(rhs)

        u_new_flat = spsolve(self.A_impl, rhs)
        self.state = u_new_flat.reshape(self.grid_points)

    def _step_bulk_sources(self) -> None:
        """Solve bulk split step: (sigma* - sigma)/dt = S_rhs - S_lhs*sigma*."""
        s_rhs = self._bulk.rhs_contribution.copy()
        s_lhs = self._bulk.lhs_contribution.copy()
        self.state = (self.state + self.dt * s_rhs) / (1.0 + self.dt * s_lhs)

    def _step_agent_sources(self) -> None:
        """Solve agent split step with explicit additive update."""
        self.state += self.dt * self.agents_rhs_contribution

    def _apply_neumann_bc(self, rhs):
        """Apply Neumann BC correction to RHS vector."""
        flux = self._boundary_conditions._get_flux(self.t + self.dt)
        d = self.diffusion_coefficient
        dt = self.dt
        theta = self.theta

        def get_forcing(h):
            return (2 * theta * dt * d * flux) / h

        if self.ndim == 1:
            dx = self.dx[0]
            forcing = get_forcing(dx)
            rhs[0] -= forcing
            rhs[-1] += forcing

        elif self.ndim == 2:
            nx, ny = self.grid_points
            dx, dy = self.dx
            fx, fy = get_forcing(dx), get_forcing(dy)

            idx_l = np.arange(ny)
            idx_r = np.arange((nx - 1) * ny, nx * ny)
            rhs[idx_l] -= fx
            rhs[idx_r] += fx

            idx_b = np.arange(0, nx * ny, ny)
            idx_t = np.arange(ny - 1, nx * ny, ny)
            rhs[idx_b] -= fy
            rhs[idx_t] += fy

        elif self.ndim == 3:
            nx, ny, nz = self.grid_points
            dx, dy, dz = self.dx
            sx, sy = ny * nz, nz
            fx, fy, fz = get_forcing(dx), get_forcing(dy), get_forcing(dz)

            idx_l, idx_r = np.arange(sx), np.arange((nx - 1) * sx, nx * sx)
            rhs[idx_l] -= fx
            rhs[idx_r] += fx

            base_y = np.arange(nz)
            idx_f = np.concatenate([base_y + i * sx for i in range(nx)])
            idx_bk = idx_f + (ny - 1) * sy
            rhs[idx_f] -= fy
            rhs[idx_bk] += fy

            idx_bt = np.arange(0, nx * ny * nz, nz)
            idx_tp = np.arange(nz - 1, nx * ny * nz, nz)
            rhs[idx_bt] -= fz
            rhs[idx_tp] += fz

        return rhs

    def _apply_dirichlet_bc(self, rhs):
        """Apply Dirichlet BC values directly to RHS vector."""
        value = self._boundary_conditions._get_value(self.t + self.dt)

        if self.ndim == 1:
            rhs[0] = value
            rhs[-1] = value

        elif self.ndim == 2:
            nx, ny = self.grid_points
            for j in range(ny):
                idx = j * nx
                rhs[idx] = value
                idx2 = j * nx + nx - 1
                rhs[idx2] = value
            for i in range(nx):
                idx = i
                rhs[idx] = value
                idx2 = (ny - 1) * nx + i
                rhs[idx2] = value

        elif self.ndim == 3:
            nx, ny, nz = self.grid_points
            for j in range(ny):
                for k in range(nz):
                    idx = 0 * ny * nz + j * nz + k
                    rhs[idx] = value
                    idx = (nx - 1) * ny * nz + j * nz + k
                    rhs[idx] = value
            for i in range(nx):
                for k in range(nz):
                    idx = i * ny * nz + 0 * nz + k
                    rhs[idx] = value
                    idx = i * ny * nz + (ny - 1) * nz + k
                    rhs[idx] = value
            for i in range(nx):
                for j in range(ny):
                    idx = i * ny * nz + j * nz + 0
                    rhs[idx] = value
                    idx = i * ny * nz + j * nz + (nz - 1)
                    rhs[idx] = value

        return rhs

    def _compute_laplacian(self, u: np.ndarray) -> np.ndarray:
        if self.ndim == 1:
            return self._laplacian_1d(u)
        elif self.ndim == 2:
            return self._laplacian_2d(u)
        elif self.ndim == 3:
            return self._laplacian_3d(u)
        else:
            raise ValueError(f"Unsupported dimensions: {self.ndim}")

    def _laplacian_1d(self, u: np.ndarray) -> np.ndarray:
        laplacian = np.zeros_like(u)
        dx = self.dx[0]
        laplacian[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx**2)

        is_neumann = isinstance(self._boundary_conditions, NeumannBC)
        is_dirichlet = isinstance(self._boundary_conditions, DirichletBC)

        if is_neumann:
            g = self._boundary_conditions._get_flux(self.t)
            laplacian[0] = 2 * (u[1] - u[0] - g * dx) / (dx**2)
            laplacian[-1] = 2 * (u[-2] - u[-1] + g * dx) / (dx**2)
        elif is_dirichlet:
            laplacian[0] = 0
            laplacian[-1] = 0
        else:
            laplacian[0] = (u[1] - 2 * u[0] + u[1]) / (dx**2)
            laplacian[-1] = (u[-2] - 2 * u[-1] + u[-2]) / (dx**2)
        return laplacian

    def _laplacian_2d(self, u: np.ndarray) -> np.ndarray:
        laplacian = np.zeros_like(u)
        dx, dy = self.dx

        laplacian[1:-1, 1:-1] = (
            (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx**2)
            + (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy**2)
        )

        is_neumann = isinstance(self._boundary_conditions, NeumannBC)
        is_dirichlet = isinstance(self._boundary_conditions, DirichletBC)

        if is_neumann:
            flux = self._boundary_conditions._get_flux(self.t)
            laplacian[0, 1:-1] = (2 * (u[1, 1:-1] - u[0, 1:-1]) / dx**2 - 2 * flux / dx) + (u[0, 2:] - 2 * u[0, 1:-1] + u[0, :-2]) / dy**2
            laplacian[-1, 1:-1] = (2 * (u[-2, 1:-1] - u[-1, 1:-1]) / dx**2 + 2 * flux / dx) + (u[-1, 2:] - 2 * u[-1, 1:-1] + u[-1, :-2]) / dy**2
            laplacian[1:-1, 0] = (u[2:, 0] - 2 * u[1:-1, 0] + u[:-2, 0]) / dx**2 + (2 * (u[1:-1, 1] - u[1:-1, 0]) / dy**2 - 2 * flux / dy)
            laplacian[1:-1, -1] = (u[2:, -1] - 2 * u[1:-1, -1] + u[:-2, -1]) / dx**2 + (2 * (u[1:-1, -2] - u[1:-1, -1]) / dy**2 + 2 * flux / dy)
            laplacian[0, 0] = (2 * (u[1, 0] - u[0, 0]) / dx**2 - 2 * flux / dx) + (2 * (u[0, 1] - u[0, 0]) / dy**2 - 2 * flux / dy)
            laplacian[0, -1] = (2 * (u[1, -1] - u[0, -1]) / dx**2 - 2 * flux / dx) + (2 * (u[0, -2] - u[0, -1]) / dy**2 + 2 * flux / dy)
            laplacian[-1, 0] = (2 * (u[-2, 0] - u[-1, 0]) / dx**2 + 2 * flux / dx) + (2 * (u[-1, 1] - u[-1, 0]) / dy**2 - 2 * flux / dy)
            laplacian[-1, -1] = (2 * (u[-2, -1] - u[-1, -1]) / dx**2 + 2 * flux / dx) + (2 * (u[-1, -2] - u[-1, -1]) / dy**2 + 2 * flux / dy)
        elif is_dirichlet:
            laplacian[0, :] = 0
            laplacian[-1, :] = 0
            laplacian[:, 0] = 0
            laplacian[:, -1] = 0
        else:
            laplacian[0, 1:-1] = (2 * (u[1, 1:-1] - u[0, 1:-1]) / dx**2) + (u[0, 2:] - 2 * u[0, 1:-1] + u[0, :-2]) / dy**2
            laplacian[-1, 1:-1] = (2 * (u[-2, 1:-1] - u[-1, 1:-1]) / dx**2) + (u[-1, 2:] - 2 * u[-1, 1:-1] + u[-1, :-2]) / dy**2
            laplacian[1:-1, 0] = (u[2:, 0] - 2 * u[1:-1, 0] + u[:-2, 0]) / dx**2 + (2 * (u[1:-1, 1] - u[1:-1, 0]) / dy**2)
            laplacian[1:-1, -1] = (u[2:, -1] - 2 * u[1:-1, -1] + u[:-2, -1]) / dx**2 + (2 * (u[1:-1, -2] - u[1:-1, -1]) / dy**2)
            laplacian[0, 0] = 2 * (u[1, 0] - u[0, 0]) / dx**2 + 2 * (u[0, 1] - u[0, 0]) / dy**2
            laplacian[0, -1] = 2 * (u[1, -1] - u[0, -1]) / dx**2 + 2 * (u[0, -2] - u[0, -1]) / dy**2
            laplacian[-1, 0] = 2 * (u[-2, 0] - u[-1, 0]) / dx**2 + 2 * (u[-1, 1] - u[-1, 0]) / dy**2
            laplacian[-1, -1] = 2 * (u[-2, -1] - u[-1, -1]) / dx**2 + 2 * (u[-1, -2] - u[-1, -1]) / dy**2

        return laplacian

    def _laplacian_3d(self, u: np.ndarray) -> np.ndarray:
        laplacian = np.zeros_like(u)
        dx, dy, dz = self.dx

        laplacian[1:-1, 1:-1, 1:-1] = (
            (u[2:, 1:-1, 1:-1] - 2 * u[1:-1, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]) / (dx**2)
            + (u[1:-1, 2:, 1:-1] - 2 * u[1:-1, 1:-1, 1:-1] + u[1:-1, :-2, 1:-1]) / (dy**2)
            + (u[1:-1, 1:-1, 2:] - 2 * u[1:-1, 1:-1, 1:-1] + u[1:-1, 1:-1, :-2]) / (dz**2)
        )

        is_neumann = isinstance(self._boundary_conditions, NeumannBC)
        is_dirichlet = isinstance(self._boundary_conditions, DirichletBC)

        if is_neumann:
            flux = self._boundary_conditions._get_flux(self.t)
            laplacian[0, 1:-1, 1:-1] = (
                (2 * (u[1, 1:-1, 1:-1] - u[0, 1:-1, 1:-1]) / dx**2 - 2 * flux / dx)
                + (u[0, 2:, 1:-1] - 2 * u[0, 1:-1, 1:-1] + u[0, :-2, 1:-1]) / dy**2
                + (u[0, 1:-1, 2:] - 2 * u[0, 1:-1, 1:-1] + u[0, 1:-1, :-2]) / dz**2
            )
            laplacian[-1, 1:-1, 1:-1] = (
                (2 * (u[-2, 1:-1, 1:-1] - u[-1, 1:-1, 1:-1]) / dx**2 + 2 * flux / dx)
                + (u[-1, 2:, 1:-1] - 2 * u[-1, 1:-1, 1:-1] + u[-1, :-2, 1:-1]) / dy**2
                + (u[-1, 1:-1, 2:] - 2 * u[-1, 1:-1, 1:-1] + u[-1, 1:-1, :-2]) / dz**2
            )
            laplacian[1:-1, 0, 1:-1] = (
                (u[2:, 0, 1:-1] - 2 * u[1:-1, 0, 1:-1] + u[:-2, 0, 1:-1]) / dx**2
                + (2 * (u[1:-1, 1, 1:-1] - u[1:-1, 0, 1:-1]) / dy**2 - 2 * flux / dy)
                + (u[1:-1, 0, 2:] - 2 * u[1:-1, 0, 1:-1] + u[1:-1, 0, :-2]) / dz**2
            )
            laplacian[1:-1, -1, 1:-1] = (
                (u[2:, -1, 1:-1] - 2 * u[1:-1, -1, 1:-1] + u[:-2, -1, 1:-1]) / dx**2
                + (2 * (u[1:-1, -2, 1:-1] - u[1:-1, -1, 1:-1]) / dy**2 + 2 * flux / dy)
                + (u[1:-1, -1, 2:] - 2 * u[1:-1, -1, 1:-1] + u[1:-1, -1, :-2]) / dz**2
            )
            laplacian[1:-1, 1:-1, 0] = (
                (u[2:, 1:-1, 0] - 2 * u[1:-1, 1:-1, 0] + u[:-2, 1:-1, 0]) / dx**2
                + (u[1:-1, 2:, 0] - 2 * u[1:-1, 1:-1, 0] + u[1:-1, :-2, 0]) / dy**2
                + (2 * (u[1:-1, 1:-1, 1] - u[1:-1, 1:-1, 0]) / dz**2 - 2 * flux / dz)
            )
            laplacian[1:-1, 1:-1, -1] = (
                (u[2:, 1:-1, -1] - 2 * u[1:-1, 1:-1, -1] + u[:-2, 1:-1, -1]) / dx**2
                + (u[1:-1, 2:, -1] - 2 * u[1:-1, 1:-1, -1] + u[1:-1, :-2, -1]) / dy**2
                + (2 * (u[1:-1, 1:-1, -2] - u[1:-1, 1:-1, -1]) / dz**2 + 2 * flux / dz)
            )
        elif is_dirichlet:
            laplacian[0, :, :] = 0
            laplacian[-1, :, :] = 0
            laplacian[:, 0, :] = 0
            laplacian[:, -1, :] = 0
            laplacian[:, :, 0] = 0
            laplacian[:, :, -1] = 0

        return laplacian

    def set_diffusion_coefficient(self, value: float) -> None:
        super().set_diffusion_coefficient(value)
        self._build_system_matrices()

    def set_decay_rate(self, value: float) -> None:
        super().set_decay_rate(value)
        self._build_system_matrices()

    def set_boundary_conditions(self, boundary_conditions) -> None:
        super().set_boundary_conditions(boundary_conditions)

        if not isinstance(boundary_conditions, (DirichletBC, NeumannBC)):
            raise ValueError("Boundary conditions must be either DirichletBC or NeumannBC.")

        a = self.A_impl.copy().tolil()

        if isinstance(boundary_conditions, DirichletBC):
            if self.ndim == 1:
                a[0, :] = 0
                a[0, 0] = 1
                a[-1, :] = 0
                a[-1, -1] = 1

            elif self.ndim == 2:
                nx, ny = self.grid_points
                for j in range(ny):
                    idx = j * nx
                    a[idx, :] = 0
                    a[idx, idx] = 1
                    idx2 = j * nx + nx - 1
                    a[idx2, :] = 0
                    a[idx2, idx2] = 1
                for i in range(nx):
                    idx = i
                    a[idx, :] = 0
                    a[idx, idx] = 1
                    idx2 = (ny - 1) * nx + i
                    a[idx2, :] = 0
                    a[idx2, idx2] = 1

            elif self.ndim == 3:
                nx, ny, nz = self.grid_points
                for j in range(ny):
                    for k in range(nz):
                        idx = 0 * ny * nz + j * nz + k
                        a[idx, :] = 0
                        a[idx, idx] = 1
                        idx = (nx - 1) * ny * nz + j * nz + k
                        a[idx, :] = 0
                        a[idx, idx] = 1
                for i in range(nx):
                    for k in range(nz):
                        idx = i * ny * nz + 0 * nz + k
                        a[idx, :] = 0
                        a[idx, idx] = 1
                        idx = i * ny * nz + (ny - 1) * nz + k
                        a[idx, :] = 0
                        a[idx, idx] = 1
                for i in range(nx):
                    for j in range(ny):
                        idx = i * ny * nz + j * nz + 0
                        a[idx, :] = 0
                        a[idx, idx] = 1
                        idx = i * ny * nz + j * nz + (nz - 1)
                        a[idx, :] = 0
                        a[idx, idx] = 1

        elif isinstance(boundary_conditions, NeumannBC):
            d = self.diffusion_coefficient
            dt = self.dt
            theta = self.theta

            def get_alpha(h):
                return (theta * dt * d) / (h**2)

            if self.ndim == 1:
                dx = self.dx[0]
                alpha = get_alpha(dx)
                a[0, 1] = -2 * alpha
                a[-1, -2] = -2 * alpha

            elif self.ndim == 2:
                nx, ny = self.grid_points
                dx, dy = self.dx
                alpha_x, alpha_y = get_alpha(dx), get_alpha(dy)

                idx_l = np.arange(ny)
                idx_r = np.arange((nx - 1) * ny, nx * ny)
                a[idx_l, idx_l + ny] = -2 * alpha_x
                a[idx_r, idx_r - ny] = -2 * alpha_x

                idx_b = np.arange(0, nx * ny, ny)
                idx_t = np.arange(ny - 1, nx * ny, ny)
                a[idx_b, idx_b + 1] = -2 * alpha_y
                a[idx_t, idx_t - 1] = -2 * alpha_y

            elif self.ndim == 3:
                nx, ny, nz = self.grid_points
                dx, dy, dz = self.dx
                sx, sy = ny * nz, nz
                alpha_x, alpha_y, alpha_z = get_alpha(dx), get_alpha(dy), get_alpha(dz)

                idx_l, idx_r = np.arange(sx), np.arange((nx - 1) * sx, nx * sx)
                a[idx_l, idx_l + sx] = -2 * alpha_x
                a[idx_r, idx_r - sx] = -2 * alpha_x

                base_y = np.arange(nz)
                idx_f = np.concatenate([base_y + i * sx for i in range(nx)])
                idx_bk = idx_f + (ny - 1) * sy
                a[idx_f, idx_f + sy] = -2 * alpha_y
                a[idx_bk, idx_bk - sy] = -2 * alpha_y

                idx_bt = np.arange(0, nx * ny * nz, nz)
                idx_tp = np.arange(nz - 1, nx * ny * nz, nz)
                a[idx_bt, idx_bt + 1] = -2 * alpha_z
                a[idx_tp, idx_tp - 1] = -2 * alpha_z

        self.A_impl = a.tocsr()
