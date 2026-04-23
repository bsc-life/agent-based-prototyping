"""
Crank-Nicolson method for diffusion equation using LOD splitting.

This module implements the Crank-Nicolson finite difference scheme with
Alternating Direction Implicit splitting and operator-split source updates.
"""

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from diffusion_schemas.base import Schema
from diffusion_schemas.utils.boundary import DirichletBC, NeumannBC


class CrankNicolsonLODBCOSSchema(Schema):
    """
    Crank-Nicolson method for the diffusion equation using LOD and OS sources.

    The diffusion/decay operator is advanced via CN-LOD sweeps while bulk and
    agent contributions are applied in separate operator-splitting substeps.
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
        """Precompute boundary indices mask for the current grid shape."""
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
        """Build sparse 1D implicit operators used by each LOD sweep."""
        if self.ndim == 1:
            self.A_impl_x = self._build_matrices_1d()
        elif self.ndim == 2:
            self.A_impl_x, self.A_impl_y = self._build_matrices_2d()
        elif self.ndim == 3:
            self.A_impl_x, self.A_impl_y, self.A_impl_z = self._build_matrices_3d()
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.ndim}")

    def _build_matrices_1d(self):
        """Build 1D implicit sweep matrix."""
        n = self.grid_points[0]
        dx = self.dx[0]

        diag_main = -2 * np.ones(n) / (dx**2)
        diag_off = np.ones(n - 1) / (dx**2)

        l = diags([diag_off, diag_main, diag_off], [-1, 0, 1], shape=(n, n), format="csr")
        i = eye(n, format="csr")

        a_impl = i - self.theta * self.dt * self.diffusion_coefficient * l + self.theta * self.dt * self.decay_rate * i
        return a_impl

    def _build_matrices_2d(self):
        """Build 2D implicit sweep matrices (x and y)."""
        nx, ny = self.grid_points
        dx, dy = self.dx
        factor = 1 / 2

        diag_main_x = -2 * np.ones(nx) / (dx**2)
        diag_off_x = np.ones(nx - 1) / (dx**2)
        lx = diags([diag_off_x, diag_main_x, diag_off_x], [-1, 0, 1], shape=(nx, nx), format="csr")

        diag_main_y = -2 * np.ones(ny) / (dy**2)
        diag_off_y = np.ones(ny - 1) / (dy**2)
        ly = diags([diag_off_y, diag_main_y, diag_off_y], [-1, 0, 1], shape=(ny, ny), format="csr")

        ix = eye(nx, format="csr")
        iy = eye(ny, format="csr")

        a_impl_x = ix - self.theta * self.dt * self.diffusion_coefficient * lx + self.theta * factor * self.dt * self.decay_rate * ix
        a_impl_y = iy - self.theta * self.dt * self.diffusion_coefficient * ly + self.theta * factor * self.dt * self.decay_rate * iy
        return a_impl_x, a_impl_y

    def _build_matrices_3d(self):
        """Build 3D implicit sweep matrices (x, y, z)."""
        nx, ny, nz = self.grid_points
        dx, dy, dz = self.dx
        factor = 1 / 3

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

        a_impl_x = ix - self.theta * self.dt * self.diffusion_coefficient * lx + self.theta * factor * self.dt * self.decay_rate * ix
        a_impl_y = iy - self.theta * self.dt * self.diffusion_coefficient * ly + self.theta * factor * self.dt * self.decay_rate * iy
        a_impl_z = iz - self.theta * self.dt * self.diffusion_coefficient * lz + self.theta * factor * self.dt * self.decay_rate * iz
        return a_impl_x, a_impl_y, a_impl_z

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
        """Solve diffusion/decay CN-LOD step without source contributions."""
        laplacian_n = self._compute_laplacian(self.state)
        rhs_grid = self.state + (1 - self.theta) * self.dt * (
            self.diffusion_coefficient * laplacian_n - self.decay_rate * self.state
        )

        # In OS mode, bulk LHS is handled only in _step_bulk_sources.
        self.state = self.step_lod(rhs_grid)

    def _step_bulk_sources(self) -> None:
        """Solve bulk split step: (sigma* - sigma)/dt = S_rhs - S_lhs*sigma*."""
        s_rhs = self._bulk.rhs_contribution.copy()
        s_lhs = self._bulk.lhs_contribution.copy()
        self.state = (self.state + self.dt * s_rhs) / (1.0 + self.dt * s_lhs)

    def _step_agent_sources(self) -> None:
        """Solve agent split step with explicit additive update."""
        self.state += self.dt * self.agents_rhs_contribution

    def step_lod(self, rhs):
        """Solve one CN-LOD implicit stage using batched directional sweeps."""
        if self.ndim == 1:
            ax = self.A_impl_x.copy().tolil()
            nx = self.grid_points[0]
            rhs_x = rhs.reshape(nx, 1)

            rhs_x = self._apply_bc_to_sweep(ax, rhs_x, self.dx[0])
            self.state = spsolve(ax.tocsr(), rhs_x).reshape(self.grid_points)

        elif self.ndim == 2:
            ax = self.A_impl_x.copy().tolil()
            ay = self.A_impl_y.copy().tolil()
            nx, ny = self.grid_points
            rhs = rhs.reshape(nx, ny)

            rhs_x = rhs
            rhs_x = self._apply_bc_to_sweep(ax, rhs_x, self.dx[0])
            u_star = spsolve(ax.tocsr(), rhs_x)

            rhs_y = u_star.T
            rhs_y = self._apply_bc_to_sweep(ay, rhs_y, self.dx[1])
            u_new_t = spsolve(ay.tocsr(), rhs_y)

            self.state = u_new_t.T

        elif self.ndim == 3:
            ax = self.A_impl_x.copy().tolil()
            ay = self.A_impl_y.copy().tolil()
            az = self.A_impl_z.copy().tolil()
            nx, ny, nz = self.grid_points
            rhs = rhs.reshape(nx, ny, nz)

            rhs_x = rhs.reshape(nx, ny * nz)
            rhs_x = self._apply_bc_to_sweep(ax, rhs_x, self.dx[0])
            u_star = spsolve(ax.tocsr(), rhs_x).reshape(nx, ny, nz)

            rhs_y = u_star.transpose(1, 0, 2).reshape(ny, nx * nz)
            rhs_y = self._apply_bc_to_sweep(ay, rhs_y, self.dx[1])
            u_star_star = spsolve(ay.tocsr(), rhs_y).reshape(ny, nx, nz).transpose(1, 0, 2)

            rhs_z = u_star_star.transpose(2, 0, 1).reshape(nz, nx * ny)
            rhs_z = self._apply_bc_to_sweep(az, rhs_z, self.dx[2])
            u_new_t = spsolve(az.tocsr(), rhs_z)

            self.state = u_new_t.reshape(nz, nx, ny).transpose(1, 2, 0)

        return self.state

    def _apply_bc_to_sweep(self, matrix, rhs_array, h):
        """Apply boundary conditions to one 1D sweep system."""
        if self._boundary_conditions is None:
            return rhs_array

        d = self.diffusion_coefficient
        dt = self.dt
        theta = self.theta

        if isinstance(self._boundary_conditions, NeumannBC):
            flux = self._boundary_conditions._get_flux(self.t + self.dt)
            alpha = (theta * dt * d) / (h**2)
            forcing = (2 * theta * dt * d * flux) / h

            matrix[0, 1] = -2 * alpha
            rhs_array[0, :] -= forcing

            matrix[-1, -2] = -2 * alpha
            rhs_array[-1, :] += forcing

        elif isinstance(self._boundary_conditions, DirichletBC):
            val = self._boundary_conditions._get_value(self.t + self.dt)

            matrix[0, :] = 0
            matrix[0, 0] = 1
            rhs_array[0, :] = val

            matrix[-1, :] = 0
            matrix[-1, -1] = 1
            rhs_array[-1, :] = val

        return rhs_array

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

        if isinstance(self._boundary_conditions, NeumannBC):
            g = self._boundary_conditions._get_flux(self.t)
            laplacian[0] = 2 * (u[1] - u[0] - g * dx) / (dx**2)
            laplacian[-1] = 2 * (u[-2] - u[-1] + g * dx) / (dx**2)
        elif isinstance(self._boundary_conditions, DirichletBC):
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

        if isinstance(self._boundary_conditions, NeumannBC):
            flux = self._boundary_conditions._get_flux(self.t)
            laplacian[0, 1:-1] = (2 * (u[1, 1:-1] - u[0, 1:-1]) / (dx**2) - 2 * flux / dx) + (u[0, 2:] - 2 * u[0, 1:-1] + u[0, :-2]) / (dy**2)
            laplacian[-1, 1:-1] = (2 * (u[-2, 1:-1] - u[-1, 1:-1]) / (dx**2) + 2 * flux / dx) + (u[-1, 2:] - 2 * u[-1, 1:-1] + u[-1, :-2]) / (dy**2)
            laplacian[1:-1, 0] = (u[2:, 0] - 2 * u[1:-1, 0] + u[:-2, 0]) / (dx**2) + (2 * (u[1:-1, 1] - u[1:-1, 0]) / (dy**2) - 2 * flux / dy)
            laplacian[1:-1, -1] = (u[2:, -1] - 2 * u[1:-1, -1] + u[:-2, -1]) / (dx**2) + (2 * (u[1:-1, -2] - u[1:-1, -1]) / (dy**2) + 2 * flux / dy)
            laplacian[0, 0] = (2 * (u[1, 0] - u[0, 0]) / dx**2 - 2 * flux / dx) + (2 * (u[0, 1] - u[0, 0]) / dy**2 - 2 * flux / dy)
            laplacian[0, -1] = (2 * (u[1, -1] - u[0, -1]) / dx**2 - 2 * flux / dx) + (2 * (u[0, -2] - u[0, -1]) / dy**2 + 2 * flux / dy)
            laplacian[-1, 0] = (2 * (u[-2, 0] - u[-1, 0]) / dx**2 + 2 * flux / dx) + (2 * (u[-1, 1] - u[-1, 0]) / dy**2 - 2 * flux / dy)
            laplacian[-1, -1] = (2 * (u[-2, -1] - u[-1, -1]) / dx**2 + 2 * flux / dx) + (2 * (u[-1, -2] - u[-1, -1]) / dy**2 + 2 * flux / dy)
        elif isinstance(self._boundary_conditions, DirichletBC):
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

        if isinstance(self._boundary_conditions, NeumannBC):
            flux = self._boundary_conditions._get_flux(self.t)
            laplacian[0, 1:-1, 1:-1] = (2 * (u[1, 1:-1, 1:-1] - u[0, 1:-1, 1:-1]) / (dx**2) - 2 * flux / dx) + (u[0, 2:, 1:-1] - 2 * u[0, 1:-1, 1:-1] + u[0, :-2, 1:-1]) / (dy**2) + (u[0, 1:-1, 2:] - 2 * u[0, 1:-1, 1:-1] + u[0, 1:-1, :-2]) / (dz**2)
            laplacian[-1, 1:-1, 1:-1] = (2 * (u[-2, 1:-1, 1:-1] - u[-1, 1:-1, 1:-1]) / (dx**2) + 2 * flux / dx) + (u[-1, 2:, 1:-1] - 2 * u[-1, 1:-1, 1:-1] + u[-1, :-2, 1:-1]) / (dy**2) + (u[-1, 1:-1, 2:] - 2 * u[-1, 1:-1, 1:-1] + u[-1, 1:-1, :-2]) / (dz**2)
            laplacian[1:-1, 0, 1:-1] = (u[2:, 0, 1:-1] - 2 * u[1:-1, 0, 1:-1] + u[:-2, 0, 1:-1]) / (dx**2) + (2 * (u[1:-1, 1, 1:-1] - u[1:-1, 0, 1:-1]) / (dy**2) - 2 * flux / dy) + (u[1:-1, 0, 2:] - 2 * u[1:-1, 0, 1:-1] + u[1:-1, 0, :-2]) / (dz**2)
            laplacian[1:-1, -1, 1:-1] = (u[2:, -1, 1:-1] - 2 * u[1:-1, -1, 1:-1] + u[:-2, -1, 1:-1]) / (dx**2) + (2 * (u[1:-1, -2, 1:-1] - u[1:-1, -1, 1:-1]) / (dy**2) + 2 * flux / dy) + (u[1:-1, -1, 2:] - 2 * u[1:-1, -1, 1:-1] + u[1:-1, -1, :-2]) / (dz**2)
            laplacian[1:-1, 1:-1, 0] = (u[2:, 1:-1, 0] - 2 * u[1:-1, 1:-1, 0] + u[:-2, 1:-1, 0]) / (dx**2) + (u[1:-1, 2:, 0] - 2 * u[1:-1, 1:-1, 0] + u[1:-1, :-2, 0]) / (dy**2) + (2 * (u[1:-1, 1:-1, 1] - u[1:-1, 1:-1, 0]) / (dz**2) - 2 * flux / dz)
            laplacian[1:-1, 1:-1, -1] = (u[2:, 1:-1, -1] - 2 * u[1:-1, 1:-1, -1] + u[:-2, 1:-1, -1]) / (dx**2) + (u[1:-1, 2:, -1] - 2 * u[1:-1, 1:-1, -1] + u[1:-1, :-2, -1]) / (dy**2) + (2 * (u[1:-1, 1:-1, -2] - u[1:-1, 1:-1, -1]) / (dz**2) + 2 * flux / dz)

            laplacian[1:-1, 0, 0] = (u[2:, 0, 0] - 2 * u[1:-1, 0, 0] + u[:-2, 0, 0]) / dx**2 + (2 * (u[1:-1, 1, 0] - u[1:-1, 0, 0]) / dy**2 - 2 * flux / dy) + (2 * (u[1:-1, 0, 1] - u[1:-1, 0, 0]) / dz**2 - 2 * flux / dz)
            laplacian[1:-1, 0, -1] = (u[2:, 0, -1] - 2 * u[1:-1, 0, -1] + u[:-2, 0, -1]) / dx**2 + (2 * (u[1:-1, 1, -1] - u[1:-1, 0, -1]) / dy**2 - 2 * flux / dy) + (2 * (u[1:-1, 0, -2] - u[1:-1, 0, -1]) / dz**2 + 2 * flux / dz)
            laplacian[1:-1, -1, 0] = (u[2:, -1, 0] - 2 * u[1:-1, -1, 0] + u[:-2, -1, 0]) / dx**2 + (2 * (u[1:-1, -2, 0] - u[1:-1, -1, 0]) / dy**2 + 2 * flux / dy) + (2 * (u[1:-1, -1, 1] - u[1:-1, -1, 0]) / dz**2 - 2 * flux / dz)
            laplacian[1:-1, -1, -1] = (u[2:, -1, -1] - 2 * u[1:-1, -1, -1] + u[:-2, -1, -1]) / dx**2 + (2 * (u[1:-1, -2, -1] - u[1:-1, -1, -1]) / dy**2 + 2 * flux / dy) + (2 * (u[1:-1, -1, -2] - u[1:-1, -1, -1]) / dz**2 + 2 * flux / dz)

            laplacian[0, 1:-1, 0] = (2 * (u[1, 1:-1, 0] - u[0, 1:-1, 0]) / dx**2 - 2 * flux / dx) + (u[0, 2:, 0] - 2 * u[0, 1:-1, 0] + u[0, :-2, 0]) / dy**2 + (2 * (u[0, 1:-1, 1] - u[0, 1:-1, 0]) / dz**2 - 2 * flux / dz)
            laplacian[0, 1:-1, -1] = (2 * (u[1, 1:-1, -1] - u[0, 1:-1, -1]) / dx**2 - 2 * flux / dx) + (u[0, 2:, -1] - 2 * u[0, 1:-1, -1] + u[0, :-2, -1]) / dy**2 + (2 * (u[0, 1:-1, -2] - u[0, 1:-1, -1]) / dz**2 + 2 * flux / dz)
            laplacian[-1, 1:-1, 0] = (2 * (u[-2, 1:-1, 0] - u[-1, 1:-1, 0]) / dx**2 + 2 * flux / dx) + (u[-1, 2:, 0] - 2 * u[-1, 1:-1, 0] + u[-1, :-2, 0]) / dy**2 + (2 * (u[-1, 1:-1, 1] - u[-1, 1:-1, 0]) / dz**2 - 2 * flux / dz)
            laplacian[-1, 1:-1, -1] = (2 * (u[-2, 1:-1, -1] - u[-1, 1:-1, -1]) / dx**2 + 2 * flux / dx) + (u[-1, 2:, -1] - 2 * u[-1, 1:-1, -1] + u[-1, :-2, -1]) / dy**2 + (2 * (u[-1, 1:-1, -2] - u[-1, 1:-1, -1]) / dz**2 + 2 * flux / dz)

            laplacian[0, 0, 1:-1] = (2 * (u[1, 0, 1:-1] - u[0, 0, 1:-1]) / dx**2 - 2 * flux / dx) + (2 * (u[0, 1, 1:-1] - u[0, 0, 1:-1]) / dy**2 - 2 * flux / dy) + (u[0, 0, 2:] - 2 * u[0, 0, 1:-1] + u[0, 0, :-2]) / dz**2
            laplacian[0, -1, 1:-1] = (2 * (u[1, -1, 1:-1] - u[0, -1, 1:-1]) / dx**2 - 2 * flux / dx) + (2 * (u[0, -2, 1:-1] - u[0, -1, 1:-1]) / dy**2 + 2 * flux / dy) + (u[0, -1, 2:] - 2 * u[0, -1, 1:-1] + u[0, -1, :-2]) / dz**2
            laplacian[-1, 0, 1:-1] = (2 * (u[-2, 0, 1:-1] - u[-1, 0, 1:-1]) / dx**2 + 2 * flux / dx) + (2 * (u[-1, 1, 1:-1] - u[-1, 0, 1:-1]) / dy**2 - 2 * flux / dy) + (u[-1, 0, 2:] - 2 * u[-1, 0, 1:-1] + u[-1, 0, :-2]) / dz**2
            laplacian[-1, -1, 1:-1] = (2 * (u[-2, -1, 1:-1] - u[-1, -1, 1:-1]) / dx**2 + 2 * flux / dx) + (2 * (u[-1, -2, 1:-1] - u[-1, -1, 1:-1]) / dy**2 + 2 * flux / dy) + (u[-1, -1, 2:] - 2 * u[-1, -1, 1:-1] + u[-1, -1, :-2]) / dz**2

            laplacian[0, 0, 0] = (2 * (u[1, 0, 0] - u[0, 0, 0]) / dx**2 - 2 * flux / dx) + (2 * (u[0, 1, 0] - u[0, 0, 0]) / dy**2 - 2 * flux / dy) + (2 * (u[0, 0, 1] - u[0, 0, 0]) / dz**2 - 2 * flux / dz)
            laplacian[-1, 0, 0] = (2 * (u[-2, 0, 0] - u[-1, 0, 0]) / dx**2 + 2 * flux / dx) + (2 * (u[-1, 1, 0] - u[-1, 0, 0]) / dy**2 - 2 * flux / dy) + (2 * (u[-1, 0, 1] - u[-1, 0, 0]) / dz**2 - 2 * flux / dz)
            laplacian[0, -1, 0] = (2 * (u[1, -1, 0] - u[0, -1, 0]) / dx**2 - 2 * flux / dx) + (2 * (u[0, -2, 0] - u[0, -1, 0]) / dy**2 + 2 * flux / dy) + (2 * (u[0, -1, 1] - u[0, -1, 0]) / dz**2 - 2 * flux / dz)
            laplacian[-1, -1, 0] = (2 * (u[-2, -1, 0] - u[-1, -1, 0]) / dx**2 + 2 * flux / dx) + (2 * (u[-1, -2, 0] - u[-1, -1, 0]) / dy**2 + 2 * flux / dy) + (2 * (u[-1, -1, 1] - u[-1, -1, 0]) / dz**2 - 2 * flux / dz)
            laplacian[0, 0, -1] = (2 * (u[1, 0, -1] - u[0, 0, -1]) / dx**2 - 2 * flux / dx) + (2 * (u[0, 1, -1] - u[0, 0, -1]) / dy**2 - 2 * flux / dy) + (2 * (u[0, 0, -2] - u[0, 0, -1]) / dz**2 + 2 * flux / dz)
            laplacian[-1, 0, -1] = (2 * (u[-2, 0, -1] - u[-1, 0, -1]) / dx**2 + 2 * flux / dx) + (2 * (u[-1, 1, -1] - u[-1, 0, -1]) / dy**2 - 2 * flux / dy) + (2 * (u[-1, 0, -2] - u[-1, 0, -1]) / dz**2 + 2 * flux / dz)
            laplacian[0, -1, -1] = (2 * (u[1, -1, -1] - u[0, -1, -1]) / dx**2 - 2 * flux / dx) + (2 * (u[0, -2, -1] - u[0, -1, -1]) / dy**2 + 2 * flux / dy) + (2 * (u[0, -1, -2] - u[0, -1, -1]) / dz**2 + 2 * flux / dz)
            laplacian[-1, -1, -1] = (2 * (u[-2, -1, -1] - u[-1, -1, -1]) / dx**2 + 2 * flux / dx) + (2 * (u[-1, -2, -1] - u[-1, -1, -1]) / dy**2 + 2 * flux / dy) + (2 * (u[-1, -1, -2] - u[-1, -1, -1]) / dz**2 + 2 * flux / dz)

        elif isinstance(self._boundary_conditions, DirichletBC):
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
