"""Unified Crank-Nicolson LOD schemas with variant selection."""

from typing import Optional, Tuple
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve, splu
from scipy.linalg import solve_banded

from diffusion_schemas.base import Schema
from diffusion_schemas.utils.boundary import DirichletBC, NeumannBC


class _CrankNicolsonLODUnified(Schema):
    """Crank-Nicolson LOD core with selectable behavior variants."""

    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0,
        theta=0.5,
        variant="base",
    ):
        super().__init__(domain_size, grid_points, dt, diffusion_coefficient, decay_rate)

        if not 0 <= theta <= 1:
            raise ValueError("theta must be in [0, 1]")
        self.theta = theta
        self._variant = variant

        self._build_system_matrices()

        if variant in {"bci", "bcos", "bci_opt"}:
            self._boundary_mask = self._compute_boundary_mask()

    def _compute_boundary_mask(self) -> np.ndarray:
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
            self.A_impl_x = self._build_matrices_1d()
        elif self.ndim == 2:
            self.A_impl_x, self.A_impl_y = self._build_matrices_2d()
        elif self.ndim == 3:
            self.A_impl_x, self.A_impl_y, self.A_impl_z = self._build_matrices_3d()
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.ndim}")
        if self._variant == "bci_opt":
            self._lu_x = splu(self.A_impl_x.tocsc())
            if self.ndim >= 2:
                self._lu_y = splu(self.A_impl_y.tocsc())
            if self.ndim == 3:
                self._lu_z = splu(self.A_impl_z.tocsc())

        self._diffusion_dirty = False

    def _ensure_system_matrices_current(self) -> None:
        if self.diffusion_is_time_dependent or self._diffusion_dirty:
            self._build_system_matrices()

    def _build_line_banded(
        self,
        d_line: np.ndarray,
        h: float,
        dt_factor: float,
        decay_term: float,
        source_lhs_line: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = d_line.shape[0]
        d_face = 0.5 * (d_line[:-1] + d_line[1:])

        d_minus = np.empty(n)
        d_plus = np.empty(n)
        d_minus[0] = d_face[0]
        d_minus[1:] = d_face
        d_plus[-1] = d_face[-1]
        d_plus[:-1] = d_face

        main = 1.0 + dt_factor * (d_minus + d_plus) / (h**2) + decay_term
        if source_lhs_line is not None:
            main = main + source_lhs_line

        ab = np.zeros((3, n))
        ab[0, 1:] = -(dt_factor * d_plus[:-1]) / (h**2)
        ab[2, :-1] = -(dt_factor * d_minus[1:]) / (h**2)
        ab[1, :] = main

        return ab, d_face

    def _apply_bc_to_banded(
        self,
        ab: np.ndarray,
        rhs_line: np.ndarray,
        h: float,
        dt_factor: float,
        t_eval: float,
        d_face: Optional[np.ndarray] = None,
    ) -> None:
        if self._boundary_conditions is None:
            return

        if isinstance(self._boundary_conditions, NeumannBC):
            flux = self._boundary_conditions._get_flux(t_eval)
            forcing = (2 * dt_factor * flux) / h

            if d_face is None:
                d_left = float(self._diffusion_value(t_eval))
                d_right = d_left
            else:
                d_left = d_face[0]
                d_right = d_face[-1]

            alpha_left = dt_factor * d_left / (h**2)
            alpha_right = dt_factor * d_right / (h**2)

            ab[0, 1] = -2 * alpha_left
            rhs_line[0] -= forcing

            ab[2, -2] = -2 * alpha_right
            rhs_line[-1] += forcing

        elif isinstance(self._boundary_conditions, DirichletBC):
            val = self._boundary_conditions._get_value(t_eval)
            ab[1, 0] = 1.0
            ab[0, 1] = 0.0
            rhs_line[0] = val

            ab[1, -1] = 1.0
            ab[2, -2] = 0.0
            rhs_line[-1] = val

    def _solve_line_banded(
        self,
        d_line: np.ndarray,
        rhs_line: np.ndarray,
        h: float,
        dt_factor: float,
        decay_term: float,
        t_eval: float,
        source_lhs_line: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ab, d_face = self._build_line_banded(d_line, h, dt_factor, decay_term, source_lhs_line)
        self._apply_bc_to_banded(ab, rhs_line, h, dt_factor, t_eval, d_face)
        return solve_banded((1, 1), ab, rhs_line)


    def _build_matrices_1d(self):
        n = self.grid_points[0]
        dx = self.dx[0]

        if not self.diffusion_is_scalar():
            d = float(np.mean(self._diffusion_field()))
        else:
            d = float(self._diffusion_value())

        diag_main = -2 * np.ones(n) / (dx**2)
        diag_off = np.ones(n - 1) / (dx**2)

        l = diags([diag_off, diag_main, diag_off], [-1, 0, 1], shape=(n, n), format="csr")
        i = eye(n, format="csr")

        a_impl = i - self.theta * self.dt * d * l + self.theta * self.dt * self.decay_rate * i

        self.Lx = l
        return a_impl

    def _build_matrices_2d(self):
        nx, ny = self.grid_points
        dx, dy = self.dx
        factor = 1 / 2

        if not self.diffusion_is_scalar():
            d = float(np.mean(self._diffusion_field()))
        else:
            d = float(self._diffusion_value())

        diag_main_x = -2 * np.ones(nx) / (dx**2)
        diag_off_x = np.ones(nx - 1) / (dx**2)
        lx = diags([diag_off_x, diag_main_x, diag_off_x], [-1, 0, 1], shape=(nx, nx), format="csr")

        diag_main_y = -2 * np.ones(ny) / (dy**2)
        diag_off_y = np.ones(ny - 1) / (dy**2)
        ly = diags([diag_off_y, diag_main_y, diag_off_y], [-1, 0, 1], shape=(ny, ny), format="csr")

        ix = eye(nx, format="csr")
        iy = eye(ny, format="csr")

        a_impl_x = ix - self.theta * self.dt * d * lx + self.theta * factor * self.dt * self.decay_rate * ix
        a_impl_y = iy - self.theta * self.dt * d * ly + self.theta * factor * self.dt * self.decay_rate * iy

        self.Lx = lx
        self.Ly = ly

        return a_impl_x, a_impl_y

    def _build_matrices_3d(self):
        nx, ny, nz = self.grid_points
        dx, dy, dz = self.dx
        factor = 1 / 3

        if not self.diffusion_is_scalar():
            d = float(np.mean(self._diffusion_field()))
        else:
            d = float(self._diffusion_value())

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

        a_impl_x = ix - self.theta * self.dt * d * lx + self.theta * factor * self.dt * self.decay_rate * ix
        a_impl_y = iy - self.theta * self.dt * d * ly + self.theta * factor * self.dt * self.decay_rate * iy
        a_impl_z = iz - self.theta * self.dt * d * lz + self.theta * factor * self.dt * self.decay_rate * iz

        self.Lx = lx
        self.Ly = ly
        self.Lz = lz

        return a_impl_x, a_impl_y, a_impl_z

    def step(self) -> None:
        self._ensure_system_matrices_current()
        if self._variant == "base":
            self._step_base()
        elif self._variant == "bc":
            self._step_bc()
        elif self._variant == "bci":
            self._step_bci()
        elif self._variant == "bcos":
            self._step_bcos()
        elif self._variant == "bci_opt":
            self._step_bci() # same as bci but with pre-factorized matrices
        else:
            raise ValueError(f"Unsupported Crank-Nicolson LOD variant: {self._variant}")

    def _step_base(self) -> None:
        source_n = self._compute_source_term()
        source_np1 = source_n

        explicit_term = self._step_explicit_base()
        rhs_grid = self.state + explicit_term + self.dt * source_np1

        self.state = self._step_lod_base(rhs_grid)

        if self._boundary_conditions is not None:
            self.state = self._apply_boundary_conditions(self.state)

        self.t += self.dt

    def _step_bc(self) -> None:
        source_n = self._compute_source_term()
        source_np1 = source_n

        explicit_term = self._step_explicit_bc()
        rhs_grid = self.state + explicit_term + self.dt * ((1 - self.theta) * source_n + self.theta * source_np1)

        self.state = self._step_lod_bc(rhs_grid)
        self.t += self.dt

    def _step_bci(self) -> None:
        source_n = self._compute_source_term()

        t_next = self.t + self.dt
        agent_source_np1 = self._compute_source_term(implicit=True, t=t_next)
        bulk_rhs_np1 = np.zeros_like(self.state)
        bulk_lhs_np1 = np.zeros_like(self.state)

        if self._bulk is not None:
            bulk_rhs_np1 = self._bulk.rhs_contribution
            bulk_lhs_np1 = self._bulk.lhs_contribution.copy()
            if isinstance(self._boundary_conditions, DirichletBC):
                bulk_lhs_np1[self._boundary_mask] = 0.0

        diffusion_n = self._compute_diffusion_term(self.state)

        rhs_grid = self.state + (1 - self.theta) * self.dt * (
            diffusion_n - self.decay_rate * self.state + source_n
        ) + self.theta * self.dt * (bulk_rhs_np1 + agent_source_np1)

        if self._variant == "bci_opt":
            self.state = self._step_lod_bci_opt(rhs_grid, bulk_lhs_np1)
        else:
            self.state = self._step_lod_bci(rhs_grid, bulk_lhs_np1)

        self.t += self.dt

    def _step_bcos(self) -> None:
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
        diffusion_n = self._compute_diffusion_term(self.state)
        rhs_grid = self.state + (1 - self.theta) * self.dt * (
            diffusion_n - self.decay_rate * self.state
        )

        self.state = self._step_lod_bc(rhs_grid)

    def _step_bulk_sources(self) -> None:
        s_rhs = self._bulk.rhs_contribution.copy()
        s_lhs = self._bulk.lhs_contribution.copy()
        self.state = (self.state + self.dt * s_rhs) / (1.0 + self.dt * s_lhs)

    def _step_agent_sources(self) -> None:
        self.state += self.dt * self.agents_rhs_contribution

    def _compute_diffusion_term(self, u: np.ndarray) -> np.ndarray:
        if self.diffusion_is_scalar():
            d = float(self._diffusion_value())
            return d * self._compute_laplacian(u)
        d_field = self._diffusion_field()
        return self._compute_conservative_diffusion(u, d_field)

    def _compute_conservative_diffusion(self, u: np.ndarray, d_field: np.ndarray) -> np.ndarray:
        if self.ndim == 1:
            return self._conservative_diffusion_1d(u, d_field)
        if self.ndim == 2:
            return self._conservative_diffusion_2d(u, d_field)
        if self.ndim == 3:
            return self._conservative_diffusion_3d(u, d_field)
        raise ValueError(f"Unsupported dimensions: {self.ndim}")

    def _conservative_diffusion_1d(self, u: np.ndarray, d_field: np.ndarray) -> np.ndarray:
        dx = self.dx[0]
        n = self.grid_points[0]
        d_face = 0.5 * (d_field[:-1] + d_field[1:])
        flux_face = np.zeros(n + 1)
        flux_face[1:-1] = d_face * (u[1:] - u[:-1]) / dx

        if isinstance(self._boundary_conditions, NeumannBC):
            flux_val = self._boundary_conditions._get_flux(self.t)
        else:
            flux_val = 0.0

        flux_face[0] = -flux_val
        flux_face[-1] = flux_val

        div = (flux_face[1:] - flux_face[:-1]) / dx

        if isinstance(self._boundary_conditions, DirichletBC):
            div[0] = 0.0
            div[-1] = 0.0

        return div

    def _conservative_diffusion_2d(self, u: np.ndarray, d_field: np.ndarray) -> np.ndarray:
        dx, dy = self.dx
        nx, ny = self.grid_points
        d_x, d_y = self._diffusion_faces(d_field)

        flux_x = d_x * (u[1:, :] - u[:-1, :]) / dx
        flux_y = d_y * (u[:, 1:] - u[:, :-1]) / dy

        flux_x_ext = np.zeros((nx + 1, ny))
        flux_y_ext = np.zeros((nx, ny + 1))
        flux_x_ext[1:-1, :] = flux_x
        flux_y_ext[:, 1:-1] = flux_y

        if isinstance(self._boundary_conditions, NeumannBC):
            flux_val = self._boundary_conditions._get_flux(self.t)
        else:
            flux_val = 0.0

        flux_x_ext[0, :] = -flux_val
        flux_x_ext[-1, :] = flux_val
        flux_y_ext[:, 0] = -flux_val
        flux_y_ext[:, -1] = flux_val

        div = (flux_x_ext[1:, :] - flux_x_ext[:-1, :]) / dx + (flux_y_ext[:, 1:] - flux_y_ext[:, :-1]) / dy

        if isinstance(self._boundary_conditions, DirichletBC):
            div[0, :] = 0.0
            div[-1, :] = 0.0
            div[:, 0] = 0.0
            div[:, -1] = 0.0

        return div

    def _conservative_diffusion_3d(self, u: np.ndarray, d_field: np.ndarray) -> np.ndarray:
        dx, dy, dz = self.dx
        nx, ny, nz = self.grid_points
        d_x, d_y, d_z = self._diffusion_faces(d_field)

        flux_x = d_x * (u[1:, :, :] - u[:-1, :, :]) / dx
        flux_y = d_y * (u[:, 1:, :] - u[:, :-1, :]) / dy
        flux_z = d_z * (u[:, :, 1:] - u[:, :, :-1]) / dz

        flux_x_ext = np.zeros((nx + 1, ny, nz))
        flux_y_ext = np.zeros((nx, ny + 1, nz))
        flux_z_ext = np.zeros((nx, ny, nz + 1))

        flux_x_ext[1:-1, :, :] = flux_x
        flux_y_ext[:, 1:-1, :] = flux_y
        flux_z_ext[:, :, 1:-1] = flux_z

        if isinstance(self._boundary_conditions, NeumannBC):
            flux_val = self._boundary_conditions._get_flux(self.t)
        else:
            flux_val = 0.0

        flux_x_ext[0, :, :] = -flux_val
        flux_x_ext[-1, :, :] = flux_val
        flux_y_ext[:, 0, :] = -flux_val
        flux_y_ext[:, -1, :] = flux_val
        flux_z_ext[:, :, 0] = -flux_val
        flux_z_ext[:, :, -1] = flux_val

        div = (
            (flux_x_ext[1:, :, :] - flux_x_ext[:-1, :, :]) / dx
            + (flux_y_ext[:, 1:, :] - flux_y_ext[:, :-1, :]) / dy
            + (flux_z_ext[:, :, 1:] - flux_z_ext[:, :, :-1]) / dz
        )

        if isinstance(self._boundary_conditions, DirichletBC):
            div[0, :, :] = 0.0
            div[-1, :, :] = 0.0
            div[:, 0, :] = 0.0
            div[:, -1, :] = 0.0
            div[:, :, 0] = 0.0
            div[:, :, -1] = 0.0

        return div

    def _step_explicit_base(self):
        u = self.state

        if self.diffusion_is_scalar():
            if self.ndim == 1:
                laplacian = self.Lx.dot(u)

            elif self.ndim == 2:
                laplacian = self.Lx.dot(u) + self.Ly.dot(u.T).T

            elif self.ndim == 3:
                nx, ny, nz = self.grid_points
                diff_x = self.Lx.dot(u.reshape(nx, -1)).reshape(nx, ny, nz)
                diff_y = self.Ly.dot(u.transpose(1, 0, 2).reshape(ny, -1)).reshape(ny, nx, nz).transpose(1, 0, 2)
                diff_z = self.Lz.dot(u.transpose(2, 0, 1).reshape(nz, -1)).reshape(nz, nx, ny).transpose(1, 2, 0)
                laplacian = diff_x + diff_y + diff_z

            d = float(self._diffusion_value())
            diffusion_term = d * laplacian
        else:
            diffusion_term = self._compute_diffusion_term(u)

        explicit_term = self.dt * (1 - self.theta) * (
            diffusion_term - self.decay_rate * u
        )

        return explicit_term

    def _step_explicit_bc(self):
        u = self.state
        diffusion_term = self._compute_diffusion_term(u)

        explicit_term = self.dt * (1 - self.theta) * (
            diffusion_term - self.decay_rate * u
        )

        return explicit_term

    def _step_lod_base(self, rhs):
        if not self.diffusion_is_scalar():
            return self._step_lod_base_variable(rhs)

        if self.ndim == 1:
            self.state = spsolve(self.A_impl_x, rhs)

        elif self.ndim == 2:
            ax, ay = self.A_impl_x, self.A_impl_y
            nx, ny = self.grid_points
            rhs = rhs.reshape(nx, ny)

            u_star = spsolve(ax, rhs)
            if self._boundary_conditions is not None:
                u_star = self._apply_boundary_conditions(u_star)

            self.state = spsolve(ay, u_star.T).T

        elif self.ndim == 3:
            ax, ay, az = self.A_impl_x, self.A_impl_y, self.A_impl_z
            nx, ny, nz = self.grid_points
            rhs = rhs.reshape(nx, ny, nz)

            rhs_x = rhs.reshape(nx, ny * nz)
            u_star = spsolve(ax, rhs_x)
            u_star = u_star.reshape(nx, ny, nz)
            if self._boundary_conditions is not None:
                u_star = self._apply_boundary_conditions(u_star)

            rhs_y = u_star.transpose(1, 0, 2).reshape(ny, nx * nz)
            u_star_star = spsolve(ay, rhs_y)
            u_star_star = u_star_star.reshape(ny, nx, nz).transpose(1, 0, 2)
            if self._boundary_conditions is not None:
                u_star_star = self._apply_boundary_conditions(u_star_star)

            rhs_z = u_star_star.transpose(2, 0, 1).reshape(nz, nx * ny)
            self.state = spsolve(az, rhs_z)
            self.state = self.state.reshape(nz, nx, ny).transpose(1, 2, 0)

        return self.state

    def _step_lod_base_variable(self, rhs):
        t_eval = self.t + self.dt
        d_field = self._diffusion_field()

        if self.ndim == 1:
            dt_factor = self.theta * self.dt
            decay_term = self.theta * self.dt * self.decay_rate
            rhs_line = rhs.reshape(self.grid_points[0]).copy()
            self.state = self._solve_line_banded(
                d_field,
                rhs_line,
                self.dx[0],
                dt_factor,
                decay_term,
                t_eval,
            ).reshape(self.grid_points)

        elif self.ndim == 2:
            nx, ny = self.grid_points
            dt_factor = self.theta * self.dt
            decay_term = self.theta * 0.5 * self.dt * self.decay_rate
            rhs = rhs.reshape(nx, ny)

            u_star = np.zeros((nx, ny))
            for j in range(ny):
                rhs_j = rhs[:, j].copy()
                u_star[:, j] = self._solve_line_banded(
                    d_field[:, j],
                    rhs_j,
                    self.dx[0],
                    dt_factor,
                    decay_term,
                    t_eval,
                )

            if self._boundary_conditions is not None:
                u_star = self._apply_boundary_conditions(u_star)

            u_new = np.zeros((nx, ny))
            for i in range(nx):
                rhs_i = u_star[i, :].copy()
                u_new[i, :] = self._solve_line_banded(
                    d_field[i, :],
                    rhs_i,
                    self.dx[1],
                    dt_factor,
                    decay_term,
                    t_eval,
                )
            self.state = u_new

        elif self.ndim == 3:
            nx, ny, nz = self.grid_points
            dt_factor = self.theta * self.dt
            decay_term = self.theta * self.dt * self.decay_rate / 3.0
            rhs = rhs.reshape(nx, ny, nz)

            u_star = np.zeros((nx, ny, nz))
            for j in range(ny):
                for k in range(nz):
                    rhs_jk = rhs[:, j, k].copy()
                    u_star[:, j, k] = self._solve_line_banded(
                        d_field[:, j, k],
                        rhs_jk,
                        self.dx[0],
                        dt_factor,
                        decay_term,
                        t_eval,
                    )

            if self._boundary_conditions is not None:
                u_star = self._apply_boundary_conditions(u_star)

            u_star_star = np.zeros((nx, ny, nz))
            for i in range(nx):
                for k in range(nz):
                    rhs_ik = u_star[i, :, k].copy()
                    u_star_star[i, :, k] = self._solve_line_banded(
                        d_field[i, :, k],
                        rhs_ik,
                        self.dx[1],
                        dt_factor,
                        decay_term,
                        t_eval,
                    )

            if self._boundary_conditions is not None:
                u_star_star = self._apply_boundary_conditions(u_star_star)

            u_new = np.zeros((nx, ny, nz))
            for i in range(nx):
                for j in range(ny):
                    rhs_ij = u_star_star[i, j, :].copy()
                    u_new[i, j, :] = self._solve_line_banded(
                        d_field[i, j, :],
                        rhs_ij,
                        self.dx[2],
                        dt_factor,
                        decay_term,
                        t_eval,
                    )
            self.state = u_new

        return self.state

    def _step_lod_bc(self, rhs):
        if not self.diffusion_is_scalar():
            return self._step_lod_base_variable(rhs)

        if self.ndim == 1:
            ax = self.A_impl_x.copy().tolil()
            rhs_x = rhs.reshape(self.grid_points[0], 1)

            rhs_x = self._apply_bc_to_sweep(ax, rhs_x, self.dx[0])
            self.state = spsolve(ax.tocsr(), rhs_x).reshape(self.grid_points)

        elif self.ndim == 2:
            ax = self.A_impl_x.copy().tolil()
            ay = self.A_impl_y.copy().tolil()
            nx, ny = self.grid_points
            rhs = rhs.reshape(nx, ny)

            rhs_x = self._apply_bc_to_sweep(ax, rhs, self.dx[0])
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

    def _step_lod_bci(self, rhs, bulk_lhs_np1):
        if not self.diffusion_is_scalar():
            return self._step_lod_bci_variable(rhs, bulk_lhs_np1)

        if self.ndim == 1:
            ax = self.A_impl_x.copy().tolil()
            nx = self.grid_points[0]
            rhs_x = rhs.reshape(nx, 1)

            source_diag = diags([self.theta * self.dt * bulk_lhs_np1.reshape(nx)], [0], shape=(nx, nx), format="csr")
            ax_eff = (ax + source_diag).tolil()

            rhs_x = self._apply_bc_to_sweep(ax_eff, rhs_x, self.dx[0])
            self.state = spsolve(ax_eff.tocsr(), rhs_x).reshape(self.grid_points)

        elif self.ndim == 2:
            ax = self.A_impl_x.copy().tolil()
            ay = self.A_impl_y.copy().tolil()
            nx, ny = self.grid_points
            rhs = rhs.reshape(nx, ny)

            u_star = np.zeros((nx, ny))
            for j in range(ny):
                source_diag = diags([(self.theta * self.dt / 2.0) * bulk_lhs_np1[:, j]], [0], shape=(nx, nx), format="csr")
                ax_j = (ax + source_diag).tolil()

                rhs_j = rhs[:, j].reshape(nx, 1)
                rhs_j = self._apply_bc_to_sweep(ax_j, rhs_j, self.dx[0])
                u_star[:, j] = spsolve(ax_j.tocsr(), rhs_j).flatten()

            u_new = np.zeros((nx, ny))
            for i in range(nx):
                source_diag = diags([(self.theta * self.dt / 2.0) * bulk_lhs_np1[i, :]], [0], shape=(ny, ny), format="csr")
                ay_i = (ay + source_diag).tolil()

                rhs_i = u_star[i, :].reshape(ny, 1)
                rhs_i = self._apply_bc_to_sweep(ay_i, rhs_i, self.dx[1])
                u_new[i, :] = spsolve(ay_i.tocsr(), rhs_i).flatten()

            self.state = u_new

        elif self.ndim == 3:
            ax = self.A_impl_x.copy().tolil()
            ay = self.A_impl_y.copy().tolil()
            az = self.A_impl_z.copy().tolil()
            nx, ny, nz = self.grid_points
            rhs = rhs.reshape(nx, ny, nz)

            u_star = np.zeros((nx, ny, nz))
            for j in range(ny):
                for k in range(nz):
                    source_diag = diags([(self.theta * self.dt / 3.0) * bulk_lhs_np1[:, j, k]], [0], shape=(nx, nx), format="csr")
                    ax_jk = (ax + source_diag).tolil()

                    rhs_jk = rhs[:, j, k].reshape(nx, 1)
                    rhs_jk = self._apply_bc_to_sweep(ax_jk, rhs_jk, self.dx[0])
                    u_star[:, j, k] = spsolve(ax_jk.tocsr(), rhs_jk).flatten()

            u_star_star = np.zeros((nx, ny, nz))
            for i in range(nx):
                for k in range(nz):
                    source_diag = diags([(self.theta * self.dt / 3.0) * bulk_lhs_np1[i, :, k]], [0], shape=(ny, ny), format="csr")
                    ay_ik = (ay + source_diag).tolil()

                    rhs_ik = u_star[i, :, k].reshape(ny, 1)
                    rhs_ik = self._apply_bc_to_sweep(ay_ik, rhs_ik, self.dx[1])
                    u_star_star[i, :, k] = spsolve(ay_ik.tocsr(), rhs_ik).flatten()

            u_new = np.zeros((nx, ny, nz))
            for i in range(nx):
                for j in range(ny):
                    source_diag = diags([(self.theta * self.dt / 3.0) * bulk_lhs_np1[i, j, :]], [0], shape=(nz, nz), format="csr")
                    az_ij = (az + source_diag).tolil()

                    rhs_ij = u_star_star[i, j, :].reshape(nz, 1)
                    rhs_ij = self._apply_bc_to_sweep(az_ij, rhs_ij, self.dx[2])
                    u_new[i, j, :] = spsolve(az_ij.tocsr(), rhs_ij).flatten()

            self.state = u_new

        return self.state

    def _step_lod_bci_variable(self, rhs, bulk_lhs_np1):
        t_eval = self.t + self.dt
        d_field = self._diffusion_field()

        if self.ndim == 1:
            dt_factor = self.theta * self.dt
            decay_term = self.theta * self.dt * self.decay_rate
            rhs_line = rhs.reshape(self.grid_points[0]).copy()
            source_line = (self.theta * self.dt) * bulk_lhs_np1.reshape(self.grid_points[0])
            self.state = self._solve_line_banded(
                d_field,
                rhs_line,
                self.dx[0],
                dt_factor,
                decay_term,
                t_eval,
                source_line,
            ).reshape(self.grid_points)

        elif self.ndim == 2:
            nx, ny = self.grid_points
            dt_factor = self.theta * self.dt
            decay_term = self.theta * 0.5 * self.dt * self.decay_rate
            rhs = rhs.reshape(nx, ny)

            u_star = np.zeros((nx, ny))
            for j in range(ny):
                rhs_j = rhs[:, j].copy()
                source_line = (self.theta * self.dt / 2.0) * bulk_lhs_np1[:, j]
                u_star[:, j] = self._solve_line_banded(
                    d_field[:, j],
                    rhs_j,
                    self.dx[0],
                    dt_factor,
                    decay_term,
                    t_eval,
                    source_line,
                )

            u_new = np.zeros((nx, ny))
            for i in range(nx):
                rhs_i = u_star[i, :].copy()
                source_line = (self.theta * self.dt / 2.0) * bulk_lhs_np1[i, :]
                u_new[i, :] = self._solve_line_banded(
                    d_field[i, :],
                    rhs_i,
                    self.dx[1],
                    dt_factor,
                    decay_term,
                    t_eval,
                    source_line,
                )
            self.state = u_new

        elif self.ndim == 3:
            nx, ny, nz = self.grid_points
            dt_factor = self.theta * self.dt
            decay_term = self.theta * self.dt * self.decay_rate / 3.0
            rhs = rhs.reshape(nx, ny, nz)

            u_star = np.zeros((nx, ny, nz))
            for j in range(ny):
                for k in range(nz):
                    rhs_jk = rhs[:, j, k].copy()
                    source_line = (self.theta * self.dt / 3.0) * bulk_lhs_np1[:, j, k]
                    u_star[:, j, k] = self._solve_line_banded(
                        d_field[:, j, k],
                        rhs_jk,
                        self.dx[0],
                        dt_factor,
                        decay_term,
                        t_eval,
                        source_line,
                    )

            u_star_star = np.zeros((nx, ny, nz))
            for i in range(nx):
                for k in range(nz):
                    rhs_ik = u_star[i, :, k].copy()
                    source_line = (self.theta * self.dt / 3.0) * bulk_lhs_np1[i, :, k]
                    u_star_star[i, :, k] = self._solve_line_banded(
                        d_field[i, :, k],
                        rhs_ik,
                        self.dx[1],
                        dt_factor,
                        decay_term,
                        t_eval,
                        source_line,
                    )

            u_new = np.zeros((nx, ny, nz))
            for i in range(nx):
                for j in range(ny):
                    rhs_ij = u_star_star[i, j, :].copy()
                    source_line = (self.theta * self.dt / 3.0) * bulk_lhs_np1[i, j, :]
                    u_new[i, j, :] = self._solve_line_banded(
                        d_field[i, j, :],
                        rhs_ij,
                        self.dx[2],
                        dt_factor,
                        decay_term,
                        t_eval,
                        source_line,
                    )
            self.state = u_new

        return self.state

    def _step_lod_bci_opt(self, rhs, bulk_lhs_np1):
        if not self.diffusion_is_scalar():
            return self._step_lod_bci(rhs, bulk_lhs_np1)

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
        

    def _apply_bc_to_sweep(self, matrix, rhs_array, h):
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


class CrankNicolsonLODSchema(_CrankNicolsonLODUnified):
    """Crank-Nicolson LOD method."""

    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0,
        theta=0.5,
    ):
        super().__init__(
            domain_size,
            grid_points,
            dt,
            diffusion_coefficient=diffusion_coefficient,
            decay_rate=decay_rate,
            theta=theta,
            variant="base",
        )


class CrankNicolsonLODBCSchema(_CrankNicolsonLODUnified):
    """Crank-Nicolson LOD method with boundary conditions."""

    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0,
        theta=0.5,
    ):
        super().__init__(
            domain_size,
            grid_points,
            dt,
            diffusion_coefficient=diffusion_coefficient,
            decay_rate=decay_rate,
            theta=theta,
            variant="bc",
        )


class CrankNicolsonLODBCISchema(_CrankNicolsonLODUnified):
    """Crank-Nicolson LOD method with implicit sources and BCs."""

    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0,
        theta=0.5,
    ):
        super().__init__(
            domain_size,
            grid_points,
            dt,
            diffusion_coefficient=diffusion_coefficient,
            decay_rate=decay_rate,
            theta=theta,
            variant="bci",
        )


class CrankNicolsonLODBCOSSchema(_CrankNicolsonLODUnified):
    """Crank-Nicolson LOD method with operator-split sources and BCs."""

    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0,
        theta=0.5,
    ):
        super().__init__(
            domain_size,
            grid_points,
            dt,
            diffusion_coefficient=diffusion_coefficient,
            decay_rate=decay_rate,
            theta=theta,
            variant="bcos",
        )

class CrankNicolsonLODBCIOptSchema(_CrankNicolsonLODUnified):
    """Crank-Nicolson LOD method with optimized implicit sources and BCs."""

    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0,
        theta=0.5,
    ):
        super().__init__(
            domain_size,
            grid_points,
            dt,
            diffusion_coefficient=diffusion_coefficient,
            decay_rate=decay_rate,
            theta=theta,
            variant="bci_opt",
        )