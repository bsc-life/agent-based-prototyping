"""Unified ADI schemas with variant selection."""

from typing import Optional, Tuple

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded

from diffusion_schemas.base import Schema
from diffusion_schemas.utils.boundary import DirichletBC, NeumannBC


class _ADIUnified(Schema):
    """ADI core with selectable behavior variants."""

    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0,
        variant="base",
    ):
        super().__init__(domain_size, grid_points, dt, diffusion_coefficient, decay_rate)
        self._variant = variant

        self._build_system_matrix()

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

    def _build_system_matrix(self) -> None:
        if self.diffusion_is_scalar():
            d = float(self._diffusion_value())
        else:
            d = float(np.mean(self._diffusion_field()))

        if self._variant == "base":
            if self.ndim == 1:
                self.system_matrix = self._build_matrix_1d_base(d)
            elif self.ndim == 2:
                self.system_matrix = self._build_matrix_2d_base(d)
            elif self.ndim == 3:
                self.system_matrix = self._build_matrix_3d_base(d)
            else:
                raise ValueError(f"Unsupported number of dimensions: {self.ndim}")
            self._diffusion_dirty = False
            return

        if self.ndim == 1:
            self.system_matrix = self._build_matrix_1d_bc(d)
        elif self.ndim == 2:
            self.system_matrix = self._build_matrix_2d_bc(d)
        elif self.ndim == 3:
            self.system_matrix = self._build_matrix_3d_bc(d)
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.ndim}")

        self._diffusion_dirty = False

    def _ensure_system_matrix_current(self) -> None:
        if self.diffusion_is_time_dependent or self._diffusion_dirty:
            self._build_system_matrix()

    def _build_matrix_1d_base(self, d):
        n = self.grid_points[0]
        dx = self.dx[0]

        diag_main = -2 * np.ones(n) / (dx**2)
        diag_off = np.ones(n - 1) / (dx**2)
        l = diags([diag_off, diag_main, diag_off], [-1, 0, 1], shape=(n, n), format="csr")
        i = eye(n, format="csr")

        return i - self.dt * d * l + self.dt * self.decay_rate * i

    def _build_matrix_2d_base(self, d):
        nx, ny = self.grid_points
        dx, dy = self.dx

        dt_half = self.dt / 2.0
        decay_term = (self.decay_rate / 2.0) * dt_half

        diag_main_x = -2 * np.ones(nx) / (dx**2)
        diag_off_x = np.ones(nx - 1) / (dx**2)
        lx = diags([diag_off_x, diag_main_x, diag_off_x], [-1, 0, 1], shape=(nx, nx), format="csr")
        ix = eye(nx, format="csr")

        lhs_x = ix - dt_half * d * lx + decay_term * ix
        rhs_x = ix + dt_half * d * lx - decay_term * ix

        diag_main_y = -2 * np.ones(ny) / (dy**2)
        diag_off_y = np.ones(ny - 1) / (dy**2)
        ly = diags([diag_off_y, diag_main_y, diag_off_y], [-1, 0, 1], shape=(ny, ny), format="csr")
        iy = eye(ny, format="csr")

        lhs_y = iy - dt_half * d * ly + decay_term * iy
        rhs_y = iy + dt_half * d * ly - decay_term * iy

        return lhs_x, rhs_x, lhs_y, rhs_y

    def _build_matrix_3d_base(self, d):
        nx, ny, nz = self.grid_points
        dx, dy, dz = self.dx

        decay_term = (self.decay_rate / 3.0) * self.dt

        lx = diags([np.ones(nx - 1) / dx**2, -2 * np.ones(nx) / dx**2, np.ones(nx - 1) / dx**2], [-1, 0, 1], shape=(nx, nx), format="csr")
        ix = eye(nx, format="csr")
        lhs_x = ix - self.dt * d * lx + decay_term * ix
        a_x = self.dt * d * lx - decay_term * ix

        ly = diags([np.ones(ny - 1) / dy**2, -2 * np.ones(ny) / dy**2, np.ones(ny - 1) / dy**2], [-1, 0, 1], shape=(ny, ny), format="csr")
        iy = eye(ny, format="csr")
        lhs_y = iy - self.dt * d * ly + decay_term * iy
        a_y = self.dt * d * ly - decay_term * iy

        lz = diags([np.ones(nz - 1) / dz**2, -2 * np.ones(nz) / dz**2, np.ones(nz - 1) / dz**2], [-1, 0, 1], shape=(nz, nz), format="csr")
        iz = eye(nz, format="csr")
        lhs_z = iz - self.dt * d * lz + decay_term * iz
        a_z = self.dt * d * lz - decay_term * iz

        return lhs_x, a_x, lhs_y, a_y, lhs_z, a_z

    def _build_matrix_1d_bc(self, d):
        n = self.grid_points[0]
        dx = self.dx[0]

        diag_main = -2 * np.ones(n) / (dx**2)
        diag_off = np.ones(n - 1) / (dx**2)
        l = diags([diag_off, diag_main, diag_off], [-1, 0, 1], shape=(n, n), format="csr")
        i = eye(n, format="csr")

        return i - self.dt * d * l + self.dt * self.decay_rate * i

    def _build_matrix_2d_bc(self, d):
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
        lhs_x = (ix - dt_half * d * lx + decay_term * ix).tolil()
        rhs_x = (ix + dt_half * d * lx - decay_term * ix).tolil()

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
        lhs_y = (iy - dt_half * d * ly + decay_term * iy).tolil()
        rhs_y = (iy + dt_half * d * ly - decay_term * iy).tolil()

        if isinstance(self._boundary_conditions, DirichletBC):
            for row in [0, -1]:
                lhs_y[row, :] = 0
                lhs_y[row, row] = 1
                rhs_y[row, :] = 0
                rhs_y[row, row] = 1

        return lhs_x, rhs_x, lhs_y, rhs_y

    def _build_matrix_3d_bc(self, d):
        nx, ny, nz = self.grid_points
        dx, dy, dz = self.dx

        decay_term = (self.decay_rate / 3.0) * self.dt

        lx = diags([np.ones(nx - 1) / dx**2, -2 * np.ones(nx) / dx**2, np.ones(nx - 1) / dx**2], [-1, 0, 1], shape=(nx, nx), format="csr").tolil()
        ix = eye(nx, format="csr")
        if isinstance(self._boundary_conditions, NeumannBC):
            lx[0, 0], lx[0, 1] = -2 / (dx**2), 2 / (dx**2)
            lx[-1, -1], lx[-1, -2] = -2 / (dx**2), 2 / (dx**2)
        lx, ix = lx.tocsr(), ix.tocsr()
        lhs_x = (ix - self.dt * d * lx + decay_term * ix).tolil()
        a_x = (self.dt * d * lx - decay_term * ix).tolil()

        ly = diags([np.ones(ny - 1) / dy**2, -2 * np.ones(ny) / dy**2, np.ones(ny - 1) / dy**2], [-1, 0, 1], shape=(ny, ny), format="csr").tolil()
        iy = eye(ny, format="csr")
        if isinstance(self._boundary_conditions, NeumannBC):
            ly[0, 0], ly[0, 1] = -2 / (dy**2), 2 / (dy**2)
            ly[-1, -1], ly[-1, -2] = -2 / (dy**2), 2 / (dy**2)
        ly, iy = ly.tocsr(), iy.tocsr()
        lhs_y = (iy - self.dt * d * ly + decay_term * iy).tolil()
        a_y = (self.dt * d * ly - decay_term * iy).tolil()

        lz = diags([np.ones(nz - 1) / dz**2, -2 * np.ones(nz) / dz**2, np.ones(nz - 1) / dz**2], [-1, 0, 1], shape=(nz, nz), format="csr").tolil()
        iz = eye(nz, format="csr")
        if isinstance(self._boundary_conditions, NeumannBC):
            lz[0, 0], lz[0, 1] = -2 / (dz**2), 2 / (dz**2)
            lz[-1, -1], lz[-1, -2] = -2 / (dz**2), 2 / (dz**2)
        lz, iz = lz.tocsr(), iz.tocsr()
        lhs_z = (iz - self.dt * d * lz + decay_term * iz).tolil()
        a_z = (self.dt * d * lz - decay_term * iz).tolil()

        if isinstance(self._boundary_conditions, DirichletBC):
            for op_lhs, op_a in [(lhs_x, a_x), (lhs_y, a_y), (lhs_z, a_z)]:
                for row in [0, -1]:
                    op_lhs[row, :] = 0
                    op_lhs[row, row] = 1
                    op_a[row, :] = 0
                    op_a[row, row] = 0

        return lhs_x, a_x, lhs_y, a_y, lhs_z, a_z

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
            if d_face is None:
                d_left = float(self._diffusion_value(t_eval))
                d_right = d_left
            else:
                d_left = d_face[0]
                d_right = d_face[-1]

            alpha_left = dt_factor * d_left / (h**2)
            alpha_right = dt_factor * d_right / (h**2)
            forcing_left = (2 * dt_factor * d_left * flux) / h
            forcing_right = (2 * dt_factor * d_right * flux) / h

            ab[0, 1] = -2 * alpha_left
            rhs_line[0] -= forcing_left

            ab[2, -2] = -2 * alpha_right
            rhs_line[-1] += forcing_right

        elif isinstance(self._boundary_conditions, DirichletBC):
            val = self._boundary_conditions._get_value(t_eval)
            ab[1, 0] = 1.0
            ab[0, 1] = 0.0
            rhs_line[0] = val
            ab[1, -1] = 1.0
            ab[2, -2] = 0.0
            rhs_line[-1] = val

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

    def step(self) -> None:
        self._ensure_system_matrix_current()
        if self._variant == "base":
            return self._step_base()
        if self._variant == "bc":
            return self._step_bc()
        if self._variant == "bci":
            return self._step_bci()
        if self._variant == "bcos":
            return self._step_bcos()
        if self._variant == "bci_opt":
            return self._step_bci_opt()
        raise ValueError(f"Unsupported ADI variant: {self._variant}")

    def _step_base(self) -> None:
        if not self.diffusion_is_scalar():
            source = self._compute_source_term()
            if source is None:
                source = np.zeros_like(self.state)
            return self._step_base_variable(source)

        source = self._compute_source_term()

        if self.ndim == 1:
            rhs = self.state + self.dt * source if source is not None else self.state
            rhs = rhs.reshape(self.grid_points[0], 1)
            self.state = spsolve(self.system_matrix, rhs).reshape(self.grid_points)
            if self._boundary_conditions is not None:
                self.state = self._apply_boundary_conditions(self.state)

        elif self.ndim == 2:
            dt_half = self.dt / 2.0

            lhs_x, rhs_x, lhs_y, rhs_y = self.system_matrix
            half_source = dt_half * source if source is not None else 0.0

            rhs_1 = (rhs_y @ self.state.T).T + half_source
            u_star = spsolve(lhs_x, rhs_1)
            if self._boundary_conditions is not None:
                u_star = self._apply_boundary_conditions(u_star)

            rhs_2 = (rhs_x @ u_star) + half_source
            rhs_2_t = rhs_2.T
            u_new_t = spsolve(lhs_y, rhs_2_t)

            self.state = u_new_t.T
            if self._boundary_conditions is not None:
                self.state = self._apply_boundary_conditions(self.state)

        elif self.ndim == 3:
            nx, ny, nz = self.grid_points
            lhs_x, a_x, lhs_y, a_y, lhs_z, a_z = self.system_matrix
            un = self.state

            full_source = self.dt * source if source is not None else 0.0

            # Pre-calculate explicit operations on u^n
            a_y_un = (a_y @ un.transpose(1, 0, 2).reshape(ny, nx * nz)).reshape(ny, nx, nz).transpose(1, 0, 2)
            a_z_un = (a_z @ un.transpose(2, 0, 1).reshape(nz, nx * ny)).reshape(nz, nx, ny).transpose(1, 2, 0)

            # --- SWEEP 1: X-direction ---
            rhs_1 = un + a_y_un + a_z_un + full_source
            rhs_1_x = rhs_1.reshape(nx, ny * nz)
            u_star = spsolve(lhs_x, rhs_1_x).reshape(nx, ny, nz)
            
            if self._boundary_conditions is not None:
                u_star = self._apply_boundary_conditions(u_star)

            # --- SWEEP 2: Y-direction ---
            rhs_2 = u_star - a_y_un
            rhs_2_y = rhs_2.transpose(1, 0, 2).reshape(ny, nx * nz)
            u_star_star = spsolve(lhs_y, rhs_2_y).reshape(ny, nx, nz).transpose(1, 0, 2)

            if self._boundary_conditions is not None:
                u_star_star = self._apply_boundary_conditions(u_star_star)

            # --- SWEEP 3: Z-direction ---
            rhs_3 = u_star_star - a_z_un
            rhs_3_z = rhs_3.transpose(2, 0, 1).reshape(nz, nx * ny)
            u_new = spsolve(lhs_z, rhs_3_z).reshape(nz, nx, ny).transpose(1, 2, 0)

            self.state = u_new
            if self._boundary_conditions is not None:
                self.state = self._apply_boundary_conditions(self.state)

        self.t += self.dt

    def _step_bc(self) -> None:
        if self.ndim not in (1, 2, 3):
            raise NotImplementedError(f"{self.ndim}D ADI is not implemented yet")

        if not self.diffusion_is_scalar():
            source = self._compute_source_term()
            if source is None:
                source = np.zeros_like(self.state)
            return self._step_base_variable(source)

        source = self._compute_source_term()

        if self.ndim == 1:
            rhs = self.state + self.dt * source
            ax = self.system_matrix.copy().tolil()
            rhs = rhs.reshape(self.grid_points[0], 1)
            rhs = self._apply_bc_to_sweep(ax, rhs, self.dx[0], self.dt, t_eval=self.t + self.dt)
            self.state = spsolve(ax.tocsr(), rhs).reshape(self.grid_points)

        elif self.ndim == 2:
            dt_half = self.dt / 2.0
            d = self.diffusion_coefficient
            dx, dy = self.dx

            if isinstance(self._boundary_conditions, DirichletBC):
                source = source.copy()
                source[0, :] = 0
                source[-1, :] = 0
                source[:, 0] = 0
                source[:, -1] = 0

            half_source = dt_half * source

            lhs_x, rhs_x, lhs_y, rhs_y = self.system_matrix

            rhs_1 = (rhs_y @ self.state.T).T + half_source

            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(self.t)
                explicit_y_forcing = dt_half * d * 2 * flux / dy
                rhs_1[:, 0] -= explicit_y_forcing
                rhs_1[:, -1] += explicit_y_forcing

            lhs_x_lil = lhs_x.copy().tolil()
            rhs_1 = self._apply_bc_to_sweep(lhs_x_lil, rhs_1, dx, dt_half, t_eval=self.t + dt_half)
            u_star = spsolve(lhs_x_lil.tocsr(), rhs_1)

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(self.t + dt_half)
                u_star[0, :] = val
                u_star[-1, :] = val
                u_star[:, 0] = val
                u_star[:, -1] = val

            rhs_2 = (rhs_x @ u_star) + half_source

            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(self.t + dt_half)
                explicit_x_forcing = dt_half * d * 2 * flux / dx
                rhs_2[0, :] -= explicit_x_forcing
                rhs_2[-1, :] += explicit_x_forcing

            rhs_2_t = rhs_2.T
            lhs_y_lil = lhs_y.copy().tolil()
            rhs_2_t = self._apply_bc_to_sweep(lhs_y_lil, rhs_2_t, dy, dt_half, t_eval=self.t + self.dt)
            u_new_t = spsolve(lhs_y_lil.tocsr(), rhs_2_t)

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(self.t + self.dt)
                u_new = u_new_t.T
                u_new[0, :] = val
                u_new[-1, :] = val
                u_new[:, 0] = val
                u_new[:, -1] = val
                self.state = u_new
            else:
                self.state = u_new_t.T

        elif self.ndim == 3:
            t_next = self.t + self.dt
            d = self.diffusion_coefficient
            dx, dy, dz = self.dx
            nx, ny, nz = self.grid_points

            lhs_x, a_x, lhs_y, a_y, lhs_z, a_z = self.system_matrix
            un = self.state

            if isinstance(self._boundary_conditions, DirichletBC):
                source = source.copy()
                source[0, :, :] = 0
                source[-1, :, :] = 0
                source[:, 0, :] = 0
                source[:, -1, :] = 0
                source[:, :, 0] = 0
                source[:, :, -1] = 0

            a_y_un = (a_y @ un.transpose(1, 0, 2).reshape(ny, nx * nz)).reshape(ny, nx, nz).transpose(1, 0, 2)
            a_z_un = (a_z @ un.transpose(2, 0, 1).reshape(nz, nx * ny)).reshape(nz, nx, ny).transpose(1, 2, 0)

            rhs_1 = un + a_y_un + a_z_un + self.dt * source

            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(t_next)
                rhs_1[:, 0, :] -= self.dt * d * 2 * flux / dy
                rhs_1[:, -1, :] += self.dt * d * 2 * flux / dy
                rhs_1[:, :, 0] -= self.dt * d * 2 * flux / dz
                rhs_1[:, :, -1] += self.dt * d * 2 * flux / dz

            rhs_1_x = rhs_1.reshape(nx, ny * nz)
            lhs_x_lil = lhs_x.copy().tolil()
            rhs_1_x = self._apply_bc_to_sweep(lhs_x_lil, rhs_1_x, dx, self.dt, t_eval=t_next)
            u_star = spsolve(lhs_x_lil.tocsr(), rhs_1_x).reshape(nx, ny, nz)

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(t_next)
                u_star[0, :, :] = val
                u_star[-1, :, :] = val
                u_star[:, 0, :] = val
                u_star[:, -1, :] = val
                u_star[:, :, 0] = val
                u_star[:, :, -1] = val

            rhs_2 = u_star - a_y_un

            rhs_2_y = rhs_2.transpose(1, 0, 2).reshape(ny, nx * nz)
            lhs_y_lil = lhs_y.copy().tolil()
            rhs_2_y = self._apply_bc_to_sweep(lhs_y_lil, rhs_2_y, dy, self.dt, t_eval=t_next)
            u_star_star = spsolve(lhs_y_lil.tocsr(), rhs_2_y).reshape(ny, nx, nz).transpose(1, 0, 2)

            if isinstance(self._boundary_conditions, DirichletBC):
                u_star_star[0, :, :] = val
                u_star_star[-1, :, :] = val
                u_star_star[:, 0, :] = val
                u_star_star[:, -1, :] = val
                u_star_star[:, :, 0] = val
                u_star_star[:, :, -1] = val

            rhs_3 = u_star_star - a_z_un

            rhs_3_z = rhs_3.transpose(2, 0, 1).reshape(nz, nx * ny)
            lhs_z_lil = lhs_z.copy().tolil()
            rhs_3_z = self._apply_bc_to_sweep(lhs_z_lil, rhs_3_z, dz, self.dt, t_eval=t_next)
            u_new = spsolve(lhs_z_lil.tocsr(), rhs_3_z).reshape(nz, nx, ny).transpose(1, 2, 0)

            if isinstance(self._boundary_conditions, DirichletBC):
                u_new[0, :, :] = val
                u_new[-1, :, :] = val
                u_new[:, 0, :] = val
                u_new[:, -1, :] = val
                u_new[:, :, 0] = val
                u_new[:, :, -1] = val

            self.state = u_new

        self.t += self.dt

    def _step_bci(self) -> None:
        if self.ndim not in (1, 2, 3):
            raise NotImplementedError(f"{self.ndim}D ADI is not implemented yet")

        if not self.diffusion_is_scalar():
            t_eval = self.t + self.dt
            source_explicit = self._compute_source_term(implicit=True, t=t_eval)
            source_rhs = np.zeros_like(self.state)
            if self._bulk is not None:
                source_rhs = self._bulk.rhs_contribution
            source = source_rhs + source_explicit if source_explicit is not None else source_rhs
            return self._step_base_variable(source)

        if self.ndim == 1:
            t_next = self.t + self.dt
            n = self.grid_points[0]

            source_explicit = self._compute_source_term(implicit=True, t=t_next)
            source_rhs = np.zeros_like(self.state)
            source_lhs = np.zeros_like(self.state)
            if self._bulk is not None:
                source_rhs = self._bulk.rhs_contribution
                source_lhs = self._bulk.lhs_contribution
                if isinstance(self._boundary_conditions, DirichletBC):
                    source_lhs[self._boundary_mask] = 0.0

            rhs = self.state + self.dt * (source_rhs + source_explicit)

            source_diag = diags([self.dt * source_lhs], [0], shape=(n, n), format="csr")
            ax = (self.system_matrix + source_diag).tolil()
            rhs = rhs.reshape(n, 1)
            rhs = self._apply_bc_to_sweep(ax, rhs, self.dx[0], self.dt, t_eval=t_next)
            self.state = spsolve(ax.tocsr(), rhs).reshape(self.grid_points)

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(t_next)
                self.state[0] = val
                self.state[-1] = val

        elif self.ndim == 2:
            dt_half = self.dt / 2.0
            t_mid = self.t + dt_half

            d = self.diffusion_coefficient
            dx, dy = self.dx
            nx, ny = self.grid_points

            lhs_x, rhs_x, lhs_y, rhs_y = self.system_matrix

            source_explicit = self._compute_source_term(implicit=True, t=t_mid)
            source_rhs = np.zeros_like(self.state)
            source_lhs = np.zeros_like(self.state)
            if self._bulk is not None:
                source_rhs = self._bulk.rhs_contribution
                source_lhs = self._bulk.lhs_contribution
                if isinstance(self._boundary_conditions, DirichletBC):
                    source_lhs[self._boundary_mask] = 0.0

            rhs_1 = (rhs_y @ self.state.T).T + dt_half * (source_rhs + source_explicit)

            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(t_mid)
                explicit_y_forcing = dt_half * d * 2 * flux / dy
                rhs_1[:, 0] -= explicit_y_forcing
                rhs_1[:, -1] += explicit_y_forcing

            u_star = np.zeros((nx, ny))
            for j in range(ny):
                source_diag = diags([dt_half * source_lhs[:, j]], [0], shape=(nx, nx), format="csr")
                lhs_x_j = (lhs_x + source_diag).tolil()

                rhs_1_j = rhs_1[:, j].reshape(nx, 1)
                rhs_1_j = self._apply_bc_to_sweep(lhs_x_j, rhs_1_j, dx, dt_half, t_eval=t_mid)

                u_star[:, j] = spsolve(lhs_x_j.tocsr(), rhs_1_j).flatten()

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(self.t + dt_half)
                u_star[0, :] = val
                u_star[-1, :] = val
                u_star[:, 0] = val
                u_star[:, -1] = val

            source_explicit = self._compute_source_term(state=u_star, implicit=True, t=self.t + self.dt)
            source_rhs = np.zeros_like(self.state)
            source_lhs = np.zeros_like(self.state)
            if self._bulk is not None:
                source_rhs = self._bulk.rhs_contribution
                source_lhs = self._bulk.lhs_contribution
                if isinstance(self._boundary_conditions, DirichletBC):
                    source_lhs[self._boundary_mask] = 0.0

            rhs_2 = (rhs_x @ u_star) + dt_half * (source_rhs + source_explicit)

            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(self.t + self.dt)
                explicit_x_forcing = dt_half * d * 2 * flux / dx
                rhs_2[0, :] -= explicit_x_forcing
                rhs_2[-1, :] += explicit_x_forcing

            u_new = np.zeros((nx, ny))
            for i in range(nx):
                source_diag = diags([dt_half * source_lhs[i, :]], [0], shape=(ny, ny), format="csr")
                lhs_y_i = (lhs_y + source_diag).tolil()

                rhs_2_i = rhs_2[i, :].reshape(ny, 1)
                rhs_2_i = self._apply_bc_to_sweep(lhs_y_i, rhs_2_i, dy, dt_half, t_eval=self.t + self.dt)

                u_new[i, :] = spsolve(lhs_y_i.tocsr(), rhs_2_i).flatten()

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(self.t + self.dt)
                u_new[0, :] = val
                u_new[-1, :] = val
                u_new[:, 0] = val
                u_new[:, -1] = val

            self.state = u_new

        elif self.ndim == 3:
            t_next = self.t + self.dt

            d = self.diffusion_coefficient
            dx, dy, dz = self.dx
            nx, ny, nz = self.grid_points

            lhs_x, a_x, lhs_y, a_y, lhs_z, a_z = self.system_matrix
            un = self.state

            a_y_un = (a_y @ un.transpose(1, 0, 2).reshape(ny, nx * nz)).reshape(ny, nx, nz).transpose(1, 0, 2)
            a_z_un = (a_z @ un.transpose(2, 0, 1).reshape(nz, nx * ny)).reshape(nz, nx, ny).transpose(1, 2, 0)

            source_explicit = self._compute_source_term(implicit=True, t=t_next)
            source_rhs = np.zeros_like(self.state)
            source_lhs = np.zeros_like(self.state)
            if self._bulk is not None:
                source_rhs = self._bulk.rhs_contribution
                source_lhs = self._bulk.lhs_contribution
                if isinstance(self._boundary_conditions, DirichletBC):
                    source_lhs[self._boundary_mask] = 0.0

            rhs_1 = un + a_y_un + a_z_un + self.dt * (source_rhs + source_explicit)

            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(t_next)
                rhs_1[:, 0, :] -= self.dt * d * 2 * flux / dy
                rhs_1[:, -1, :] += self.dt * d * 2 * flux / dy
                rhs_1[:, :, 0] -= self.dt * d * 2 * flux / dz
                rhs_1[:, :, -1] += self.dt * d * 2 * flux / dz

            u_star = np.zeros((nx, ny, nz))
            for j in range(ny):
                for k in range(nz):
                    source_diag = diags([self.dt * source_lhs[:, j, k] / 3.0], [0], shape=(nx, nx), format="csr")
                    lhs_x_jk = (lhs_x + source_diag).tolil()
                    rhs_1_jk = rhs_1[:, j, k].reshape(nx, 1)
                    rhs_1_jk = self._apply_bc_to_sweep(lhs_x_jk, rhs_1_jk, dx, self.dt, t_eval=t_next)
                    u_star[:, j, k] = spsolve(lhs_x_jk.tocsr(), rhs_1_jk).flatten()

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(t_next)
                u_star[0, :, :] = val
                u_star[-1, :, :] = val
                u_star[:, 0, :] = val
                u_star[:, -1] = val
                u_star[:, :, 0] = val
                u_star[:, :, -1] = val

            rhs_2 = u_star - a_y_un

            u_star_star = np.zeros((nx, ny, nz))
            for i in range(nx):
                for k in range(nz):
                    source_diag = diags([self.dt * source_lhs[i, :, k] / 3.0], [0], shape=(ny, ny), format="csr")
                    lhs_y_ik = (lhs_y + source_diag).tolil()
                    rhs_2_ik = rhs_2[i, :, k].reshape(ny, 1)
                    rhs_2_ik = self._apply_bc_to_sweep(lhs_y_ik, rhs_2_ik, dy, self.dt, t_eval=t_next)
                    u_star_star[i, :, k] = spsolve(lhs_y_ik.tocsr(), rhs_2_ik).flatten()

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(t_next)
                u_star_star[0, :, :] = val
                u_star_star[-1, :, :] = val
                u_star_star[:, 0, :] = val
                u_star_star[:, -1, :] = val
                u_star_star[:, :, 0] = val
                u_star_star[:, :, -1] = val

            rhs_3 = u_star_star - a_z_un

            u_new = np.zeros((nx, ny, nz))
            for i in range(nx):
                for j in range(ny):
                    source_diag = diags([self.dt * source_lhs[i, j, :] / 3.0], [0], shape=(nz, nz), format="csr")
                    lhs_z_ij = (lhs_z + source_diag).tolil()
                    rhs_3_ij = rhs_3[i, j, :].reshape(nz, 1)
                    rhs_3_ij = self._apply_bc_to_sweep(lhs_z_ij, rhs_3_ij, dz, self.dt, t_eval=t_next)
                    u_new[i, j, :] = spsolve(lhs_z_ij.tocsr(), rhs_3_ij).flatten()

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(t_next)
                u_new[0, :, :] = val
                u_new[-1, :, :] = val
                u_new[:, 0, :] = val
                u_new[:, -1, :] = val
                u_new[:, :, 0] = val
                u_new[:, :, -1] = val

            self.state = u_new

        self.t += self.dt

    def _step_bcos(self) -> None:
        if not self.diffusion_is_scalar():
            source = self._compute_source_term(implicit=True, t=self.t + self.dt)
            if source is None:
                source = np.zeros_like(self.state)
            return self._step_base_variable(source)

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
        if not self.diffusion_is_scalar():
            source = np.zeros_like(self.state)
            return self._step_base_variable(source)

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
            t_next = self.t + self.dt

            d = self.diffusion_coefficient
            dx, dy, dz = self.dx
            nx, ny, nz = self.grid_points

            lhs_x, a_x, lhs_y, a_y, lhs_z, a_z = self.system_matrix
            un = self.state.copy()

            a_y_un = (a_y @ un.transpose(1, 0, 2).reshape(ny, nx * nz)).reshape(ny, nx, nz).transpose(1, 0, 2)
            a_z_un = (a_z @ un.transpose(2, 0, 1).reshape(nz, nx * ny)).reshape(nz, nx, ny).transpose(1, 2, 0)

            rhs_1 = un + a_y_un + a_z_un

            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(t_next)
                rhs_1[:, 0, :] -= self.dt * d * 2 * flux / dy
                rhs_1[:, -1, :] += self.dt * d * 2 * flux / dy
                rhs_1[:, :, 0] -= self.dt * d * 2 * flux / dz
                rhs_1[:, :, -1] += self.dt * d * 2 * flux / dz

            lhs_x_eff = lhs_x.copy().tolil()
            rhs_1_x = rhs_1.reshape(nx, ny * nz)
            rhs_1_x = self._apply_bc_to_sweep(lhs_x_eff, rhs_1_x, dx, self.dt, t_eval=t_next)
            u_star = spsolve(lhs_x_eff.tocsr(), rhs_1_x).reshape(nx, ny, nz)

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(t_next)
                u_star[0, :, :] = val
                u_star[-1, :, :] = val
                u_star[:, 0, :] = val
                u_star[:, -1, :] = val
                u_star[:, :, 0] = val
                u_star[:, :, -1] = val

            rhs_2 = u_star - a_y_un

            lhs_y_eff = lhs_y.copy().tolil()
            rhs_2_y = rhs_2.transpose(1, 0, 2).reshape(ny, nx * nz)
            rhs_2_y = self._apply_bc_to_sweep(lhs_y_eff, rhs_2_y, dy, self.dt, t_eval=t_next)
            u_star_star = spsolve(lhs_y_eff.tocsr(), rhs_2_y).reshape(ny, nx, nz).transpose(1, 0, 2)

            if isinstance(self._boundary_conditions, DirichletBC):
                u_star_star[0, :, :] = val
                u_star_star[-1, :, :] = val
                u_star_star[:, 0, :] = val
                u_star_star[:, -1, :] = val
                u_star_star[:, :, 0] = val
                u_star_star[:, :, -1] = val

            rhs_3 = u_star_star - a_z_un

            lhs_z_eff = lhs_z.copy().tolil()
            rhs_3_z = rhs_3.transpose(2, 0, 1).reshape(nz, nx * ny)
            rhs_3_z = self._apply_bc_to_sweep(lhs_z_eff, rhs_3_z, dz, self.dt, t_eval=t_next)
            u_new = spsolve(lhs_z_eff.tocsr(), rhs_3_z).reshape(nz, nx, ny).transpose(1, 2, 0)

            if isinstance(self._boundary_conditions, DirichletBC):
                u_new[0, :, :] = val
                u_new[-1, :, :] = val
                u_new[:, 0, :] = val
                u_new[:, -1, :] = val
                u_new[:, :, 0] = val
                u_new[:, :, -1] = val

            self.state = u_new

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
        return self._axis_diffusion(u, 0) + self._axis_diffusion(u, 1) + self._axis_diffusion(u, 2)

    def _axis_diffusion(self, u: np.ndarray, axis: int) -> np.ndarray:
        d_field = self._diffusion_field()

        if self.ndim == 1:
            return self._axis_diffusion_1d(u, d_field)
        if self.ndim == 2:
            return self._axis_diffusion_2d(u, d_field, axis)
        if self.ndim == 3:
            return self._axis_diffusion_3d(u, d_field, axis)
        raise ValueError(f"Unsupported dimensions: {self.ndim}")

    def _axis_diffusion_1d(self, u: np.ndarray, d_field: np.ndarray) -> np.ndarray:
        dx = self.dx[0]
        n = self.grid_points[0]
        d_face = 0.5 * (d_field[:-1] + d_field[1:])
        flux_face = np.zeros(n + 1)
        flux_face[1:-1] = d_face * (u[1:] - u[:-1]) / dx

        if isinstance(self._boundary_conditions, NeumannBC):
            flux_val = self._boundary_conditions._get_flux(self.t)
            flux_face[0] = -flux_val
            flux_face[-1] = flux_val

        div = (flux_face[1:] - flux_face[:-1]) / dx
        if isinstance(self._boundary_conditions, DirichletBC):
            div[0] = 0.0
            div[-1] = 0.0
        return div

    def _axis_diffusion_2d(self, u: np.ndarray, d_field: np.ndarray, axis: int) -> np.ndarray:
        dx, dy = self.dx
        nx, ny = self.grid_points
        d_x, d_y = self._diffusion_faces(d_field)

        if axis == 0:
            flux = d_x * (u[1:, :] - u[:-1, :]) / dx
            flux_ext = np.zeros((nx + 1, ny))
            flux_ext[1:-1, :] = flux
            if isinstance(self._boundary_conditions, NeumannBC):
                flux_val = self._boundary_conditions._get_flux(self.t)
                flux_ext[0, :] = -flux_val
                flux_ext[-1, :] = flux_val
            div = (flux_ext[1:, :] - flux_ext[:-1, :]) / dx
            if isinstance(self._boundary_conditions, DirichletBC):
                div[0, :] = 0.0
                div[-1, :] = 0.0
            return div

        flux = d_y * (u[:, 1:] - u[:, :-1]) / dy
        flux_ext = np.zeros((nx, ny + 1))
        flux_ext[:, 1:-1] = flux
        if isinstance(self._boundary_conditions, NeumannBC):
            flux_val = self._boundary_conditions._get_flux(self.t)
            flux_ext[:, 0] = -flux_val
            flux_ext[:, -1] = flux_val
        div = (flux_ext[:, 1:] - flux_ext[:, :-1]) / dy
        if isinstance(self._boundary_conditions, DirichletBC):
            div[:, 0] = 0.0
            div[:, -1] = 0.0
        return div

    def _axis_diffusion_3d(self, u: np.ndarray, d_field: np.ndarray, axis: int) -> np.ndarray:
        dx, dy, dz = self.dx
        nx, ny, nz = self.grid_points
        d_x, d_y, d_z = self._diffusion_faces(d_field)

        if axis == 0:
            flux = d_x * (u[1:, :, :] - u[:-1, :, :]) / dx
            flux_ext = np.zeros((nx + 1, ny, nz))
            flux_ext[1:-1, :, :] = flux
            if isinstance(self._boundary_conditions, NeumannBC):
                flux_val = self._boundary_conditions._get_flux(self.t)
                flux_ext[0, :, :] = -flux_val
                flux_ext[-1, :, :] = flux_val
            div = (flux_ext[1:, :, :] - flux_ext[:-1, :, :]) / dx
            if isinstance(self._boundary_conditions, DirichletBC):
                div[0, :, :] = 0.0
                div[-1, :, :] = 0.0
            return div

        if axis == 1:
            flux = d_y * (u[:, 1:, :] - u[:, :-1, :]) / dy
            flux_ext = np.zeros((nx, ny + 1, nz))
            flux_ext[:, 1:-1, :] = flux
            if isinstance(self._boundary_conditions, NeumannBC):
                flux_val = self._boundary_conditions._get_flux(self.t)
                flux_ext[:, 0, :] = -flux_val
                flux_ext[:, -1, :] = flux_val
            div = (flux_ext[:, 1:, :] - flux_ext[:, :-1, :]) / dy
            if isinstance(self._boundary_conditions, DirichletBC):
                div[:, 0, :] = 0.0
                div[:, -1, :] = 0.0
            return div

        flux = d_z * (u[:, :, 1:] - u[:, :, :-1]) / dz
        flux_ext = np.zeros((nx, ny, nz + 1))
        flux_ext[:, :, 1:-1] = flux
        if isinstance(self._boundary_conditions, NeumannBC):
            flux_val = self._boundary_conditions._get_flux(self.t)
            flux_ext[:, :, 0] = -flux_val
            flux_ext[:, :, -1] = flux_val
        div = (flux_ext[:, :, 1:] - flux_ext[:, :, :-1]) / dz
        if isinstance(self._boundary_conditions, DirichletBC):
            div[:, :, 0] = 0.0
            div[:, :, -1] = 0.0
        return div

    def _step_base_variable(self, source: np.ndarray) -> None:
        d_field = self._diffusion_field()
        t_next = self.t + self.dt

        if self.ndim == 1:
            rhs = self.state + self.dt * source
            rhs_line = rhs.reshape(self.grid_points[0]).copy()
            self.state = self._solve_line_banded(
                d_field,
                rhs_line,
                self.dx[0],
                self.dt,
                self.dt * self.decay_rate,
                t_next,
            ).reshape(self.grid_points)

        elif self.ndim == 2:
            dt_half = self.dt / 2.0
            decay_term = (self.decay_rate / 2.0) * dt_half
            half_source = dt_half * source
            rhs_1 = self.state + self._axis_diffusion(self.state, 1) * dt_half - decay_term * self.state + half_source

            nx, ny = self.grid_points
            u_star = np.zeros((nx, ny))
            for j in range(ny):
                rhs_j = rhs_1[:, j].copy()
                u_star[:, j] = self._solve_line_banded(
                    d_field[:, j],
                    rhs_j,
                    self.dx[0],
                    dt_half,
                    decay_term,
                    t_next,
                )

            if self._boundary_conditions is not None:
                u_star = self._apply_boundary_conditions(u_star)

            rhs_2 = u_star + self._axis_diffusion(u_star, 0) * dt_half - decay_term * u_star + half_source
            u_new = np.zeros((nx, ny))
            for i in range(nx):
                rhs_i = rhs_2[i, :].copy()
                u_new[i, :] = self._solve_line_banded(
                    d_field[i, :],
                    rhs_i,
                    self.dx[1],
                    dt_half,
                    decay_term,
                    t_next,
                )
            self.state = u_new

        elif self.ndim == 3:
            nx, ny, nz = self.grid_points
            decay_term = (self.decay_rate / 3.0) * self.dt
            full_source = self.dt * source

            a_y_un = self.dt * self._axis_diffusion(self.state, 1) - decay_term * self.state
            a_z_un = self.dt * self._axis_diffusion(self.state, 2) - decay_term * self.state

            rhs_1 = self.state + a_y_un + a_z_un + full_source

            u_star = np.zeros((nx, ny, nz))
            for j in range(ny):
                for k in range(nz):
                    rhs_jk = rhs_1[:, j, k].copy()
                    u_star[:, j, k] = self._solve_line_banded(
                        d_field[:, j, k],
                        rhs_jk,
                        self.dx[0],
                        self.dt,
                        decay_term,
                        t_next,
                    )

            if self._boundary_conditions is not None:
                u_star = self._apply_boundary_conditions(u_star)

            rhs_2 = u_star - a_y_un
            u_star_star = np.zeros((nx, ny, nz))
            for i in range(nx):
                for k in range(nz):
                    rhs_ik = rhs_2[i, :, k].copy()
                    u_star_star[i, :, k] = self._solve_line_banded(
                        d_field[i, :, k],
                        rhs_ik,
                        self.dx[1],
                        self.dt,
                        decay_term,
                        t_next,
                    )

            if self._boundary_conditions is not None:
                u_star_star = self._apply_boundary_conditions(u_star_star)

            rhs_3 = u_star_star - a_z_un
            u_new = np.zeros((nx, ny, nz))
            for i in range(nx):
                for j in range(ny):
                    rhs_ij = rhs_3[i, j, :].copy()
                    u_new[i, j, :] = self._solve_line_banded(
                        d_field[i, j, :],
                        rhs_ij,
                        self.dx[2],
                        self.dt,
                        decay_term,
                        t_next,
                    )
            self.state = u_new

        if self._boundary_conditions is not None:
            self.state = self._apply_boundary_conditions(self.state)

    def _step_bci_opt(self) -> None:
        if self.ndim not in (1, 2, 3):
            raise NotImplementedError(f"{self.ndim}D ADI is not implemented yet")

        if not self.diffusion_is_scalar():
            source = self._compute_source_term(implicit=True, t=self.t + self.dt)
            if source is None:
                source = np.zeros_like(self.state)
            return self._step_base_variable(source)

        if self.ndim == 1:
            t_next = self.t + self.dt
            n = self.grid_points[0]

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

            ab_x = np.zeros((3, n))
            ab_x[0, 1:] = -alpha
            ab_x[2, :-1] = -alpha
            ab_x[1, :] = 1.0 + 2*alpha + decay_term + self.dt * source_lhs.flatten()

            self._apply_bc_to_banded(ab_x, rhs_flat, self.dx[0], self.dt, t_next)
            self.state = solve_banded((1, 1), ab_x, rhs_flat).reshape(self.grid_points)

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(t_next)
                self.state[0] = val
                self.state[-1] = val

        elif self.ndim == 2:
            dt_half = self.dt / 2.0
            t_mid = self.t + dt_half
            
            d = self.diffusion_coefficient
            dx, dy = self.dx
            nx, ny = self.grid_points

            # Unpack RHS explicitly, ignoring LHS
            _, rhs_x, _, rhs_y = self.system_matrix

            alpha_x = dt_half * d / (dx**2)
            alpha_y = dt_half * d / (dy**2)
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

            rhs_1 = (rhs_y @ self.state.T).T + dt_half * (source_rhs + source_explicit)

            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(t_mid)
                explicit_y_forcing = dt_half * d * 2 * flux / dy
                rhs_1[:, 0]  -= explicit_y_forcing
                rhs_1[:, -1] += explicit_y_forcing

            u_star = np.zeros((nx, ny))
            for j in range(ny):
                ab_x = np.zeros((3, nx))
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

            rhs_2 = (rhs_x @ u_star) + dt_half * (source_rhs + source_explicit)

            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(self.t + self.dt)
                explicit_x_forcing = dt_half * d * 2 * flux / dx
                rhs_2[0, :]  -= explicit_x_forcing
                rhs_2[-1, :] += explicit_x_forcing

            u_new_T = np.zeros((ny, nx))
            for i in range(nx):
                ab_y = np.zeros((3, ny))
                ab_y[0, 1:] = -alpha_y
                ab_y[2, :-1] = -alpha_y
                ab_y[1, :] = 1.0 + 2*alpha_y + decay_term + dt_half * source_lhs[i, :]
                
                rhs_2_i = rhs_2[i, :].copy()
                self._apply_bc_to_banded(ab_y, rhs_2_i, dy, dt_half, t_eval=self.t + self.dt)
                u_new_T[:, i] = solve_banded((1, 1), ab_y, rhs_2_i)  
            u_new = u_new_T.T 

            if isinstance(self._boundary_conditions, DirichletBC):
                val = self._boundary_conditions._get_value(self.t + self.dt)
                u_new[0, :] = val; u_new[-1, :] = val
                u_new[:, 0] = val; u_new[:, -1] = val

            self.state = u_new

        elif self.ndim == 3:
            t_next = self.t + self.dt

            d = self.diffusion_coefficient
            dx, dy, dz = self.dx
            nx, ny, nz = self.grid_points

            # Unpack full-step explicit matrices (ignore LHS)
            _, a_x, _, a_y, _, a_z = self.system_matrix
            
            alpha_x = self.dt * d / (dx**2)
            alpha_y = self.dt * d / (dy**2)
            alpha_z = self.dt * d / (dz**2)
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
            a_y_un = (a_y @ un.transpose(1, 0, 2).reshape(ny, nx * nz)).reshape(ny, nx, nz).transpose(1, 0, 2)
            a_z_un = (a_z @ un.transpose(2, 0, 1).reshape(nz, nx * ny)).reshape(nz, nx, ny).transpose(1, 2, 0)

            # --- SWEEP 1: X-direction ---
            rhs_1 = un + a_y_un + a_z_un + self.dt * (source_rhs + source_explicit)

            if isinstance(self._boundary_conditions, NeumannBC):
                flux = self._boundary_conditions._get_flux(t_next)
                rhs_1[:, 0, :]  -= self.dt * d * 2 * flux / dy
                rhs_1[:, -1, :] += self.dt * d * 2 * flux / dy
                rhs_1[:, :, 0]  -= self.dt * d * 2 * flux / dz
                rhs_1[:, :, -1] += self.dt * d * 2 * flux / dz

            u_star = np.zeros((nx, ny, nz))
            for j in range(ny):
                for k in range(nz):
                    ab_x = np.zeros((3, nx))
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
            rhs_2 = u_star - a_y_un

            u_star_star = np.zeros((nx, ny, nz))
            for i in range(nx):
                for k in range(nz):
                    ab_y = np.zeros((3, ny))
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
            rhs_3 = u_star_star - a_z_un

            u_new = np.zeros((nx, ny, nz))
            for i in range(nx):
                for j in range(ny):
                    ab_z = np.zeros((3, nz))
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

    def _apply_bc_to_sweep(self, matrix, rhs_array: np.ndarray, h: float, dt_sweep: float, t_eval: float = None) -> np.ndarray:
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

    def _apply_bc_to_banded(
        self,
        ab: np.ndarray,
        rhs_array: np.ndarray,
        h: float,
        dt_sweep: float,
        t_eval: float,
        d_face: Optional[np.ndarray] = None,
    ) -> None:
        """Apply boundary conditions to the banded matrix and 1D RHS array in-place."""
        if self._boundary_conditions is None:
            return

        if isinstance(self._boundary_conditions, NeumannBC):
            flux = self._boundary_conditions._get_flux(t_eval)
            if d_face is None:
                d_left = float(self._diffusion_value(t_eval))
                d_right = d_left
            else:
                d_left = float(d_face[0])
                d_right = float(d_face[-1])

            alpha_left = (dt_sweep * d_left) / (h**2)
            alpha_right = (dt_sweep * d_right) / (h**2)
            forcing_left = (2 * dt_sweep * d_left * flux) / h
            forcing_right = (2 * dt_sweep * d_right * flux) / h
            
            # Left Boundary
            ab[0, 1] = -2 * alpha_left
            rhs_array[0] -= forcing_left
            
            # Right Boundary
            ab[2, -2] = -2 * alpha_right
            rhs_array[-1] += forcing_right

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

    def set_boundary_conditions(self, boundary_conditions) -> None:
        super().set_boundary_conditions(boundary_conditions)
        if self._variant in {"bc", "bci", "bcos", "bci_opt"}:
            self._build_system_matrix()

    def set_diffusion_coefficient(self, value: float) -> None:
        super().set_diffusion_coefficient(value)
        self._build_system_matrix()

    def set_decay_rate(self, value: float) -> None:
        super().set_decay_rate(value)
        self._build_system_matrix()


class ADISchema(_ADIUnified):
    """ADI base schema."""

    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0,
    ):
        super().__init__(
            domain_size,
            grid_points,
            dt,
            diffusion_coefficient=diffusion_coefficient,
            decay_rate=decay_rate,
            variant="base",
        )


class ADIBCSchema(_ADIUnified):
    """ADI schema with boundary conditions."""

    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0,
    ):
        super().__init__(
            domain_size,
            grid_points,
            dt,
            diffusion_coefficient=diffusion_coefficient,
            decay_rate=decay_rate,
            variant="bc",
        )


class ADIBCISchema(_ADIUnified):
    """ADI schema with implicit sources and BCs."""

    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0,
    ):
        super().__init__(
            domain_size,
            grid_points,
            dt,
            diffusion_coefficient=diffusion_coefficient,
            decay_rate=decay_rate,
            variant="bci",
        )


class ADIBCOSSchema(_ADIUnified):
    """ADI schema with operator-split sources and BCs."""

    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0,
    ):
        super().__init__(
            domain_size,
            grid_points,
            dt,
            diffusion_coefficient=diffusion_coefficient,
            decay_rate=decay_rate,
            variant="bcos",
        )

class ADIBCIOptSchema(_ADIUnified):
    """ADI schema with implicit sources and BCs (Optimized Banded Solver)."""

    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0,
    ):
        super().__init__(
            domain_size,
            grid_points,
            dt,
            diffusion_coefficient=diffusion_coefficient,
            decay_rate=decay_rate,
            variant="bci_opt",
        )