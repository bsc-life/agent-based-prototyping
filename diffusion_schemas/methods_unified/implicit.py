"""Unified implicit Euler schemas with variant selection."""

import numpy as np
from scipy.sparse import diags, kron, eye, csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from diffusion_schemas.base import Schema
from diffusion_schemas.utils.boundary import DirichletBC, NeumannBC


class _ImplicitEulerUnified(Schema):
    """Implicit Euler core with selectable behavior variants."""

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

        if variant in ("bci", "bcos"):
            self._boundary_idx = self._compute_boundary_indices()

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

        return np.flatnonzero(mask.ravel())

    def _build_system_matrix(self) -> None:
        """Build the sparse system matrix for the implicit scheme."""
        if self.ndim == 1:
            self.system_matrix = self._build_matrix_1d()
        elif self.ndim == 2:
            self.system_matrix = self._build_matrix_2d()
        elif self.ndim == 3:
            self.system_matrix = self._build_matrix_3d()
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.ndim}")

        self._diffusion_dirty = False

    def _ensure_system_matrix_current(self) -> None:
        if self.diffusion_is_time_dependent or self._diffusion_dirty:
            self._build_system_matrix()

    def _build_conservative_operator_1d(self, d_field: np.ndarray) -> csr_matrix:
        n = self.grid_points[0]
        dx = self.dx[0]
        d_face = 0.5 * (d_field[:-1] + d_field[1:])

        lower = d_face / (dx**2)
        upper = d_face / (dx**2)

        diag = np.zeros(n)
        diag[1:-1] = -(d_face[1:] + d_face[:-1]) / (dx**2)
        diag[0] = -2 * d_face[0] / (dx**2)
        diag[-1] = -2 * d_face[-1] / (dx**2)

        return diags([lower, diag, upper], [-1, 0, 1], shape=(n, n), format="csr")

    def _build_conservative_operator_2d(self, d_field: np.ndarray) -> csr_matrix:
        nx, ny = self.grid_points
        dx, dy = self.dx
        d_x, d_y = self._diffusion_faces(d_field)

        rows = []
        cols = []
        data = []

        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j

                d_x_minus = d_x[i - 1, j] if i > 0 else d_x[0, j]
                d_x_plus = d_x[i, j] if i < nx - 1 else d_x[-1, j]
                d_y_minus = d_y[i, j - 1] if j > 0 else d_y[i, 0]
                d_y_plus = d_y[i, j] if j < ny - 1 else d_y[i, -1]

                diag = -(
                    (d_x_minus + d_x_plus) / (dx**2)
                    + (d_y_minus + d_y_plus) / (dy**2)
                )
                rows.append(idx)
                cols.append(idx)
                data.append(diag)

                if i > 0:
                    rows.append(idx)
                    cols.append(idx - ny)
                    data.append(d_x_minus / (dx**2))
                if i < nx - 1:
                    rows.append(idx)
                    cols.append(idx + ny)
                    data.append(d_x_plus / (dx**2))
                if j > 0:
                    rows.append(idx)
                    cols.append(idx - 1)
                    data.append(d_y_minus / (dy**2))
                if j < ny - 1:
                    rows.append(idx)
                    cols.append(idx + 1)
                    data.append(d_y_plus / (dy**2))

        return csr_matrix((data, (rows, cols)), shape=(nx * ny, nx * ny))

    def _build_conservative_operator_3d(self, d_field: np.ndarray) -> csr_matrix:
        nx, ny, nz = self.grid_points
        dx, dy, dz = self.dx
        d_x, d_y, d_z = self._diffusion_faces(d_field)

        rows = []
        cols = []
        data = []

        stride_y = nz
        stride_x = ny * nz

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    idx = (i * ny + j) * nz + k

                    d_x_minus = d_x[i - 1, j, k] if i > 0 else d_x[0, j, k]
                    d_x_plus = d_x[i, j, k] if i < nx - 1 else d_x[-1, j, k]
                    d_y_minus = d_y[i, j - 1, k] if j > 0 else d_y[i, 0, k]
                    d_y_plus = d_y[i, j, k] if j < ny - 1 else d_y[i, -1, k]
                    d_z_minus = d_z[i, j, k - 1] if k > 0 else d_z[i, j, 0]
                    d_z_plus = d_z[i, j, k] if k < nz - 1 else d_z[i, j, -1]

                    diag = -(
                        (d_x_minus + d_x_plus) / (dx**2)
                        + (d_y_minus + d_y_plus) / (dy**2)
                        + (d_z_minus + d_z_plus) / (dz**2)
                    )
                    rows.append(idx)
                    cols.append(idx)
                    data.append(diag)

                    if i > 0:
                        rows.append(idx)
                        cols.append(idx - stride_x)
                        data.append(d_x_minus / (dx**2))
                    if i < nx - 1:
                        rows.append(idx)
                        cols.append(idx + stride_x)
                        data.append(d_x_plus / (dx**2))
                    if j > 0:
                        rows.append(idx)
                        cols.append(idx - stride_y)
                        data.append(d_y_minus / (dy**2))
                    if j < ny - 1:
                        rows.append(idx)
                        cols.append(idx + stride_y)
                        data.append(d_y_plus / (dy**2))
                    if k > 0:
                        rows.append(idx)
                        cols.append(idx - 1)
                        data.append(d_z_minus / (dz**2))
                    if k < nz - 1:
                        rows.append(idx)
                        cols.append(idx + 1)
                        data.append(d_z_plus / (dz**2))

        return csr_matrix((data, (rows, cols)), shape=(nx * ny * nz, nx * ny * nz))

    def _build_matrix_1d(self) -> csr_matrix:
        """Build the 1D system matrix (I - dt*D*L + dt*lambda*I)."""
        n = self.grid_points[0]
        dx = self.dx[0]

        if not self.diffusion_is_scalar():
            d_field = self._diffusion_field()
            l = self._build_conservative_operator_1d(d_field)
            i = eye(n, format="csr")
            return i - self.dt * l + self.dt * self.decay_rate * i

        d = float(self._diffusion_value())

        diag_main = -2 * np.ones(n) / (dx**2)
        diag_off = np.ones(n - 1) / (dx**2)

        l = diags([diag_off, diag_main, diag_off], [-1, 0, 1], shape=(n, n), format="csr")
        i = eye(n, format="csr")

        return i - self.dt * d * l + self.dt * self.decay_rate * i

    def _build_matrix_2d(self) -> csr_matrix:
        """Build the 2D system matrix using Kronecker products."""
        nx, ny = self.grid_points
        dx, dy = self.dx

        if not self.diffusion_is_scalar():
            d_field = self._diffusion_field()
            l = self._build_conservative_operator_2d(d_field)
            i = eye(nx * ny, format="csr")
            return i - self.dt * l + self.dt * self.decay_rate * i

        d = float(self._diffusion_value())

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

        return i - self.dt * d * l + self.dt * self.decay_rate * i

    def _build_matrix_3d(self) -> csr_matrix:
        """Build the 3D system matrix using Kronecker products."""
        nx, ny, nz = self.grid_points
        dx, dy, dz = self.dx

        if not self.diffusion_is_scalar():
            d_field = self._diffusion_field()
            l = self._build_conservative_operator_3d(d_field)
            i = eye(nx * ny * nz, format="csr")
            return i - self.dt * l + self.dt * self.decay_rate * i

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

        l = kron(kron(lx, iy), iz) + kron(kron(ix, ly), iz) + kron(kron(ix, iy), lz)
        i = eye(nx * ny * nz, format="csr")

        return i - self.dt * d * l + self.dt * self.decay_rate * i

    def step(self) -> None:
        self._ensure_system_matrix_current()
        if self._variant == "base":
            self._step_base()
        elif self._variant == "bc":
            self._step_bc()
        elif self._variant == "bci":
            self._step_bci()
        elif self._variant == "bcos":
            self._step_bcos()
        else:
            raise ValueError(f"Unsupported implicit variant: {self._variant}")

    def _step_base(self) -> None:
        source = self._compute_source_term()
        rhs = self.state.flatten() + self.dt * source.flatten()

        u_new_flat = spsolve(self.system_matrix, rhs)
        self.state = u_new_flat.reshape(self.grid_points)

        if self._boundary_conditions is not None:
            self.state = self._apply_boundary_conditions(self.state)

        self.t += self.dt

    def _step_bc(self) -> None:
        source = self._compute_source_term()
        rhs = self.state.flatten() + self.dt * source.flatten()

        if isinstance(self._boundary_conditions, DirichletBC):
            rhs = self._apply_dirichlet_bc_simple(rhs)
        elif isinstance(self._boundary_conditions, NeumannBC):
            rhs = self._apply_neumann_bc_simple(rhs)

        u_new_flat = spsolve(self.system_matrix, rhs)
        self.state = u_new_flat.reshape(self.grid_points)
        self.t += self.dt

    def _step_bci(self) -> None:
        t_next = self.t + self.dt

        source_explicit = self._compute_source_term(implicit=True, t=t_next)
        source_rhs = np.zeros_like(self.state)
        source_lhs = np.zeros_like(self.state)
        if self._bulk is not None:
            source_rhs = self._bulk.rhs_contribution
            source_lhs = self._bulk.lhs_contribution

            if isinstance(self._boundary_conditions, DirichletBC):
                source_lhs[self._boundary_idx] = 0.0

        rhs = self.state.flatten() + self.dt * (source_explicit.flatten() + source_rhs.flatten())

        if self._bulk is not None:
            lhs = source_lhs.ravel().copy()

            if isinstance(self._boundary_conditions, DirichletBC):
                lhs[self._boundary_idx] = 0.0

            system_matrix = self.system_matrix + diags(self.dt * lhs, 0, format="csr")
        else:
            system_matrix = self.system_matrix

        if isinstance(self._boundary_conditions, DirichletBC):
            rhs = self._apply_dirichlet_bc_simple(rhs)
        elif isinstance(self._boundary_conditions, NeumannBC):
            rhs = self._apply_neumann_bc_simple(rhs)

        u_new_flat = spsolve(system_matrix, rhs)
        self.state = u_new_flat.reshape(self.grid_points)
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
            flat_state = self.state.flatten()
            flat_state[self._boundary_idx] = value
            self.state = flat_state.reshape(self.grid_points)
            
        self.t += self.dt

    def _step_diffusion_decay(self) -> None:
        rhs = self.state.flatten()

        if isinstance(self._boundary_conditions, DirichletBC):
            rhs, _ = self._apply_dirichlet_bc(rhs)
        elif isinstance(self._boundary_conditions, NeumannBC):
            rhs, _ = self._apply_neumann_bc(rhs)

        u_new_flat = spsolve(self.system_matrix, rhs)
        self.state = u_new_flat.reshape(self.grid_points)

    def _step_bulk_sources(self) -> None:
        s_rhs = self._bulk.rhs_contribution.copy()
        s_lhs = self._bulk.lhs_contribution.copy()

        self.state = (self.state + self.dt * s_rhs) / (1.0 + self.dt * s_lhs)

    def _step_agent_sources(self) -> None:
        self.state += self.dt * self.agents_rhs_contribution

    def _apply_dirichlet_bc_simple(self, rhs):
        value = self._boundary_conditions._get_value(self.t)

        if self.ndim == 1:
            rhs[0] = value
            rhs[-1] = value

        if self.ndim == 2:
            nx, ny = self.grid_points

            for j in range(ny):
                idx = j * nx
                rhs[idx] = value

                idx2 = j * nx + (nx - 1)
                rhs[idx2] = value

            for i in range(nx):
                idx = i
                rhs[idx] = value

                idx2 = (ny - 1) * nx + i
                rhs[idx2] = value

        if self.ndim == 3:
            nx, ny, nz = self.grid_points

            for j in range(ny):
                for k in range(nz):
                    idx_left = 0 * ny * nz + j * nz + k
                    idx_right = (nx - 1) * ny * nz + j * nz + k
                    rhs[idx_left] = value
                    rhs[idx_right] = value

            for i in range(nx):
                for k in range(nz):
                    idx_bottom = i * ny * nz + 0 * nz + k
                    idx_top = i * ny * nz + (ny - 1) * nz + k
                    rhs[idx_bottom] = value
                    rhs[idx_top] = value

            for i in range(nx):
                for j in range(ny):
                    idx_front = i * ny * nz + j * nz + 0
                    idx_back = i * ny * nz + j * nz + (nz - 1)
                    rhs[idx_front] = value
                    rhs[idx_back] = value

        return rhs

    def _apply_neumann_bc_simple(self, rhs):
        flux = self._boundary_conditions._get_flux(self.t + self.dt)
        d = self.diffusion_coefficient
        dt = self.dt

        if self.ndim == 1:
            dx = self.dx[0]
            forcing = (2 * dt * d * flux) / dx
            rhs[0] -= forcing
            rhs[-1] += forcing

        elif self.ndim == 2:
            nx, ny = self.grid_points
            dx, dy = self.dx
            force_x = (2 * dt * d * flux) / dx
            force_y = (2 * dt * d * flux) / dy

            idx_l = np.arange(ny)
            idx_r = np.arange((nx - 1) * ny, nx * ny)
            rhs[idx_l] -= force_x
            rhs[idx_r] += force_x

            idx_b = np.arange(0, nx * ny, ny)
            idx_t = np.arange(ny - 1, nx * ny, ny)
            rhs[idx_b] -= force_y
            rhs[idx_t] += force_y

        elif self.ndim == 3:
            nx, ny, nz = self.grid_points
            dx, dy, dz = self.dx
            sx, sy = ny * nz, nz
            fx, fy, fz = (2 * dt * d * flux) / dx, (2 * dt * d * flux) / dy, (2 * dt * d * flux) / dz

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

    def _apply_dirichlet_bc(self, rhs, matrix=None):
        value = self._boundary_conditions._get_value(self.t)

        if matrix is not None:
            a = matrix.tolil(copy=True)
        else:
            a = None

        if self.ndim == 1:
            rhs[0] = value
            if a is not None:
                a[0, :] = 0
                a[0, 0] = 1
            rhs[-1] = value
            if a is not None:
                a[-1, :] = 0
                a[-1, -1] = 1

        if self.ndim == 2:
            nx, ny = self.grid_points

            for j in range(ny):
                idx = j * nx
                rhs[idx] = value
                if a is not None:
                    a[idx, :] = 0
                    a[idx, idx] = 1

                idx2 = j * nx + (nx - 1)
                rhs[idx2] = value
                if a is not None:
                    a[idx2, :] = 0
                    a[idx2, idx2] = 1

            for i in range(nx):
                idx = i
                rhs[idx] = value
                if a is not None:
                    a[idx, :] = 0
                    a[idx, idx] = 1

                idx2 = (ny - 1) * nx + i
                rhs[idx2] = value
                if a is not None:
                    a[idx2, :] = 0
                    a[idx2, idx2] = 1

        if self.ndim == 3:
            nx, ny, nz = self.grid_points

            for j in range(ny):
                for k in range(nz):
                    idx_left = 0 * ny * nz + j * nz + k
                    idx_right = (nx - 1) * ny * nz + j * nz + k
                    rhs[idx_left] = value
                    if a is not None:
                        a[idx_left, :] = 0
                        a[idx_left, idx_left] = 1
                    rhs[idx_right] = value
                    if a is not None:
                        a[idx_right, :] = 0
                        a[idx_right, idx_right] = 1

            for i in range(nx):
                for k in range(nz):
                    idx_bottom = i * ny * nz + 0 * nz + k
                    idx_top = i * ny * nz + (ny - 1) * nz + k
                    if a is not None:
                        a[idx_bottom, :] = 0
                        a[idx_bottom, idx_bottom] = 1
                    rhs[idx_bottom] = value
                    if a is not None:
                        a[idx_top, :] = 0
                        a[idx_top, idx_top] = 1
                    rhs[idx_top] = value

            for i in range(nx):
                for j in range(ny):
                    idx_front = i * ny * nz + j * nz + 0
                    idx_back = i * ny * nz + j * nz + (nz - 1)
                    if a is not None:
                        a[idx_front, :] = 0
                        a[idx_front, idx_front] = 1
                    rhs[idx_front] = value
                    if a is not None:
                        a[idx_back, :] = 0
                        a[idx_back, idx_back] = 1
                    rhs[idx_back] = value

        if a is not None:
            matrix = a.tocsr()

        return rhs, matrix

    def _apply_neumann_bc(self, rhs, matrix=None):
        if matrix is not None:
            a = matrix.tolil(copy=True)
        else:
            a = None

        flux = self._boundary_conditions._get_flux(self.t + self.dt)
        d = self.diffusion_coefficient
        dt = self.dt

        if self.ndim == 1:
            dx = self.dx[0]
            alpha = (dt * d) / (dx**2)
            forcing = (2 * dt * d * flux) / dx

            if a is not None:
                a[0, 1] = -2 * alpha
            rhs[0] -= forcing

            if a is not None:
                a[-1, -2] = -2 * alpha
            rhs[-1] += forcing

        elif self.ndim == 2:
            nx, ny = self.grid_points
            dx, dy = self.dx
            alpha_x = (dt * d) / (dx**2)
            alpha_y = (dt * d) / (dy**2)
            force_x = (2 * dt * d * flux) / dx
            force_y = (2 * dt * d * flux) / dy

            idx_l = np.arange(ny)
            idx_r = np.arange((nx - 1) * ny, nx * ny)
            if a is not None:
                a[idx_l, idx_l + ny] = -2 * alpha_x
                a[idx_r, idx_r - ny] = -2 * alpha_x
            rhs[idx_l] -= force_x
            rhs[idx_r] += force_x

            idx_b = np.arange(0, nx * ny, ny)
            idx_t = np.arange(ny - 1, nx * ny, ny)
            if a is not None:
                a[idx_b, idx_b + 1] = -2 * alpha_y
                a[idx_t, idx_t - 1] = -2 * alpha_y
            rhs[idx_b] -= force_y
            rhs[idx_t] += force_y

        elif self.ndim == 3:
            nx, ny, nz = self.grid_points
            dx, dy, dz = self.dx
            sx, sy = ny * nz, nz
            fx, fy, fz = (2 * dt * d * flux) / dx, (2 * dt * d * flux) / dy, (2 * dt * d * flux) / dz
            alpha_x, alpha_y, alpha_z = (dt * d) / dx**2, (dt * d) / dy**2, (dt * d) / dz**2

            idx_l, idx_r = np.arange(sx), np.arange((nx - 1) * sx, nx * sx)
            if a is not None:
                a[idx_l, idx_l + sx] = -2 * alpha_x
                a[idx_r, idx_r - sx] = -2 * alpha_x
            rhs[idx_l] -= fx
            rhs[idx_r] += fx

            base_y = np.arange(nz)
            idx_f = np.concatenate([base_y + i * sx for i in range(nx)])
            idx_bk = idx_f + (ny - 1) * sy
            if a is not None:
                a[idx_f, idx_f + sy] = -2 * alpha_y
                a[idx_bk, idx_bk - sy] = -2 * alpha_y
            rhs[idx_f] -= fy
            rhs[idx_bk] += fy

            idx_bt = np.arange(0, nx * ny * nz, nz)
            idx_tp = np.arange(nz - 1, nx * ny * nz, nz)
            if a is not None:
                a[idx_bt, idx_bt + 1] = -2 * alpha_z
                a[idx_tp, idx_tp - 1] = -2 * alpha_z
            rhs[idx_bt] -= fz
            rhs[idx_tp] += fz

        if a is not None:
            matrix = a.tocsr()

        return rhs, matrix

    def set_boundary_conditions(self, boundary_conditions) -> None:
        super().set_boundary_conditions(boundary_conditions)

        if not isinstance(boundary_conditions, (DirichletBC, NeumannBC)):
            raise ValueError("Boundary conditions must be either DirichletBC or NeumannBC.")

        a = self.system_matrix.copy().tolil()

        if isinstance(boundary_conditions, DirichletBC):
            if self.ndim == 1:
                a[0, :] = 0
                a[0, 0] = 1
                a[-1, :] = 0
                a[-1, -1] = 1

            if self.ndim == 2:
                nx, ny = self.grid_points
                for j in range(ny):
                    idx = j * nx
                    a[idx, :] = 0
                    a[idx, idx] = 1

                    idx2 = j * nx + (nx - 1)
                    a[idx2, :] = 0
                    a[idx2, idx2] = 1

                for i in range(nx):
                    idx = i
                    a[idx, :] = 0
                    a[idx, idx] = 1

                    idx2 = (ny - 1) * nx + i
                    a[idx2, :] = 0
                    a[idx2, idx2] = 1

            if self.ndim == 3:
                nx, ny, nz = self.grid_points
                for j in range(ny):
                    for k in range(nz):
                        idx_left = 0 * ny * nz + j * nz + k
                        idx_right = (nx - 1) * ny * nz + j * nz + k
                        a[idx_left, :] = 0
                        a[idx_left, idx_left] = 1
                        a[idx_right, :] = 0
                        a[idx_right, idx_right] = 1

                for i in range(nx):
                    for k in range(nz):
                        idx_bottom = i * ny * nz + 0 * nz + k
                        idx_top = i * ny * nz + (ny - 1) * nz + k
                        a[idx_bottom, :] = 0
                        a[idx_bottom, idx_bottom] = 1
                        a[idx_top, :] = 0
                        a[idx_top, idx_top] = 1

                for i in range(nx):
                    for j in range(ny):
                        idx_front = i * ny * nz + j * nz + 0
                        idx_back = i * ny * nz + j * nz + (nz - 1)
                        a[idx_front, :] = 0
                        a[idx_front, idx_front] = 1
                        a[idx_back, :] = 0
                        a[idx_back, idx_back] = 1

        elif isinstance(boundary_conditions, NeumannBC):
            d = self.diffusion_coefficient
            dt = self.dt

            if self.ndim == 1:
                dx = self.dx[0]
                alpha = (dt * d) / (dx**2)
                a[0, 1] = -2 * alpha
                a[-1, -2] = -2 * alpha

            elif self.ndim == 2:
                nx, ny = self.grid_points
                dx, dy = self.dx
                alpha_x = (dt * d) / (dx**2)
                alpha_y = (dt * d) / (dy**2)

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
                alpha_x, alpha_y, alpha_z = (dt * d) / dx**2, (dt * d) / dy**2, (dt * d) / dz**2

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

        self.system_matrix = a.tocsr()

    def set_diffusion_coefficient(self, value: float) -> None:
        super().set_diffusion_coefficient(value)
        self._build_system_matrix()

    def set_decay_rate(self, value: float) -> None:
        super().set_decay_rate(value)
        self._build_system_matrix()


class ImplicitEulerSchema(_ImplicitEulerUnified):
    """Implicit Euler method for the diffusion equation."""

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


class ImplicitEulerBCSchema(_ImplicitEulerUnified):
    """Implicit Euler method with boundary conditions."""

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


class ImplicitEulerBCISchema(_ImplicitEulerUnified):
    """Implicit Euler method with implicit sources and BCs."""

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


class ImplicitEulerBCOSSchema(_ImplicitEulerUnified):
    """Implicit Euler method with operator-split sources and BCs."""

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
