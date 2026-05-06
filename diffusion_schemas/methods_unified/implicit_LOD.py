"""Unified implicit LOD schemas with variant selection."""

import numpy as np
from scipy.sparse import diags, eye, csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve, splu
from scipy.linalg import solve_banded

from diffusion_schemas.base import Schema
from diffusion_schemas.utils.boundary import DirichletBC, NeumannBC


class _ImplicitLODUnified(Schema):
    """Implicit LOD core with selectable behavior variants."""

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

        if variant in ("bci", "bcos", "bci_opt"):
            self._boundary_mask = self._compute_boundary_indices()

        if variant == "bci_opt":
            self._prepare_solvers()

    def _compute_boundary_indices(self) -> np.ndarray:
        """Precompute boundary mask for the current grid shape."""
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

        return mask

    def _build_system_matrix(self) -> None:
        """Build the sparse system matrices for the implicit scheme."""
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
        nx, ny = self.grid_points
        dx, dy = self.dx
        factor = 1 / 2

        diag_main_x = -2 * np.ones(nx) / (dx**2)
        diag_off_x = np.ones(nx - 1) / (dx**2)
        lx = diags([diag_off_x, diag_main_x, diag_off_x], [-1, 0, 1], shape=(nx, nx), format="csr")
        ix = eye(nx, format="csr")
        ax = ix - self.dt * self.diffusion_coefficient * lx + factor * self.dt * self.decay_rate * ix

        diag_main_y = -2 * np.ones(ny) / (dy**2)
        diag_off_y = np.ones(ny - 1) / (dy**2)
        ly = diags([diag_off_y, diag_main_y, diag_off_y], [-1, 0, 1], shape=(ny, ny), format="csr")
        iy = eye(ny, format="csr")
        ay = iy - self.dt * self.diffusion_coefficient * ly + factor * self.dt * self.decay_rate * iy

        return ax, ay

    def _build_matrix_3d(self):
        nx, ny, nz = self.grid_points
        dx, dy, dz = self.dx
        factor = 1 / 3

        lx = diags(
            [np.ones(nx - 1) / dx**2, -2 * np.ones(nx) / dx**2, np.ones(nx - 1) / dx**2],
            [-1, 0, 1],
            shape=(nx, nx),
            format="csr",
        )
        ix = eye(nx, format="csr")
        ax = ix - self.dt * self.diffusion_coefficient * lx + factor * self.dt * self.decay_rate * ix

        ly = diags(
            [np.ones(ny - 1) / dy**2, -2 * np.ones(ny) / dy**2, np.ones(ny - 1) / dy**2],
            [-1, 0, 1],
            shape=(ny, ny),
            format="csr",
        )
        iy = eye(ny, format="csr")
        ay = iy - self.dt * self.diffusion_coefficient * ly + factor * self.dt * self.decay_rate * iy

        lz = diags(
            [np.ones(nz - 1) / dz**2, -2 * np.ones(nz) / dz**2, np.ones(nz - 1) / dz**2],
            [-1, 0, 1],
            shape=(nz, nz),
            format="csr",
        )
        iz = eye(nz, format="csr")
        az = iz - self.dt * self.diffusion_coefficient * lz + factor * self.dt * self.decay_rate * iz

        return ax, ay, az

    def _prepare_solvers(self) -> None:
        self._lu = None
        self._lu_x = None
        self._lu_y = None
        self._lu_z = None

        try:
            if self.ndim == 1:
                if hasattr(self.system_matrix, 'tocsc'):
                    self._lu = splu(self.system_matrix.tocsc())
            elif self.ndim == 2:
                ax, ay = self.system_matrix
                if hasattr(ax, 'tocsc'):
                    self._lu_x = splu(ax.tocsc())
                if hasattr(ay, 'tocsc'):
                    self._lu_y = splu(ay.tocsc())
            elif self.ndim == 3:
                ax, ay, az = self.system_matrix
                if hasattr(ax, 'tocsc'):
                    self._lu_x = splu(ax.tocsc())
                if hasattr(ay, 'tocsc'):
                    self._lu_y = splu(ay.tocsc())
                if hasattr(az, 'tocsc'):
                    self._lu_z = splu(az.tocsc())
        except Exception:
            self._lu = None
            self._lu_x = None
            self._lu_y = None
            self._lu_z = None

    def step(self) -> None:
        if self._variant == "base":
            self._step_base()
        elif self._variant == "bc":
            self._step_bc()
        elif self._variant == "bci":
            self._step_bci()
        elif self._variant == "bcos":
            self._step_bcos()
        elif self._variant == "bci_opt":
            self._step_bci_opt()        
        else:
            raise ValueError(f"Unsupported implicit LOD variant: {self._variant}")

    def _step_base(self) -> None:
        source = self._compute_source_term()
        rhs = self.state + self.dt * source

        if self.ndim == 1:
            ax = self.system_matrix
            self.state = spsolve(ax, rhs)

        elif self.ndim == 2:
            ax, ay = self.system_matrix
            u_star = spsolve(ax, rhs)
            if self._boundary_conditions is not None:
                u_star = self._apply_boundary_conditions(u_star)
            self.state = spsolve(ay, u_star.T).T

        elif self.ndim == 3:
            ax, ay, az = self.system_matrix
            nx, ny, nz = self.grid_points
            rhs_x = rhs.reshape(nx, ny * nz)
            u_star = spsolve(ax, rhs_x).reshape(nx, ny, nz)
            if self._boundary_conditions is not None:
                u_star = self._apply_boundary_conditions(u_star)

            rhs_y = u_star.transpose(1, 0, 2).reshape(ny, nx * nz)
            u_star_star = spsolve(ay, rhs_y).reshape(ny, nx, nz).transpose(1, 0, 2)
            if self._boundary_conditions is not None:
                u_star_star = self._apply_boundary_conditions(u_star_star)

            rhs_z = u_star_star.transpose(2, 0, 1).reshape(nz, nx * ny)
            self.state = spsolve(az, rhs_z).reshape(nz, nx, ny).transpose(1, 2, 0)

        if self._boundary_conditions is not None:
            self.state = self._apply_boundary_conditions(self.state)

        self.t += self.dt

    def _step_bc(self) -> None:
        source = self._compute_source_term()
        rhs = self.state + self.dt * source

        if self.ndim == 1:
            ax = self.system_matrix.copy().tolil()
            rhs = rhs.reshape(self.grid_points[0], 1)
            rhs = self._apply_bc_to_sweep(ax, rhs, self.dx[0])
            self.state = spsolve(ax.tocsr(), rhs).reshape(self.grid_points)

        elif self.ndim == 2:
            ax, ay = [m.copy().tolil() for m in self.system_matrix]
            nx, ny = self.grid_points

            rhs = self._apply_bc_to_sweep(ax, rhs, self.dx[0])
            u_star = spsolve(ax.tocsr(), rhs)

            rhs_y = u_star.T
            rhs_y = self._apply_bc_to_sweep(ay, rhs_y, self.dx[1])
            u_new_t = spsolve(ay.tocsr(), rhs_y)
            self.state = u_new_t.T

        elif self.ndim == 3:
            ax, ay, az = [m.copy().tolil() for m in self.system_matrix]
            nx, ny, nz = self.grid_points

            rhs_x = rhs.reshape(nx, ny * nz)
            rhs_x = self._apply_bc_to_sweep(ax, rhs_x, self.dx[0])
            u_star = spsolve(ax.tocsr(), rhs_x).reshape(nx, ny, nz)

            rhs_y = u_star.transpose(1, 0, 2).reshape(ny, nx * nz)
            rhs_y = self._apply_bc_to_sweep(ay, rhs_y, self.dx[1])
            u_star_star = spsolve(ay.tocsr(), rhs_y).reshape(ny, nx, nz).transpose(1, 0, 2)

            rhs_z = u_star_star.transpose(2, 0, 1).reshape(nz, nx * ny)
            rhs_z = self._apply_bc_to_sweep(az, rhs_z, self.dx[2])
            self.state = spsolve(az.tocsr(), rhs_z).reshape(nz, nx, ny).transpose(1, 2, 0)

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
                source_lhs[self._boundary_mask] = 0.0

        rhs = self.state.flatten() + self.dt * (source_explicit.flatten() + source_rhs.flatten())

        if self.ndim == 1:
            ax = self.system_matrix.copy().tolil()
            nx = self.grid_points[0]
            rhs = rhs.reshape(nx, 1)
            rhs = self._apply_bc_to_sweep(ax, rhs, self.dx[0])
            source_diag = diags([self.dt * source_lhs], [0], shape=(nx, nx), format="csr")
            self.state = spsolve((ax + source_diag).tocsr(), rhs).reshape(self.grid_points)

        elif self.ndim == 2:
            ax, ay = [m.copy().tolil() for m in self.system_matrix]
            nx, ny = self.grid_points
            rhs_2d = rhs.reshape(nx, ny)

            u_star = np.zeros((nx, ny))
            for j in range(ny):
                source_diag = diags([(self.dt / 2.0) * source_lhs[:, j]], [0], shape=(nx, nx), format="csr")
                ax_j = (ax + source_diag).tolil()
                rhs_j = rhs_2d[:, j].reshape(nx, 1)
                rhs_j = self._apply_bc_to_sweep(ax_j, rhs_j, self.dx[0])
                u_star[:, j] = spsolve(ax_j.tocsr(), rhs_j).flatten()

            u_new = np.zeros((nx, ny))
            for i in range(nx):
                source_diag = diags([(self.dt / 2.0) * source_lhs[i, :]], [0], shape=(ny, ny), format="csr")
                ay_i = (ay + source_diag).tolil()
                rhs_i = u_star[i, :].reshape(ny, 1)
                rhs_i = self._apply_bc_to_sweep(ay_i, rhs_i, self.dx[1])
                u_new[i, :] = spsolve(ay_i.tocsr(), rhs_i).flatten()

            self.state = u_new

        elif self.ndim == 3:
            ax, ay, az = [m.copy().tolil() for m in self.system_matrix]
            nx, ny, nz = self.grid_points
            rhs_x = rhs.reshape(nx, ny, nz)

            u_star = np.zeros((nx, ny, nz))
            for j in range(ny):
                for k in range(nz):
                    source_diag = diags([(self.dt / 3.0) * source_lhs[:, j, k]], [0], shape=(nx, nx), format="csr")
                    ax_jk = (ax + source_diag).tolil()
                    rhs_jk = rhs_x[:, j, k].reshape(nx, 1)
                    rhs_jk = self._apply_bc_to_sweep(ax_jk, rhs_jk, self.dx[0])
                    u_star[:, j, k] = spsolve(ax_jk.tocsr(), rhs_jk).flatten()

            u_star_star = np.zeros((nx, ny, nz))
            for i in range(nx):
                for k in range(nz):
                    source_diag = diags([(self.dt / 3.0) * source_lhs[i, :, k]], [0], shape=(ny, ny), format="csr")
                    ay_ik = (ay + source_diag).tolil()
                    rhs_ik = u_star[i, :, k].reshape(ny, 1)
                    rhs_ik = self._apply_bc_to_sweep(ay_ik, rhs_ik, self.dx[1])
                    u_star_star[i, :, k] = spsolve(ay_ik.tocsr(), rhs_ik).flatten()

            u_new = np.zeros((nx, ny, nz))
            for i in range(nx):
                for j in range(ny):
                    source_diag = diags([(self.dt / 3.0) * source_lhs[i, j, :]], [0], shape=(nz, nz), format="csr")
                    az_ij = (az + source_diag).tolil()
                    rhs_ij = u_star_star[i, j, :].reshape(nz, 1)
                    rhs_ij = self._apply_bc_to_sweep(az_ij, rhs_ij, self.dx[2])
                    u_new[i, j, :] = spsolve(az_ij.tocsr(), rhs_ij).flatten()

            self.state = u_new

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
        rhs = self.state.copy()

        if self.ndim == 1:
            nx = self.grid_points[0]
            ax = self.system_matrix.copy().tolil()
            rhs = rhs.reshape(nx, 1)
            rhs = self._apply_bc_to_sweep(ax, rhs, self.dx[0])
            self.state = spsolve(ax.tocsr(), rhs).reshape(self.grid_points)

        elif self.ndim == 2:
            ax, ay = [m.copy().tolil() for m in self.system_matrix]
            nx, ny = self.grid_points
            rhs_x = rhs.reshape(nx, ny)
            rhs_x = self._apply_bc_to_sweep(ax, rhs_x, self.dx[0])
            u_star = spsolve(ax.tocsr(), rhs_x)

            rhs_y = u_star.T
            rhs_y = self._apply_bc_to_sweep(ay, rhs_y, self.dx[1])
            u_new_t = spsolve(ay.tocsr(), rhs_y)
            self.state = u_new_t.T

        elif self.ndim == 3:
            ax, ay, az = [m.copy().tolil() for m in self.system_matrix]
            nx, ny, nz = self.grid_points
            rhs_x = rhs.reshape(nx, ny * nz)
            rhs_x = self._apply_bc_to_sweep(ax, rhs_x, self.dx[0])
            u_star = spsolve(ax.tocsr(), rhs_x).reshape(nx, ny, nz)

            rhs_y = u_star.transpose(1, 0, 2).reshape(ny, nx * nz)
            rhs_y = self._apply_bc_to_sweep(ay, rhs_y, self.dx[1])
            u_star_star = spsolve(ay.tocsr(), rhs_y).reshape(ny, nx, nz).transpose(1, 0, 2)

            rhs_z = u_star_star.transpose(2, 0, 1).reshape(nz, nx * ny)
            rhs_z = self._apply_bc_to_sweep(az, rhs_z, self.dx[2])
            self.state = spsolve(az.tocsr(), rhs_z).reshape(nz, nx, ny).transpose(1, 2, 0)

    def _step_bulk_sources(self) -> None:
        s_rhs = self._bulk.rhs_contribution.copy()
        s_lhs = self._bulk.lhs_contribution.copy()
        self.state = (self.state + self.dt * s_rhs) / (1.0 + self.dt * s_lhs)

    def _step_agent_sources(self) -> None:
        self.state += self.dt * self.agents_rhs_contribution

    def step(self) -> None:
        """Perform one LOD time step with integrated BCs."""
        t_next = self.t + self.dt
        
        source_explicit = self._compute_source_term(implicit=True, t=t_next)
        source_rhs = np.zeros_like(self.state)
        source_lhs = np.zeros_like(self.state)
        if self._bulk is not None:
            source_rhs = self._bulk.rhs_contribution
            source_lhs = self._bulk.lhs_contribution

            # Only Dirichlet rows are identity-constrained; 
            # keep Neumann boundary source contributions active
            if isinstance(self._boundary_conditions, DirichletBC):
                source_lhs[self._boundary_mask] = 0.0

        # Right-hand side: u^n + dt*(S_explicit + S_rhs)
        rhs = self.state.flatten() + self.dt * (source_explicit.flatten() + source_rhs.flatten())
        
        # --------------------- 1D CASE ---------------------
        if self.ndim == 1:
            Nx = self.grid_points[0]

            has_per_node_source = np.any(source_lhs)
            has_dirichlet = isinstance(self._boundary_conditions, DirichletBC)
            is_neumann = isinstance(self._boundary_conditions, NeumannBC)

            if (not has_per_node_source) and (not has_dirichlet) and getattr(self, '_lu', None) is not None:
                # Solve A * u_new = rhs
                u_new = self._lu.solve(rhs)
                
            else:
                u_new = np.zeros_like((Nx,))
                alpha_x = self.dt * self.diffusion_coefficient / (self.dx[0]**2)
                decay_term = self.dt * self.decay_rate

                # Pre-fetch BC values 
                if is_neumann:
                    bc_val_x = self._boundary_conditions._get_flux(self.t + self.dt)
                    forcing_x = (2 * self.dt * self.diffusion_coefficient * bc_val_x) / self.dx[0]
                elif has_dirichlet:
                    bc_val_x = self._boundary_conditions._get_value(self.t + self.dt)

                # Build banded matrix (3, Nx)
                ab_x = np.zeros((3, Nx))
                ab_x[0, 1:] = -alpha_x
                ab_x[2, :-1] = -alpha_x
                ab_x[1, :] = 1.0 + 2*alpha_x + decay_term + self.dt * source_lhs.flatten() # Main diagonal
                rhs_copy = rhs.copy()

                # Apply Boundary Conditions
                if is_neumann:
                    ab_x[0, 1] = -2 * alpha_x
                    ab_x[2, -2] = -2 * alpha_x
                    rhs_copy[0] -= forcing_x
                    rhs_copy[-1] += forcing_x
                elif has_dirichlet:
                    ab_x[1, 0] = 1.0; ab_x[0, 1] = 0.0; rhs_copy[0] = bc_val_x
                    ab_x[1, -1] = 1.0; ab_x[2, -2] = 0.0; rhs_copy[-1] = bc_val_x

                u_star[:, j] = solve_banded((1, 1), ab_x, rhs_copy)

        # --------------------- 2D CASE ---------------------
        elif self.ndim == 2:
                         
            Nx, Ny = self.grid_points
            rhs_2d = rhs.reshape(Nx, Ny)
            
            has_per_node_source = np.any(source_lhs)
            has_dirichlet = isinstance(self._boundary_conditions, DirichletBC)
            is_neumann = isinstance(self._boundary_conditions, NeumannBC)

            # --- SWEEP 1: X-Direction ---
            if (not has_per_node_source) and (not has_dirichlet) and getattr(self, '_lu_x', None) is not None:
                # Solve Ax * U = rhs_2d (all columns at once)
                u_star = self._lu_x.solve(rhs_2d)
            else:
                u_star = np.zeros((Nx, Ny))
                alpha_x = self.dt * self.diffusion_coefficient / (self.dx[0]**2)
                decay_term = 0.5 * self.dt * self.decay_rate
                
                # Pre-fetch BC values 
                if is_neumann:
                    bc_val_x = self._boundary_conditions._get_flux(self.t + self.dt)
                    forcing_x = (2 * self.dt * self.diffusion_coefficient * bc_val_x) / self.dx[0]
                elif has_dirichlet:
                    bc_val_x = self._boundary_conditions._get_value(self.t + self.dt)

                for j in range(Ny):
                    # 1. Build banded matrix (3, Nx)
                    ab_x = np.zeros((3, Nx))
                    ab_x[0, 1:] = -alpha_x  # Upper
                    ab_x[2, :-1] = -alpha_x # Lower
                    ab_x[1, :] = 1.0 + 2*alpha_x + decay_term + (self.dt / 2.0) * source_lhs[:, j] # Main
                    
                    rhs_j = rhs_2d[:, j].copy()

                    # 2. Apply Boundary Conditions
                    if is_neumann:
                        ab_x[0, 1] = -2 * alpha_x
                        ab_x[2, -2] = -2 * alpha_x
                        rhs_j[0] -= forcing_x
                        rhs_j[-1] += forcing_x
                    elif has_dirichlet:
                        ab_x[1, 0] = 1.0; ab_x[0, 1] = 0.0; rhs_j[0] = bc_val_x
                        ab_x[1, -1] = 1.0; ab_x[2, -2] = 0.0; rhs_j[-1] = bc_val_x

                    # 3. Solve and store
                    u_star[:, j] = solve_banded((1, 1), ab_x, rhs_j)

            # --- SWEEP 2: Y-Direction ---
            if (not has_per_node_source) and (not has_dirichlet) and getattr(self, '_lu_y', None) is not None:
                # Solve Ay * U_y = u_star.T  -> result has shape (Ny, Nx)
                u_new_T = self._lu_y.solve(u_star.T)
                u_new = u_new_T.T
            else:
                u_new = np.zeros((Nx, Ny))
                alpha_y = self.dt * self.diffusion_coefficient / (self.dx[1]**2)
                
                # Pre-fetch BC values 
                if is_neumann:
                    bc_val_y = self._boundary_conditions._get_flux(self.t + self.dt)
                    forcing_y = (2 * self.dt * self.diffusion_coefficient * bc_val_y) / self.dx[1]
                elif has_dirichlet:
                    bc_val_y = self._boundary_conditions._get_value(self.t + self.dt)

                for i in range(Nx):
                    # 1. Build banded matrix (3, Ny)
                    ab_y = np.zeros((3, Ny))
                    ab_y[0, 1:] = -alpha_y
                    ab_y[2, :-1] = -alpha_y
                    ab_y[1, :] = 1.0 + 2*alpha_y + decay_term + (self.dt / 2.0) * source_lhs[i, :]
                    
                    rhs_i = u_star[i, :].copy()

                    # 2. Apply Boundary Conditions
                    if is_neumann:
                        ab_y[0, 1] = -2 * alpha_y
                        ab_y[2, -2] = -2 * alpha_y
                        rhs_i[0] -= forcing_y
                        rhs_i[-1] += forcing_y
                    elif has_dirichlet:
                        ab_y[1, 0] = 1.0; ab_y[0, 1] = 0.0; rhs_i[0] = bc_val_y
                        ab_y[1, -1] = 1.0; ab_y[2, -2] = 0.0; rhs_i[-1] = bc_val_y

                    # 3. Solve and store
                    u_new[i, :] = solve_banded((1, 1), ab_y, rhs_i)

            self.state = u_new

        # --------------------- 3D CASE ---------------------
        elif self.ndim == 3:
            Nx, Ny, Nz = self.grid_points
            rhs_x = rhs.reshape(Nx, Ny, Nz)

            has_per_node_source = np.any(source_lhs)
            has_dirichlet = isinstance(self._boundary_conditions, DirichletBC)
            is_neumann = isinstance(self._boundary_conditions, NeumannBC)
            
            decay_term = self.dt * self.decay_rate / 3.0

            # --- SWEEP 1: X-Direction ---
            if (not has_per_node_source) and (not has_dirichlet) and getattr(self, '_lu_x', None) is not None:
                rhs_x_flat = rhs_x.reshape(Nx, Ny * Nz)
                u_star_flat = self._lu_x.solve(rhs_x_flat)
                u_star = u_star_flat.reshape(Nx, Ny, Nz)
            else:
                u_star = np.zeros((Nx, Ny, Nz))
                alpha_x = self.dt * self.diffusion_coefficient / (self.dx[0]**2)
                
                if is_neumann:
                    bc_val_x = self._boundary_conditions._get_flux(self.t + self.dt)
                    forcing_x = (2 * self.dt * self.diffusion_coefficient * bc_val_x) / self.dx[0]
                elif has_dirichlet:
                    bc_val_x = self._boundary_conditions._get_value(self.t + self.dt)

                for j in range(Ny):
                    for k in range(Nz):
                        ab_x = np.zeros((3, Nx))
                        ab_x[0, 1:] = -alpha_x
                        ab_x[2, :-1] = -alpha_x
                        ab_x[1, :] = 1.0 + 2*alpha_x + decay_term + (self.dt / 3.0) * source_lhs[:, j, k]
                        
                        rhs_jk = rhs_x[:, j, k].copy()

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
                alpha_y = self.dt * self.diffusion_coefficient / (self.dx[1]**2)
                
                if is_neumann:
                    bc_val_y = self._boundary_conditions._get_flux(self.t + self.dt)
                    forcing_y = (2 * self.dt * self.diffusion_coefficient * bc_val_y) / self.dx[1]
                elif has_dirichlet:
                    bc_val_y = self._boundary_conditions._get_value(self.t + self.dt)

                for i in range(Nx):
                    for k in range(Nz):
                        ab_y = np.zeros((3, Ny))
                        ab_y[0, 1:] = -alpha_y
                        ab_y[2, :-1] = -alpha_y
                        ab_y[1, :] = 1.0 + 2*alpha_y + decay_term + (self.dt / 3.0) * source_lhs[i, :, k]
                        
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
                u_new = u_final_flat.reshape(Nz, Nx, Ny).transpose(1, 2, 0)
            else:
                u_new = np.zeros((Nx, Ny, Nz))
                alpha_z = self.dt * self.diffusion_coefficient / (self.dx[2]**2)
                
                if is_neumann:
                    bc_val_z = self._boundary_conditions._get_flux(self.t + self.dt)
                    forcing_z = (2 * self.dt * self.diffusion_coefficient * bc_val_z) / self.dx[2]
                elif has_dirichlet:
                    bc_val_z = self._boundary_conditions._get_value(self.t + self.dt)

                for i in range(Nx):
                    for j in range(Ny):
                        ab_z = np.zeros((3, Nz))
                        ab_z[0, 1:] = -alpha_z
                        ab_z[2, :-1] = -alpha_z
                        ab_z[1, :] = 1.0 + 2*alpha_z + decay_term + (self.dt / 3.0) * source_lhs[i, j, :]
                        
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
        
        # Update time
        self.t += self.dt

    def _apply_bc_to_sweep(self, matrix: csr_matrix, rhs_array: np.ndarray, h: float, apply_dirichlet: bool = True) -> np.ndarray:
        if self._boundary_conditions is None:
            return rhs_array

        d = self.diffusion_coefficient
        dt = self.dt

        if isinstance(self._boundary_conditions, NeumannBC):
            flux = self._boundary_conditions._get_flux(self.t + self.dt)
            alpha = (dt * d) / (h**2)
            forcing = (2 * dt * d * flux) / h

            matrix[0, 1] = -2 * alpha
            rhs_array[0, :] -= forcing

            matrix[-1, -2] = -2 * alpha
            rhs_array[-1, :] += forcing

        elif isinstance(self._boundary_conditions, DirichletBC) and apply_dirichlet:
            val = self._boundary_conditions._get_value(self.t + self.dt)
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


class ImplicitLODSchema(_ImplicitLODUnified):
    """Implicit LOD method for the diffusion equation."""

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


class ImplicitLODBCSchema(_ImplicitLODUnified):
    """Implicit LOD method with boundary conditions."""

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


class ImplicitLODBCISchema(_ImplicitLODUnified):
    """Implicit LOD method with implicit sources and BCs."""

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


class ImplicitLODBCOSSchema(_ImplicitLODUnified):
    """Implicit LOD method with operator-split sources and BCs."""

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

class ImplicitLODBCIOptSchema(_ImplicitLODUnified):
    """Implicit LOD method with implicit sources, BCs, and optimized solvers."""

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