"""
Implicit Euler with Alternating Direction Implicit (ADI) method.

This module implements the LOD splitting scheme. It reduces multidimensional
problems into a sequence of efficient 1D tridiagonal solves while maintaining
unconditional stability.
"""

import numpy as np
from scipy.sparse import diags, eye, csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve, splu
from scipy.linalg import solve_banded
from diffusion_schemas.base import Schema
from diffusion_schemas.utils.boundary import DirichletBC, NeumannBC


class ImplicitLODBCISchema(Schema):
    """
    Alternating Direction Implicit (ADI) method for the diffusion equation.
    
    This method utilizes operator splitting to solve multidimensional diffusion
    problems as a sequence of 1D problems.
    
    Scheme (First-order splitting):
    (I - dt*Ax) * u* = u^n + dt*S
    (I - dt*Ay) * u** = u*
    (I - dt*Az) * u^(n+1) = u**
    
    Where Ax, Ay, Az are the diffusion/decay operators for each dimension.
    
    Key Features:
    - Unconditionally stable.
    - Solves N decoupled tridiagonal systems (very fast).
    - Boundary conditions are applied integrated into each 1D sweep.
    
    Parameters
    ----------
    domain_size : Union[float, Tuple[float, ...]]
        Size of the domain in each dimension.
    grid_points : Union[int, Tuple[int, ...]]
        Number of grid points in each dimension.
    dt : float
        Time step size.
    diffusion_coefficient : float, optional
        Diffusion coefficient D. Default is 1.0.
    decay_rate : float, optional
        Decay rate λ. Default is 0.0.
    """
    
    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0
    ):
        """Initialize the LOD schema."""
        super().__init__(domain_size, grid_points, dt, diffusion_coefficient, decay_rate)
        
        # Build the system matrices (Ax, Ay, Az)
        # Note: These are small 1D matrices (Nx x Nx, etc.)
        self._build_system_matrix()
        self._boundary_mask = self._compute_boundary_indices()
        # Prepare cached sparse factorizations for LHS matrices where possible.
        # These are reused each time-step to avoid repeated factorization.
        self._prepare_solvers()

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
        """Build the sparse system matrices for the implicit scheme."""
        if self.ndim == 1:
            # For 1D, LOD is just standard Implicit Euler
            self.system_matrix = self._build_matrix_1d()
        elif self.ndim == 2:
            self.system_matrix = self._build_matrix_2d()
        elif self.ndim == 3:
            self.system_matrix = self._build_matrix_3d()
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.ndim}")
    
    def _build_matrix_1d(self) -> csr_matrix:
        """Build 1D system matrix."""
        N = self.grid_points[0]
        dx = self.dx[0]
        
        diag_main = -2 * np.ones(N) / (dx**2)
        diag_off = np.ones(N-1) / (dx**2)
        L = diags([diag_off, diag_main, diag_off], [-1, 0, 1], shape=(N, N), format='csr')
        I = eye(N, format='csr')
        
        return I - self.dt * self.diffusion_coefficient * L + self.dt * self.decay_rate * I
    
    def _build_matrix_2d(self):
        """Build 2D splitting matrices (Ax, Ay)."""
        Nx, Ny = self.grid_points
        dx, dy = self.dx
        factor = 1 / 2  # Split decay term equally
        
        # X-Operator
        diag_main_x = -2 * np.ones(Nx) / (dx**2)
        diag_off_x = np.ones(Nx-1) / (dx**2)
        Lx = diags([diag_off_x, diag_main_x, diag_off_x], [-1, 0, 1], shape=(Nx, Nx), format='csr')
        Ix = eye(Nx, format='csr')
        Ax = Ix - self.dt * self.diffusion_coefficient * Lx + factor * self.dt * self.decay_rate * Ix
        
        # Y-Operator
        diag_main_y = -2 * np.ones(Ny) / (dy**2)
        diag_off_y = np.ones(Ny-1) / (dy**2)
        Ly = diags([diag_off_y, diag_main_y, diag_off_y], [-1, 0, 1], shape=(Ny, Ny), format='csr')
        Iy = eye(Ny, format='csr')
        Ay = Iy - self.dt * self.diffusion_coefficient * Ly + factor * self.dt * self.decay_rate * Iy

        return Ax, Ay
    
    def _build_matrix_3d(self):
        """Build 3D splitting matrices (Ax, Ay, Az)."""
        Nx, Ny, Nz = self.grid_points
        dx, dy, dz = self.dx
        factor = 1 / 3  # Split decay term
        
        # X-Operator
        Lx = diags([np.ones(Nx-1)/dx**2, -2*np.ones(Nx)/dx**2, np.ones(Nx-1)/dx**2], [-1, 0, 1], shape=(Nx, Nx), format='csr')
        Ix = eye(Nx, format='csr')
        Ax = Ix - self.dt * self.diffusion_coefficient * Lx + factor * self.dt * self.decay_rate * Ix
        
        # Y-Operator
        Ly = diags([np.ones(Ny-1)/dy**2, -2*np.ones(Ny)/dy**2, np.ones(Ny-1)/dy**2], [-1, 0, 1], shape=(Ny, Ny), format='csr')
        Iy = eye(Ny, format='csr')
        Ay = Iy - self.dt * self.diffusion_coefficient * Ly + factor * self.dt * self.decay_rate * Iy
        
        # Z-Operator
        Lz = diags([np.ones(Nz-1)/dz**2, -2*np.ones(Nz)/dz**2, np.ones(Nz-1)/dz**2], [-1, 0, 1], shape=(Nz, Nz), format='csr')
        Iz = eye(Nz, format='csr')
        Az = Iz - self.dt * self.diffusion_coefficient * Lz + factor * self.dt * self.decay_rate * Iz
        
        return Ax, Ay, Az

    def _prepare_solvers(self) -> None:
        """Create cached sparse factorizations (SuperLU) for 1D operators.

        This is a conservative, low-risk optimization: cached solvers are used
        only when boundary conditions/source diagonals do not require per-step
        structural changes. If factorization fails, solver attributes are None
        and code falls back to existing behavior.
        """
        # Initialize solver attributes
        self._lu = None
        self._lu_x = None
        self._lu_y = None
        self._lu_z = None

        try:
            # 1D
            if self.ndim == 1:
                A = self.system_matrix
                if hasattr(A, 'tocsc'):
                    self._lu = splu(A.tocsc())

            # 2D
            elif self.ndim == 2:
                Ax, Ay = self.system_matrix
                if hasattr(Ax, 'tocsc'):
                    self._lu_x = splu(Ax.tocsc())
                if hasattr(Ay, 'tocsc'):
                    self._lu_y = splu(Ay.tocsc())

            # 3D
            elif self.ndim == 3:
                Ax, Ay, Az = self.system_matrix
                if hasattr(Ax, 'tocsc'):
                    self._lu_x = splu(Ax.tocsc())
                if hasattr(Ay, 'tocsc'):
                    self._lu_y = splu(Ay.tocsc())
                if hasattr(Az, 'tocsc'):
                    self._lu_z = splu(Az.tocsc())
        except Exception:
            # If anything goes wrong (e.g. matrix singularities), fall back
            # to None so existing spsolve-based code is used.
            self._lu = None
            self._lu_x = None
            self._lu_y = None
            self._lu_z = None

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

    def _apply_bc_to_sweep(self, matrix: csr_matrix, rhs_array: np.ndarray, h: float) -> np.ndarray:
        """
        Apply boundary conditions to the 1D sweep system.
        
        This modifies the matrix (LHS) and rhs_array (RHS) in-place to account
        for the boundary conditions (Neumann Ghost Points or Dirichlet Values).
        
        Parameters
        ----------
        matrix : csr_matrix
            The 1D system matrix (Ax, Ay, or Az) for the current sweep.
        rhs_array : np.ndarray
            The RHS for the solve, shape (N_sweep, N_others).
            Each column represents an independent 1D line being solved.
        h : float
            The grid spacing (dx, dy, or dz) for this dimension.
        """
        if self._boundary_conditions is None:
            return rhs_array

        D = self.diffusion_coefficient
        dt = self.dt
        
        # --- NEUMANN BC ---
        if isinstance(self._boundary_conditions, NeumannBC):
            flux = self._boundary_conditions._get_flux(self.t + self.dt)
            
            # Ghost Point Logic:
            # -alpha * u_{-1} + (1+2alpha)*u_0 - alpha*u_1 = ...
            # Substitute u_{-1} = u_1 - 2*h*flux/D
            # Becomes: (1+2alpha)*u_0 - 2*alpha*u_1 = RHS + forcing
            
            # Calculate alpha (the off-diagonal weight in the matrix)
            # The matrix was built as (I - dt*D*L).
            # L has 1/h^2 on off-diagonals.
            # So off-diagonal weight is: - (dt * D) / h^2
            alpha = (dt * D) / (h**2)
            
            # Forcing term
            # From ghost point: u_{-1} contribution adds (2*dt*D*flux)/h to RHS
            forcing = (2 * dt * D * flux) / h
            
            # Modify Matrix (Left Boundary i=0)
            # We want the (0,1) term to be -2*alpha. Currently it is -alpha.
            matrix[0, 1] = -2 * alpha
            
            # Modify RHS (Left Boundary)
            # Subtract forcing (based on sign convention established in previous implicit code)
            rhs_array[0, :] -= forcing
            
            # Modify Matrix (Right Boundary i=N-1)
            matrix[-1, -2] = -2 * alpha
            
            # Modify RHS (Right Boundary)
            rhs_array[-1, :] += forcing

        # --- DIRICHLET BC ---
        elif isinstance(self._boundary_conditions, DirichletBC):
            val = self._boundary_conditions._get_value(self.t + self.dt)
            
            # Zero out boundary rows
            # Note: This is slow on CSR, but these matrices are small (1D)
            # Efficient way for 1D: set data to 0, diag to 1
            
            # Left Boundary (i=0)
            matrix[0, :] = 0
            matrix[0, 0] = 1
            rhs_array[0, :] = val
            
            # Right Boundary (i=N-1)
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