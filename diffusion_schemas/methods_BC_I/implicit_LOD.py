"""
Implicit Euler with Alternating Direction Implicit (ADI) method.

This module implements the LOD splitting scheme. It reduces multidimensional
problems into a sequence of efficient 1D tridiagonal solves while maintaining
unconditional stability.
"""

import numpy as np
from scipy.sparse import diags, eye, csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
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

    def step(self) -> None:
        """Perform one LOD time step with integrated BCs."""
        t_next = self.t + self.dt
        
        source_explicit = self._compute_source_term(implicit=True, t=t_next)
        source_rhs = np.zeros_like(self.state)
        source_lhs = np.zeros_like(self.state)
        if self._bulk is not None:
            source_rhs = self._bulk.rhs_contribution
            source_lhs = self._bulk.lhs_contribution

            source_lhs[self._boundary_mask] = 0.0

        # Right-hand side: u^n + dt*(S_explicit + S_rhs)
        rhs = self.state.flatten() + self.dt * (source_explicit.flatten() + source_rhs.flatten())
        
        # --------------------- 1D CASE ---------------------
        if self.ndim == 1:
            Ax = self.system_matrix.copy().tolil() # LIL for efficient BC modification
            Nx = self.grid_points[0]
            
            # Apply BC to the single system
            rhs = rhs.reshape(Nx, 1) # Ensure 2D column for helper
            rhs = self._apply_bc_to_sweep(Ax, rhs, self.dx[0])

            source_diag = diags([self.dt * source_lhs], [0], shape=(Nx,Nx), format='csr')
            self.state = spsolve((Ax + source_diag).tocsr(), rhs).reshape(self.grid_points)

        # --------------------- 2D CASE ---------------------
        elif self.ndim == 2:
            Ax, Ay = [m.copy().tolil() for m in self.system_matrix]  # LIL for efficient BC modification
            Nx, Ny = self.grid_points
            
            # Reshape rhs to (Nx, Ny) for sweep operations
            rhs_2d = rhs.reshape(Nx, Ny)
            
            # --- SWEEP 1: X-Direction ---
            # Solve (Ax + source_lhs diagonal) * u* = rhs for each y-line
            u_star = np.zeros((Nx, Ny))
            for j in range(Ny):
                # Create modified LHS with diagonal source term
                source_diag = diags([(self.dt / 2.0) * source_lhs[:, j]], [0], shape=(Nx, Nx), format='csr')
                Ax_j = (Ax + source_diag).tolil()
                
                # Extract RHS for this column
                rhs_j = rhs_2d[:, j].reshape(Nx, 1)
                
                # Apply boundary conditions
                rhs_j = self._apply_bc_to_sweep(Ax_j, rhs_j, self.dx[0])
                
                # Solve and store
                u_star[:, j] = spsolve(Ax_j.tocsr(), rhs_j).flatten()
            
            # --- SWEEP 2: Y-Direction ---
            # Solve (Ay + source_lhs diagonal) * u^(n+1) = u* for each x-line
            u_new = np.zeros((Nx, Ny))
            for i in range(Nx):
                # Create modified LHS with diagonal source term
                source_diag = diags([(self.dt / 2.0) * source_lhs[i, :]], [0], shape=(Ny, Ny), format='csr')
                Ay_i = (Ay + source_diag).tolil()
                
                # Extract RHS for this row (as column vector)
                rhs_i = u_star[i, :].reshape(Ny, 1)
                
                # Apply boundary conditions
                rhs_i = self._apply_bc_to_sweep(Ay_i, rhs_i, self.dx[1])
                
                # Solve and store
                u_new[i, :] = spsolve(Ay_i.tocsr(), rhs_i).flatten()
            
            self.state = u_new

        # --------------------- 3D CASE ---------------------
        elif self.ndim == 3:
            Ax, Ay, Az = [m.copy().tolil() for m in self.system_matrix]  # LIL for efficient BC modification
            Nx, Ny, Nz = self.grid_points
            
            # --- SWEEP 1: X-Direction ---
            # Solve (Ax + source_lhs diagonal) * u* for each (y,z) line
            rhs_x = rhs.reshape(Nx, Ny, Nz)
            u_star = np.zeros((Nx, Ny, Nz))
            
            for j in range(Ny):
                for k in range(Nz):
                    # Create modified LHS with diagonal source term
                    source_diag = diags([(self.dt / 3.0) * source_lhs[:, j, k]], [0], shape=(Nx, Nx), format='csr')
                    Ax_jk = (Ax + source_diag).tolil()
                    
                    # Extract RHS for this line
                    rhs_jk = rhs_x[:, j, k].reshape(Nx, 1)
                    
                    # Apply boundary conditions
                    rhs_jk = self._apply_bc_to_sweep(Ax_jk, rhs_jk, self.dx[0])
                    
                    # Solve and store
                    u_star[:, j, k] = spsolve(Ax_jk.tocsr(), rhs_jk).flatten()
            
            # --- SWEEP 2: Y-Direction ---
            # Solve (Ay + source_lhs diagonal) * u** for each (x,z) line
            u_star_star = np.zeros((Nx, Ny, Nz))
            
            for i in range(Nx):
                for k in range(Nz):
                    # Create modified LHS with diagonal source term
                    source_diag = diags([(self.dt / 3.0) * source_lhs[i, :, k]], [0], shape=(Ny, Ny), format='csr')
                    Ay_ik = (Ay + source_diag).tolil()
                    
                    # Extract RHS for this line
                    rhs_ik = u_star[i, :, k].reshape(Ny, 1)
                    
                    # Apply boundary conditions
                    rhs_ik = self._apply_bc_to_sweep(Ay_ik, rhs_ik, self.dx[1])
                    
                    # Solve and store
                    u_star_star[i, :, k] = spsolve(Ay_ik.tocsr(), rhs_ik).flatten()
            
            # --- SWEEP 3: Z-Direction ---
            # Solve (Az + source_lhs diagonal) * u^(n+1) for each (x,y) line
            u_new = np.zeros((Nx, Ny, Nz))
            
            for i in range(Nx):
                for j in range(Ny):
                    # Create modified LHS with diagonal source term
                    source_diag = diags([(self.dt / 3.0) * source_lhs[i, j, :]], [0], shape=(Nz, Nz), format='csr')
                    Az_ij = (Az + source_diag).tolil()
                    
                    # Extract RHS for this line
                    rhs_ij = u_star_star[i, j, :].reshape(Nz, 1)
                    
                    # Apply boundary conditions
                    rhs_ij = self._apply_bc_to_sweep(Az_ij, rhs_ij, self.dx[2])
                    
                    # Solve and store
                    u_new[i, j, :] = spsolve(Az_ij.tocsr(), rhs_ij).flatten()
            
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