"""
Crank-Nicolson method for diffusion equation.

This module implements the Crank-Nicolson finite difference scheme for solving
the diffusion equation. The method is unconditionally stable and second-order
accurate in time.
"""

import numpy as np
from scipy.sparse import diags, eye, csr_matrix
from scipy.sparse.linalg import spsolve
from diffusion_schemas.base import Schema


class CrankNicolsonLODSchema(Schema):
    """
    Crank-Nicolson method for the diffusion equation.
    
    Implements the Crank-Nicolson (θ-method with θ=0.5) finite difference scheme:
        u^(n+1) = u^n + (dt/2) * (D*∇²u^n - λ*u^n + D*∇²u^(n+1) - λ*u^(n+1) + S^n + S^(n+1))
    
    Rearranging:
        (I - dt*D*L/2 + dt*λ*I/2) * u^(n+1) = (I + dt*D*L/2 - dt*λ*I/2) * u^n + dt*(S^n + S^(n+1))/2
    
    where L is the discrete Laplacian operator.
    
    The method is:
        - Unconditionally stable
        - Second-order accurate in time (vs. first-order for explicit/implicit Euler)
        - Requires solving a sparse linear system at each step
    
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
    theta : float, optional
        Weighting parameter (0 = explicit, 0.5 = Crank-Nicolson, 1 = implicit).
        Default is 0.5.
    """
    
    def __init__(
        self,
        domain_size,
        grid_points,
        dt,
        diffusion_coefficient=1.0,
        decay_rate=0.0,
        theta=0.5
    ):
        """Initialize the Crank-Nicolson schema."""
        super().__init__(domain_size, grid_points, dt, diffusion_coefficient, decay_rate)
        
        if not 0 <= theta <= 1:
            raise ValueError("theta must be in [0, 1]")
        self.theta = theta
        
        # Build the system matrices
        self._build_system_matrices()
    
    def _build_system_matrices(self) -> None:
        """Build the sparse system matrices for the Crank-Nicolson scheme."""
        if self.ndim == 1:
            self.A_impl_x = self._build_matrices_1d()
        elif self.ndim == 2:
            self.A_impl_x, self.A_impl_y = self._build_matrices_2d()
        elif self.ndim == 3:
            self.A_impl_x, self.A_impl_y, self.A_impl_z = self._build_matrices_3d()
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.ndim}")
    
    def _build_matrices_1d(self):
        """Build the 1D system matrices."""
        N = self.grid_points[0]
        dx = self.dx[0]
        
        # Discrete Laplacian in 1D
        diag_main = -2 * np.ones(N) / (dx**2)
        diag_off = np.ones(N-1) / (dx**2)
        
        L = diags([diag_off, diag_main, diag_off], [-1, 0, 1], shape=(N, N), format='csr')
        
        # Identity matrix
        I = eye(N, format='csr')
        
        # Implicit side: I - θ*dt*D*L + θ*dt*λ*I
        A_impl = I - self.theta * self.dt * self.diffusion_coefficient * L + \
                 self.theta * self.dt * self.decay_rate * I
        
        # Explicit side: I + (1-θ)*dt*D*L - (1-θ)*dt*λ*I
        # A_expl = I + (1 - self.theta) * self.dt * self.diffusion_coefficient * L - \
        #          (1 - self.theta) * self.dt * self.decay_rate * I
        
        self.Lx = L

        return A_impl
    
    def _build_matrices_2d(self):
        """Build the 2D system matrices using Kronecker products."""
        Nx, Ny = self.grid_points
        dx, dy = self.dx
        factor = 1 / 2 # LOD factor for 2D splitting
        
        # 1D Laplacian operators
        diag_main_x = -2 * np.ones(Nx) / (dx**2)
        diag_off_x = np.ones(Nx-1) / (dx**2)
        Lx = diags([diag_off_x, diag_main_x, diag_off_x], [-1, 0, 1], shape=(Nx, Nx), format='csr')
        
        diag_main_y = -2 * np.ones(Ny) / (dy**2)
        diag_off_y = np.ones(Ny-1) / (dy**2)
        Ly = diags([diag_off_y, diag_main_y, diag_off_y], [-1, 0, 1], shape=(Ny, Ny), format='csr')
        
        # 2D Laplacian
        Ix = eye(Nx, format='csr')
        Iy = eye(Ny, format='csr')
                
        # Identity matrix
        I = eye(Nx * Ny, format='csr')

        # L = kron(Lx, Iy) + kron(Ix, Ly) # still compute it for the explicit part ???
        
        # Implicit and explicit matrices
        A_impl_x = Ix - self.theta * self.dt * self.diffusion_coefficient * Lx + \
                 self.theta * factor * self.dt * self.decay_rate * Ix
        A_impl_y = Iy - self.theta * self.dt * self.diffusion_coefficient * Ly + \
                 self.theta * factor * self.dt * self.decay_rate * Iy
        
        # A_expl = I + (1 - self.theta) * self.dt * self.diffusion_coefficient * L - \
        #          (1 - self.theta) * self.dt * self.decay_rate * I
        
        self.Lx = Lx
        self.Ly = Ly

        return A_impl_x, A_impl_y
    
    def _build_matrices_3d(self):
        """Build the 3D system matrices using Kronecker products."""
        Nx, Ny, Nz = self.grid_points
        dx, dy, dz = self.dx
        factor = 1 / 3 # LOD factor for 3D splitting
        
        # 1D Laplacian operators
        diag_main_x = -2 * np.ones(Nx) / (dx**2)
        diag_off_x = np.ones(Nx-1) / (dx**2)
        Lx = diags([diag_off_x, diag_main_x, diag_off_x], [-1, 0, 1], shape=(Nx, Nx), format='csr')
        
        diag_main_y = -2 * np.ones(Ny) / (dy**2)
        diag_off_y = np.ones(Ny-1) / (dy**2)
        Ly = diags([diag_off_y, diag_main_y, diag_off_y], [-1, 0, 1], shape=(Ny, Ny), format='csr')
        
        diag_main_z = -2 * np.ones(Nz) / (dz**2)
        diag_off_z = np.ones(Nz-1) / (dz**2)
        Lz = diags([diag_off_z, diag_main_z, diag_off_z], [-1, 0, 1], shape=(Nz, Nz), format='csr')
        
        # 3D Laplacian
        Ix = eye(Nx, format='csr')
        Iy = eye(Ny, format='csr')
        Iz = eye(Nz, format='csr')
        
        # L = kron(kron(Lx, Iy), Iz) + kron(kron(Ix, Ly), Iz) + kron(kron(Ix, Iy), Lz)
        
        # Identity matrix
        I = eye(Nx * Ny * Nz, format='csr')
        
        # Implicit and explicit matrices
        A_impl_x = Ix - self.theta * self.dt * self.diffusion_coefficient * Lx + \
                 self.theta * factor * self.dt * self.decay_rate * Ix
        A_impl_y = Iy - self.theta * self.dt * self.diffusion_coefficient * Ly + \
                 self.theta * factor * self.dt * self.decay_rate * Iy
        A_impl_z = Iz - self.theta * self.dt * self.diffusion_coefficient * Lz + \
                 self.theta * factor * self.dt * self.decay_rate * Iz
        
        # A_expl = I + (1 - self.theta) * self.dt * self.diffusion_coefficient * L - \
        #          (1 - self.theta) * self.dt * self.decay_rate * I
        
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        
        return A_impl_x, A_impl_y, A_impl_z
    
    def step(self) -> None:
        """Perform one Crank-Nicolson time step."""
        # Compute source term at current time
        source_n = self._compute_source_term()
        
        # For Crank-Nicolson, we approximate S^(n+1) ≈ S^n
        # (true CN would need to evaluate at n+1, but agents are at fixed positions)
        source_np1 = source_n  # Approximation
        
        # Compute Explicit Part (Right-Hand Side)
        # explicit_term = dt * (1-theta) * (D * Lap_n - decay * u_n)
        explicit_term = self.step_explicit()

        # Right-hand side: A_expl * u^n + dt * θ * S^(n+1) + dt * (1-θ) * S^n
        # rhs = self.A_expl.dot(self.state.flatten()) + \
        #       self.dt * self.theta * source_np1.flatten() + \
        #       self.dt * (1 - self.theta) * source_n.flatten()
        rhs_grid = self.state + explicit_term + self.dt * source_np1

        # These two lines make CN and CN-LOD differ!!!
        # Should BC be implemented only at the end of the total step or after each step (implicit and explicit)???
        # if self._boundary_conditions is not None:
        #     rhs_grid = self._apply_boundary_conditions(rhs_grid)

        # Solve the linear system: A_impl * u^(n+1) = rhs
        # LOD step to solve the system more efficiently (split into x,y,z)
        u_new_grid = self.step_lod(rhs_grid) 
        
        # Reshape to grid
        self.state = u_new_grid
        
        # Apply boundary conditions
        if self._boundary_conditions is not None:
            self.state = self._apply_boundary_conditions(self.state)
        
        # Update time
        self.t += self.dt

    def step_explicit(self):

        u = self.state  # Removed unnecessary reshape
        
        # Compute Laplacian based on dimensions
        if self.ndim == 1:
            # 1D: Direct matrix-vector multiplication
            # Lx is (N, N), u is (N,). Result is (N,)
            laplacian = self.Lx.dot(u)

        elif self.ndim == 2:
            # 2D: Lx acts on cols, Ly acts on rows
            laplacian = self.Lx.dot(u) + self.Ly.dot(u.T).T

        elif self.ndim == 3:
            # 3D: Apply Lx, Ly, Lz to respective axes
            Nx, Ny, Nz = self.grid_points
            diff_x = self.Lx.dot(u.reshape(Nx, -1)).reshape(Nx, Ny, Nz)
            diff_y = self.Ly.dot(u.transpose(1,0,2).reshape(Ny, -1)).reshape(Ny, Nx, Nz).transpose(1,0,2)
            diff_z = self.Lz.dot(u.transpose(2,0,1).reshape(Nz, -1)).reshape(Nz, Nx, Ny).transpose(1,2,0)
            laplacian = diff_x + diff_y + diff_z

        # Crank-Nicolson Explicit Formula (Same for all dimensions)
        # RHS = u^n + dt * (1-theta) * [ D * Laplacian(u) - decay * u ]
        explicit_term = self.dt * (1 - self.theta) * (
            self.diffusion_coefficient * laplacian - self.decay_rate * u
        )  

        return explicit_term

    def step_lod(self, rhs):

        # Theta proportion is already accounted for in the system matrices
        # We just solve the linear system using LOD splitting.

        if self.ndim == 1:
            self.state = spsolve(self.A_impl_x, rhs)

        elif self.ndim == 2:
            Ax, Ay = self.A_impl_x, self.A_impl_y
            Nx, Ny = self.grid_points
            rhs = rhs.reshape(Nx,Ny)

            # Step 1: Solve (Ax) * u* = rhs
            u_star = spsolve(Ax, rhs)
            # Intermediate step: Apply boundary conditions 
            if self._boundary_conditions is not None: u_star = self._apply_boundary_conditions(u_star)
            # Step 2: Solve (Ay) * u^(n+1) = u*
            self.state = spsolve(Ay, u_star.T)
            # Transpose back to original shape
            self.state = self.state.T 

        elif self.ndim == 3:
            Ax, Ay, Az = self.A_impl_x, self.A_impl_y, self.A_impl_z
            Nx, Ny, Nz = self.grid_points
            rhs = rhs.reshape(Nx,Ny,Nz)

            # Reshape to (Nx, Ny*Nz) to feed in spsolve
            rhs_x = rhs.reshape(Nx, Ny * Nz)
            # Step 1: Solve (Ax) * u* = rhs
            u_star = spsolve(Ax, rhs_x)
            # Reshape back to 3D before applying BC
            u_star = u_star.reshape(Nx, Ny, Nz)
            # Intermediate step: Apply boundary conditions 
            if self._boundary_conditions is not None:u_star = self._apply_boundary_conditions(u_star)

            # Transpose and reshape to (Ny, Nz, N) to feed in spsolve
            rhs_y = u_star.transpose(1,0,2).reshape(Ny, Nx * Nz)
            # Step 2: Solve (Ay) * u** = u*
            u_star_star = spsolve(Ay, rhs_y)
            # Transpose back to original shape
            u_star_star = u_star_star.reshape(Ny, Nx, Nz).transpose(1,0,2)
            # Intermediate step: Apply boundary conditions 
            if self._boundary_conditions is not None:u_star_star = self._apply_boundary_conditions(u_star_star)

            # Reshape to (Nz, Nx*Ny) to feed in spsolve
            rhs_z = u_star_star.transpose(2,0,1).reshape(Nz, Nx * Ny)
            # Step 3: Solve (Az) * u^(n+1) = u**
            self.state = spsolve(Az, rhs_z)
            # Reshape back to 3D and transpose back to original shape
            self.state = self.state.reshape(Nz, Nx, Ny).transpose(1,2,0)

        # Fix return type in step_lod
        return self.state  # Removed flatten to maintain grid shape
    
    def set_diffusion_coefficient(self, value: float) -> None:
        """
        Set the diffusion coefficient and rebuild system matrices.
        
        Parameters
        ----------
        value : float
            Diffusion coefficient D (must be non-negative).
        """
        super().set_diffusion_coefficient(value)
        self._build_system_matrices()
    
    def set_decay_rate(self, value: float) -> None:
        """
        Set the decay rate and rebuild system matrices.
        
        Parameters
        ----------
        value : float
            Decay rate λ (must be non-negative).
        """
        super().set_decay_rate(value)
        self._build_system_matrices()
