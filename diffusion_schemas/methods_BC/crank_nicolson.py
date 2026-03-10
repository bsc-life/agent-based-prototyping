"""
Crank-Nicolson method for diffusion equation.

This module implements the Crank-Nicolson finite difference scheme for solving
the diffusion equation. The method is unconditionally stable and second-order
accurate in time.
"""

import numpy as np
from scipy.sparse import diags, kron, eye, csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from diffusion_schemas.base import Schema
from diffusion_schemas.utils.boundary import DirichletBC, NeumannBC


class CrankNicolsonBCSchema(Schema):
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
            self.A_impl, self.A_expl = self._build_matrices_1d()
        elif self.ndim == 2:
            self.A_impl, self.A_expl = self._build_matrices_2d()
        elif self.ndim == 3:
            self.A_impl, self.A_expl = self._build_matrices_3d()
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
        A_expl = I + (1 - self.theta) * self.dt * self.diffusion_coefficient * L - \
                 (1 - self.theta) * self.dt * self.decay_rate * I
        
        return A_impl, A_expl
    
    def _build_matrices_2d(self):
        """Build the 2D system matrices using Kronecker products."""
        Nx, Ny = self.grid_points
        dx, dy = self.dx
        
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
        
        L = kron(Lx, Iy) + kron(Ix, Ly) # Dimensions of L: (Nx*Ny, Nx*Ny)
        
        # Identity matrix
        I = eye(Nx * Ny, format='csr')
        
        # Implicit and explicit matrices
        A_impl = I - self.theta * self.dt * self.diffusion_coefficient * L + \
                 self.theta * self.dt * self.decay_rate * I
        
        A_expl = I + (1 - self.theta) * self.dt * self.diffusion_coefficient * L - \
                 (1 - self.theta) * self.dt * self.decay_rate * I
        
        return A_impl, A_expl
    
    def _build_matrices_3d(self):
        """Build the 3D system matrices using Kronecker products."""
        Nx, Ny, Nz = self.grid_points
        dx, dy, dz = self.dx
        
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
        
        L = kron(kron(Lx, Iy), Iz) + kron(kron(Ix, Ly), Iz) + kron(kron(Ix, Iy), Lz)
        
        # Identity matrix
        I = eye(Nx * Ny * Nz, format='csr')
        
        # Implicit and explicit matrices
        A_impl = I - self.theta * self.dt * self.diffusion_coefficient * L + \
                 self.theta * self.dt * self.decay_rate * I
        
        A_expl = I + (1 - self.theta) * self.dt * self.diffusion_coefficient * L - \
                 (1 - self.theta) * self.dt * self.decay_rate * I
        
        return A_impl, A_expl
    
    def step(self) -> None:
        """Perform one Crank-Nicolson time step."""
        
        # --- EXPLICIT PART (RHS Construction) ---
        # Instead of using A_expl (which lacks correct ghost-point BCs),
        # we calculate Laplacian using the stencil method from ExplicitEulerBCSchema.
        laplacian_n = self._compute_laplacian(self.state)
        
        # Compute source term
        source_n = self._compute_source_term()
        # Approximation for Crank-Nicolson source term
        source_np1 = source_n 
        
        # RHS = u^n + (1-theta)*dt * (D*Laplacian^n - lambda*u^n + S^n) + theta*dt*S^(n+1)
        rhs = self.state + \
              (1 - self.theta) * self.dt * (
                  self.diffusion_coefficient * laplacian_n - 
                  self.decay_rate * self.state + 
                  source_n
              ) + \
              self.theta * self.dt * source_np1
        
        rhs = rhs.flatten()

        # --- IMPLICIT PART (LHS Construction & Solve) ---
        # Copy matrix in LIL format for efficient BC modification
        A_system = self.A_impl.copy().tolil()

        # Apply boundary conditions
        if self._boundary_conditions is not None:
            if isinstance(self._boundary_conditions, DirichletBC):
                rhs = self._apply_dirichlet_bc(A_system, rhs)
            elif isinstance(self._boundary_conditions, NeumannBC):
                rhs = self._apply_neumann_bc(A_system, rhs)
        
        # Convert back to CSR for efficient solving
        A_system = A_system.tocsr()

        # Solve the linear system
        u_new_flat = spsolve(A_system, rhs)
        
        # Reshape to grid
        self.state = u_new_flat.reshape(self.grid_points)
        
        # Update time
        self.t += self.dt
    
    # =========================================================================
    # IMPLICIT BC LOGIC (From ImplicitEulerBCSchema)
    # =========================================================================
    def _apply_neumann_bc(self, matrix, rhs):
        """Apply Neumann BC to LHS matrix and RHS vector."""
        flux = self._boundary_conditions._get_flux(self.t + self.dt)
        D = self.diffusion_coefficient
        dt = self.dt
        theta = self.theta

        # Common calculation for forcing term (matches Implicit logic adjusted for theta)
        # Note: Implicit had forcing = (2*dt*D*flux)/dx. Here we scale by theta.
        def get_forcing(h): return (2 * theta * dt * D * flux) / h
        def get_alpha(h): return (theta * dt * D) / (h**2)

        if self.ndim == 1:
            N, dx = self.grid_points[0], self.dx[0]
            forcing = get_forcing(dx)
            alpha = get_alpha(dx)
            
            # Left (i=0): Neighbor (i=1) weight doubles (subtract 2*alpha from existing)
            # Existing was -alpha, we want -2alpha, so subtract alpha more.
            # But simpler: just overwrite/add. Implicit used: matrix[0,1] -= alpha (or = -2alpha)
            # Since matrix is (I - ...), off-diagonals are -theta*dt*D/h^2 = -alpha.
            # We set it to -2*alpha.
            matrix[0, 1] = -2 * alpha
            rhs[0] -= forcing
            
            # Right (i=N-1)
            matrix[-1, -2] = -2 * alpha
            rhs[-1] += forcing

        elif self.ndim == 2:
            Nx, Ny = self.grid_points
            dx, dy = self.dx
            alpha_x, alpha_y = get_alpha(dx), get_alpha(dy)
            fx, fy = get_forcing(dx), get_forcing(dy)

            # Left/Right (Stride Ny)
            idx_l = np.arange(Ny)
            idx_r = np.arange((Nx - 1) * Ny, Nx * Ny)
            matrix[idx_l, idx_l + Ny] = -2 * alpha_x
            matrix[idx_r, idx_r - Ny] = -2 * alpha_x
            rhs[idx_l] -= fx
            rhs[idx_r] += fx

            # Bottom/Top (Stride 1)
            idx_b = np.arange(0, Nx * Ny, Ny)
            idx_t = np.arange(Ny - 1, Nx * Ny, Ny)
            matrix[idx_b, idx_b + 1] = -2 * alpha_y
            matrix[idx_t, idx_t - 1] = -2 * alpha_y
            rhs[idx_b] -= fy
            rhs[idx_t] += fy

        elif self.ndim == 3:
            Nx, Ny, Nz = self.grid_points
            dx, dy, dz = self.dx
            sx, sy = Ny * Nz, Nz
            alpha_x, alpha_y, alpha_z = get_alpha(dx), get_alpha(dy), get_alpha(dz)
            fx, fy, fz = get_forcing(dx), get_forcing(dy), get_forcing(dz)

            # X-planes
            idx_l, idx_r = np.arange(sx), np.arange((Nx-1)*sx, Nx*sx)
            matrix[idx_l, idx_l + sx] = -2 * alpha_x
            matrix[idx_r, idx_r - sx] = -2 * alpha_x
            rhs[idx_l] -= fx
            rhs[idx_r] += fx

            # Y-planes
            base_y = np.arange(Nz)
            idx_f = np.concatenate([base_y + i*sx for i in range(Nx)])
            idx_bk = idx_f + (Ny-1)*sy
            matrix[idx_f, idx_f + sy] = -2 * alpha_y
            matrix[idx_bk, idx_bk - sy] = -2 * alpha_y
            rhs[idx_f] -= fy
            rhs[idx_bk] += fy

            # Z-planes
            idx_bt = np.arange(0, Nx*Ny*Nz, Nz)
            idx_tp = np.arange(Nz-1, Nx*Ny*Nz, Nz)
            matrix[idx_bt, idx_bt + 1] = -2 * alpha_z
            matrix[idx_tp, idx_tp - 1] = -2 * alpha_z
            rhs[idx_bt] -= fz
            rhs[idx_tp] += fz

        return rhs

    def _apply_dirichlet_bc(self, matrix, rhs):
        """Apply Dirichlet BC."""
        value = self._boundary_conditions._get_value(self.t + self.dt)
        
        if self.ndim == 1:
            matrix[0, :] = 0; matrix[0, 0] = 1; rhs[0] = value
            matrix[-1, :] = 0; matrix[-1, -1] = 1; rhs[-1] = value
            
        elif self.ndim == 2:
            Nx, Ny = self.grid_points
            # Left/Right
            for j in range(Ny):
                idx = j*Nx; matrix[idx,:]=0; matrix[idx,idx]=1; rhs[idx]=value
                idx2= j*Nx+Nx-1; matrix[idx2,:]=0; matrix[idx2,idx2]=1; rhs[idx2]=value
            # Bot/Top
            for i in range(Nx):
                idx = i; matrix[idx,:]=0; matrix[idx,idx]=1; rhs[idx]=value
                idx2= (Ny-1)*Nx+i; matrix[idx2,:]=0; matrix[idx2,idx2]=1; rhs[idx2]=value
                
        elif self.ndim == 3:
            Nx, Ny, Nz = self.grid_points
            # Simple iteration for brevity in this example
            # (Ideally utilize vectorized indexing like in Explicit schema if perf is critical)
            # X-faces
            for j in range(Ny):
                for k in range(Nz):
                    idx = 0*Ny*Nz + j*Nz + k; matrix[idx,:]=0; matrix[idx,idx]=1; rhs[idx]=value
                    idx = (Nx-1)*Ny*Nz + j*Nz + k; matrix[idx,:]=0; matrix[idx,idx]=1; rhs[idx]=value
            # Y-faces
            for i in range(Nx):
                for k in range(Nz):
                    idx = i*Ny*Nz + 0*Nz + k; matrix[idx,:]=0; matrix[idx,idx]=1; rhs[idx]=value
                    idx = i*Ny*Nz + (Ny-1)*Nz + k; matrix[idx,:]=0; matrix[idx,idx]=1; rhs[idx]=value
            # Z-faces
            for i in range(Nx):
                for j in range(Ny):
                    idx = i*Ny*Nz + j*Nz + 0; matrix[idx,:]=0; matrix[idx,idx]=1; rhs[idx]=value
                    idx = i*Ny*Nz + j*Nz + (Nz-1); matrix[idx,:]=0; matrix[idx,idx]=1; rhs[idx]=value

        return rhs

    # =========================================================================
    # EXPLICIT LAPLACIAN LOGIC (From ExplicitEulerBCSchema)
    # =========================================================================
    def _compute_laplacian(self, u: np.ndarray) -> np.ndarray:
        if self.ndim == 1: return self._laplacian_1d(u)
        elif self.ndim == 2: return self._laplacian_2d(u)
        elif self.ndim == 3: return self._laplacian_3d(u)
        else: raise ValueError(f"Unsupported dimensions: {self.ndim}")

    def _laplacian_1d(self, u: np.ndarray) -> np.ndarray:
        laplacian = np.zeros_like(u)
        dx = self.dx[0]
        laplacian[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
        
        is_neumann = isinstance(self._boundary_conditions, NeumannBC)
        is_dirichlet = isinstance(self._boundary_conditions, DirichletBC)

        if is_neumann:
            g = self._boundary_conditions._get_flux(self.t)
            laplacian[0] = 2 * (u[1] - u[0] - g * dx) / (dx**2)
            laplacian[-1] = 2 * (u[-2] - u[-1] + g * dx) / (dx**2)
        elif is_dirichlet:
            laplacian[0] = 0; laplacian[-1] = 0
        else:
            laplacian[0] = (u[1] - 2*u[0] + u[1]) / (dx**2)
            laplacian[-1] = (u[-2] - 2*u[-1] + u[-2]) / (dx**2)
        return laplacian

    def _laplacian_2d(self, u: np.ndarray) -> np.ndarray:
        laplacian = np.zeros_like(u)
        dx, dy = self.dx
        
        laplacian[1:-1, 1:-1] = (
            (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx**2) +
            (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy**2)
        )
        
        is_neumann = isinstance(self._boundary_conditions, NeumannBC)
        is_dirichlet = isinstance(self._boundary_conditions, DirichletBC)

        if is_neumann:
            flux = self._boundary_conditions._get_flux(self.t)
            # Left/Right
            laplacian[0, 1:-1] = (2*(u[1,1:-1]-u[0,1:-1])/dx**2 - 2*flux/dx) + (u[0,2:]-2*u[0,1:-1]+u[0,:-2])/dy**2
            laplacian[-1, 1:-1] = (2*(u[-2,1:-1]-u[-1,1:-1])/dx**2 + 2*flux/dx) + (u[-1,2:]-2*u[-1,1:-1]+u[-1,:-2])/dy**2
            # Bot/Top
            laplacian[1:-1, 0] = (u[2:,0]-2*u[1:-1,0]+u[:-2,0])/dx**2 + (2*(u[1:-1,1]-u[1:-1,0])/dy**2 - 2*flux/dy)
            laplacian[1:-1, -1] = (u[2:,-1]-2*u[1:-1,-1]+u[:-2,-1])/dx**2 + (2*(u[1:-1,-2]-u[1:-1,-1])/dy**2 + 2*flux/dy)
            # Corners
            laplacian[0,0] = (2*(u[1,0]-u[0,0])/dx**2 - 2*flux/dx) + (2*(u[0,1]-u[0,0])/dy**2 - 2*flux/dy)
            laplacian[0,-1] = (2*(u[1,-1]-u[0,-1])/dx**2 - 2*flux/dx) + (2*(u[0,-2]-u[0,-1])/dy**2 + 2*flux/dy)
            laplacian[-1,0] = (2*(u[-2,0]-u[-1,0])/dx**2 + 2*flux/dx) + (2*(u[-1,1]-u[-1,0])/dy**2 - 2*flux/dy)
            laplacian[-1,-1] = (2*(u[-2,-1]-u[-1,-1])/dx**2 + 2*flux/dx) + (2*(u[-1,-2]-u[-1,-1])/dy**2 + 2*flux/dy)
        elif is_dirichlet:
            laplacian[0,:]=0; laplacian[-1,:]=0; laplacian[:,0]=0; laplacian[:,-1]=0
        else:
            # Default Zero Flux
            laplacian[0, 1:-1] = (2*(u[1, 1:-1] - u[0, 1:-1])/dx**2) + (u[0, 2:] - 2*u[0, 1:-1] + u[0, :-2])/dy**2
            laplacian[-1, 1:-1] = (2*(u[-2, 1:-1] - u[-1, 1:-1])/dx**2) + (u[-1, 2:] - 2*u[-1, 1:-1] + u[-1, :-2])/dy**2
            laplacian[1:-1, 0] = (u[2:, 0] - 2*u[1:-1, 0] + u[:-2, 0])/dx**2 + (2*(u[1:-1, 1] - u[1:-1, 0])/dy**2)
            laplacian[1:-1, -1] = (u[2:, -1] - 2*u[1:-1, -1] + u[:-2, -1])/dx**2 + (2*(u[1:-1, -2] - u[1:-1, -1])/dy**2)
            laplacian[0,0] = 2*(u[1,0]-u[0,0])/dx**2 + 2*(u[0,1]-u[0,0])/dy**2
            laplacian[0,-1] = 2*(u[1,-1]-u[0,-1])/dx**2 + 2*(u[0,-2]-u[0,-1])/dy**2
            laplacian[-1,0] = 2*(u[-2,0]-u[-1,0])/dx**2 + 2*(u[-1,1]-u[-1,0])/dy**2
            laplacian[-1,-1] = 2*(u[-2,-1]-u[-1,-1])/dx**2 + 2*(u[-1,-2]-u[-1,-1])/dy**2
        return laplacian

    def _laplacian_3d(self, u: np.ndarray) -> np.ndarray:
        laplacian = np.zeros_like(u)
        dx, dy, dz = self.dx
        
        # Interior
        laplacian[1:-1, 1:-1, 1:-1] = (
            (u[2:, 1:-1, 1:-1] - 2*u[1:-1, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]) / (dx**2)
            + (u[1:-1, 2:, 1:-1] - 2*u[1:-1, 1:-1, 1:-1] + u[1:-1, :-2, 1:-1]) / (dy**2)
            + (u[1:-1, 1:-1, 2:] - 2*u[1:-1, 1:-1, 1:-1] + u[1:-1, 1:-1, :-2]) / (dz**2)
        )
        
        is_neumann = isinstance(self._boundary_conditions, NeumannBC)
        is_dirichlet = isinstance(self._boundary_conditions, DirichletBC)

        if is_neumann:
            flux = self._boundary_conditions._get_flux(self.t)
            # Ported 3D logic: X-boundaries
            laplacian[0, 1:-1, 1:-1] = (2*(u[1,1:-1,1:-1]-u[0,1:-1,1:-1])/dx**2 - 2*flux/dx) + \
                                       (u[0,2:,1:-1]-2*u[0,1:-1,1:-1]+u[0,:-2,1:-1])/dy**2 + \
                                       (u[0,1:-1,2:]-2*u[0,1:-1,1:-1]+u[0,1:-1,:-2])/dz**2
            laplacian[-1, 1:-1, 1:-1] = (2*(u[-2,1:-1,1:-1]-u[-1,1:-1,1:-1])/dx**2 + 2*flux/dx) + \
                                        (u[-1,2:,1:-1]-2*u[-1,1:-1,1:-1]+u[-1,:-2,1:-1])/dy**2 + \
                                        (u[-1,1:-1,2:]-2*u[-1,1:-1,1:-1]+u[-1,1:-1,:-2])/dz**2
            # Y-boundaries
            laplacian[1:-1, 0, 1:-1] = (u[2:,0,1:-1]-2*u[1:-1,0,1:-1]+u[:-2,0,1:-1])/dx**2 + \
                                       (2*(u[1:-1,1,1:-1]-u[1:-1,0,1:-1])/dy**2 - 2*flux/dy) + \
                                       (u[1:-1,0,2:]-2*u[1:-1,0,1:-1]+u[1:-1,0,:-2])/dz**2
            laplacian[1:-1, -1, 1:-1] = (u[2:,-1,1:-1]-2*u[1:-1,-1,1:-1]+u[:-2,-1,1:-1])/dx**2 + \
                                        (2*(u[1:-1,-2,1:-1]-u[1:-1,-1,1:-1])/dy**2 + 2*flux/dy) + \
                                        (u[1:-1,-1,2:]-2*u[1:-1,-1,1:-1]+u[1:-1,-1,:-2])/dz**2
            # Z-boundaries
            laplacian[1:-1, 1:-1, 0] = (u[2:,1:-1,0]-2*u[1:-1,1:-1,0]+u[:-2,1:-1,0])/dx**2 + \
                                       (u[1:-1,2:,0]-2*u[1:-1,1:-1,0]+u[1:-1,:-2,0])/dy**2 + \
                                       (2*(u[1:-1,1:-1,1]-u[1:-1,1:-1,0])/dz**2 - 2*flux/dz)
            laplacian[1:-1, 1:-1, -1] = (u[2:,1:-1,-1]-2*u[1:-1,1:-1,-1]+u[:-2,1:-1,-1])/dx**2 + \
                                        (u[1:-1,2:,-1]-2*u[1:-1,1:-1,-1]+u[1:-1,:-2,-1])/dy**2 + \
                                        (2*(u[1:-1,1:-1,-2]-u[1:-1,1:-1,-1])/dz**2 + 2*flux/dz)
            
            # EDGES
            # 1. Front-Bottom Edge (y=0, z=0)
            laplacian[1:-1, 0, 0] = (u[2:,0,0]-2*u[1:-1,0,0]+u[:-2,0,0])/dx**2 + \
                                    (2*(u[1:-1,1,0]-u[1:-1,0,0])/dy**2 - 2*flux/dy) + \
                                    (2*(u[1:-1,0,1]-u[1:-1,0,0])/dz**2 - 2*flux/dz)
            # 2. Front-Top Edge (y=0, z=-1)
            laplacian[1:-1, 0, -1] = (u[2:,0,-1]-2*u[1:-1,0,-1]+u[:-2,0,-1])/dx**2 + \
                                     (2*(u[1:-1,1,-1]-u[1:-1,0,-1])/dy**2 - 2*flux/dy) + \
                                     (2*(u[1:-1,0,-2]-u[1:-1,0,-1])/dz**2 + 2*flux/dz)
            # 3. Back-Bottom Edge (y=-1, z=0)
            laplacian[1:-1, -1, 0] = (u[2:,-1,0]-2*u[1:-1,-1,0]+u[:-2,-1,0])/dx**2 + \
                                     (2*(u[1:-1,-2,0]-u[1:-1,-1,0])/dy**2 + 2*flux/dy) + \
                                     (2*(u[1:-1,-1,1]-u[1:-1,-1,0])/dz**2 - 2*flux/dz)
            # 4. Back-Top Edge (y=-1, z=-1)
            laplacian[1:-1, -1, -1] = (u[2:,-1,-1]-2*u[1:-1,-1,-1]+u[:-2,-1,-1])/dx**2 + \
                                      (2*(u[1:-1,-2,-1]-u[1:-1,-1,-1])/dy**2 + 2*flux/dy) + \
                                      (2*(u[1:-1,-1,-2]-u[1:-1,-1,-1])/dz**2 + 2*flux/dz)

            # Edges parallel to Y-axis (x=0, x=-1, z=0, z=-1)
            # 5. Left-Bottom Edge (x=0, z=0)
            laplacian[0, 1:-1, 0] = (2*(u[1,1:-1,0]-u[0,1:-1,0])/dx**2 - 2*flux/dx) + \
                                    (u[0,2:,0]-2*u[0,1:-1,0]+u[0,:-2,0])/dy**2 + \
                                    (2*(u[0,1:-1,1]-u[0,1:-1,0])/dz**2 - 2*flux/dz)
            # 6. Left-Top Edge (x=0, z=-1)
            laplacian[0, 1:-1, -1] = (2*(u[1,1:-1,-1]-u[0,1:-1,-1])/dx**2 - 2*flux/dx) + \
                                     (u[0,2:,-1]-2*u[0,1:-1,-1]+u[0,:-2,-1])/dy**2 + \
                                     (2*(u[0,1:-1,-2]-u[0,1:-1,-1])/dz**2 + 2*flux/dz)
            # 7. Right-Bottom Edge (x=-1, z=0)
            laplacian[-1, 1:-1, 0] = (2*(u[-2,1:-1,0]-u[-1,1:-1,0])/dx**2 + 2*flux/dx) + \
                                     (u[-1,2:,0]-2*u[-1,1:-1,0]+u[-1,:-2,0])/dy**2 + \
                                     (2*(u[-1,1:-1,1]-u[-1,1:-1,0])/dz**2 - 2*flux/dz)
            # 8. Right-Top Edge (x=-1, z=-1)
            laplacian[-1, 1:-1, -1] = (2*(u[-2,1:-1,-1]-u[-1,1:-1,-1])/dx**2 + 2*flux/dx) + \
                                      (u[-1,2:,-1]-2*u[-1,1:-1,-1]+u[-1,:-2,-1])/dy**2 + \
                                      (2*(u[-1,1:-1,-2]-u[-1,1:-1,-1])/dz**2 + 2*flux/dz)

            # Edges parallel to Z-axis (x=0, x=-1, y=0, y=-1)
            # 9. Left-Front Edge (x=0, y=0)
            laplacian[0, 0, 1:-1] = (2*(u[1,0,1:-1]-u[0,0,1:-1])/dx**2 - 2*flux/dx) + \
                                    (2*(u[0,1,1:-1]-u[0,0,1:-1])/dy**2 - 2*flux/dy) + \
                                    (u[0,0,2:]-2*u[0,0,1:-1]+u[0,0,:-2])/dz**2
            # 10. Left-Back Edge (x=0, y=-1)
            laplacian[0, -1, 1:-1] = (2*(u[1,-1,1:-1]-u[0,-1,1:-1])/dx**2 - 2*flux/dx) + \
                                     (2*(u[0,-2,1:-1]-u[0,-1,1:-1])/dy**2 + 2*flux/dy) + \
                                     (u[0,-1,2:]-2*u[0,-1,1:-1]+u[0,-1,:-2])/dz**2
            # 11. Right-Front Edge (x=-1, y=0)
            laplacian[-1, 0, 1:-1] = (2*(u[-2,0,1:-1]-u[-1,0,1:-1])/dx**2 + 2*flux/dx) + \
                                     (2*(u[-1,1,1:-1]-u[-1,0,1:-1])/dy**2 - 2*flux/dy) + \
                                     (u[-1,0,2:]-2*u[-1,0,1:-1]+u[-1,0,:-2])/dz**2
            # 12. Right-Back Edge (x=-1, y=-1)
            laplacian[-1, -1, 1:-1] = (2*(u[-2,-1,1:-1]-u[-1,-1,1:-1])/dx**2 + 2*flux/dx) + \
                                      (2*(u[-1,-2,1:-1]-u[-1,-1,1:-1])/dy**2 + 2*flux/dy) + \
                                      (u[-1,-1,2:]-2*u[-1,-1,1:-1]+u[-1,-1,:-2])/dz**2

            # CORNERS
            # 1. Left-Front-Bottom (0,0,0)
            laplacian[0,0,0] = (2*(u[1,0,0]-u[0,0,0])/dx**2 - 2*flux/dx) + \
                               (2*(u[0,1,0]-u[0,0,0])/dy**2 - 2*flux/dy) + \
                               (2*(u[0,0,1]-u[0,0,0])/dz**2 - 2*flux/dz)
            # 2. Right-Front-Bottom (-1,0,0)
            laplacian[-1,0,0] = (2*(u[-2,0,0]-u[-1,0,0])/dx**2 + 2*flux/dx) + \
                                (2*(u[-1,1,0]-u[-1,0,0])/dy**2 - 2*flux/dy) + \
                                (2*(u[-1,0,1]-u[-1,0,0])/dz**2 - 2*flux/dz)
            # 3. Left-Back-Bottom (0,-1,0)
            laplacian[0,-1,0] = (2*(u[1,-1,0]-u[0,-1,0])/dx**2 - 2*flux/dx) + \
                                (2*(u[0,-2,0]-u[0,-1,0])/dy**2 + 2*flux/dy) + \
                                (2*(u[0,-1,1]-u[0,-1,0])/dz**2 - 2*flux/dz)
            # 4. Right-Back-Bottom (-1,-1,0)
            laplacian[-1,-1,0] = (2*(u[-2,-1,0]-u[-1,-1,0])/dx**2 + 2*flux/dx) + \
                                 (2*(u[-1,-2,0]-u[-1,-1,0])/dy**2 + 2*flux/dy) + \
                                 (2*(u[-1,-1,1]-u[-1,-1,0])/dz**2 - 2*flux/dz)
            # 5. Left-Front-Top (0,0,-1)
            laplacian[0,0,-1] = (2*(u[1,0,-1]-u[0,0,-1])/dx**2 - 2*flux/dx) + \
                                (2*(u[0,1,-1]-u[0,0,-1])/dy**2 - 2*flux/dy) + \
                                (2*(u[0,0,-2]-u[0,0,-1])/dz**2 + 2*flux/dz)
            # 6. Right-Front-Top (-1,0,-1)
            laplacian[-1,0,-1] = (2*(u[-2,0,-1]-u[-1,0,-1])/dx**2 + 2*flux/dx) + \
                                 (2*(u[-1,1,-1]-u[-1,0,-1])/dy**2 - 2*flux/dy) + \
                                 (2*(u[-1,0,-2]-u[-1,0,-1])/dz**2 + 2*flux/dz)
            # 7. Left-Back-Top (0,-1,-1)
            laplacian[0,-1,-1] = (2*(u[1,-1,-1]-u[0,-1,-1])/dx**2 - 2*flux/dx) + \
                                 (2*(u[0,-2,-1]-u[0,-1,-1])/dy**2 + 2*flux/dy) + \
                                 (2*(u[0,-1,-2]-u[0,-1,-1])/dz**2 + 2*flux/dz)
            # 8. Right-Back-Top (-1,-1,-1)
            laplacian[-1,-1,-1] = (2*(u[-2,-1,-1]-u[-1,-1,-1])/dx**2 + 2*flux/dx) + \
                                  (2*(u[-1,-2,-1]-u[-1,-1,-1])/dy**2 + 2*flux/dy) + \
                                  (2*(u[-1,-1,-2]-u[-1,-1,-1])/dz**2 + 2*flux/dz)
            
        elif is_dirichlet:
            laplacian[0,:,:]=0; laplacian[-1,:,:]=0; 
            laplacian[:,0,:]=0; laplacian[:,-1,:]=0; 
            laplacian[:,:,0]=0; laplacian[:,:,-1]=0
        
        return laplacian

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