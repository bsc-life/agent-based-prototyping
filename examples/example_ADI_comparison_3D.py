"""
Example 3: Comparison of Numerical Methods

Compares the accuracy and stability of different numerical methods
(Standard Implicit Euler vs ADI Implicit Euler) on the same problem.
"""

from email import errors
import numpy as np
import matplotlib.pyplot as plt
import time
# Assuming ADISchema is available in diffusion_schemas
from diffusion_schemas import ImplicitEulerSchema, ADISchema
from diffusion_schemas.utils import gaussian, NeumannBC


def analytical_solution_3d(x, y, z, t, D):
    """
    Analytical solution for 3D diffusion with Neumann BC and Gaussian IC.
    This is an approximation using the fundamental solution.
    """
    # For simplicity, use the fundamental solution (infinite domain approximation)
    # True solution with Neumann BC is more complex
    sigma_initial = 0.05
    sigma_t = np.sqrt(sigma_initial**2 + 2*D*t)
    amplitude = (sigma_initial / sigma_t) ** 3 # faster amplitude decay in 3D
    return amplitude * np.exp(-((x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2)/ (2 * sigma_t**2))


def run_comparison():
    # Parameters are now tuples for 2D
    L = (1.0, 1.0, 1.0)
    N = (25, 25, 25)
    D = 0.1
    t_final = 0.1
    
    # Time steps
    dt_implicit = 0.001     # Standard Implicit
    dt_adi = 0.001          # ADI (Implicit splitting)
    
    print("=" * 70)
    print("Numerical Methods Comparison: Implicit vs ADI")
    print("=" * 70)
    print(f"Problem: 2D diffusion with Gaussian initial condition")
    print(f"Domain: [(0, 0, 0), {L}], Grid points: {N}")
    print(f"Diffusion coefficient: {D}")
    print(f"Final time: {t_final}")
    print("=" * 70)
    
    # Create coordinate array (x and y instead of just x, and meshgrid for 2D)
    x = np.linspace(0, L[0], N[0])
    y = np.linspace(0, L[1], N[1])
    z = np.linspace(0, L[2], N[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                              
    # Initial condition
    ic = gaussian(center=(0.5, 0.5, 0.5), amplitude=1.0, width=0.05)

    # Boundary condition (Neumann: zero flux)
    bc = NeumannBC(flux=0.0)
    
    # ========== Implicit Euler (Standard) ==========
    print("\n[1/2] Running Standard Implicit Euler method...")
    print(f"  Time step: {dt_implicit}")
    
    schema_implicit = ImplicitEulerSchema(
        domain_size=L,
        grid_points=N,
        dt=dt_implicit,
        diffusion_coefficient=D
    )
    schema_implicit.set_initial_condition(ic)
    schema_implicit.set_boundary_conditions(bc)
    
    start = time.time()
    schema_implicit.solve(t_final=t_final)
    time_implicit = time.time() - start
    u_implicit = schema_implicit.get_state()
    steps_implicit = int(t_final / dt_implicit)
    
    print(f"  Completed in {time_implicit:.4f} seconds ({steps_implicit} steps)")
    
    # ========== Implicit ADI ==========
    print("\n[2/2] Running ADI Implicit method...")
    print(f"  Time step: {dt_adi}")
    
    schema_adi = ADISchema(
        domain_size=L,
        grid_points=N,
        dt=dt_adi,
        diffusion_coefficient=D
    )
    schema_adi.set_initial_condition(ic)
    schema_adi.set_boundary_conditions(bc)
    
    start = time.time()
    schema_adi.solve(t_final=t_final)
    time_adi = time.time() - start
    u_adi = schema_adi.get_state()
    steps_adi = int(t_final / dt_adi)
    
    print(f"  Completed in {time_adi:.4f} seconds ({steps_adi} steps)")
    
    # Analytical solution (approximation)
    u_analytical = analytical_solution_3d(X, Y, Z, t_final, D)
    
    # Compute errors
    error_implicit = np.linalg.norm(u_implicit - u_analytical) / np.linalg.norm(u_analytical)
    error_adi = np.linalg.norm(u_adi - u_analytical) / np.linalg.norm(u_analytical)
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'Time Step':<12} {'Steps':<8} {'Time (s)':<12} {'Rel. Error':<12}")
    print("-" * 70)
    print(f"{'Implicit Euler':<20} {dt_implicit:<12.6f} {steps_implicit:<8} {time_implicit:<12.4f} {error_implicit:<12.6e}")
    print(f"{'ADI Implicit':<20} {dt_adi:<12.6f} {steps_adi:<8} {time_adi:<12.4f} {error_adi:<12.6e}")
    print("=" * 70)
    
    # Plotting results
    fig, axes = plt.subplots(3,3, figsize=(10, 10))
    
    mid_z = u_implicit.shape[2] // 2  # Middle slice in z-direction
    mid_y = u_implicit.shape[1] // 2  # Middle slice in y-direction
    mid_x = u_implicit.shape[0] // 2  # Middle slice in x-direction

    # 1st row: Implicit Euler
    im1 = axes[0,0].imshow(u_implicit[:, mid_y, :].T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    axes[0,0].set_title("Implicit Euler (y-slice)")
    plt.colorbar(im1, ax=axes[0,0], fraction=0.046) 
    im2 = axes[0,1].imshow(u_implicit[mid_x, :, :].T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    axes[0,1].set_title("Implicit Euler (x-slice)")
    plt.colorbar(im2, ax=axes[0,1], fraction=0.046)
    im3 = axes[0,2].imshow(u_implicit[:, :, mid_z].T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    axes[0,2].set_title("Implicit Euler (z-slice)")
    plt.colorbar(im3, ax=axes[0,2], fraction=0.046)

    # 2nd row: ADI Implicit
    im4 = axes[1,0].imshow(u_adi[:, mid_y, :].T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    axes[1,0].set_title("ADI (y-slice)")
    plt.colorbar(im4, ax=axes[1,0], fraction=0.046) 
    im5 = axes[1,1].imshow(u_adi[mid_x, :, :].T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    axes[1,1].set_title("ADI (x-slice)")
    plt.colorbar(im5, ax=axes[1,1], fraction=0.046)
    im6 = axes[1,2].imshow(u_adi[:, :, mid_z].T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    axes[1,2].set_title("ADI (z-slice)")
    plt.colorbar(im6, ax=axes[1,2], fraction=0.046)

    # 3rd row: Analytical Solution
    im7 = axes[2,0].imshow(u_adi[:, mid_y, :].T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    axes[2,0].set_title("ADI (y-slice)")
    plt.colorbar(im7, ax=axes[2,0], fraction=0.046) 
    im8 = axes[2,1].imshow(u_adi[mid_x, :, :].T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    axes[2,1].set_title("ADI (x-slice)")
    plt.colorbar(im8, ax=axes[2,1], fraction=0.046)
    im9 = axes[2,2].imshow(u_adi[:, :, mid_z].T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    axes[2,2].set_title("ADI (z-slice)")
    plt.colorbar(im9, ax=axes[2,2], fraction=0.046)

    # Plotting performance
    fig = plt.figure(figsize=(5, 10))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    methods = ['Standard Implicit', 'ADI Implicit']
    times = [time_implicit, time_adi]
    colors = ['#440154', '#22a884'] # Matching viridis-style tones
    bars = ax1.bar(methods, times, color=colors, alpha=0.7, width=0.6)
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Performance Comparison (Lower is Better)')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, t in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{t:.4f}s', ha='center', va='bottom')

    errors = [error_implicit, error_adi]
    bars_err = ax2.bar(methods, errors, color=colors, alpha=0.7, width=0.6)
    ax2.set_ylabel('Relative L2 Error')
    ax2.set_title('Accuracy Comparison (Lower is Better)')
    # ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y', which='both') # 'both' shows major and minor lines
    for bar, e in zip(bars_err, errors):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), 
                 f'{e:.2e}', ha='center', va='bottom')
        
    # Clean up layout
    plt.tight_layout()
    plt.savefig('adi_vs_implicit_3d', dpi=150)
    plt.show()


if __name__ == "__main__":
    run_comparison()