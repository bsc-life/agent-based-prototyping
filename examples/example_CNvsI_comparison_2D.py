"""
Example 3: Comparison of Numerical Methods

Compares the accuracy and stability of different numerical methods
(Standard Implicit Euler vs Crank-Nicolson) on the same problem.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from diffusion_schemas import ImplicitEulerSchema, CrankNicolsonSchema
from diffusion_schemas.utils import gaussian, NeumannBC


def analytical_solution_2d(x, y, t, D):
    """
    Analytical solution for 2D diffusion with Neumann BC and Gaussian IC.
    This is an approximation using the fundamental solution.
    """
    # For simplicity, use the fundamental solution (infinite domain approximation)
    # True solution with Neumann BC is more complex
    sigma_initial = 0.05
    sigma_t = np.sqrt(sigma_initial**2 + 2*D*t)
    amplitude = (sigma_initial / sigma_t) ** 2 # faster amplitude decay in 2D
    return amplitude * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / (2 * sigma_t**2))


def run_comparison():
    # Parameters are now tuples for 2D
    L = (1.0, 1.0) 
    N = (100, 100)
    D = 0.1
    t_final = 0.1
    
    # Time steps
    dt_implicit = 0.00005   # Standard Implicit
    dt_cn = 0.00005          # Crank-Nicolson 
    
    print("=" * 70)
    print("Numerical Methods Comparison: Implicit vs CN")
    print("=" * 70)
    print(f"Problem: 2D diffusion with Gaussian initial condition")
    print(f"Domain: [(0, 0), {L}], Grid points: {N}")
    print(f"Diffusion coefficient: {D}")
    print(f"Final time: {t_final}")
    print("=" * 70)
    
    # Create coordinate array (x and y instead of just x, and meshgrid for 2D)
    x = np.linspace(0, L[0], N[0])
    y = np.linspace(0, L[1], N[1])
    X, Y = np.meshgrid(x, y, indexing='ij') 
    
    # Initial condition
    ic = gaussian(center=(0.5, 0.5), amplitude=1.0, width=0.05)

    # Boundary condition (Neumann: zero flux)
    # bc = NeumannBC(flux=0.0)
    
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
    # schema_implicit.set_boundary_conditions(bc)
    
    start = time.time()
    schema_implicit.solve(t_final=t_final)
    time_implicit = time.time() - start
    u_implicit = schema_implicit.get_state()
    steps_implicit = int(t_final / dt_implicit)
    
    print(f"  Completed in {time_implicit:.4f} seconds ({steps_implicit} steps)")
    
    # ========== Implicit CN ==========
    print("\n[2/2] Running CN Implicit method...")
    print(f"  Time step: {dt_cn}")
    
    schema_cn = CrankNicolsonSchema(
        domain_size=L,
        grid_points=N,
        dt=dt_cn,
        diffusion_coefficient=D
    )
    schema_cn.set_initial_condition(ic)
    # schema_cn.set_boundary_conditions(bc)
    
    start = time.time()
    schema_cn.solve(t_final=t_final)
    time_cn = time.time() - start
    u_cn = schema_cn.get_state()
    steps_cn = int(t_final / dt_cn)
    
    print(f"  Completed in {time_cn:.4f} seconds ({steps_cn} steps)")
    
    # Analytical solution (approximation)
    u_analytical = analytical_solution_2d(X, Y, t_final, D)
    
    # Compute errors
    error_rel_implicit = np.linalg.norm(u_implicit - u_analytical) / np.linalg.norm(u_analytical)
    error_rel_cn = np.linalg.norm(u_cn - u_analytical) / np.linalg.norm(u_analytical)
    error_implicit = np.linalg.norm(u_implicit - u_analytical) 
    error_cn = np.linalg.norm(u_cn - u_analytical)
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Method':<20} {'Time Step':<12} {'Steps':<8} {'Time (s)':<12} {'Rel. Error':<12} {'Abs. Error':<12}")
    print("-" * 80)
    print(f"{'Implicit Euler':<20} {dt_implicit:<12.6f} {steps_implicit:<8} {time_implicit:<12.4f} {error_rel_implicit:<12.6e} {error_implicit:<12.6e}")
    print(f"{'Crank-Nicolson':<20} {dt_cn:<12.6f} {steps_cn:<8} {time_cn:<12.4f} {error_rel_cn:<12.6e} {error_cn:<12.6e}")
    print("=" * 80)
    
    """     
    # Plotting results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Standard Implicit Result
    im1 = axes[0].imshow(u_implicit.T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    axes[0].set_title("Standard Implicit")
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Plot 2: Crank-Nicolson Result
    im2 = axes[1].imshow(u_cn.T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    axes[1].set_title("Crank-Nicolson Method")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # Plot 3: Difference (Crank-Nicolson - Implicit)
    # This shows the "splitting error" introduced by Crank-Nicolson vs Implicit
    diff = u_cn - u_implicit
    im3 = axes[2].imshow(diff.T, origin='lower', extent=[0,1,0,1], cmap='coolwarm')
    axes[2].set_title("Difference (Crank-Nicolson - Implicit)")
    plt.colorbar(im3, ax=axes[2], fraction=0.046) 
    """

    # Create the figure
    fig = plt.figure(figsize=(18, 10))
    
    # Define a 2-row, 6-column grid to allow for easy alignment
    # (3 columns on top = 2 grid units each; 2 columns on bottom = 3 grid units each)
    ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
    ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
    ax4 = plt.subplot2grid((2, 6), (1, 0), colspan=3)
    ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=3)

    # --- ROW 1: SOLUTIONS ---
    
    # Plot 1: Analytical Solution
    im1 = ax1.imshow(u_analytical.T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    ax1.set_title(f'Analytical Solution (t={t_final})')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    ax1.set_ylabel('Position y')
    
    # Plot 2: Standard Implicit Solution
    im2 = ax2.imshow(u_implicit.T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    ax2.set_title('Standard Implicit Euler')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Plot 3: Crank-Nicolson Implicit Solution
    im3 = ax3.imshow(u_cn.T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    ax3.set_title('Crank-Nicolson Method')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # --- ROW 2: COMPARISONS ---

    # Plot 4: Computation Time (Bar Chart)
    methods = ['Standard Implicit', 'Crank-Nicolson']
    times = [time_implicit, time_cn]
    colors = ['#440154', '#22a884'] # Matching viridis-style tones
    bars = ax4.bar(methods, times, color=colors, alpha=0.7, width=0.6)
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Performance Comparison (Lower is Better)')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, t in zip(bars, times):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{t:.4f}s', ha='center', va='bottom')

    # Plot 5: Accuracy / Error Map (Analytical - Crank-Nicolson)
    # This keeps your requested difference logic for the final plot
    error_map = u_analytical - u_cn
    im5 = ax5.imshow(error_map.T, origin='lower', extent=[0,1,0,1], cmap='coolwarm')
    ax5.set_title('Crank-Nicolson Accuracy: Analytical - Crank-Nicolson')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    ax5.set_xlabel('Position x')
    
    # Clean up layout
    plt.tight_layout()
    plt.savefig('cn_vs_implicit_2d.png', dpi=150)
    plt.show()
    
    # ========== 1D PROFILE COMPARISON ==========
    # Extract the middle of the 2D grid and plot 1D profiles
    mid_idx = N[0] // 2  # Middle index
    
    # Extract horizontal profile through the middle (y = 0.5)
    profile_analytical = u_analytical[mid_idx, :]
    profile_implicit = u_implicit[mid_idx, :]
    profile_cn = u_cn[mid_idx, :]
    
    x_coords = np.linspace(0, L[0], N[0])
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Left plot: 1D Profile Comparison
    ax1.plot(x_coords, profile_analytical, 'ko-', linewidth=2, markersize=4, label='Analytical', alpha=0.7)
    ax1.plot(x_coords, profile_implicit, 'bs--', linewidth=2, markersize=4, label='Implicit Euler', alpha=0.7)
    ax1.plot(x_coords, profile_cn, 'r^--', linewidth=2, markersize=4, label='Crank-Nicolson', alpha=0.7)
    
    ax1.set_xlabel('Position x', fontsize=12)
    ax1.set_ylabel('Concentration', fontsize=12)
    ax1.set_title(f'1D Profile Comparison at y=0.5 (t={t_final})', fontsize=12)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Middle plot: Pointwise Absolute Error in Logarithmic Scale
    error_implicit_profile = np.abs(profile_implicit - profile_analytical)
    error_cn_profile = np.abs(profile_cn - profile_analytical)
    
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-16
    error_implicit_profile = np.maximum(error_implicit_profile, epsilon)
    error_cn_profile = np.maximum(error_cn_profile, epsilon)
    
    ax2.semilogy(x_coords, error_implicit_profile, 'bs--', linewidth=2, markersize=4, label='Implicit Euler', alpha=0.7)
    ax2.semilogy(x_coords, error_cn_profile, 'r^--', linewidth=2, markersize=4, label='Crank-Nicolson', alpha=0.7)
    
    ax2.set_xlabel('Position x', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title(f'Pointwise Absolute Error at y=0.5 (t={t_final})', fontsize=12)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3, which='both')
    
    # Right plot: Pointwise Relative Error in Logarithmic Scale
    rel_error_implicit_profile = np.abs(profile_implicit - profile_analytical) / (np.abs(profile_analytical) + epsilon)
    rel_error_cn_profile = np.abs(profile_cn - profile_analytical) / (np.abs(profile_analytical) + epsilon)
    
    # Avoid log(0) by adding small epsilon
    rel_error_implicit_profile = np.maximum(rel_error_implicit_profile, epsilon)
    rel_error_cn_profile = np.maximum(rel_error_cn_profile, epsilon)
    
    ax3.semilogy(x_coords, rel_error_implicit_profile, 'bs--', linewidth=2, markersize=4, label='Implicit Euler', alpha=0.7)
    ax3.semilogy(x_coords, rel_error_cn_profile, 'r^--', linewidth=2, markersize=4, label='Crank-Nicolson', alpha=0.7)
    
    ax3.set_xlabel('Position x', fontsize=12)
    ax3.set_ylabel('Relative Error', fontsize=12)
    ax3.set_title(f'Pointwise Relative Error at y=0.5 (t={t_final})', fontsize=12)
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('cn_vs_implicit_1d_profile_theta06.png', dpi=150)
    plt.show()
    
if __name__ == "__main__":
    run_comparison()