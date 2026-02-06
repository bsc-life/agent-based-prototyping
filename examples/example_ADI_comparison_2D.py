"""
Example 3: Comparison of Numerical Methods

Compares the accuracy and stability of different numerical methods
(Standard Implicit Euler vs ADI Implicit Euler) on the same problem.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
# Assuming ADISchema is available in diffusion_schemas
from diffusion_schemas import ImplicitEulerSchema, ADISchema
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
    dt_implicit = 0.001     # Standard Implicit
    dt_adi = 0.001          # ADI (Implicit splitting)
    
    print("=" * 70)
    print("Numerical Methods Comparison: Implicit vs ADI")
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
    u_analytical = analytical_solution_2d(X, Y, t_final, D)
    
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Standard Implicit Result
    im1 = axes[0].imshow(u_implicit.T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    axes[0].set_title("Standard Implicit")
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Plot 2: ADI Result
    im2 = axes[1].imshow(u_adi.T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    axes[1].set_title("ADI Method")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # Plot 3: Difference (ADI - Implicit)
    # This shows the "splitting error" introduced by ADI
    diff = u_adi - u_implicit
    im3 = axes[2].imshow(diff.T, origin='lower', extent=[0,1,0,1], cmap='coolwarm')
    axes[2].set_title("Difference (ADI - Implicit)")
    plt.colorbar(im3, ax=axes[2], fraction=0.046)

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
    
    # Plot 3: ADI Implicit Solution
    im3 = ax3.imshow(u_adi.T, origin='lower', extent=[0,1,0,1], cmap='viridis')
    ax3.set_title('ADI Implicit Method')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # --- ROW 2: COMPARISONS ---

    # Plot 4: Computation Time (Bar Chart)
    methods = ['Standard Implicit', 'ADI Implicit']
    times = [time_implicit, time_adi]
    colors = ['#440154', '#22a884'] # Matching viridis-style tones
    bars = ax4.bar(methods, times, color=colors, alpha=0.7, width=0.6)
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Performance Comparison (Lower is Better)')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, t in zip(bars, times):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{t:.4f}s', ha='center', va='bottom')

    # Plot 5: Accuracy / Error Map (Analytical - ADI)
    # This keeps your requested difference logic for the final plot
    error_map = u_analytical - u_adi
    im5 = ax5.imshow(error_map.T, origin='lower', extent=[0,1,0,1], cmap='coolwarm')
    ax5.set_title('ADI Accuracy: Analytical - ADI')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    ax5.set_xlabel('Position x')
    
    # Clean up layout
    plt.tight_layout()
    plt.savefig('adi_vs_implicit_2d.png', dpi=150)
    plt.show()
    
if __name__ == "__main__":
    run_comparison()