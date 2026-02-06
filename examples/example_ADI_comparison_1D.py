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


def analytical_solution_1d(x, t, D, L):
    """
    Analytical solution for 1D diffusion with Neumann BC and Gaussian IC.
    This is an approximation using the fundamental solution.
    """
    # For simplicity, use the fundamental solution (infinite domain approximation)
    # True solution with Neumann BC is more complex
    sigma_initial = 0.05
    sigma_t = np.sqrt(sigma_initial**2 + 2*D*t)
    amplitude = sigma_initial / sigma_t
    return amplitude * np.exp(-(x - 0.5)**2 / (2 * sigma_t**2))


def run_comparison():
    # Parameters
    L = 1.0
    N = 100
    D = 0.1
    t_final = 0.1
    
    # Time steps
    dt_implicit = 0.001     # Standard Implicit
    dt_adi = 0.001          # ADI (Implicit splitting)
    
    print("=" * 70)
    print("Numerical Methods Comparison: Implicit vs ADI")
    print("=" * 70)
    print(f"Problem: 1D diffusion with Gaussian initial condition")
    print(f"Domain: [0, {L}], Grid points: {N}")
    print(f"Diffusion coefficient: {D}")
    print(f"Final time: {t_final}")
    print("=" * 70)
    
    # Create coordinate array
    x = np.linspace(0, L, N)
    
    # Initial condition
    ic = gaussian(center=0.5, amplitude=1.0, width=0.05)
    
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
    u_analytical = analytical_solution_1d(x, t_final, D, L)
    
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
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: All solutions
    ax = axes[0, 0]
    ax.plot(x, u_analytical, 'k--', linewidth=2, label='Analytical', alpha=0.7)
    ax.plot(x, u_implicit, 'r-', linewidth=1.5, label=f'Implicit Euler (dt={dt_implicit})')
    ax.plot(x, u_adi, 'g--', linewidth=1.5, label=f'ADI Implicit (dt={dt_adi})')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Concentration u')
    ax.set_title(f'Solution Comparison at t={t_final}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Errors
    ax = axes[0, 1]
    ax.plot(x, u_implicit - u_analytical, 'r-', label='Implicit Euler')
    ax.plot(x, u_adi - u_analytical, 'g--', label='ADI Implicit')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Position x')
    ax.set_ylabel('Error (Numerical - Analytical)')
    ax.set_title('Pointwise Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Performance comparison
    ax = axes[1, 0]
    methods = ['Implicit\nEuler', 'ADI\nImplicit']
    times = [time_implicit, time_adi]
    colors = ['red', 'green']
    bars = ax.bar(methods, times, color=colors, alpha=0.7)
    ax.set_ylabel('Computation Time (s)')
    ax.set_title('Performance Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.4f}s', ha='center', va='bottom')
    
    # Plot 4: Accuracy comparison
    ax = axes[1, 1]
    errors = [error_implicit, error_adi]
    bars = ax.bar(methods, errors, color=colors, alpha=0.7)
    ax.set_ylabel('Relative Error')
    ax.set_yscale('log')
    ax.set_title('Accuracy Comparison (Relative L2 Error)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{err:.2e}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('example_adi_1D_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'example_adi_1D_comparison.png'")
    plt.show()
    
    print("\nKey Observations:")
    print("  • Both methods are unconditionally stable")
    print("  • In 1D, ADI reduces to standard Implicit Euler (results should be identical)")
    print("  • In 2D/3D, ADI would show significant speedup due to tridiagonal solvers")


if __name__ == "__main__":
    run_comparison()