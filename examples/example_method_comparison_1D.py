"""
Example 3: Comparison of Numerical Methods

Compares the accuracy and stability of different numerical methods
(Explicit Euler, Implicit Euler, Crank-Nicolson) on the same problem.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from diffusion_schemas import ExplicitEulerSchema, ImplicitEulerSchema, CrankNicolsonSchema, ImplicitLODSchema, CrankNicolsonLODSchema, \
ExplicitEulerBCSchema, ImplicitEulerBCSchema, CrankNicolsonLODBCSchema, CrankNicolsonBCSchema, ImplicitLODBCSchema
    
from diffusion_schemas.utils import gaussian, NeumannBC


def analytical_solution_1d(x, x0, t, D):
    """
    Analytical solution for 2D diffusion with Neumann BC and Gaussian IC.
    This is an approximation using the fundamental solution.
    """
    # For simplicity, use the fundamental solution (infinite domain approximation)
    # True solution with Neumann BC is more complex
    sigma_initial = 0.05
    sigma_t = np.sqrt(sigma_initial**2 + 2*D*t)
    amplitude = (sigma_initial / sigma_t) # faster amplitude decay in 2D
    return amplitude * np.exp(-((x - x0)**2) / (2 * sigma_t**2))


def run_comparison():
    # Parameters
    L = 1.0
    N = 100
    D = 0.1
    t_final = 0.1
    
    # Different time steps for each method
    dt_explicit = 0.00005  # Small for stability
    dt_implicit = 0.00005    # Can be larger
    dt_crank = 0.00005       # Can be larger
    dt_lod = 0.00005         # Can be larger
    dt_crank_lod = 0.00005   # Can be larger

    print("=" * 70)
    print("Numerical Methods Comparison")
    print("=" * 70)
    print(f"Problem: 1D diffusion with Gaussian initial condition")
    print(f"Domain: [0, {L}], Grid points: {N}")
    print(f"Diffusion coefficient: {D}")
    print(f"Final time: {t_final}")
    print("=" * 70)
    
    # Create coordinate array
    x = np.linspace(0, L, N)
    
    # Initial condition
    x0 = 0.5
    ic = gaussian(center=x0, amplitude=1.0, width=0.05)

    # Boundary condition (Neumann: zero flux)
    # bc = NeumannBC(flux=0.0)
    
    # ========== Explicit Euler ==========
    print("\n[1/5] Running Explicit Euler method...")
    print(f"  Time step: {dt_explicit}")
    
    schema_explicit = ExplicitEulerSchema(
        domain_size=L,
        grid_points=N,
        dt=dt_explicit,
        diffusion_coefficient=D,
        check_stability=True
    )
    schema_explicit.set_initial_condition(ic)
    # schema_explicit.set_boundary_conditions(bc)
    
    start = time.time()
    schema_explicit.solve(t_final=t_final)
    time_explicit = time.time() - start
    u_explicit = schema_explicit.get_state()
    steps_explicit = int(t_final / dt_explicit)
    
    print(f"  Completed in {time_explicit:.4f} seconds ({steps_explicit} steps)")
    
    # ========== Implicit Euler ==========
    print("\n[2/5] Running Implicit Euler method...")
    print(f"  Time step: {dt_implicit}")
    
    schema_implicit = ImplicitEulerSchema(
        domain_size=L,
        grid_points=N,
        dt=dt_implicit,
        diffusion_coefficient=D
    )
    schema_implicit.set_initial_condition(ic)
    #schema_implicit.set_boundary_conditions(bc)
    
    start = time.time()
    schema_implicit.solve(t_final=t_final)
    time_implicit = time.time() - start
    u_implicit = schema_implicit.get_state()
    steps_implicit = int(t_final / dt_implicit)
    
    print(f"  Completed in {time_implicit:.4f} seconds ({steps_implicit} steps)")
    
    # ========== Crank-Nicolson ==========
    print("\n[3/5] Running Crank-Nicolson method...")
    print(f"  Time step: {dt_crank}")
    
    schema_crank = CrankNicolsonSchema(
        domain_size=L,
        grid_points=N,
        dt=dt_crank,
        diffusion_coefficient=D
    )
    schema_crank.set_initial_condition(ic)
    #schema_crank.set_boundary_conditions(bc)
    
    start = time.time()
    schema_crank.solve(t_final=t_final)
    time_crank = time.time() - start
    u_crank = schema_crank.get_state()
    steps_crank = int(t_final / dt_crank)
    
    print(f"  Completed in {time_crank:.4f} seconds ({steps_crank} steps)")

    # ========== Alternate-Direction Implicit (ADI) ==========
    print("\n[4/5] Running Alternate-Direction Implicit (ADI) method...")
    print(f"  Time step: {dt_lod}")
    
    schema_lod = ImplicitLODSchema(
        domain_size=L,
        grid_points=N,
        dt=dt_lod,
        diffusion_coefficient=D
    )
    schema_lod.set_initial_condition(ic)
    #schema_lod.set_boundary_conditions(bc)
    
    start = time.time()
    schema_lod.solve(t_final=t_final)
    time_lod = time.time() - start
    u_lod = schema_lod.get_state()
    steps_lod = int(t_final / dt_lod)
    
    print(f"  Completed in {time_lod:.4f} seconds ({steps_lod} steps)")
    
    # ========== Crank-Nicolson with LOD ==========
    print("\n[5/5] Running Crank-Nicolson with LOD method...")
    print(f"  Time step: {dt_crank_lod}")
    
    schema_crank_lod = CrankNicolsonLODSchema(
        domain_size=L,
        grid_points=N,
        dt=dt_crank_lod,
        diffusion_coefficient=D
    )
    schema_crank_lod.set_initial_condition(ic)
    # schema_crank_lod.set_boundary_conditions(bc)
    
    start = time.time()
    schema_crank_lod.solve(t_final=t_final)
    time_crank_lod = time.time() - start
    u_crank_lod = schema_crank_lod.get_state()
    steps_crank_lod = int(t_final / dt_crank_lod)
    
    print(f"  Completed in {time_crank_lod:.4f} seconds ({steps_crank_lod} steps)")
    
    # Analytical solution (approximation)
    u_analytical = analytical_solution_1d(x, x0, t_final, D)
    
    # Compute errors
    error_explicit = np.linalg.norm(u_explicit - u_analytical) / np.linalg.norm(u_analytical)
    error_implicit = np.linalg.norm(u_implicit - u_analytical) / np.linalg.norm(u_analytical)
    error_crank = np.linalg.norm(u_crank - u_analytical) / np.linalg.norm(u_analytical)
    error_lod = np.linalg.norm(u_lod - u_analytical) / np.linalg.norm(u_analytical)
    error_crank_lod = np.linalg.norm(u_crank_lod - u_analytical) / np.linalg.norm(u_analytical)
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'Time Step':<12} {'Steps':<8} {'Time (s)':<12} {'Rel. Error':<12}")
    print("-" * 70)
    print(f"{'Explicit Euler':<25} {dt_explicit:<12.6f} {steps_explicit:<8} {time_explicit:<12.4f} {error_explicit:<12.6e}")
    print(f"{'Implicit Euler':<25} {dt_implicit:<12.6f} {steps_implicit:<8} {time_implicit:<12.4f} {error_implicit:<12.6e}")
    print(f"{'Crank-Nicolson':<25} {dt_crank:<12.6f} {steps_crank:<8} {time_crank:<12.4f} {error_crank:<12.6e}")
    print(f"{'Implicit LOD':<25} {dt_lod:<12.6f} {steps_lod:<8} {time_lod:<12.4f} {error_lod:<12.6e}")
    print(f"{'Crank-Nicolson LOD':<25} {dt_crank_lod:<12.6f} {steps_crank_lod:<8} {time_crank_lod:<12.4f} {error_crank_lod:<12.6e}")
    print("=" * 70)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: All solutions
    ax = axes[0, 0]
    ax.plot(x, u_analytical, 'k--', linewidth=2, label='Analytical', alpha=0.7)
    ax.plot(x, u_explicit, 'b-', linewidth=1.5, label='Explicit')
    ax.plot(x, u_implicit, 'r-', linewidth=1.5, label='Implicit')
    ax.plot(x, u_crank, 'g-', linewidth=1.5, label='Crank-Nicolson')
    ax.plot(x, u_lod, 'm-', linewidth=1.5, label='Implicit LOD')
    ax.plot(x, u_crank_lod, 'c-', linewidth=1.5, label='Crank-Nicolson LOD')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Concentration u (y=0.5)')
    ax.set_title(f'Solution Comparison at t={t_final}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Errors
    ax = axes[0, 1]
    ax.plot(x, abs(u_explicit - u_analytical), 'b-', label='Explicit')
    ax.plot(x, abs(u_implicit - u_analytical), 'r-', label='Implicit')
    ax.plot(x, abs(u_crank - u_analytical), 'g-', label='Crank-Nicolson')
    ax.plot(x, abs(u_lod - u_analytical), 'm-', label='Implicit LOD')
    ax.plot(x, abs(u_crank_lod - u_analytical), 'c-', label='Crank-Nicolson LOD')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Position x')
    ax.set_ylabel('Asbolute Error')
    ax.set_title('Pointwise Absolute Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Performance comparison
    ax = axes[1, 0]
    methods = ['Explicit\nEuler', 'Implicit\nEuler', 'Crank-\nNicolson', 'Implicit LOD', 'CN-LOD']
    times = [time_explicit, time_implicit, time_crank, time_lod, time_crank_lod]
    colors = ['blue', 'red', 'green', 'magenta', 'cyan']
    bars = ax.bar(methods, times, color=colors, alpha=0.7)
    ax.set_ylabel('Computation Time (s)')
    ax.set_title('Performance Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.4f}s', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Accuracy comparison
    ax = axes[1, 1]
    errors = [error_explicit, error_implicit, error_crank, error_lod, error_crank_lod]
    bars = ax.bar(methods, errors, color=colors, alpha=0.7)
    ax.set_ylabel('Relative Error')
    ax.set_yscale('log')
    ax.set_title('Accuracy Comparison (Relative L2 Error)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{err:.2e}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('example_method_comparison_1D', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'example_method_comparison_1D.png'")
    plt.show()
    
    """     
    print("\nKey Observations:")
    print("  • Explicit Euler requires small time steps for stability")
    print("  • Implicit methods allow larger time steps but solve linear systems")
    print("  • Crank-Nicolson achieves best accuracy (2nd order in time)")
    print("  • LOD and CN-LOD are faster for multi-dimensional problems")
    print("  • Trade-off between step size, stability, and computational cost") 
    """


if __name__ == "__main__":
    run_comparison()
