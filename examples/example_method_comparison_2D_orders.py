import numpy as np
import matplotlib.pyplot as plt
import time
from diffusion_schemas import ExplicitEulerSchema, ImplicitEulerSchema, CrankNicolsonSchema, ADISchema, CrankNicolsonADISchema
from diffusion_schemas.utils import gaussian, NeumannBC
from itertools import product


def analytical_solution_2d(x, y, x0, y0, t, D):
    """
    Analytical solution for 2D diffusion with Neumann BC and Gaussian IC.
    This is an approximation using the fundamental solution.
    """
    # For simplicity, use the fundamental solution (infinite domain approximation)
    # True solution with Neumann BC is more complex
    sigma_initial = 0.05
    sigma_t = np.sqrt(sigma_initial**2 + 2*D*t)
    amplitude = (sigma_initial / sigma_t) ** 2 # faster amplitude decay in 2D
    return amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma_t**2))


def run_comparison(dx, dt):  

    # Assume uniform spacing and same dt for all schemes

    L = (1.0, 1.0)
    N = (int(L[0]/dx), int(L[1]/dx))  # Adjust grid points based on dx
    D = 0.1
    t_final = 0.1
    
    # Different time steps for each method
    dt_explicit = dt  # Small for stability
    dt_implicit = dt    # Can be larger
    dt_crank = dt       # Can be larger
    dt_adi = dt         # Can be larger
    dt_crank_adi = dt   # Can be larger

    print("=" * 70)
    print("Numerical Methods Comparison")
    print("=" * 70)
    print(f"Problem: 2D diffusion with Gaussian initial condition")
    print(f"Domain: [(0,0), {L}], Grid points: {N}")
    print(f"Diffusion coefficient: {D}")
    print(f"Final time: {t_final}")
    print("=" * 70)
    
    # Create coordinate array
    x = np.linspace(0, L[0], N[0])
    y = np.linspace(0, L[1], N[1])
    X, Y = np.meshgrid(x, y, indexing='ij') 
    
    # Initial condition
    x0 = 0.5
    y0 = 0.5
    ic = gaussian(center=(x0, y0), amplitude=1.0, width=0.05)

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
    print(f"  Time step: {dt_adi}")
    
    schema_adi = ADISchema(
        domain_size=L,
        grid_points=N,
        dt=dt_adi,
        diffusion_coefficient=D
    )
    schema_adi.set_initial_condition(ic)
    #schema_adi.set_boundary_conditions(bc)
    
    start = time.time()
    schema_adi.solve(t_final=t_final)
    time_adi = time.time() - start
    u_adi = schema_adi.get_state()
    steps_adi = int(t_final / dt_adi)
    
    print(f"  Completed in {time_adi:.4f} seconds ({steps_adi} steps)")
    
    # ========== Crank-Nicolson with ADI ==========
    print("\n[5/5] Running Crank-Nicolson with ADI method...")
    print(f"  Time step: {dt_crank_adi}")
    
    schema_crank_adi = CrankNicolsonADISchema(
        domain_size=L,
        grid_points=N,
        dt=dt_crank_adi,
        diffusion_coefficient=D
    )
    schema_crank_adi.set_initial_condition(ic)
    # schema_crank_adi.set_boundary_conditions(bc)
    
    start = time.time()
    schema_crank_adi.solve(t_final=t_final)
    time_crank_adi = time.time() - start
    u_crank_adi = schema_crank_adi.get_state()
    steps_crank_adi = int(t_final / dt_crank_adi)
    
    print(f"  Completed in {time_crank_adi:.4f} seconds ({steps_crank_adi} steps)")
    
    # Analytical solution (approximation)
    u_analytical = analytical_solution_2d(X, Y, x0, y0, t_final, D)
    
    # Compute errors
    error_explicit = np.sqrt(np.mean((u_explicit - u_analytical) ** 2))
    error_implicit = np.sqrt(np.mean((u_implicit - u_analytical) ** 2))
    error_crank = np.sqrt(np.mean((u_crank - u_analytical) ** 2))
    error_adi = np.sqrt(np.mean((u_adi - u_analytical) ** 2))
    error_crank_adi = np.sqrt(np.mean((u_crank_adi - u_analytical) ** 2))
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'Time Step':<12} {'Steps':<8} {'Time (s)':<12} {'Rel. Error':<12}")
    print("-" * 70)
    print(f"{'Explicit Euler':<25} {dt_explicit:<12.6f} {steps_explicit:<8} {time_explicit:<12.4f} {error_explicit:<12.6e}")
    print(f"{'Implicit Euler':<25} {dt_implicit:<12.6f} {steps_implicit:<8} {time_implicit:<12.4f} {error_implicit:<12.6e}")
    print(f"{'Crank-Nicolson':<25} {dt_crank:<12.6f} {steps_crank:<8} {time_crank:<12.4f} {error_crank:<12.6e}")
    print(f"{'ADI':<25} {dt_adi:<12.6f} {steps_adi:<8} {time_adi:<12.4f} {error_adi:<12.6e}")
    print(f"{'Crank-Nicolson ADI':<25} {dt_crank_adi:<12.6f} {steps_crank_adi:<8} {time_crank_adi:<12.4f} {error_crank_adi:<12.6e}")
    print("=" * 70)
    
    """
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    mid_y = N[1] // 2  # Index for the middle row to plot 1D slice example    

    # Plot 1: All solutions
    ax = axes[0, 0]
    # Slicing 2D arrays to 1D: [:, mid_y]
    ax.plot(x, u_analytical[:, mid_y], 'k--', linewidth=2, label='Analytical', alpha=0.7)
    ax.plot(x, u_explicit[:, mid_y], 'b-', linewidth=1.5, label='Explicit')
    ax.plot(x, u_implicit[:, mid_y], 'r-', linewidth=1.5, label='Implicit')
    ax.plot(x, u_crank[:, mid_y], 'g-', linewidth=1.5, label='Crank-Nicolson')
    ax.plot(x, u_adi[:, mid_y], 'm-', linewidth=1.5, label='ADI')
    ax.plot(x, u_crank_adi[:, mid_y], 'c-', linewidth=1.5, label='Crank-Nicolson ADI')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Concentration u (y=0.5)')
    ax.set_title(f'Solution Comparison at t={t_final}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Errors
    ax = axes[0, 1]
    ax.plot(x, abs(u_explicit[:, mid_y] - u_analytical[:, mid_y]), 'b-', label='Explicit')
    ax.plot(x, abs(u_implicit[:, mid_y] - u_analytical[:, mid_y]), 'r-', label='Implicit')
    ax.plot(x, abs(u_crank[:, mid_y] - u_analytical[:, mid_y]), 'g-', label='Crank-Nicolson')
    ax.plot(x, abs(u_adi[:, mid_y] - u_analytical[:, mid_y]), 'm-', label='ADI')
    ax.plot(x, abs(u_crank_adi[:, mid_y] - u_analytical[:, mid_y]), 'c-', label='Crank-Nicolson ADI')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Position x')
    ax.set_ylabel('Asbolute Error (y=0.5)')
    ax.set_title('Pointwise Absolute Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Performance comparison
    ax = axes[1, 0]
    methods = ['Explicit\nEuler', 'Implicit\nEuler', 'Crank-\nNicolson', 'ADI', 'CN-ADI']
    times = [time_explicit, time_implicit, time_crank, time_adi, time_crank_adi]
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
    errors = [error_explicit, error_implicit, error_crank, error_adi, error_crank_adi]
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
    plt.savefig('example_method_comparison_2D', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'example_method_comparison_2D.png'")
    plt.show()
    """
    """     
    print("\nKey Observations:")
    print("  • Explicit Euler requires small time steps for stability")
    print("  • Implicit methods allow larger time steps but solve linear systems")
    print("  • Crank-Nicolson achieves best accuracy (2nd order in time)")
    print("  • ADI and CN-ADI are faster for multi-dimensional problems")
    print("  • Trade-off between step size, stability, and computational cost") 
    """

    errors = np.array([error_explicit, error_implicit, error_crank, error_adi, error_crank_adi])
    return errors


if __name__ == "__main__":
    dx_values = np.array([0.1, 0.075, 0.05, 0.025, 0.01])
    dt_values = np.array([0.001, 0.0001, 0.00001, 0.000001])  
    
    # Initialize a dictionary to store errors for each combination of dx and dt
    error_data = {}

    # Iterate over all combinations of dx and dt
    for dx, dt in product(dx_values, dt_values):
        print(f"\nRunning comparison with dx={dx:.3f} and dt={dt:.0e}")
        errors = run_comparison(dx, dt)
        error_data[(dx, dt)] = errors  # Store errors for this combination

    # Prepare data for plotting convergence
    methods = [
        # 'Explicit Euler',  # Commented out to exclude the explicit method
        'Implicit Euler',
        'Crank-Nicolson',
        'ADI',
        'Crank-Nicolson ADI'
    ]

    # Create a figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()  # Flatten the 2x2 array of axes for easier indexing

    # Define theoretical convergence orders for each method
    theoretical_orders = {
        'Implicit Euler': 1,  # First-order method
        'Crank-Nicolson': 2,  # Second-order method
        'ADI': 2,  # Second-order method
        'Crank-Nicolson ADI': 2  # Second-order method
    }

    # Iterate over methods and create a subplot for each
    for method_idx, method in enumerate(methods):
        ax = axes[method_idx]

        # Plot error curves for each dt value
        for dt in dt_values:
            errors_for_dt = [
                error_data[(dx, dt)][method_idx + 1]  # Adjusted index to skip explicit method
                for dx in dx_values
            ]
            ax.plot(
                dx_values,
                errors_for_dt,
                marker="o",
                label=f"dt={dt:.0e}"
            )

        # Add theoretical convergence line
        theoretical_order = theoretical_orders[method]
        theoretical_errors = [dx**theoretical_order for dx in dx_values]
        ax.plot(
            dx_values,
            theoretical_errors,
            linestyle="--",
            color="grey",
            label=f"Theoretical (O(dx^{theoretical_order}))"
        )

        # Set subplot labels and title
        ax.set_xlabel("dx")
        ax.set_ylabel("Error (RMSE)")
        ax.set_title(f"{method}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig("convergence_analysis_subplots_with_theoretical.png", dpi=150, bbox_inches="tight")
    print("\nConvergence analysis plot saved as 'convergence_analysis_subplots_with_theoretical.png'")
    plt.show()