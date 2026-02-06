"""
Example 1: Basic 1D Diffusion

Demonstrates simple 1D diffusion of a Gaussian initial condition
using the explicit Euler method.
"""

import numpy as np
import matplotlib.pyplot as plt
from diffusion_schemas import ADISchema
from diffusion_schemas.utils import gaussian, DirichletBC


def main():
    # Parameters
    L = 1.0  # Domain length
    N = 100  # Number of grid points
    D = 0.1  # Diffusion coefficient
    dt = 0.0001  # Time step
    t_final = 0.1  # Final time
    
    print("=" * 60)
    print("1D Diffusion Example")
    print("=" * 60)
    print(f"Domain: [0, {L}]")
    print(f"Grid points: {N}")
    print(f"Diffusion coefficient: {D}")
    print(f"Time step: {dt}")
    print(f"Final time: {t_final}")
    print("=" * 60)
    
    # Create schema
    schema = ADISchema(
        domain_size=L,
        grid_points=N,
        dt=dt,
        diffusion_coefficient=D
    )
    
    # Set initial condition: Gaussian centered at x=0.5
    ic = gaussian(center=0.5, amplitude=1.0, width=0.05)
    schema.set_initial_condition(ic)
    
    # Store initial state
    u_initial = schema.get_state()
    
    # Set boundary conditions (Dirichlet: u=0 at boundaries)
    schema.set_boundary_conditions(DirichletBC(value=0.0))
    
    # Solve
    print(f"\nSolving from t=0 to t={t_final}...")
    history = schema.solve(t_final=t_final, store_history=True)
    print(f"Completed {len(history)} time steps")
    
    # Get final state
    u_final = schema.get_state()
    
    # Create coordinate array
    x = np.linspace(0, L, N)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot initial and final states
    ax1.plot(x, u_initial, 'b-', label='Initial (t=0)', linewidth=2)
    ax1.plot(x, u_final, 'r-', label=f'Final (t={t_final})', linewidth=2)
    ax1.set_xlabel('Position x')
    ax1.set_ylabel('Concentration u')
    ax1.set_title('1D Diffusion: Initial and Final States')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot evolution over time (every 10th frame)
    skip = max(1, len(history) // 20)
    times = np.arange(0, len(history) * dt, skip * dt)
    for i, t in enumerate(times):
        idx = int(i * skip)
        if idx < len(history):
            alpha = 0.3 + 0.7 * (i / len(times))
            color = plt.cm.viridis(i / len(times))
            ax2.plot(x, history[idx], color=color, alpha=alpha, linewidth=1)
    
    ax2.set_xlabel('Position x')
    ax2.set_ylabel('Concentration u')
    ax2.set_title('Diffusion Evolution Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar to show time
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=t_final))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label('Time')
    
    plt.tight_layout()
    plt.savefig('example_1d_diffusion_ADI.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'example_1d_diffusion_ADI.png'")
    plt.show()
    
    # Print some statistics
    print("\nStatistics:")
    print(f"  Initial max: {np.max(u_initial):.6f}")
    print(f"  Final max: {np.max(u_final):.6f}")
    print(f"  Initial integral: {np.trapezoid(u_initial, x):.6f}")
    print(f"  Final integral: {np.trapezoid(u_final, x):.6f}")
    print(f"  Mass loss: {(1 - np.trapezoid(u_final, x) / np.trapezoid(u_initial, x)) * 100:.2f}%")


if __name__ == "__main__":
    main()
