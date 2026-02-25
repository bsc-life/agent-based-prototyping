"""
Example 2: 2D Diffusion with Multiple Agents

Demonstrates 2D diffusion with multiple substrate-secreting agents
using the Crank-Nicolson method.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from diffusion_schemas import CrankNicolsonADISchema
from diffusion_schemas.methods.crank_nicolson_ADI import CrankNicolsonADISchema
from diffusion_schemas.utils import Agent, DirichletBC
from matplotlib.animation import FuncAnimation


def main():
    # Parameters
    L = 1.0  # Domain size (square domain)
    N = 50  # Grid points per dimension
    D = 0.05  # Diffusion coefficient
    dt = 0.01  # Time step
    t_final = 2.0  # Final time
    
    print("=" * 60)
    print("2D Diffusion with Agents Example")
    print("=" * 60)
    print(f"Domain: [{0}, {L}] x [{0}, {L}]")
    print(f"Grid: {N} x {N}")
    print(f"Diffusion coefficient: {D}")
    print(f"Time step: {dt}")
    print(f"Final time: {t_final}")
    print("=" * 60)
    
    # Create schema
    schema = CrankNicolsonADISchema(
        domain_size=(L, L),
        grid_points=(N, N),
        dt=dt,
        diffusion_coefficient=D,
        decay_rate=0.1  # Small decay to reach steady state
    )
    
    # Set zero initial condition
    schema.set_initial_condition(0.0)
    
    # Add substrate-secreting agents at different positions
    agents = [
        Agent(position=(0.25, 0.25), net_rate=10.0, kernel_width=0.03, name="Agent 1"),
        Agent(position=(0.75, 0.25), net_rate=15.0, kernel_width=0.03, name="Agent 2"),
        Agent(position=(0.5, 0.75), net_rate=8.0, kernel_width=0.03, name="Agent 3"),
    ]
    
    for agent in agents:
        schema.add_agent(agent)
        print(f"Added {agent}")
    
    # Set boundary conditions (zero at boundaries)
    schema.set_boundary_conditions(DirichletBC(value=0.0))
    
    # Solve with history
    print(f"\nSolving from t=0 to t={t_final}...")
    history = schema.solve(t_final=t_final, store_history=True)
    print(f"Completed {len(history)} time steps")
    
    # Create coordinate grids for plotting
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Plot final state
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Contour plot
    u_final = schema.get_state()
    levels = 20
    contour = axes[0].contourf(X, Y, u_final, levels=levels, cmap='hot')
    axes[0].contour(X, Y, u_final, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    
    # Mark agent positions
    for agent in agents:
        axes[0].plot(agent.position[0], agent.position[1], 'b*', markersize=15, 
                    markeredgecolor='white', markeredgewidth=1.5)
        axes[0].text(agent.position[0], agent.position[1] + 0.05, agent.name,
                    ha='center', color='white', fontweight='bold', fontsize=9)
    
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title(f'Concentration Field at t={t_final}')
    axes[0].set_aspect('equal')
    plt.colorbar(contour, ax=axes[0], label='Concentration')
    
    # 3D surface plot
    ax3d = fig.add_subplot(122, projection='3d')
    surf = ax3d.plot_surface(X, Y, u_final, cmap='hot', edgecolor='none', alpha=0.9)
    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('Concentration')
    ax3d.set_title('3D View of Concentration')
    plt.colorbar(surf, ax=ax3d, label='Concentration', shrink=0.5)
    
    plt.tight_layout()
    plt.savefig('example_2d_agents.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'example_2d_agents.png'")
    
    # Create evolution animation
    print("\nCreating animation of concentration evolution...")
    fig_anim, ax_anim = plt.subplots(figsize=(8, 7))
    
    # Select frames to animate (every 5th frame)
    frame_skip = max(1, len(history) // 50)
    frames_to_plot = list(range(0, len(history), frame_skip))
    
    vmin, vmax = 0, np.max([np.max(history[i]) for i in frames_to_plot])
    
    contour_anim = ax_anim.contourf(X, Y, history[0], levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    
    # Mark agents
    for agent in agents:
        ax_anim.plot(agent.position[0], agent.position[1], 'b*', markersize=12, 
                    markeredgecolor='white', markeredgewidth=1)
    
    ax_anim.set_xlabel('x')
    ax_anim.set_ylabel('y')
    ax_anim.set_aspect('equal')
    time_text = ax_anim.text(0.02, 0.98, '', transform=ax_anim.transAxes, 
                            verticalalignment='top', fontsize=12, 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.colorbar(contour_anim, ax=ax_anim, label='Concentration')
    
    def animate(frame_idx):
        idx = frames_to_plot[frame_idx]
        t = idx * dt
        ax_anim.clear()
        
        contour = ax_anim.contourf(X, Y, history[idx], levels=20, cmap='hot', vmin=vmin, vmax=vmax)
        
        # Re-mark agents
        for agent in agents:
            ax_anim.plot(agent.position[0], agent.position[1], 'b*', markersize=12, 
                        markeredgecolor='white', markeredgewidth=1)
        
        ax_anim.set_xlabel('x')
        ax_anim.set_ylabel('y')
        ax_anim.set_title(f'Concentration Evolution (t={t:.3f})')
        ax_anim.set_aspect('equal')
        
        return contour,
    
    anim = FuncAnimation(fig_anim, animate, frames=len(frames_to_plot), 
                        interval=100, blit=False, repeat=True)
    
    plt.tight_layout()
    print("Displaying animation (close window to continue)...")
    plt.show()
    
    # Print statistics
    print("\nFinal Statistics:")
    print(f"  Maximum concentration: {np.max(u_final):.6f}")
    print(f"  Mean concentration: {np.mean(u_final):.6f}")
    print(f"  Total mass: {np.sum(u_final) * (L/N)**2:.6f}")


if __name__ == "__main__":
    main()
