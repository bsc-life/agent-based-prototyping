"""
Example 4: 3D Diffusion from a Spherical Source

Demonstrates 3D diffusion with a spherical initial condition
using the Implicit Euler method.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from diffusion_schemas import ImplicitEulerSchema
from diffusion_schemas.utils import sphere, Agent


def main():
    # Parameters
    L = 1.0  # Domain size (cubic domain)
    N = 30  # Grid points per dimension (kept small for 3D)
    D = 0.05  # Diffusion coefficient
    dt = 0.01  # Time step
    t_final = 0.5  # Final time
    
    print("=" * 60)
    print("3D Diffusion from Spherical Source")
    print("=" * 60)
    print(f"Domain: [0, {L}]³")
    print(f"Grid: {N} x {N} x {N} ({N**3} points)")
    print(f"Diffusion coefficient: {D}")
    print(f"Time step: {dt}")
    print(f"Final time: {t_final}")
    print("=" * 60)
    
    # Create schema
    print("\nInitializing 3D implicit Euler schema...")
    schema = ImplicitEulerSchema(
        domain_size=(L, L, L),
        grid_points=(N, N, N),
        dt=dt,
        diffusion_coefficient=D,
        decay_rate=0.05
    )
    
    # Set initial condition: sphere at center
    ic = sphere(
        center=(0.5, 0.5, 0.5),
        radius=0.2,
        value_inside=10.0,
        value_outside=0.0
    )
    schema.set_initial_condition(ic)
    
    u_initial = schema.get_state()
    
    # Alternative: use an agent as a continuous source
    # agent = Agent(position=(0.5, 0.5, 0.5), secretion_rate=50.0, kernel_width=0.1)
    # schema.add_agent(agent)
    
    # Solve
    print(f"\nSolving 3D diffusion problem...")
    start_time = __import__('time').time()
    schema.solve(t_final=t_final)
    elapsed = __import__('time').time() - start_time
    
    print(f"Completed in {elapsed:.2f} seconds ({int(t_final/dt)} time steps)")
    
    # Get final state
    u_final = schema.get_state()
    
    # Create coordinate grids
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    z = np.linspace(0, L, N)
    
    # Visualize using 2D slices through the center
    fig = plt.figure(figsize=(16, 10))
    
    center_idx = N // 2
    
    # XY slice (z = L/2)
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(u_initial[:, :, center_idx].T, extent=[0, L, 0, L], 
                     origin='lower', cmap='hot', aspect='auto')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Initial: XY Slice at z={L/2:.2f}')
    plt.colorbar(im1, ax=ax1, label='Concentration')
    
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(u_final[:, :, center_idx].T, extent=[0, L, 0, L], 
                     origin='lower', cmap='hot', aspect='auto')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f'Final (t={t_final}): XY Slice at z={L/2:.2f}')
    plt.colorbar(im2, ax=ax2, label='Concentration')
    
    # XZ slice (y = L/2)
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(u_initial[:, center_idx, :].T, extent=[0, L, 0, L], 
                     origin='lower', cmap='hot', aspect='auto')
    ax3.set_xlabel('x')
    ax3.set_ylabel('z')
    ax3.set_title(f'Initial: XZ Slice at y={L/2:.2f}')
    plt.colorbar(im3, ax=ax3, label='Concentration')
    
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(u_final[:, center_idx, :].T, extent=[0, L, 0, L], 
                     origin='lower', cmap='hot', aspect='auto')
    ax4.set_xlabel('x')
    ax4.set_ylabel('z')
    ax4.set_title(f'Final (t={t_final}): XZ Slice at y={L/2:.2f}')
    plt.colorbar(im4, ax=ax4, label='Concentration')
    
    # YZ slice (x = L/2)
    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.imshow(u_initial[center_idx, :, :].T, extent=[0, L, 0, L], 
                     origin='lower', cmap='hot', aspect='auto')
    ax5.set_xlabel('y')
    ax5.set_ylabel('z')
    ax5.set_title(f'Initial: YZ Slice at x={L/2:.2f}')
    plt.colorbar(im5, ax=ax5, label='Concentration')
    
    ax6 = plt.subplot(2, 3, 6)
    im6 = ax6.imshow(u_final[center_idx, :, :].T, extent=[0, L, 0, L], 
                     origin='lower', cmap='hot', aspect='auto')
    ax6.set_xlabel('y')
    ax6.set_ylabel('z')
    ax6.set_title(f'Final (t={t_final}): YZ Slice at x={L/2:.2f}')
    plt.colorbar(im6, ax=ax6, label='Concentration')
    
    plt.tight_layout()
    plt.savefig('example_3d_sphere_slices.png', dpi=150, bbox_inches='tight')
    print(f"\n2D slices plot saved as 'example_3d_sphere_slices.png'")
    
    # 3D isosurface plot
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # Create voxel plot for regions above threshold
    threshold = np.max(u_final) * 0.3
    mask = u_final > threshold
    
    # Get coordinates of voxels above threshold
    pos = np.where(mask)
    
    # Create color array based on concentration
    colors = plt.cm.hot(u_final[mask] / np.max(u_final))
    
    # Plot voxels
    ax_3d.scatter(
        x[pos[0]], y[pos[1]], z[pos[2]],
        c=colors, marker='s', s=50, alpha=0.6
    )
    
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('z')
    ax_3d.set_title(f'3D Concentration (threshold={threshold:.2f})')
    ax_3d.set_xlim([0, L])
    ax_3d.set_ylim([0, L])
    ax_3d.set_zlim([0, L])
    
    plt.tight_layout()
    plt.savefig('example_3d_sphere_isosurface.png', dpi=150, bbox_inches='tight')
    print(f"3D isosurface plot saved as 'example_3d_sphere_isosurface.png'")
    
    # Print statistics
    print("\nStatistics:")
    print(f"  Initial max concentration: {np.max(u_initial):.6f}")
    print(f"  Final max concentration: {np.max(u_final):.6f}")
    print(f"  Initial total mass: {np.sum(u_initial) * (L/N)**3:.6f}")
    print(f"  Final total mass: {np.sum(u_final) * (L/N)**3:.6f}")
    print(f"  Concentration at center: {u_final[center_idx, center_idx, center_idx]:.6f}")
    
    plt.show()


if __name__ == "__main__":
    main()
