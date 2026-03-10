import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from diffusion_schemas import ImplicitEulerBCSchema, ImplicitEulerSchema, \
    ExplicitEulerBCSchema, ExplicitEulerSchema, \
    CrankNicolsonBCSchema, CrankNicolsonSchema, \
    ImplicitLODBCSchema, ImplicitLODSchema,\
    CrankNicolsonLODBCSchema, CrankNicolsonLODSchema
from diffusion_schemas.utils import gaussian, NeumannBC, DirichletBC

def main():
    # Parameters
    L = (1.0, 1.0)        # Domain length
    N = (64, 64)         # Number of grid points (reduced for faster solve)
    dx = L[0] / (N[0] - 1)  
    dy = L[1] / (N[1] - 1)
    D = 0.1        # Diffusion coefficient
    # dt = 0.005     # Larger time step (Implicit is stable!)
    # dt_threshold = dx ** 2 / (2 * D)
    dt = 0.000001     # Smaller time step (Explicit is conditionally stable!)

    t_final = 0.1  # Final time
    
    # Grid for IC and plotting
    x = np.linspace(0, L[0], N[0])
    y = np.linspace(0, L[1], N[1])
    X, Y = np.meshgrid(x, y)
    ic = gaussian(center=(0.5, 0.5), amplitude=1.0, width=0.1)
    ic_array = ic(X, Y)

    methods = {
        "Implicit (Integrated - BC)": ImplicitEulerBCSchema(
            domain_size=L, grid_points=N, dt=dt, diffusion_coefficient=D
        ),
        "Implicit (Operator Splitting - BC)": ImplicitEulerSchema(
            domain_size=L, grid_points=N, dt=dt, diffusion_coefficient=D
        )
    }

    methods2 = {
        "Explicit (Forward)": ExplicitEulerSchema(
            domain_size=L, grid_points=N, dt=dt, diffusion_coefficient=D,
            spatial_discretization='forward_1'
        ),
        "Explicit (Backward)": ExplicitEulerSchema(
            domain_size=L, grid_points=N, dt=dt, diffusion_coefficient=D,
            spatial_discretization='backward_1'
        )
    }

    methods3 = {
        "Crank-Nicolson (Integrated - BC)": CrankNicolsonBCSchema(
            domain_size=L, grid_points=N, dt=dt, diffusion_coefficient=D
        ),
        "Crank-Nicolson (Operator Splitting - BC)": CrankNicolsonSchema(
            domain_size=L, grid_points=N, dt=dt, diffusion_coefficient=D
        )
    }

    methods4 = {
        "ADI (Integrated - BC)": ImplicitLODBCSchema(
            domain_size=L, grid_points=N, dt=dt, diffusion_coefficient=D
        ),
        "ADI (Operator Splitting - BC)": ImplicitLODSchema(
            domain_size=L, grid_points=N, dt=dt, diffusion_coefficient=D
        )
    }

    methods5 = {
        "Crank-Nicolson LOD (Integrated - BC)": CrankNicolsonLODBCSchema(
            domain_size=L, grid_points=N, dt=dt, diffusion_coefficient=D
        ),
        "Crank-Nicolson LOD (Operator Splitting - BC)": CrankNicolsonLODSchema(
            domain_size=L, grid_points=N, dt=dt, diffusion_coefficient=D
        )
    }

    results = {}
    stats = []

    for name, schema in methods2.items():
        print(f"Running {name}...")
        
        # Setup
        schema.set_initial_condition(ic)
        u_initial = methods["Implicit (Integrated - BC)"].get_state()  # Get initial state from one of the schemas
        # schema.set_boundary_conditions(NeumannBC(flux=0.0))
        schema.set_boundary_conditions(DirichletBC(value=0.0))


        # Solve
        start_time = time.time()
        schema.solve(t_final=t_final)
        elapsed = time.time() - start_time
        
        # Store results
        final_state = schema.get_state()
        results[name] = final_state
        
        # Calculate Stats
        total_mass = np.sum(final_state) * (L[0]/N[0]) * (L[1]/N[1])
        initial_mass = np.sum(ic_array) * dx * dy
        final_mass = np.sum(final_state) * dx * dy
        mass_change = (1 - final_mass / initial_mass) * 100        
        
        stats.append({
            "Method": name,
            "Runtime (s)": round(elapsed, 4),
            "Max Val": round(np.max(final_state), 6),
            "Min Val": round(np.min(final_state), 6),
            "Total Mass": round(total_mass, 6),
            "Mass Change (%)": f"{mass_change:.4f}%"
        })


    # --- Print Statistics Table ---
    print("\n" + "="*70)
    print("Comparison")
    print("="*70)
    df = pd.DataFrame(stats).set_index("Method")
    print(df.transpose()) # Transposed for easier side-by-side reading
    print("="*70)

    # --- Plotting 2D Maps ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, (name, data) in zip(axes, results.items()):
        im = ax.imshow(data, extent=[0, L[0], 0, L[1]], origin='lower', cmap='magma')
        ax.set_title(f"Final State: {name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, label='Concentration u', fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    plt.savefig('comparison_2d_diffusion.png', dpi=150)
    print(f"\nPlot saved as 'comparison_2d_diffusion.png'")
    plt.show()

if __name__ == "__main__":
    main()