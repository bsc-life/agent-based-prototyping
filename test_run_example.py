import matplotlib.pyplot as plt
from test_suite import FlexibleSimulation
from diffusion_schemas import ImplicitEulerBCSchema, ImplicitEulerSchema, \
    ExplicitEulerBCSchema, ExplicitEulerSchema, \
    CrankNicolsonBCSchema, CrankNicolsonSchema, \
    ADIBCSchema, ADISchema,\
    CrankNicolsonADIBCSchema, CrankNicolsonADISchema
from matplotlib.animation import FuncAnimation
from diffusion_schemas.utils import gaussian, uniform, step_function, checkerboard, sphere, \
    DirichletBC, NeumannBC, \
    Agent

def main():
    
    ic = gaussian(center=(0.5, 0.5), amplitude=10.0, width=0.1)
    bc = DirichletBC(value=0.0)
    agents = [
            Agent((0.3, 0.3), secretion_rate = 5.0, kernel_width = 0.2),
            Agent((0.7, 0.7), secretion_rate = 5.0, kernel_width= 0.1)
        ]

    # Definition of test parameters
    # These will be used to setup_physics when .initialize()
    
    test_params = {
        'L': (1.0, 1.0),
        'N': (50, 50),
        'D': 0.05, 
        'dt': 0.01, 
        't_final': 2.0,
        'decay': 0.1,
        'store_history': True,
        'bc': bc,                   # optional
        'ic': ic,                   # optional
        'agents': agents            # optional
    }

    # Definition of test case to be run (test case is a subclass of SimulationScenario abstract class)
    test_case = FlexibleSimulation(
        name = "Flexible Simulation with Custom IC, BC, and Agents",
        schema_class = ImplicitEulerSchema,
        params = test_params
    )

    # Execute methods to initialize and run the test case
    test_case.initialize()
    results = test_case.run()

    # Visualization of results and statistics
    print("\nTest Results:")
    print(f"Duration: {results['duration']:.4f} s")
    print(f"Max Conc: {results['max_concentration']:.4f}")
    print(f"Min Conc: {results['min_concentration']:.4f}")
    
    # Simple plot to verify
    plt.imshow(results['final_state'].T, origin='lower', cmap='hot')
    plt.colorbar()
    plt.title(f"{test_case.name} \n Solver: {test_case.schema_class.__name__}")
    plt.show()

    if results['history'] is not None:

        def animate_results(results):
            history = results['history']
            fig, ax = plt.subplots()
            
            # Initialize the plot with the first frame (t=0)
            # We transpose (.T) to match (x, y) coordinates to (row, col)
            im = ax.imshow(history[0].T, origin='lower', cmap='hot', interpolation='gaussian',
                        vmin = 0, vmax = results['max_concentration'])
            plt.colorbar(im)
            
            def update(frame):
                # Update the data for each frame in history
                im.set_array(history[frame].T)
                ax.set_title(f"Time Step: {frame}")
                return [im]

            # Create the animation object
            ani = FuncAnimation(fig, update, frames=len(history), interval=1, blit=False)
            
            plt.show()

        # Comment or uncomment to enable or disable animation
        animate_results(results)

if __name__ == "__main__":
    main()