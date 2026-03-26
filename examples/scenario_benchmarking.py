from unittest import runner

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import copy

# New framework imports - cleaner with __init__.py!
from benchmarking import (
    BenchmarkRunner,
    create_scenario_with_numerical_reference,
    create_scenario_with_numerical_reference_cached,
    get_default_scenarios,
    get_scenario_by_name,
    create_scenario,
    GaussianDiffusion2D,
    FlexibleSimulation,
    ValidationScenario,
    plot_scenario
)

# Schema imports
from diffusion_schemas import (
    ExplicitEulerSchema, ImplicitEulerSchema, CrankNicolsonSchema,
    ImplicitLODSchema, CrankNicolsonLODSchema, ADIBCSchema,
    ExplicitEulerBCSchema, ImplicitEulerBCSchema, CrankNicolsonBCSchema,
    ImplicitLODBCSchema, CrankNicolsonLODBCSchema, ADIBCSchema,
    ImplicitEulerBCISchema, CrankNicolsonBCISchema,
    ImplicitLODBCISchema, CrankNicolsonLODBCISchema, ADIBCISchema,
    ImplicitEulerBCOSSchema
)

# Utilities for backward compatibility example
from diffusion_schemas.utils import gaussian, uniform, \
DirichletBC, NeumannBC, Agent

def gaussian_pulses():

    print("\n" + "=" * 70)
    print("Gaussian Pulse Benchmarking")
    print("=" * 70)

    gaussian_pulse_1d = get_scenario_by_name('gaussian_pulse_1d') 
    gaussian_pulse_2d = get_scenario_by_name('gaussian_pulse_2d')  
    gaussian_pulse_3d = get_scenario_by_name('gaussian_pulse_3d')  

    runner = BenchmarkRunner()

    runner.add_schema(ExplicitEulerBCSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerBCSchema, "Implicit Euler")
    runner.add_schema(ImplicitLODBCSchema, "Implicit LOD")
    runner.add_schema(CrankNicolsonBCSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonLODBCSchema, "Crank-Nicolson LOD")

    runner.add_scenario(gaussian_pulse_1d)
    runner.add_scenario(gaussian_pulse_2d)
    runner.add_scenario(gaussian_pulse_3d)
    
    results = runner.run(
        output_dir='benchmark_results/gaussian_pulses_bc',
        store_history=True,
        generate_plots=True
    )
    
    summary = runner.generate_summary_report(
        output_path='benchmark_results/gaussian_pulses_bc/summary.csv'
    )
    
    return results, summary

def step_functions():

    print("\n" + "=" * 70)
    print("Step Function Benchmarking")
    print("=" * 70)

    step_function_1d = get_scenario_by_name('step_function_1d') 
    step_function_2d = get_scenario_by_name('step_function_2d')  

    runner = BenchmarkRunner()

    runner.add_schema(ExplicitEulerBCSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerBCSchema, "Implicit Euler")
    runner.add_schema(ImplicitLODBCSchema, "Implicit LOD")
    runner.add_schema(CrankNicolsonBCSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonLODBCSchema, "Crank-Nicolson LOD")

    runner.add_scenario(step_function_1d)
    runner.add_scenario(step_function_2d)
    
    results = runner.run(
        output_dir='benchmark_results/step_functions_bc',
        store_history=True,
        generate_plots=True
    )
    
    summary = runner.generate_summary_report(
        output_path='benchmark_results/step_functions_bc/summary.csv'
    )
    
    return results, summary

def steady_state_agents():

    print("\n" + "=" * 70)
    print("Steady State Agents Benchmarking")
    print("=" * 70)

    steady_state_agent_1d = get_scenario_by_name('steady_state_agent_1d') 
    steady_state_agent_2d = get_scenario_by_name('steady_state_agent_2d')  

    runner = BenchmarkRunner()

    runner.add_schema(ExplicitEulerSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerSchema, "Implicit Euler")
    runner.add_schema(ImplicitLODSchema, "Implicit LOD")
    runner.add_schema(CrankNicolsonSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonLODSchema, "Crank-Nicolson LOD")

    runner.add_scenario(steady_state_agent_1d)
    runner.add_scenario(steady_state_agent_2d)
    
    results = runner.run(
        output_dir='benchmark_results/steady_state_agents',
        store_history=True,
        generate_plots=True
    )
    
    summary = runner.generate_summary_report(
        output_path='benchmark_results/steady_state_agents/summary.csv'
    )
    
    return results, summary

def exponential_decay():

    print("\n" + "=" * 70)
    print("Exponential Decay Benchmarking")
    print("=" * 70)

    # steady_state_agent_1d = get_scenario_by_name('steady_state_agent_1d') 
    exponential_decay_1d = get_scenario_by_name('exponential_decay_1d')  

    runner = BenchmarkRunner()

    runner.add_schema(ExplicitEulerSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerSchema, "Implicit Euler")
    runner.add_schema(ImplicitLODSchema, "Implicit LOD")
    runner.add_schema(CrankNicolsonSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonLODSchema, "Crank-Nicolson LOD")

    runner.add_scenario(exponential_decay_1d)
    
    results = runner.run(
        output_dir='benchmark_results/exponential_decay',
        store_history=True,
        generate_plots=True
    )
    
    summary = runner.generate_summary_report(
        output_path='benchmark_results/exponential_decay/summary.csv'
    )
    
    return results, summary

def sine_decay():

    print("\n" + "=" * 70)
    print("Sine Decay Benchmarking")
    print("=" * 70)

    sine_decay_1d = get_scenario_by_name('sine_decay_1d')  

    runner = BenchmarkRunner()

    runner.add_schema(ExplicitEulerBCSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerBCSchema, "Implicit Euler")
    runner.add_schema(ImplicitLODBCSchema, "Implicit LOD")
    runner.add_schema(CrankNicolsonBCSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonLODBCSchema, "Crank-Nicolson LOD")

    runner.add_scenario(sine_decay_1d)
    
    results = runner.run(
        output_dir='benchmark_results/sine_decay_bc',
        store_history=True,
        generate_plots=True
    )
    
    summary = runner.generate_summary_report(
        output_path='benchmark_results/sine_decay_bc/summary.csv'
    )
    
    return results, summary

def cosine_diffusion():
    
    print("\n" + "=" * 70)
    print("Convergence Test 1 from BioFVM Benchmarking")
    print("=" * 70)

    base_scenario = copy.deepcopy(get_scenario_by_name('cosine_diffusion_2d'))

    dt_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
    scenarios = []
    for dt in dt_values:
        scenario = copy.deepcopy(base_scenario)
        scenario["dt"] = dt
        scenario["name"] = f"cosine_diffusion_2d_dt{dt}_dx5"
        scenario["name"] = scenario["name"].replace('.','')
        scenarios.append(scenario)

    runner = BenchmarkRunner()

    # runner.add_schema(ExplicitEulerSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerBCSchema, "Implicit Euler")
    runner.add_schema(ImplicitLODBCSchema, "Implicit LOD")
    runner.add_schema(CrankNicolsonBCSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonLODBCSchema, "Crank-Nicolson LOD")
    runner.add_schema(ADIBCSchema, "ADI")

    for scenario in scenarios:
        runner.add_scenario(scenario=scenario)
    
    results = runner.run(
        output_dir='benchmark_results/cosine_diffusion',
        store_history=False,
        generate_plots=True
    )
    
    summary = runner.generate_summary_report(
        output_path='benchmark_results/cosine_diffusion/summary_2d.csv'
    )

    return results, summary

def custom_scenario():

    # Define a fully customizable scenario

    scenario = create_scenario_with_numerical_reference(
        name='Random Scenario',
        # Schema to take as numerical reference
        schema_class=ImplicitLODBCSchema, 
        dx_ref=0.001,
        dt_ref=0.001,
        # Simulation parameters
        domain_size=(1.0, 1.0),
        grid_points=(50, 50),
        dt=0.001,
        t_final=0.04,
        # Physical parameters
        diffusion_coefficient=0.01,
        decay_rate=0.0,
        # Initial and Boundary Conditions
        # Note IC can be given as a lamba function too
        # initial_condition=lambda x: np.sin(2*np.pi*x/0.1), 
        initial_condition=uniform(1.0),
        boundary_condition={'type': 'neumann', 'flux': 0.0},
        # Agent addition (could be None)
        # List of dictionaries indicating agent parameters
        agents = [
            # {
            #     'position': [0.35,0.8],
            #     'secretion_rate': 0.0,
            #     'uptake_rate': 1.0,
            #     'saturation_density': 0.98,
            #     'kernel_width': 0.05
            # },
            # {
            #     'position': [0.25,0.25],
            #     'net_rate': 3,
            #     'kernel_width': 0.1
            # }
            ],
        # Bulk region addition (could be None)
        # List of dictionaries indicating bulk region parameters
        bulk={
            'regions': [
                {   
                    'type': 'sphere',
                    'center': (0.8,0.8),
                    'radius': 0.15,
                    'net_rate': -5
                },
                {
                    'type': 'rectangle',
                    'origin': (0.2, 0.3),
                    'size': (0.1, 0.5),
                    'linear_rate': 10
                },
                {
                    'type': 'sphere',
                    'center': (0.6,0.3),
                    'radius': 0.15,
                    'linear_rate': -100,
                    'rho_target': 0.5
                }
            ]
        }
    )

    runner = BenchmarkRunner()

    runner.add_schema(ImplicitLODBCSchema, "Implicit LOD")
    runner.add_schema(ExplicitEulerBCSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerBCSchema, "Implicit Euler")
    runner.add_schema(CrankNicolsonBCSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonLODBCSchema, "Crank-Nicolson LOD")


    runner.add_scenario(scenario=scenario)
    
    results = runner.run(
        output_dir='benchmark_results/custom_scenarios',
        store_history=True,
        generate_plots=True
    )
    
    summary = runner.generate_summary_report(
        output_path='benchmark_results/custom_scenarios/summary.csv'
    )

    return results, summary

def single_tumor():

    print("\n" + "=" * 70)
    print("Single Tumor 2D Benchmarking")
    print("=" * 70)

    base_scenario = get_scenario_by_name('single_tumor_2d')
    store = False
    base_scenario["store_history"] = store
    base_scenario["t_final"] = 10

    base_scenario["bulk"]["regions"][0]["linear_rate"] = base_scenario["bulk"]["regions"][0].pop("net_rate")

    dt_values = [0.5, 0.1]
    scenarios = []
    
    for dt in dt_values:
        scenario = copy.deepcopy(base_scenario)
        scenario["dt"] = dt
        scenario["name"] = f"single_tumor_2d_dt{dt}".replace(".", "")
        scenarios.append(scenario)
    
    runner = BenchmarkRunner()
    # runner.add_schema(ExplicitEulerBCSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerBCSchema, "Implicit Euler")
    runner.add_schema(ImplicitEulerBCISchema, "Implicit Euler with Implicit Source")
    runner.add_schema(ImplicitEulerBCOSSchema, "Implicit Euler with Operator Splitting")
    # runner.add_schema(ImplicitLODBCSchema, "Implicit LOD")
    # runner.add_schema(ImplicitLODBCISchema, "Implicit LOD with Implicit Source")
    # runner.add_schema(CrankNicolsonBCSchema, "Crank-Nicolson")
    # runner.add_schema(CrankNicolsonBCISchema, "Crank-Nicolson with Implicit Source")
    # runner.add_schema(CrankNicolsonLODBCSchema, "Crank-Nicolson LOD")
    # runner.add_schema(CrankNicolsonLODBCISchema, "Crank-Nicolson LOD with Implicit Source")
    # runner.add_schema(ADIBCSchema, "ADI")
    # runner.add_schema(ADIBCISchema, "ADI with Implicit Source")

    
    for scenario in scenarios:
        runner.add_scenario(scenario)
    
    results = runner.run(
        output_dir='benchmark_results/single_tumor',
        store_history=store,
        generate_plots=True
    )
    
    summary = runner.generate_summary_report(
        output_path='benchmark_results/single_tumor/summary.csv'
    )
    
    return results, summary

def single_tumor_on_off():

    print("\n" + "=" * 70)
    print("Single Tumor 2D On-Off Benchmarking")
    print("=" * 70)

    base_scenario = get_scenario_by_name('single_tumor_2d')
    store = True
    base_scenario["store_history"] = store
    base_scenario["t_final"] = 10
    base_scenario["bulk"]["regions"][0]["linear_rate"] = lambda t: 0.0 if int(np.floor(t + 1e-10)) % 2 == 0 else -10.0
    base_scenario["golden_solution"]["dt_ref"] = 0.01
    base_scenario["golden_solution"]["schema_class"] = CrankNicolsonBCSchema

    dt_values = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    scenarios = []
    
    for dt in dt_values:
        scenario = copy.deepcopy(base_scenario)
        scenario["dt"] = dt
        scenario["name"] = f"single_tumor_2d_dt{dt}".replace(".", "")
        scenarios.append(scenario)
    
    runner = BenchmarkRunner()
    # runner.add_schema(ExplicitEulerBCSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerBCSchema, "Implicit Euler")
    runner.add_schema(ImplicitLODBCSchema, "Implicit LOD")
    runner.add_schema(CrankNicolsonBCSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonLODBCSchema, "Crank-Nicolson LOD")
    runner.add_schema(ADIBCSchema, "ADI")
    
    for scenario in scenarios:
        runner.add_scenario(scenario)
    
    results = runner.run(
        output_dir='benchmark_results/single_tumor_on_off',
        store_history=store,
        generate_plots=True
    )
    
    summary = runner.generate_summary_report(
        output_path='benchmark_results/single_tumor_on_off/summary_dt_dx20_t10_2.csv'
    )
    
    return results, summary

def single_tumor_gradual():

    print("\n" + "=" * 70)
    print("Single Tumor 2D Gradual Benchmarking")
    print("=" * 70)

    base_scenario = get_scenario_by_name('single_tumor_2d')
    store = True
    base_scenario["store_history"] = store
    base_scenario["t_final"] = 10
    base_scenario["bulk"]["regions"][0]["linear_rate"] = lambda t: - 2 * (np.abs(t - 5)) 
    base_scenario["golden_solution"]["dt_ref"] = 0.01
    base_scenario["golden_solution"]["schema_class"] = CrankNicolsonBCSchema
    # a lambda funciton makes the golden solution hash change every time

    # dt_values = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    dt_values = [0.5, 0.1]
    scenarios = []
    
    for dt in dt_values:
        scenario = copy.deepcopy(base_scenario)
        scenario["dt"] = dt
        scenario["name"] = f"single_tumor_2d_dt{dt}".replace(".", "")
        scenarios.append(scenario)
    
    runner = BenchmarkRunner()
    # runner.add_schema(ExplicitEulerBCSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerBCSchema, "Implicit Euler")
    runner.add_schema(ImplicitEulerBCISchema, "Implicit Euler with Implicit Source")
    runner.add_schema(ImplicitEulerBCOSSchema, "Implicit Euler with Operator Splitting")
    # runner.add_schema(ImplicitLODBCSchema, "Implicit LOD")
    # runner.add_schema(ImplicitLODBCISchema, "Implicit LOD with Implicit Source")
    # runner.add_schema(CrankNicolsonBCSchema, "Crank-Nicolson")
    # runner.add_schema(CrankNicolsonBCISchema, "Crank-Nicolson with Implicit Source")
    # runner.add_schema(CrankNicolsonLODBCSchema, "Crank-Nicolson LOD")
    # runner.add_schema(CrankNicolsonLODBCISchema, "Crank-Nicolson LOD with Implicit Source")
    # runner.add_schema(ADIBCSchema, "ADI")
    # runner.add_schema(ADIBCISchema, "ADI with Implicit Source")
    
    for scenario in scenarios:
        runner.add_scenario(scenario)
    
    results = runner.run(
        output_dir='benchmark_results/single_tumor_gradual',
        store_history=store,
        generate_plots=True
    )
    
    summary = runner.generate_summary_report(
        output_path='benchmark_results/single_tumor_gradual/summary_dt_dx20_t10.csv'
    )
    
    return results, summary

def multiple_tumor():

    print("\n" + "=" * 70)
    print("Multiple Tumor 2D Benchmarking")
    print("=" * 70)

    # First generate the bulk dictionary witlh multiple tumor regions
    # Positions will be set randomly within the domain, 
    # ensuring they do not overlap and are fully contained

    # Create scenarios by modifying base scenario parameters
    base_scenario = get_scenario_by_name('multiple_tumor_2d')
    store = True
    base_scenario["golden_solution"]["dt_ref"] = 0.001
    base_scenario["store_history"] = store
    base_scenario["t_final"] = 10
    
    dx_values = [20]
    dt_values = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    scenarios = []

    for dt in dt_values:
        for dx in dx_values:
            scenario = copy.deepcopy(base_scenario)
            scenario["dt"] = dt
            scenario["grid_points"] = tuple(int(s / dx) for s in scenario["domain_size"])
            scenario["name"] = f"multiple_tumor_2d_dt{dt}_dx{dx}".replace(".", "")
            scenarios.append(scenario)

    # Parametric sweep over dx values, keeping dt fixed at 0.01
    dt_values = [0.001,0.005,0.01,0.05,0.1,0.5,1,5]
    dt_values2 = [0.2,0.3,0.4,2,3,4]

    scenarios = []
    for dt in dt_values2.append(dt_values):
        scenario = copy.deepcopy(base_scenario)
        scenario["dt"] = dt
        scenario["t_final"] = 10
        scenario["grid_points"] = tuple(int(s / 20) for s in scenario["domain_size"])
        scenario["name"] = f"multiple_tumor_2d_dt{dt}_dx20_R50"
        scenario["name"] = scenario["name"].replace('.','')
        scenarios.append(scenario)

    runner = BenchmarkRunner()

    # runner.add_schema(ExplicitEulerBCSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerBCSchema, "Implicit Euler")
    runner.add_schema(ImplicitLODBCSchema, "Implicit LOD")
    runner.add_schema(CrankNicolsonBCSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonLODBCSchema, "Crank-Nicolson LOD")
    runner.add_schema(ADIBCSchema, "ADI")

    for scenario in scenarios:
        runner.add_scenario(scenario)

    # fig = plot_scenario(
    #     scenario=scenarios[0],
    # )
    # plt.show()

    results = runner.run(
        output_dir='benchmark_results/multiple_tumor',
        store_history=store,
        generate_plots=True
    )
    
    summary = runner.generate_summary_report(
        output_path='benchmark_results/multiple_tumor/summary_dt_dx20_t10_R50_2.csv'
    )

    return results, summary

def main():

    results, summary = single_tumor()

if __name__ == "__main__":
    main()