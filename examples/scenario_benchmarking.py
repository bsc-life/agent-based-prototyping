import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# New framework imports - cleaner with __init__.py!
from benchmarking import (
    BenchmarkRunner,
    create_scenario_with_numerical_reference,
    get_default_scenarios,
    get_scenario_by_name,
    create_scenario,
    GaussianDiffusion2D,
    FlexibleSimulation,
    ValidationScenario
)

# Schema imports
from diffusion_schemas import (
    ExplicitEulerSchema, ImplicitEulerSchema, CrankNicolsonSchema,
    ADISchema, CrankNicolsonADISchema,
    ExplicitEulerBCSchema, ImplicitEulerBCSchema, CrankNicolsonBCSchema,
    ADIBCSchema, CrankNicolsonADIBCSchema
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
    runner.add_schema(ADIBCSchema, "ADI")
    runner.add_schema(CrankNicolsonBCSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonADIBCSchema, "Crank-Nicolson ADI")

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
    runner.add_schema(ADIBCSchema, "ADI")
    runner.add_schema(CrankNicolsonBCSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonADIBCSchema, "Crank-Nicolson ADI")

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
    runner.add_schema(ADISchema, "ADI")
    runner.add_schema(CrankNicolsonSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonADISchema, "Crank-Nicolson ADI")

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
    runner.add_schema(ADISchema, "ADI")
    runner.add_schema(CrankNicolsonSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonADISchema, "Crank-Nicolson ADI")

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
    runner.add_schema(ADIBCSchema, "ADI")
    runner.add_schema(CrankNicolsonBCSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonADIBCSchema, "Crank-Nicolson ADI")

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

    convergence_test_1 = get_scenario_by_name('cosine_diffusion_1d')  

    runner = BenchmarkRunner()

    runner.add_schema(ExplicitEulerBCSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerBCSchema, "Implicit Euler")
    runner.add_schema(ADIBCSchema, "ADI")
    runner.add_schema(CrankNicolsonBCSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonADIBCSchema, "Crank-Nicolson ADI")

    runner.add_scenario(convergence_test_1)
    
    results = runner.run(
        output_dir='benchmark_results/convergence_test_1_bc',
        store_history=True,
        generate_plots=True
    )
    
    summary = runner.generate_summary_report(
        output_path='benchmark_results/convergence_test_1_bc/summary.csv'
    )

    return results, summary

def custom_agent_scenario():

    # Define a fully customizable scenario

    scenario = create_scenario_with_numerical_reference(
        name='Random Scenario',

        # Schema to take as numerical reference
        schema_class=ADIBCSchema, 
        dx_ref=0.001,
        dt_ref=0.001,

        # Simulation parameters
        domain_size=(1.0,1.0),
        grid_points=(50,50),
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
        #     {
        #         'position': [0.25,0.85],
        #         'secretion_rate': 1.0,
        #         'uptake_rate': 2.0,
        #         'saturation_density': 0.98,
        #         'kernel_width': 0.05
        #     }, 
        #     {
        #         'position': [0.35,0.35],
        #         'secretion_rate': 0.0,
        #         'uptake_rate': 1.0,
        #         'saturation_density': 0.98,
        #         'kernel_width': 0.05
        #     },
        #     {
        #         'position': [0.25,0.25],
        #         'net_rate': 0.01,
        #         'kernel_width': 0
        #     }
            ],

        # Bulk region addition (could be None)
        # List of dictionaries indicating bulk region parameters
        bulk={
            'regions': [
                # {
                #     'type': 'rectangle',
                #     'origin': (0.5,0.25),
                #     'size': (0.1,0.4),
                #     'net_rate': 50
                # },
                {   
                    'type': 'sphere',
                    'center': (0.8,0.8),
                    'radius': 0.15,
                    'net_rate': -3
                }
            ]
        },
        
    )

    runner = BenchmarkRunner()

    runner.add_schema(ADIBCSchema, "ADI")

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

def main():

    print("\n" + "=" * 70)
    print("Diffusion schema benchmarking")
    print("=" * 70)

    results, summary = custom_agent_scenario()

if __name__ == "__main__":
    main()