import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

def custom_scenario():

    # Define a fully customizable scenario

    scenario = create_scenario_with_numerical_reference(
        name='Random Scenario',
        # Schema to take as numerical reference
        schema_class=ADIBCSchema, 
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
            {
                'position': [0.35,0.8],
                'secretion_rate': 0.0,
                'uptake_rate': 1.0,
                'saturation_density': 0.98,
                'kernel_width': 0.05
            },
            {
                'position': [0.25,0.25],
                'net_rate': 3,
                'kernel_width': 0.1
            }
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
                }
            ]
        }
    )

    runner = BenchmarkRunner()

    runner.add_schema(ADIBCSchema, "ADI")
    runner.add_schema(ExplicitEulerBCSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerBCSchema, "Implicit Euler")
    runner.add_schema(CrankNicolsonBCSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonADIBCSchema, "Crank-Nicolson ADI")


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

def single_tumour():

    print("\n" + "=" * 70)
    print("Single Tumor 2D Benchmarking")
    print("=" * 70)

    # single_tumor_2d = get_scenario_by_name('single_tumor_2d')  
    single_tumor_2d_dt01 = create_scenario_with_numerical_reference_cached(
        name='single_tumor_2d_dt01',
        schema_class=ADIBCSchema, # reference schema class for golden solution
        domain_size=(2000.0, 2000.0), # assuming in micrometers for a 2mm x 2mm tissue section
        grid_points=(100, 100), # dx forced to 20 micrometers, leading to a 100x100 grid
        dt=0.1, # minutes
        t_final=5, # minutes (64800 minutes = 45 days)
        initial_condition={
            'type': 'uniform',
            'value': 38.0 # mmHg
        },
        diffusion_coefficient=float(1e5), # μm^2/min
        decay_rate=0.1, # per minute
        boundary_condition={
            'type': 'dirichlet',
            'value': 38.0 # mmHg
        },
        bulk={
            'regions': [
                {
                    'type': 'sphere',
                    'center': (1000.0, 1000.0), # center of the domain
                    'radius': 250.0, # micrometers
                    'net_rate': -10.0,
                    'name': 'tumor_region'

                }
            ]
        },
        dx_ref=10.0,
        dt_ref=0.0001,
        description='2D diffusion with decay and a single tumor region uptaking continuously'
    )

    single_tumor_2d_dt001 = create_scenario_with_numerical_reference_cached(
        name='single_tumor_2d_dt001',
        schema_class=ADIBCSchema, # reference schema class for golden solution
        domain_size=(2000.0, 2000.0), # assuming in micrometers for a 2mm x 2mm tissue section
        grid_points=(100, 100), # dx forced to 20 micrometers, leading to a 100x100 grid
        dt=0.01, # minutes
        t_final=5, # minutes (64800 minutes = 45 days)
        initial_condition={
            'type': 'uniform',
            'value': 38.0 # mmHg
        },
        diffusion_coefficient=float(1e5), # μm^2/min
        decay_rate=0.1, # per minute
        boundary_condition={
            'type': 'dirichlet',
            'value': 38.0 # mmHg
        },
        bulk={
            'regions': [
                {
                    'type': 'sphere',
                    'center': (1000.0, 1000.0), # center of the domain
                    'radius': 250.0, # micrometers
                    'net_rate': -10.0,
                    'name': 'tumor_region'

                }
            ]
        },
        dx_ref=10.0,
        dt_ref=0.0001,
        description='2D diffusion with decay and a single tumor region uptaking continuously'
    )

    single_tumor_2d_dt0001 = create_scenario_with_numerical_reference_cached(
        name='single_tumor_2d_dt0001',
        schema_class=ADIBCSchema, # reference schema class for golden solution
        domain_size=(2000.0, 2000.0), # assuming in micrometers for a 2mm x 2mm tissue section
        grid_points=(100, 100), # dx forced to 20 micrometers, leading to a 100x100 grid
        dt=0.001, # minutes
        t_final=5, # minutes (64800 minutes = 45 days)
        initial_condition={
            'type': 'uniform',
            'value': 38.0 # mmHg
        },
        diffusion_coefficient=float(1e5), # μm^2/min
        decay_rate=0.1, # per minute
        boundary_condition={
            'type': 'dirichlet',
            'value': 38.0 # mmHg
        },
        bulk={
            'regions': [
                {
                    'type': 'sphere',
                    'center': (1000.0, 1000.0), # center of the domain
                    'radius': 250.0, # micrometers
                    'net_rate': -10.0,
                    'name': 'tumor_region'

                }
            ]
        },
        dx_ref=10.0,
        dt_ref=0.0001,
        description='2D diffusion with decay and a single tumor region uptaking continuously'
    )

    runner = BenchmarkRunner()

    # runner.add_schema(ExplicitEulerBCSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerBCSchema, "Implicit Euler")
    runner.add_schema(ADIBCSchema, "ADI")
    runner.add_schema(CrankNicolsonBCSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonADIBCSchema, "Crank-Nicolson ADI")

    runner.add_scenario(single_tumor_2d_dt01)
    runner.add_scenario(single_tumor_2d_dt001)
    runner.add_scenario(single_tumor_2d_dt0001)
    
    results = runner.run(
        output_dir='benchmark_results/single_tumor',
        store_history=True,
        generate_plots=True
    )
    
    summary = runner.generate_summary_report(
        output_path='benchmark_results/single_tumor/summary.csv'
    )
    
    return results, summary

def multiple_tumour():

    print("\n" + "=" * 70)
    print("Multiple Tumor 2D Benchmarking")
    print("=" * 70)

    # First generate the bulk dictionary witlh multiple tumor regions
    # Positions will be set randomly within the domain, 
    # ensuring they do not overlap and are fully contained

    np.random.seed(20)  # for reproducibility
    tumours = []
    n = 30
    radius = 10.0
    for i in range(n):
        while True:
            center = (np.random.uniform(250, 1750), np.random.uniform(250, 1750))
            # Check for overlap with existing tumors
            if all(np.linalg.norm(np.array(center) - np.array(t['center'])) > 2*radius for t in tumours):
                tumours.append({
                    'type': 'sphere',
                    'center': center,
                    'radius': radius,
                    'net_rate': -10.0,
                    'name': f'tumor_region_{i+1}'
                })
                break

    multiple_tumor_2d = create_scenario_with_numerical_reference_cached(
        name='multiple_tumor_2d',
        schema_class=ADIBCSchema, # reference schema class for golden solution
        domain_size=(2000.0, 2000.0), # assuming in micrometers for a 2mm x 2mm tissue section
        grid_points=(100, 100), # dx forced to 20 micrometers, leading to a 100x100 grid
        dt=0.05, # minutes
        t_final=5, # minutes (64800 minutes = 45 days)
        initial_condition={
            'type': 'uniform',
            'value': 38.0 # mmHg
        },
        diffusion_coefficient=float(1e5), # μm^2/min
        decay_rate=0.1, # per minute
        boundary_condition={
            'type': 'dirichlet',
            'value': 38.0 # mmHg
        },
        bulk={
            'regions': tumours  # directly pass the generated list of dictionaries for tumor regions
        },
        dx_ref=10.0,
        dt_ref=0.0001,
        description=f'2D diffusion with decay and multiple ({len(tumours)}) non-overlapping tumor regions'
    )

    runner = BenchmarkRunner()

    # runner.add_schema(ExplicitEulerBCSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerBCSchema, "Implicit Euler")
    runner.add_schema(ADIBCSchema, "ADI")
    runner.add_schema(CrankNicolsonBCSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonADIBCSchema, "Crank-Nicolson ADI")

    runner.add_scenario(multiple_tumor_2d)
    
    results = runner.run(
        output_dir='benchmark_results/multiple_tumor',
        store_history=True,
        generate_plots=True
    )
    
    summary = runner.generate_summary_report(
        output_path='benchmark_results/multiple_tumor/summary.csv'
    )
    
    return results, summary

def main():

    results, summary = single_tumour()

if __name__ == "__main__":
    main()