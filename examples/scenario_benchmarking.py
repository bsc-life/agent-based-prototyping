import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# New framework imports - cleaner with __init__.py!
from benchmarking import (
    BenchmarkRunner,
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
from diffusion_schemas.utils import gaussian, DirichletBC, NeumannBC, Agent

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

    runner.add_schema(ExplicitEulerSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerSchema, "Implicit Euler")
    runner.add_schema(ADISchema, "ADI")
    runner.add_schema(CrankNicolsonSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonADISchema, "Crank-Nicolson ADI")

    runner.add_scenario(sine_decay_1d)
    
    results = runner.run(
        output_dir='benchmark_results/sine_decay',
        store_history=True,
        generate_plots=True
    )
    
    summary = runner.generate_summary_report(
        output_path='benchmark_results/sine_decay/summary.csv'
    )
    
    return results, summary

def main():

    print("\n" + "=" * 70)
    print("Diffusion schema benchmarking")
    print("=" * 70)

    results, summary = sine_decay()

if __name__ == "__main__":
    main()