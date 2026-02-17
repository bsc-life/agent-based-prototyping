"""
Example usage of the benchmarking framework for testing diffusion schemas.

This script demonstrates:
1. Using the new BenchmarkRunner framework with default scenarios
2. Running multiple schemas against scenarios with automatic error computation
3. Performing convergence analysis
4. Generating comprehensive visualizations
5. Backward compatibility with the original test_suite approach
"""

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


def run_basic_benchmark():
    """
    Example 1: Basic benchmark using default scenarios.
    
    This demonstrates the simplest usage: test multiple schemas against
    predefined scenarios with automatic error computation and visualization.
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic Benchmark with Default Scenarios")
    print("=" * 70)
    
    # Create benchmark runner
    runner = BenchmarkRunner()
    
    # Add schemas to test
    runner.add_schema(ExplicitEulerSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerSchema, "Implicit Euler")
    runner.add_schema(ADISchema, "ADI")
    runner.add_schema(CrankNicolsonSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonADISchema, "Crank-Nicolson ADI")
    
    # Add default 2D Gaussian pulse scenario
    scenario = get_scenario_by_name('gaussian_pulse_2d')
    runner.add_scenario(scenario)
    
    # Run benchmarks
    results = runner.run(
        output_dir='benchmark_results/basic',
        store_history=True,
        generate_plots=True
    )
    
    # Generate summary report
    summary = runner.generate_summary_report(
        output_path='benchmark_results/basic/summary.csv'
    )
    
    return results, summary


def run_1d_benchmark():
    """
    Example 2: 1D benchmark for faster testing and clearer visualization.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: 1D Gaussian Pulse Benchmark")
    print("=" * 70)
    
    runner = BenchmarkRunner()
    
    # Add schemas
    runner.add_schema(ExplicitEulerSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerSchema, "Implicit Euler")
    runner.add_schema(ADISchema, "ADI")
    runner.add_schema(CrankNicolsonSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonADISchema, "Crank-Nicolson ADI")
    
    # Add 1D scenario
    scenario = get_scenario_by_name('gaussian_pulse_1d')
    runner.add_scenario(scenario)
    
    # Run
    results = runner.run(
        output_dir='benchmark_results/basic_1d',
        store_history=True,
        generate_plots=True
    )
    
    summary = runner.generate_summary_report(
        output_path='benchmark_results/basic_1d/summary.csv'
    )
    
    return results, summary

def run_convergence_analysis():
    """
    Example 3: Convergence analysis to verify order of accuracy.
    
    This tests how error decreases as dt is refined, confirming that
    methods achieve their expected convergence rates.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Convergence Analysis")
    print("=" * 70)
    
    runner = BenchmarkRunner()
    
    # Get base scenario
    scenario = get_scenario_by_name('gaussian_pulse_2d')
    
    # Test temporal convergence for Crank-Nicolson (should be 2nd order)
    print("\nTemporal convergence for Crank-Nicolson:")
    conv_results_cn = runner.run_convergence_analysis(
        schema_class=CrankNicolsonSchema,
        schema_name="Crank-Nicolson",
        scenario_base=scenario,
        refinement_type='dt',
        refinement_factors=[0.01, 0.005, 0.0025, 0.00125],
        output_dir='benchmark_results/convergence_2D'
    )
    
    # Test temporal convergence for Implicit Euler (should be 1st order)
    print("\nTemporal convergence for Implicit Euler:")
    conv_results_ie = runner.run_convergence_analysis(
        schema_class=ImplicitEulerSchema,
        schema_name="Implicit Euler",
        scenario_base=scenario,
        refinement_type='dt',
        refinement_factors=[0.01, 0.005, 0.0025, 0.00125],
        output_dir='benchmark_results/convergence_2D'
    )

    print("\nTemporal convergence for ADI:")
    conv_results_adi = runner.run_convergence_analysis(
        schema_class=ADISchema,
        schema_name="ADI",
        scenario_base=scenario,
        refinement_type='dt',
        refinement_factors=[0.01, 0.005, 0.0025, 0.00125],
        output_dir='benchmark_results/convergence_2D'
    )
    
    return conv_results_cn, conv_results_ie, conv_results_adi

def run_convergence_analysis_spatial():
    """
    Example 3: Convergence analysis to verify order of accuracy.
    
    This tests how error decreases as dx is refined, confirming that
    methods achieve their expected convergence rates.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Convergence Analysis")
    print("=" * 70)
    
    runner = BenchmarkRunner()
    
    # Get base scenario
    scenario = get_scenario_by_name('gaussian_pulse_1d')
    
    # Test spatial convergence for Crank-Nicolson (should be 2nd order)
    print("\nSpatial convergence for Crank-Nicolson:")
    conv_results_cn = runner.run_convergence_analysis(
        schema_class=CrankNicolsonSchema,
        schema_name="Crank-Nicolson",
        scenario_base=scenario,
        refinement_type='dx',
        refinement_factors=None,
        output_dir='benchmark_results/convergence_2D_spatial'
    )
    
    # Test spatial convergence for Implicit Euler (should be 1st order)
    print("\nSpatial convergence for Implicit Euler:")
    conv_results_ie = runner.run_convergence_analysis(
        schema_class=ImplicitEulerSchema,
        schema_name="Implicit Euler",
        scenario_base=scenario,
        refinement_type='dx',
        refinement_factors=None,
        output_dir='benchmark_results/convergence_2D_spatial'
    )

    print("\nSpatial convergence for ADI:")
    conv_results_adi = runner.run_convergence_analysis(
        schema_class=ADISchema,
        schema_name="ADI",
        scenario_base=scenario,
        refinement_type='dx',
        refinement_factors=None,
        output_dir='benchmark_results/convergence_2D_spatial'
    )
    
    return conv_results_cn, conv_results_ie, conv_results_adi

def run_custom_scenario():
    """
    Example 4: Custom scenario with specific parameters.
    
    This shows how to create a custom test scenario programmatically.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Custom Scenario")
    print("=" * 70)
    
    # Create custom scenario with different parameters
    custom_scenario = create_scenario(
        name='custom_2d_faster_diffusion',
        domain_size=(2.0, 2.0),  # Larger domain
        grid_points=(40, 40),
        dt=0.002,
        t_final=0.3,
        diffusion_coefficient=0.05,  # Faster diffusion
        decay_rate=0.0,
        initial_condition={
            'type': 'gaussian',
            'center': (1.0, 1.0),
            'amplitude': 2.0,
            'width': 0.15
        },
        boundary_condition={
            'type': 'neumann',
            'flux': 0.0
        },
        agents=None,
        golden_solution={
            'type': 'gaussian_2d',
            'center': (1.0, 1.0),
            'amplitude': 2.0,
            'initial_width': 0.15,
            'diffusion_coefficient': 0.05
        },
        description='Custom scenario with faster diffusion on larger domain'
    )
    
    # Test with ADI method (efficient for 2D)
    runner = BenchmarkRunner()
    runner.add_schema(ExplicitEulerSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerSchema, "Implicit Euler")
    runner.add_schema(ADISchema, "ADI")
    runner.add_schema(CrankNicolsonSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonADISchema, "Crank-Nicolson ADI")
    runner.add_scenario(custom_scenario)
    
    results = runner.run(
        output_dir='benchmark_results/custom',
        store_history=True,
        generate_plots=True
    )
    
    summary = runner.generate_summary_report(
        output_path='benchmark_results/custom/summary.csv'
    )
    
    return results, summary

def run_step_function_1d():
    
    print("\n" + "=" * 70)
    print("EXAMPLE 8a: Step Function 1D")
    print("=" * 70)

    step_function_1d = get_scenario_by_name('step_function_1d')  # Alternatively, retrieve from registry

    runner = BenchmarkRunner()

    runner.add_schema(ExplicitEulerBCSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerBCSchema, "Implicit Euler")
    runner.add_schema(ADIBCSchema, "ADI")
    runner.add_schema(CrankNicolsonBCSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonADIBCSchema, "Crank-Nicolson ADI")

    runner.add_scenario(step_function_1d)
    
    results = runner.run(
        output_dir='benchmark_results/step_function_1d_bc',
        store_history=True,
        generate_plots=True
    )
    
    summary = runner.generate_summary_report(
        output_path='benchmark_results/step_function_1d_BC/summary.csv'
    )
    
    return results, summary


def run_step_function_2d():
    
    print("\n" + "=" * 70)
    print("EXAMPLE 8b: Step Function 2D")
    print("=" * 70)

    step_function_2d = get_scenario_by_name('step_function_2d')  # Alternatively, retrieve from registry

    runner = BenchmarkRunner()

    runner.add_schema(ExplicitEulerSchema, "Explicit Euler")
    runner.add_schema(ImplicitEulerSchema, "Implicit Euler")
    runner.add_schema(ADISchema, "ADI")
    runner.add_schema(CrankNicolsonSchema, "Crank-Nicolson")
    runner.add_schema(CrankNicolsonADISchema, "Crank-Nicolson ADI")

    runner.add_scenario(step_function_2d)
    
    results = runner.run(
        output_dir='benchmark_results/step_function_2d',
        store_history=True,
        generate_plots=True
    )
    
    summary = runner.generate_summary_report(
        output_path='benchmark_results/step_function_2d/summary.csv'
    )
    
    return results, summary

def run_all_dimensions():
    """
    Example 5: Test across 1D, 2D, and 3D scenarios.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Multi-Dimensional Benchmark")
    print("=" * 70)
    
    runner = BenchmarkRunner()
    
    # Add schemas (ADI methods work well in multiple dimensions)
    runner.add_schema(ImplicitEulerSchema, "Implicit Euler")
    runner.add_schema(CrankNicolsonSchema, "Crank-Nicolson")
    runner.add_schema(ADISchema, "ADI")
    
    # Add all default scenarios (1D, 2D, 3D)
    scenarios = get_default_scenarios()
    for scenario in scenarios:
        runner.add_scenario(scenario)
    
    # Run
    results = runner.run(
        output_dir='benchmark_results/multi_dim',
        store_history=True,
        generate_plots=True
    )
    
    summary = runner.generate_summary_report(
        output_path='benchmark_results/multi_dim/summary.csv'
    )
    
    return results, summary


def run_backward_compatible_example():
    """
    Example 6: Backward compatibility with original test_suite approach.
    
    This shows that existing code using the old API still works.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Backward Compatibility Example")
    print("=" * 70)
    
    # Original approach (still works)
    ic = gaussian(center=(0.5, 0.5), amplitude=1.0, width=0.1)
    bc = NeumannBC(flux=0.0)
    
    test_params = {
        'L': (1.0, 1.0),
        'N': (50, 50),
        'D': 0.01,
        'dt': 0.001,
        't_final': 0.5,
        'store_history': False,
        'bc': bc,
        'ic': ic,
        'agents': None
    }
    
    test_case = FlexibleSimulation(
        name="Original API Example",
        schema_class=ImplicitEulerSchema,
        params=test_params
    )
    
    test_case.initialize()
    results = test_case.run()
    
    print(f"\nResults using original API:")
    print(f"  Duration: {results['duration']:.4f} s")
    print(f"  Max Concentration: {results['max_concentration']:.4f}")
    print(f"  Min Concentration: {results['min_concentration']:.4f}")
    
    return results


def run_validation_scenario_example():
    """
    Example 7: Using ValidationScenario with dict-based specification.
    
    This bridges the old and new APIs.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: ValidationScenario with Dict Specification")
    print("=" * 70)
    
    # Get scenario dict
    scenario_dict = get_scenario_by_name('gaussian_pulse_2d')
    
    # Create ValidationScenario using from_dict
    val_scenario = ValidationScenario.from_dict(
        scenario_dict,
        schema_class=CrankNicolsonSchema
    )
    
    # Run
    val_scenario.initialize()
    results = val_scenario.run()
    
    print(f"\nResults with ValidationScenario:")
    print(f"  Duration: {results['duration']:.4f} s")
    print(f"  L2 Error: {results['errors']['l2_relative']:.6e}")
    print(f"  L∞ Error: {results['errors']['linf_relative']:.6e}")
    
    return results


def main():
    """
    Main function to run examples.
    
    Uncomment the examples you want to run.
    """
    print("\n" + "=" * 70)
    print("DIFFUSION SCHEMA BENCHMARKING FRAMEWORK")
    print("Example Usage Demonstration")
    print("=" * 70)
    
    # Example 1: Basic benchmark (recommended starting point)
    # results1, summary1 = run_basic_benchmark()
    
    # Example 2: 1D benchmark (faster, good for testing)
    # results2, summary2 = run_1d_benchmark()
    
    # Example 3: Convergence analysis (verifies order of accuracy)
    # conv_cn, conv_ie, conv_adi = run_convergence_analysis()

    # Examle 3b: Spatial convergence analysis
    # conv_cn_spatial, conv_ie_spatial, conv_adi_spatial = run_convergence_analysis_spatial()
    
    # Example 4: Custom scenario
    # results4, summary4 = run_custom_scenario()
    
    # Example 5: Multi-dimensional test
    # results5, summary5 = run_all_dimensions()
    
    # Example 6: Backward compatibility
    # results6 = run_backward_compatible_example()
    
    # Example 7: ValidationScenario
    # results7 = run_validation_scenario_example()

    # Example 8: Step functions
    results8, summary8 = run_step_function_1d()
    # results8, summary8 = run_step_function_2d()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("Check the 'benchmark_results' directory for outputs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
