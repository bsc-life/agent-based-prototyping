"""
Benchmarking framework for testing diffusion schemas against analytical solutions.

This package provides tools for:
- Defining test scenarios declaratively
- Running benchmarks with automatic error computation
- Generating comprehensive visualizations
- Performing convergence analysis

Quick Start
-----------
>>> from benchmarking import BenchmarkRunner, get_scenario_by_name
>>> from diffusion_schemas import ImplicitEulerSchema
>>> 
>>> runner = BenchmarkRunner()
>>> runner.add_schema(ImplicitEulerSchema)
>>> runner.add_scenario(get_scenario_by_name('gaussian_pulse_2d'))
>>> results = runner.run()

Main Components
---------------
BenchmarkRunner : class
    Main orchestrator for running benchmarks
ValidationScenario : class
    Scenario with analytical solution for validation
create_scenario : function
    Create custom scenario from parameters
get_default_scenarios : function
    Get list of predefined scenarios
get_scenario_by_name : function
    Retrieve specific default scenario

Error Metrics
-------------
compute_l2_error : function
    L2 relative error computation
compute_linf_error : function
    L-infinity (max pointwise) error
compute_convergence_rate : function
    Convergence rate from refinement studies

Analytical Solutions
--------------------
GaussianDiffusion1D/2D/3D : class
    Gaussian fundamental solutions
ExponentialDecay : class
    Pure decay solution
"""

# Main framework components
from benchmarking.benchmark_runner import BenchmarkRunner
from benchmarking.test_suite import (
    SimulationScenario,
    FlexibleSimulation,
    ValidationScenario,
    TwoAgentInteraction
)

# Scenario creation and defaults
from benchmarking.scenarios import (
    create_scenario,
    create_scenario_with_numerical_reference,
    get_default_scenarios,
    get_scenario_by_name,
    build_scenario_components,
    GAUSSIAN_PULSE_1D,
    GAUSSIAN_PULSE_2D,
    GAUSSIAN_PULSE_3D,
    STEP_FUNCTION_1D,
    STEP_FUNCTION_2D,
    STEADY_STATE_AGENT_1D,
    STEADY_STATE_AGENT_2D,
    EXPONENTIAL_DECAY_1D,
    SINE_DECAY_1D
)

# Analytical solutions
from benchmarking.golden_solutions import (
    GoldenSolution,
    GaussianDiffusion1D,
    GaussianDiffusion2D,
    GaussianDiffusion3D,
    ExponentialDecay,
    SteadyStateAgentDiffusion,
    StepFunctionDiffusion1D,
    StepFunctionDiffusion2D,
    SineDecay1D,
    create_golden_solution_from_dict
)

# Error metrics
from benchmarking.error_metrics import (
    compute_l2_error,
    compute_linf_error,
    compute_mass_conservation_error,
    compute_convergence_rate,
    compute_all_errors,
    compute_pointwise_error,
    compute_relative_pointwise_error
)

# Visualization functions
from benchmarking.visualization import (
    plot_final_comparison,
    plot_error_distribution,
    plot_time_evolution,
    plot_error_vs_time,
    plot_convergence_analysis
)

# Public API
__all__ = [
    # Main runner
    'BenchmarkRunner',
    
    # Scenario classes
    'SimulationScenario',
    'FlexibleSimulation',
    'ValidationScenario',
    'TwoAgentInteraction',
    
    # Scenario creation
    'create_scenario',
    'create_scenario_with_numerical_reference',
    'get_default_scenarios',
    'get_scenario_by_name',
    'build_scenario_components',
    
    # Default scenarios
    'GAUSSIAN_PULSE_1D',
    'GAUSSIAN_PULSE_2D',
    'GAUSSIAN_PULSE_3D',
    
    # Golden solutions
    'GoldenSolution',
    'GaussianDiffusion1D',
    'GaussianDiffusion2D',
    'GaussianDiffusion3D',
    'ExponentialDecay',
    'SteadyStateAgentDiffusion',
    'create_golden_solution_from_dict',
    
    # Error metrics
    'compute_l2_error',
    'compute_linf_error',
    'compute_mass_conservation_error',
    'compute_convergence_rate',
    'compute_all_errors',
    'compute_pointwise_error',
    'compute_relative_pointwise_error',
    
    # Visualization
    'plot_final_comparison',
    'plot_error_distribution',
    'plot_time_evolution',
    'plot_error_vs_time',
    'plot_convergence_analysis',
]

__version__ = '1.0.0'
