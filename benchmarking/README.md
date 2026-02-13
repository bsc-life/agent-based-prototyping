# Diffusion Schema Testing Framework

A comprehensive framework for testing and validating diffusion schemas against analytical solutions.

## Overview

This framework provides:

- **Declarative scenario specification** - Define test scenarios as dictionaries
- **Automatic error computation** - L2, L∞, and mass conservation errors
- **Comprehensive visualizations** - Comparison plots, error distributions, time evolution, etc.
- **Convergence analysis** - Verify order of accuracy
- **Multiple dimensions** - Support for 1D, 2D, and 3D tests
- **Default scenarios** - Ready-to-use test cases with analytical solutions

## Quick Start

```python
from benchmarking import BenchmarkRunner, get_scenario_by_name
from diffusion_schemas import ImplicitEulerSchema, CrankNicolsonSchema

# Create benchmark runner
runner = BenchmarkRunner()

# Add schemas to test
runner.add_schema(ImplicitEulerSchema, "Implicit Euler")
runner.add_schema(CrankNicolsonSchema, "Crank-Nicolson")

# Add scenario
scenario = get_scenario_by_name('gaussian_pulse_2d')
runner.add_scenario(scenario)

# Run benchmarks
results = runner.run(output_dir='results', generate_plots=True)

# Generate summary report
summary = runner.generate_summary_report('results/summary.csv')
```

## Module Organization

### Core Modules

- **`scenarios.py`** - Scenario specification and factory functions
- **`golden_solutions.py`** - Analytical solutions for validation
- **`error_metrics.py`** - Error computation functions
- **`visualization.py`** - Plotting and visualization functions
- **`benchmark_runner.py`** - Main benchmark orchestration
- **`test_suite.py`** - Scenario base classes (original + new ValidationScenario)

### Usage Example

See `../examples/example_benchmark.py` for comprehensive examples including:

1. Basic benchmarking with default scenarios
2. 1D/2D/3D testing
3. Convergence analysis
4. Custom scenario creation
5. Backward compatibility with original API

## Scenario Specification

Scenarios are dict-based specifications that include:

```python
scenario = {
    'name': 'my_test',
    'domain_size': (1.0, 1.0),      # Physical domain size
    'grid_points': (50, 50),         # Grid resolution
    'dt': 0.001,                     # Time step
    't_final': 0.5,                  # Simulation end time
    'diffusion_coefficient': 0.01,   # D
    'decay_rate': 0.0,               # λ
    'initial_condition': {...},      # IC specification
    'boundary_condition': {...},     # BC specification
    'agents': [...],                 # Agent list (optional)
    'golden_solution': {...}         # Analytical solution
}
```

### Default Scenarios

Three default scenarios are provided:

- **`gaussian_pulse_1d`** - 1D Gaussian diffusion
- **`gaussian_pulse_2d`** - 2D Gaussian diffusion (default)
- **`gaussian_pulse_3d`** - 3D Gaussian diffusion

Access via:
```python
from benchmarking.scenarios import get_scenario_by_name, get_default_scenarios

scenario = get_scenario_by_name('gaussian_pulse_2d')
all_scenarios = get_default_scenarios()
```

## Analytical Solutions

Available golden solutions in `golden_solutions.py`:

- **`GaussianDiffusion1D/2D/3D`** - Fundamental Gaussian solution
- **`ExponentialDecay`** - Pure decay (no diffusion)
- **`SteadyStateAgentDiffusion`** - Steady-state with point source

## Error Metrics

The framework computes:

- **L2 error** - Integrated error norm (relative and absolute)
- **L∞ error** - Maximum pointwise error
- **Mass conservation** - Total mass drift (for zero-flux BC)
- **Convergence rate** - Order of accuracy from refinement studies

## Visualizations

Auto-generated plots include:

1. **Final comparison** - Side-by-side numerical vs analytical
2. **Error distribution** - Spatial distribution of errors
3. **Time evolution** - Snapshots at multiple times
4. **Error vs time** - Temporal error evolution
5. **Convergence plots** - Log-log error vs refinement

All figures saved in `output_dir/{scenario_name}/`.

## Convergence Analysis

Test order of accuracy:

```python
runner = BenchmarkRunner()
scenario = get_scenario_by_name('gaussian_pulse_1d')

# Temporal convergence
conv_results = runner.run_convergence_analysis(
    schema_class=CrankNicolsonSchema,
    schema_name="Crank-Nicolson",
    scenario_base=scenario,
    refinement_type='dt',
    refinement_factors=[0.01, 0.005, 0.0025, 0.00125]
)

print(f"Convergence rate: {conv_results['convergence_l2']['convergence_rate']:.2f}")
# Expected: ~2.0 for Crank-Nicolson
```

## Custom Scenarios

Create custom test cases:

```python
from benchmarking.scenarios import create_scenario

custom = create_scenario(
    name='my_scenario',
    domain_size=(2.0, 2.0),
    grid_points=(60, 60),
    dt=0.001,
    t_final=1.0,
    diffusion_coefficient=0.02,
    decay_rate=0.1,
    initial_condition={
        'type': 'gaussian',
        'center': (1.0, 1.0),
        'amplitude': 1.0,
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
        'amplitude': 1.0,
        'initial_width': 0.15,
        'diffusion_coefficient': 0.02
    }
)

runner.add_scenario(custom)
```

## Running Examples

```bash
cd benchmarking
python test_run_example.py
```

Results will be saved in `benchmark_results/` with subdirectories for each scenario containing:

- Comparison plots
- Error distributions
- Time evolution animations
- Error vs time plots
- Summary CSV files

## Backward Compatibility

The original `SimulationScenario` API still works:

```python
from benchmarking.test_suite import FlexibleSimulation
from diffusion_schemas import ImplicitEulerSchema

scenario = FlexibleSimulation(
    name="My Test",
    schema_class=ImplicitEulerSchema,
    params={'L': 1.0, 'N': 50, 'dt': 0.01, ...}
)
scenario.initialize()
results = scenario.run()
```

New `ValidationScenario` adds error computation:

```python
from benchmarking.test_suite import ValidationScenario

scenario_dict = get_scenario_by_name('gaussian_pulse_2d')
val_scenario = ValidationScenario.from_dict(scenario_dict, ImplicitEulerSchema)
val_scenario.initialize()
results = val_scenario.run()

print(f"L2 Error: {results['errors']['l2_relative']:.6e}")
```

## Summary Report

Generate comparison tables:

```python
summary_df = runner.generate_summary_report('output/summary.csv')
```

Output:
```
Schema             Scenario            Duration (s)  L2 Error    L∞ Error
Explicit Euler     gaussian_pulse_2d      0.1234    1.23e-03    2.45e-03
Implicit Euler     gaussian_pulse_2d      0.4567    3.45e-04    5.67e-04
Crank-Nicolson     gaussian_pulse_2d      0.5678    8.90e-05    1.23e-04
```
