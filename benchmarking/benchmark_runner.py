"""
Test runner framework for benchmarking diffusion schemas.

This module provides the BenchmarkRunner class that orchestrates the testing
of diffusion schemas against defined scenarios, computing error metrics and
generating visualizations.
"""

import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type, Tuple
from collections import defaultdict
import pandas as pd

from diffusion_schemas.base import Schema
from benchmarking.scenarios import build_scenario_components
from benchmarking.error_metrics import (
    compute_all_errors, compute_convergence_rate, 
    compute_l2_error, compute_linf_error
)
from benchmarking.visualization import (
    plot_final_comparison, plot_error_distribution,
    plot_time_evolution, plot_error_vs_time, plot_convergence_analysis,
    plot_method_comparison
)


class BenchmarkRunner:
    """
    Framework for running benchmarks on diffusion schemas.
    
    This class manages the execution of multiple schemas against multiple scenarios,
    computes error metrics, generates visualizations, and collects results.
    """
    
    def __init__(self):
        """Initialize the benchmark runner."""
        self.scenarios = []
        self.schemas = []
        self.results = {}
        
    def add_scenario(self, scenario: Dict[str, Any]):
        """
        Add a test scenario.
        
        Parameters
        ----------
        scenario : dict
            Scenario specification (from scenarios.create_scenario() or default scenarios).
        """
        self.scenarios.append(scenario)
    
    def add_schema(self, schema_class: Type[Schema], name: Optional[str] = None):
        """
        Add a schema to test.
        
        Parameters
        ----------
        schema_class : type
            Schema class (e.g., ExplicitEulerSchema, CrankNicolsonSchema).
        name : str, optional
            Custom name for the schema. If None, uses class name.
        """
        if name is None:
            name = schema_class.__name__
        self.schemas.append((schema_class, name))
    
    def run(self, output_dir: Union[str, Path] = 'benchmark_results', 
            store_history: bool = True, generate_plots: bool = True) -> Dict[str, Any]:
        """
        Run all benchmarks.
        
        Executes each schema on each scenario, computes errors, and generates visualizations.
        
        Parameters
        ----------
        output_dir : str or Path, optional
            Directory for saving results and plots (default 'benchmark_results').
        store_history : bool, optional
            Whether to store full time history (needed for time evolution plots).
        generate_plots : bool, optional
            Whether to generate visualization plots.
            
        Returns
        -------
        dict
            Results dictionary with structure:
            {(schema_name, scenario_name): {
                'errors': {...},
                'duration': float,
                'final_state': ndarray,
                'history': list (if store_history=True),
                'figures': list of paths
            }}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Running benchmarks: {len(self.schemas)} schema(s) × {len(self.scenarios)} scenario(s)")
        print("=" * 70)
        
        for schema_class, schema_name in self.schemas:
            for scenario in self.scenarios:
                scenario_name = scenario['name']
                print(f"\nTesting: {schema_name} on {scenario_name}")
                print("-" * 70)
                
                # Run benchmark
                result = self._run_single_benchmark(
                    schema_class, schema_name, scenario, 
                    output_dir, store_history, generate_plots
                )
                
                # Store results
                key = (schema_name, scenario_name)
                self.results[key] = result
                
                # Print summary
                print(f"  Duration: {result['duration']:.4f} s")
                if 'errors' in result:
                    print(f"  L2 error: {result['errors'].get('l2_relative', 'N/A'):.6e}")
                    print(f"  L∞ error: {result['errors'].get('linf_relative', 'N/A'):.6e}")
        
        # Store final states for each schema for later comparison of single runs
        if generate_plots:
            # Group results by scenario
            # Structure: grouped_results['ScenarioName'] = {'SchemaName': result_dict, ...}
            grouped_results = defaultdict(dict)
            for (schema_name, scenario_name), res in self.results.items():
                grouped_results[scenario_name][schema_name] = res

            # Loop through scenarios and plot if applicable
            for sc_name, schema_dict in grouped_results.items():
                # Only compare if we have multiple schemas
                if len(schema_dict) <= 1:
                    continue

                print(f"\nGenerating comparison plot for: {sc_name}")
                print("-" * 70)

                fig_path = output_dir / sc_name / "method_comparison.png"
                plot_method_comparison(sc_name, schema_dict, fig_path)
                
                for res in schema_dict.values():
                    res['figures'].append(str(fig_path))

        print("\n" + "=" * 70)
        print("Benchmark complete!")
        
        return self.results
    
    def _run_single_benchmark(self, schema_class: Type[Schema], schema_name: str,
                             scenario: Dict[str, Any], output_dir: Path,
                             store_history: bool, generate_plots: bool) -> Dict[str, Any]:
        """Run a single schema-scenario benchmark."""
        
        # Build scenario components
        built_scenario = build_scenario_components(scenario, store_history=store_history)
        
        # Initialize schema
        schema = schema_class(
            domain_size=scenario['domain_size'],
            grid_points=scenario['grid_points'],
            dt=scenario['dt'],
            diffusion_coefficient=scenario['diffusion_coefficient'],
            decay_rate=scenario['decay_rate']
        )
        
        # Set initial condition
        schema.set_initial_condition(built_scenario['initial_condition'])
        
        # Set boundary condition
        if built_scenario['boundary_condition'] is not None:
            schema.set_boundary_conditions(built_scenario['boundary_condition'])
        
        # Add agents
        if built_scenario['agents'] is not None:
            for agent in built_scenario['agents']:
                schema.add_agent(agent)

        # Add bulk regions
        if built_scenario['bulk'] is not None:
            schema.set_bulk(built_scenario['bulk'])
        
        # Store initial mass for conservation check
        initial_mass = np.sum(schema.state)
        if schema.dx is not None:
            if isinstance(schema.dx, (list, tuple)):
                dV = np.prod(schema.dx)
            else:
                dV = schema.dx ** schema.ndim
            initial_mass *= dV
        else:
            dV = None
        
        # Run simulation
        start_time = time.perf_counter()
        history, times = schema.solve(scenario['t_final'], store_history=store_history, progress = True)
        duration = time.perf_counter() - start_time
        
        # Get final state
        final_state = schema.get_state()
        
        if 'golden_solution' in built_scenario and built_scenario['golden_solution'] is not None:
            # Evaluate golden solution at final time
            golden_solution = built_scenario['golden_solution']
            coordinates = schema._create_coordinate_grids()
            
            if hasattr(golden_solution, 'evaluate'):
                analytical_final = golden_solution.evaluate(coordinates, scenario['t_final'])
            else:
                analytical_final = golden_solution(*coordinates, scenario['t_final'])
            
            # Compute errors
            errors = compute_all_errors(
                final_state, analytical_final, 
                dx=schema.dx, initial_mass=initial_mass
            )
        
        else:
            analytical_final = None
            errors = {}
        
        # Prepare result
        result = {
            'duration': duration,
            'errors': errors,
            'final_state': final_state,
            'analytical_final': analytical_final,
            'figures': []
        }
        
        if store_history:
            result['history'] = history
            result['times'] = times
            
            # Compute error vs time
            error_timeseries = self._compute_error_timeseries(
                history, times, golden_solution, coordinates, schema.dx
            )
            result['error_timeseries'] = error_timeseries
        
        # Generate plots
        if generate_plots:
            scenario_dir = output_dir / scenario['name']
            scenario_dir.mkdir(parents=True, exist_ok=True)
            
            # Final comparison plot
            fig_path = scenario_dir / f"{schema_name}_comparison.png"
            plot_final_comparison(
                final_state, analytical_final, coordinates,
                schema_name, scenario['name'], output_path=fig_path
            )
            result['figures'].append(str(fig_path))
            
            # Error distribution plot
            fig_path = scenario_dir / f"{schema_name}_error_dist.png"
            plot_error_distribution(
                final_state, analytical_final, coordinates,
                schema_name, scenario['name'], output_path=fig_path
            )
            result['figures'].append(str(fig_path))
            
            # Time evolution plot (if history available)
            if store_history and len(history) > 1:
                fig_path = scenario_dir / f"{schema_name}_evolution.png"
                plot_time_evolution(
                    history, times, golden_solution, coordinates,
                    schema_name, scenario['name'], output_path=fig_path
                )
                result['figures'].append(str(fig_path))
                
                # Error vs time plot
                fig_path = scenario_dir / f"{schema_name}_error_vs_time.png"
                plot_error_vs_time(
                    times, error_timeseries,
                    schema_name, scenario['name'], output_path=fig_path
                )
                result['figures'].append(str(fig_path))
        
        return result
    
    def _compute_error_timeseries(self, history: List[np.ndarray], times: List[float],
                                  golden_solution, coordinates, dx) -> Dict[str, List[float]]:
        """Compute error metrics at each time step."""
        error_timeseries = defaultdict(list)
        
        for state, t in zip(history, times):
            # Evaluate analytical solution
            if hasattr(golden_solution, 'evaluate'):
                analytical = golden_solution.evaluate(coordinates, t)
            else:
                analytical = golden_solution(*coordinates, t)
            
            # Compute errors
            l2_err = compute_l2_error(state, analytical, dx)
            linf_err = compute_linf_error(state, analytical)
            
            error_timeseries['l2_relative'].append(l2_err['l2_relative'])
            error_timeseries['l2_absolute'].append(l2_err['l2_absolute'])
            error_timeseries['linf_relative'].append(linf_err['linf_relative'])
            error_timeseries['linf_absolute'].append(linf_err['linf_absolute'])
        
        return dict(error_timeseries)
    
    def run_convergence_analysis(self, schema_class: Type[Schema], schema_name: str,
                                 scenario_base: Dict[str, Any],
                                 refinement_type: str = 'dt',
                                 refinement_factors: List[float] = None,
                                 output_dir: Union[str, Path] = 'convergence_results') -> Dict[str, Any]:
        """
        Run convergence analysis by varying dt or grid spacing.
        
        Parameters
        ----------
        schema_class : type
            Schema class to test.
        schema_name : str
            Name of the schema.
        scenario_base : dict
            Base scenario specification.
        refinement_type : str, optional
            Type of refinement: 'dt' for temporal, 'spatial' for grid refinement.
        refinement_factors : list of float, optional
            Refinement values to test. If None, uses default sequence.
        output_dir : str or Path, optional
            Directory for saving results.
            
        Returns
        -------
        dict
            Convergence analysis results with computed convergence rates.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if refinement_factors is None:
            if refinement_type == 'dt':
                # Default dt sequence
                dt_base = scenario_base['dt']
                refinement_factors = [dt_base * 2**(-i) for i in range(5)]
            else:  # spatial
                # Default grid refinement
                N_base = scenario_base['grid_points']
                if isinstance(N_base, int):
                    refinement_factors = [N_base * 2**i for i in range(4)]
                else:
                    refinement_factors = [tuple(n * 2**i for n in N_base) for i in range(4)]
        
        print(f"\nConvergence Analysis: {schema_name} on {scenario_base['name']}")
        print(f"Refinement type: {refinement_type}")
        print(f"Refinement values: {refinement_factors}")
        print("=" * 70)
        
        errors_l2 = []
        errors_linf = []
        durations = []
        
        for refinement in refinement_factors:
            # Modify scenario
            scenario = scenario_base.copy()
            
            if refinement_type == 'dt':
                scenario['dt'] = refinement
                h_value = refinement
            else:  # spatial
                scenario['grid_points'] = refinement
                # Compute effective dx
                domain_size = scenario['domain_size']
                if isinstance(refinement, int):
                    h_value = domain_size / (refinement - 1)
                else:
                    h_value = domain_size[0] / (refinement[0] - 1) 
                    # h_value = domain_size[0] / (refinement - 1)
            
            print(f"\n  Testing with {refinement_type}={refinement}")
            
            # Run benchmark
            result = self._run_single_benchmark(
                schema_class, schema_name, scenario,
                output_dir, store_history=False, generate_plots=False
            )
            
            errors_l2.append(result['errors']['l2_relative'])
            errors_linf.append(result['errors']['linf_relative'])
            durations.append(result['duration'])
            
            print(f"    L2 error: {result['errors']['l2_relative']:.6e}")
            print(f"    L∞ error: {result['errors']['linf_relative']:.6e}")
        
        # Compute convergence rates
        if refinement_type == 'dt':
            h_values = refinement_factors
        else:
            # For spatial, extract dx values
            h_values = []
            domain_size = scenario_base['domain_size']
            for N in refinement_factors:
                if isinstance(N, int):
                    h_values.append(domain_size / (N - 1))
                else:
                    h_values.append(domain_size[0] / (N[0] - 1))
        
        conv_l2 = compute_convergence_rate(errors_l2, h_values)
        conv_linf = compute_convergence_rate(errors_linf, h_values)
        
        print("\n" + "=" * 70)
        print(f"Convergence Rates:")
        print(f"  L2:  {conv_l2['convergence_rate']:.3f} (R²={conv_l2['r_squared']:.4f})")
        print(f"  L∞:  {conv_linf['convergence_rate']:.3f} (R²={conv_linf['r_squared']:.4f})")
        
        # Generate convergence plot
        errors_dict = {
            'L2 relative': errors_l2,
            'L∞ relative': errors_linf
        }
        rates_dict = {
            'L2 relative': conv_l2['convergence_rate'],
            'L∞ relative': conv_linf['convergence_rate']
        }
        
        fig_path = output_dir / f"{schema_name}_{scenario_base['name']}_convergence_{refinement_type}.png"
        plot_convergence_analysis(
            h_values, errors_dict, rates_dict,
            schema_name, scenario_base['name'],
            output_path=fig_path, refinement_type=refinement_type
        )
        
        # Return results
        return {
            'refinement_type': refinement_type,
            'refinement_values': refinement_factors,
            'h_values': h_values,
            'errors_l2': errors_l2,
            'errors_linf': errors_linf,
            'durations': durations,
            'convergence_l2': conv_l2,
            'convergence_linf': conv_linf,
            'figure': str(fig_path)
        }
    
    def generate_summary_report(self, output_path: Union[str, Path] = None) -> pd.DataFrame:
        """
        Generate summary report of all benchmark results.
        
        Parameters
        ----------
        output_path : str or Path, optional
            If provided, save report to CSV file.
            
        Returns
        -------
        pandas.DataFrame
            Summary table with schema, scenario, errors, and duration.
        """
        if not self.results:
            print("No results to report. Run benchmarks first.")
            return None
        
        # Build summary table
        rows = []
        for (schema_name, scenario_name), result in self.results.items():
            row = {
                'Schema': schema_name,
                'Scenario': scenario_name,
                'Duration (s)': result['duration'],
                'L2 Error': result['errors'].get('l2_relative', np.nan),
                'Linf Error': result['errors'].get('linf_relative', np.nan),
            }
            
            if 'mass_conservation_relative' in result['errors']:
                row['Mass Conservation Error'] = result['errors']['mass_conservation_relative']
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by scenario then schema
        df = df.sort_values(['Scenario', 'Schema'])
        
        # Print to console
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(df.to_string(index=False))
        
        # Save to file
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"\nSummary saved to: {output_path}")
        
        return df
    
    def clear_results(self):
        """Clear stored results."""
        self.results = {}
    
    def clear_scenarios(self):
        """Clear registered scenarios."""
        self.scenarios = []
    
    def clear_schemas(self):
        """Clear registered schemas."""
        self.schemas = []
