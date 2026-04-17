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
        self.dt_threshold_results = {}
        self.dt_eval_results = {}

    @staticmethod
    def _extract_errors_at_eval_times(times: List[float], error_timeseries: Dict[str, List[float]],
                                      eval_times: List[float]) -> Dict[str, List[float]]:
        """Extract relative errors at requested times using nearest available simulation time."""
        l2_errors = []
        linf_errors = []

        for t_eval in eval_times:
            idx = min(range(len(times)), key=lambda i: abs(times[i] - t_eval))
            l2_errors.append(float(error_timeseries['l2_relative'][idx]))
            linf_errors.append(float(error_timeseries['linf_relative'][idx]))

        return {
            'l2_relative': l2_errors,
            'linf_relative': linf_errors
        }
        
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
                                 refinement_factors: Optional[List[float]] = None,
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
            else:  # spatial
                scenario['grid_points'] = refinement
                # Compute effective dx
                domain_size = scenario['domain_size']
            
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

    ### Methods to search for threshold dt 

    @staticmethod
    def _first_index_at_or_above(values: List[float], threshold: float) -> Optional[int]:
        """Return index of first value >= threshold, or None if not found."""
        for i, value in enumerate(values):
            if value >= threshold:
                return i
        return None

    def _run_dt_threshold_for_schema(self, schema_class: Type[Schema], schema_name: str,
                                     scenario_base: Dict[str, Any], output_dir: Path,
                                     target_error: float, growth_factor: float,
                                     max_iterations: int, dt_start: Optional[float],
                                     dt_max: Optional[float]) -> Dict[str, Any]:
        """
        Increase dt progressively and estimate the largest dt that keeps
        L2/Linf relative error <= target.

        This method is intentionally additive and does not alter the behavior of existing
        benchmark flows.
        """
        if growth_factor <= 1.0:
            raise ValueError("growth_factor must be greater than 1.0 when increasing dt")

        initial_dt = float(dt_start if dt_start is not None else scenario_base['dt'])
        if initial_dt <= 0.0:
            raise ValueError(f"Invalid dt_start={initial_dt}. dt must be positive")

        if dt_max is not None and dt_max <= 0.0:
            raise ValueError("dt_max must be positive when provided")

        dt_values: List[float] = []
        l2_values: List[float] = []
        linf_values: List[float] = []
        durations: List[float] = []
        errors_trace: List[Dict[str, float]] = []

        def evaluate_dt(dt_value: float, probe_idx: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
            """Run one probe at a given dt and return (result, failure_message)."""
            scenario = scenario_base.copy()
            scenario['dt'] = float(dt_value)
            scenario['name'] = f"{scenario_base['name']}_dt_probe_{probe_idx}".replace('.', '')
            try:
                result = self._run_single_benchmark(
                    schema_class, schema_name, scenario,
                    output_dir=output_dir,
                    store_history=False,
                    generate_plots=False
                )
                return result, None
            except Exception as exc:
                return None, str(exc)

        status = 'completed'
        failure_message = None
        dt = initial_dt
        probe_idx = 1
        dt_start_retry_applied = False

        # Coarse geometric search: expand dt quickly to bracket the crossing zone.
        for _ in range(max_iterations):
            if dt_max is not None and dt > dt_max:
                status = 'dt_max_reached'
                break

            result, failure = evaluate_dt(dt, probe_idx)
            probe_idx += 1

            if failure is not None:
                status = 'failed'
                failure_message = failure
                break

            errors = result.get('errors', {})
            l2_rel = float(errors.get('l2_relative', np.nan))
            linf_rel = float(errors.get('linf_relative', np.nan))

            if not np.isfinite(l2_rel) or not np.isfinite(linf_rel):
                status = 'non_finite_error'
                break

            dt_values.append(float(dt))
            l2_values.append(l2_rel)
            linf_values.append(linf_rel)
            durations.append(float(result['duration']))
            errors_trace.append({
                'l2_relative': l2_rel,
                'linf_relative': linf_rel
            })
            print(f"    probe {probe_idx - 1}: dt={dt:.6g}, L2={l2_rel:.6e}, Linf={linf_rel:.6e}")

            # Fast-fail: if the first probe is already above target for both metrics,
            # continuing the geometric increase cannot recover a safe dt.
            if len(dt_values) == 1 and l2_rel > target_error and linf_rel > target_error:
                status = 'dt_start_above_target'
                failure_message = (
                    "Initial dt is above target for both L2 and Linf;"
                )
                break
            
            # One-time recovery: if first probe is above target for both metrics,
            # retry once with dt_start / 2 before continuing geometric growth.
            # if (not dt_start_retry_applied and len(dt_values) == 1
            #         and l2_rel > target_error and linf_rel > target_error):
            #     dt = 0.5 * dt
            #     dt_start_retry_applied = True
            #     continue

            dt *= growth_factor

        def first_index_above(values: List[float], threshold: float) -> Optional[int]:
            for i, value in enumerate(values):
                if value > threshold:
                    return i
            return None

        def last_index_at_or_below(values: List[float], threshold: float) -> Optional[int]:
            idx = None
            for i, value in enumerate(values):
                if value <= threshold:
                    idx = i
            return idx

        l2_safe_idx = last_index_at_or_below(l2_values, target_error)
        linf_safe_idx = last_index_at_or_below(linf_values, target_error)

        l2_cross_idx = first_index_above(l2_values, target_error)
        linf_cross_idx = first_index_above(linf_values, target_error)

        def refine_largest_safe_dt(metric_key: str, safe_idx: int, cross_idx: int,
                                   base_error: float) -> Tuple[float, float]:
            """
            Local bisection in [dt_safe, dt_unsafe] to refine largest dt with error <= target.
            """
            print(f"Starting local refinement for metric: {metric_key}")
            lo_dt = dt_values[safe_idx]
            hi_dt = dt_values[cross_idx]
            best_dt = lo_dt
            best_err = base_error
            bisect_steps = 8

            nonlocal status, failure_message, probe_idx

            for _ in range(bisect_steps):
                mid_dt = 0.5 * (lo_dt + hi_dt)
                result, failure = evaluate_dt(mid_dt, probe_idx)
                probe_idx += 1

                if failure is not None:
                    status = 'failed'
                    failure_message = failure
                    break

                errors = result.get('errors', {})
                l2_mid = float(errors.get('l2_relative', np.nan))
                linf_mid = float(errors.get('linf_relative', np.nan))

                if not np.isfinite(l2_mid) or not np.isfinite(linf_mid):
                    status = 'non_finite_error'
                    break

                mid_err = l2_mid if metric_key == 'l2_relative' else linf_mid

                dt_values.append(float(mid_dt))
                l2_values.append(l2_mid)
                linf_values.append(linf_mid)
                durations.append(float(result['duration']))
                errors_trace.append({
                    'l2_relative': l2_mid,
                    'linf_relative': linf_mid
                })
                print(f"    probe {probe_idx - 1}: dt={mid_dt:.6g}, L2={l2_mid:.6e}, Linf={linf_mid:.6e}")

                if mid_err <= target_error:
                    lo_dt = mid_dt
                    best_dt = mid_dt
                    best_err = mid_err
                else:
                    hi_dt = mid_dt

            return best_dt, best_err

        if l2_safe_idx is not None and l2_cross_idx is not None and l2_cross_idx > l2_safe_idx:
            l2_dt_refined, l2_err_refined = refine_largest_safe_dt(
                metric_key='l2_relative',
                safe_idx=l2_safe_idx,
                cross_idx=l2_cross_idx,
                base_error=l2_values[l2_safe_idx]
            )
        else:
            l2_dt_refined = dt_values[l2_safe_idx] if l2_safe_idx is not None else None
            l2_err_refined = l2_values[l2_safe_idx] if l2_safe_idx is not None else None

        if linf_safe_idx is not None and linf_cross_idx is not None and linf_cross_idx > linf_safe_idx:
            linf_dt_refined, linf_err_refined = refine_largest_safe_dt(
                metric_key='linf_relative',
                safe_idx=linf_safe_idx,
                cross_idx=linf_cross_idx,
                base_error=linf_values[linf_safe_idx]
            )
        else:
            linf_dt_refined = dt_values[linf_safe_idx] if linf_safe_idx is not None else None
            linf_err_refined = linf_values[linf_safe_idx] if linf_safe_idx is not None else None

        if status == 'completed' and len(dt_values) >= max_iterations and (l2_cross_idx is None or linf_cross_idx is None):
            status = 'max_iterations_reached'

        result_dict = {
            'schema': schema_name,
            'scenario': scenario_base['name'],
            'target_error': target_error,
            'growth_factor': growth_factor,
            'max_iterations': max_iterations,
            'dt_start': initial_dt,
            'dt_max': dt_max,
            'status': status,
            'failure_message': failure_message,
            'trace': {
                'dt': dt_values,
                'l2_relative': l2_values,
                'linf_relative': linf_values,
                'duration': durations,
                'errors': errors_trace
            },
            'threshold_l2': {
                'reached': l2_safe_idx is not None,
                'index': l2_safe_idx,
                'dt': l2_dt_refined,
                'error': l2_err_refined,
            },
            'threshold_linf': {
                'reached': linf_safe_idx is not None,
                'index': linf_safe_idx,
                'dt': linf_dt_refined,
                'error': linf_err_refined,
            }
        }
        return result_dict

    def run_dt_threshold_search(self, scenario_base: Dict[str, Any],
                                target_error: float = 0.05,
                                growth_factor: float = 1.5,
                                max_iterations: int = 20,
                                dt_start: Optional[float] = None,
                                dt_max: Optional[float] = None,
                                output_dir: Union[str, Path] = 'benchmark_results',
                                output_csv: Optional[Union[str, Path]] = None,
                                generate_plots: bool = True) -> Dict[str, Any]:
        """
        Run an opt-in dt threshold search for all registered schemas on one scenario.

        dt is increased geometrically until each metric reaches the target error.
        Existing benchmark methods are untouched and remain the default workflow.
        """
        if not self.schemas:
            raise ValueError("No schemas registered. Use add_schema() before running threshold search")

        if target_error <= 0.0:
            raise ValueError("target_error must be positive")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scenario_name = scenario_base['name']
        print(f"\nDT Threshold Search: scenario={scenario_name}, target_error={target_error:.2%}")
        print(f"Schemas to test: {len(self.schemas)}")
        print("=" * 70)

        run_results = {}
        threshold_plot_results = {}
        for schema_class, schema_name in self.schemas:
            print(f"\nTesting threshold for {schema_name}")
            print("-" * 70)

            schema_result = self._run_dt_threshold_for_schema(
                schema_class=schema_class,
                schema_name=schema_name,
                scenario_base=scenario_base,
                output_dir=output_dir,
                target_error=target_error,
                growth_factor=growth_factor,
                max_iterations=max_iterations,
                dt_start=dt_start,
                dt_max=dt_max
            )

            schema_result['figures'] = []

            run_results[schema_name] = schema_result
            self.dt_threshold_results[(schema_name, scenario_name)] = schema_result

            l2_info = schema_result['threshold_l2']
            linf_info = schema_result['threshold_linf']

            l2_msg = f"dt={l2_info['dt']:.6g}, err={l2_info['error']:.6e}" if l2_info['reached'] else "not reached"
            linf_msg = f"dt={linf_info['dt']:.6g}, err={linf_info['error']:.6e}" if linf_info['reached'] else "not reached"

            print(f"  status: {schema_result['status']}")
            print(f"  L2 threshold: {l2_msg}")
            print(f"  Linf threshold: {linf_msg}")

            if schema_result['failure_message']:
                print(f"  failure: {schema_result['failure_message']}")

            if generate_plots:
                chosen_dt = None
                if schema_result['threshold_l2'].get('dt') is not None:
                    chosen_dt = float(schema_result['threshold_l2']['dt'])
                elif schema_result['threshold_linf'].get('dt') is not None:
                    chosen_dt = float(schema_result['threshold_linf']['dt'])

                if chosen_dt is not None:
                    scenario = scenario_base.copy()
                    scenario['dt'] = chosen_dt
                    scenario['name'] = scenario_name

                    try:
                        plotted_result = self._run_single_benchmark(
                            schema_class=schema_class,
                            schema_name=schema_name,
                            scenario=scenario,
                            output_dir=output_dir,
                            store_history=False,
                            generate_plots=True
                        )
                        schema_result['figures'].extend(plotted_result.get('figures', []))
                        threshold_plot_results[schema_name] = plotted_result
                    except Exception as exc:
                        print(f"  threshold plot generation failed: {exc}")

        if generate_plots and len(threshold_plot_results) > 1:
            scenario_dir = output_dir / scenario_name
            scenario_dir.mkdir(parents=True, exist_ok=True)

            comparison_path = scenario_dir / "method_comparison_dt_threshold.png"
            plot_method_comparison(scenario_name, threshold_plot_results, comparison_path)

            for schema_name, schema_result in run_results.items():
                if schema_name in threshold_plot_results:
                    schema_result['figures'].append(str(comparison_path))

        if output_csv:
            self.generate_dt_threshold_report(output_path=output_csv, scenario_name=scenario_name)

        return run_results

    def run_dt_eval_times_grid(self, scenario_base: Dict[str, Any],
                               dt_values: List[float],
                               eval_times: List[float],
                               output_dir: Union[str, Path] = 'benchmark_results',
                               output_csv: Optional[Union[str, Path]] = None,
                               generate_plots: bool = True) -> pd.DataFrame:
        """
        Run benchmarks for a fixed list of dt values and store errors at eval_times.

        Produces one row per (schema, scenario, dt) with list-valued columns for
        eval_times and corresponding relative errors.
        """
        if not self.schemas:
            raise ValueError("No schemas registered. Use add_schema() before running dt eval-time grid")

        if not dt_values:
            raise ValueError("dt_values must be a non-empty list")

        if not eval_times:
            raise ValueError("eval_times must be a non-empty list")

        if any(float(dt) <= 0.0 for dt in dt_values):
            raise ValueError("All dt_values must be positive")

        scenario_t_final = float(scenario_base['t_final'])
        if any(float(t) < 0.0 or float(t) > scenario_t_final for t in eval_times):
            raise ValueError("All eval_times must satisfy 0 <= t <= scenario['t_final']")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scenario_name = scenario_base['name']
        print(f"\nDT Eval-Time Grid: scenario={scenario_name}")
        print(f"Schemas to test: {len(self.schemas)}")
        print(f"dt values: {dt_values}")
        print(f"eval_times: {eval_times}")
        print("=" * 70)

        rows: List[Dict[str, Any]] = []
        schema_completed_results: Dict[str, List[Tuple[float, Dict[str, Any]]]] = defaultdict(list)
        for schema_class, schema_name in self.schemas:
            print(f"\nTesting eval-time grid for {schema_name}")
            print("-" * 70)

            for i, dt in enumerate(dt_values, start=1):
                scenario = scenario_base.copy()
                scenario['dt'] = float(dt)
                scenario['name'] = f"{scenario_base['name']}_dt_eval_{i}".replace('.', '')

                try:
                    result = self._run_single_benchmark(
                        schema_class=schema_class,
                        schema_name=schema_name,
                        scenario=scenario,
                        output_dir=output_dir,
                        store_history=True,
                        generate_plots=generate_plots
                    )
                except Exception as exc:
                    rows.append({
                        'Schema': schema_name,
                        'Scenario': scenario_base['name'],
                        'dt': float(dt),
                        'eval_times': [float(t) for t in eval_times],
                        'l2_errors': np.nan,
                        'linf_errors': np.nan,
                        'Duration (s)': np.nan,
                        'Status': 'failed',
                        'Failure Message': str(exc)
                    })
                    continue

                errors_at_eval = self._extract_errors_at_eval_times(
                    times=result['times'],
                    error_timeseries=result['error_timeseries'],
                    eval_times=[float(t) for t in eval_times]
                )

                rows.append({
                    'Schema': schema_name,
                    'Scenario': scenario_base['name'],
                    'dt': float(dt),
                    'eval_times': [float(t) for t in eval_times],
                    'l2_errors': errors_at_eval['l2_relative'],
                    'linf_errors': errors_at_eval['linf_relative'],
                    'Duration (s)': float(result['duration']),
                    'Status': 'completed',
                    'Failure Message': None
                })
                schema_completed_results[schema_name].append((float(dt), result))

        if not rows:
            print("No dt eval-time rows were produced.")
            return None

        df = pd.DataFrame(rows)
        df = df.sort_values(['Scenario', 'Schema', 'dt'])

        self.dt_eval_results[scenario_name] = rows

        print("\n" + "=" * 70)
        print("DT EVAL-TIME GRID SUMMARY")
        print("=" * 70)
        print(df.to_string(index=False))

        if generate_plots:
            comparison_inputs = {}
            for schema_name, results_for_schema in schema_completed_results.items():
                if not results_for_schema:
                    continue

                base_dt = float(scenario_base['dt'])
                selected_dt, selected_result = min(
                    results_for_schema,
                    key=lambda item: abs(item[0] - base_dt)
                )
                comparison_inputs[schema_name] = selected_result

            if len(comparison_inputs) > 1:
                scenario_dir = output_dir / scenario_name
                scenario_dir.mkdir(parents=True, exist_ok=True)

                comparison_path = scenario_dir / "method_comparison_dt_eval.png"
                plot_method_comparison(scenario_name, comparison_inputs, comparison_path)
                df['Comparison Figure'] = str(comparison_path)

        if output_csv:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)
            print(f"\nDT eval-time grid saved to: {output_csv}")

        return df

    def generate_dt_threshold_report(self, output_path: Union[str, Path] = None,
                                     scenario_name: Optional[str] = None) -> pd.DataFrame:
        """
        Generate a compact summary table for dt threshold searches.

        Parameters
        ----------
        output_path : str or Path, optional
            If provided, save report to CSV file.
        scenario_name : str, optional
            If provided, only include rows for this scenario.
        """
        if not self.dt_threshold_results:
            print("No dt threshold results to report. Run run_dt_threshold_search() first.")
            return None

        rows = []
        for (_, sc_name), result in self.dt_threshold_results.items():
            if scenario_name is not None and sc_name != scenario_name:
                continue

            trace = result.get('trace', {})
            l2_info = result.get('threshold_l2', {})
            linf_info = result.get('threshold_linf', {})

            row = {
                'Schema': result.get('schema'),
                'Scenario': result.get('scenario'),
                'Target Error': result.get('target_error'),
                'Status': result.get('status'),
                'Iterations': len(trace.get('dt', [])),
                'L2 Threshold Reached': l2_info.get('reached', False),
                'L2 Threshold dt': l2_info.get('dt', np.nan),
                'L2 Error at Threshold': l2_info.get('error', np.nan),
                'Linf Threshold Reached': linf_info.get('reached', False),
                'Linf Threshold dt': linf_info.get('dt', np.nan),
                'Linf Error at Threshold': linf_info.get('error', np.nan),
                'Failure Message': result.get('failure_message')
            }
            rows.append(row)

        if not rows:
            print("No dt threshold rows to report for the selected filters.")
            return None

        df = pd.DataFrame(rows)
        df = df.sort_values(['Scenario', 'Schema'])

        print("\n" + "=" * 70)
        print("DT THRESHOLD SUMMARY")
        print("=" * 70)
        print(df.to_string(index=False))

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"\nDT threshold summary saved to: {output_path}")

        return df

    ###
    
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
