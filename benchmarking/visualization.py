"""
Visualization tools for comparing numerical and analytical solutions.

This module provides plotting functions to visualize test results, including
comparisons between numerical and analytical solutions, error distributions,
and time evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any, Optional
import os


def plot_final_comparison(numerical: np.ndarray, 
                         analytical: np.ndarray,
                         coordinates: Union[np.ndarray, Tuple[np.ndarray, ...]],
                         schema_name: str,
                         scenario_name: str,
                         output_path: Optional[Union[str, Path]] = None,
                         title: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot side-by-side comparison of numerical and analytical solutions.
    
    Creates 1D line plots or 2D heatmaps depending on dimensionality.
    
    Parameters
    ----------
    numerical : np.ndarray
        Numerical solution.
    analytical : np.ndarray
        Analytical solution.
    coordinates : np.ndarray or tuple of np.ndarray
        Coordinate arrays (1D) or meshgrids (2D/3D).
    schema_name : str
        Name of the numerical schema.
    scenario_name : str
        Name of the test scenario.
    output_path : str or Path, optional
        If provided, save figure to this path.
    title : str, optional
        Custom title for the figure.
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    ndim = numerical.ndim
    
    if ndim == 1:
        # 1D line plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Handle coordinates - might be list/tuple containing single array
        if isinstance(coordinates, (list, tuple)):
            x = coordinates[0]
        else:
            x = coordinates
        
        # Ensure x is 1D
        if x.ndim > 1:
            x = x.flatten()
        
        axes[0].plot(x, numerical, 'b-', label='Numerical', linewidth=2)
        axes[0].plot(x, analytical, 'r--', label='Analytical', linewidth=2)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('Concentration')
        axes[0].set_title('Solution Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Error plot
        error = numerical - analytical
        axes[1].plot(x, error, 'k-', linewidth=2)
        axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('Error')
        axes[1].set_title('Pointwise Error')
        axes[1].grid(True, alpha=0.3)
        
    elif ndim == 2:
        # 2D heatmap
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        x, y = coordinates
        
        # Determine common color scale
        vmin = min(numerical.min(), analytical.min())
        vmax = max(numerical.max(), analytical.max())
        
        im0 = axes[0].contourf(x, y, numerical, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_title('Numerical')
        axes[0].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].contourf(x, y, analytical, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_title('Analytical')
        axes[1].set_aspect('equal')
        plt.colorbar(im1, ax=axes[1])
        
        # Error
        error = numerical - analytical
        im2 = axes[2].contourf(x, y, error, levels=20, cmap='RdBu_r')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        axes[2].set_title('Error (Numerical - Analytical)')
        axes[2].set_aspect('equal')
        plt.colorbar(im2, ax=axes[2])
        
    elif ndim == 3:
        # 3D: show center slice
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        x, y, z = coordinates
        center_idx = numerical.shape[2] // 2
        
        # Get center slice
        num_slice = numerical[:, :, center_idx]
        ana_slice = analytical[:, :, center_idx]
        
        vmin = min(num_slice.min(), ana_slice.min())
        vmax = max(num_slice.max(), ana_slice.max())
        
        # XY plane
        im00 = axes[0, 0].contourf(x[:, :, center_idx], y[:, :, center_idx], num_slice, 
                                    levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        axes[0, 0].set_title('Numerical (XY center slice)')
        axes[0, 0].set_aspect('equal')
        plt.colorbar(im00, ax=axes[0, 0])
        
        im01 = axes[0, 1].contourf(x[:, :, center_idx], y[:, :, center_idx], ana_slice, 
                                    levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        axes[0, 1].set_title('Analytical (XY center slice)')
        axes[0, 1].set_aspect('equal')
        plt.colorbar(im01, ax=axes[0, 1])
        
        error_slice = num_slice - ana_slice
        im02 = axes[0, 2].contourf(x[:, :, center_idx], y[:, :, center_idx], error_slice, 
                                    levels=20, cmap='RdBu_r')
        axes[0, 2].set_xlabel('x')
        axes[0, 2].set_ylabel('y')
        axes[0, 2].set_title('Error (XY center slice)')
        axes[0, 2].set_aspect('equal')
        plt.colorbar(im02, ax=axes[0, 2])
        
        # Show 1D profile through center
        center_y = numerical.shape[1] // 2
        num_profile = numerical[:, center_y, center_idx]
        ana_profile = analytical[:, center_y, center_idx]
        x_profile = x[:, center_y, center_idx]
        
        axes[1, 0].plot(x_profile, num_profile, 'b-', label='Numerical', linewidth=2)
        axes[1, 0].plot(x_profile, ana_profile, 'r--', label='Analytical', linewidth=2)
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('Concentration')
        axes[1, 0].set_title('1D Profile (center)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(x_profile, num_profile - ana_profile, 'k-', linewidth=2)
        axes[1, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].set_title('Profile Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Hide unused subplot
        axes[1, 2].axis('off')
    
    else:
        raise ValueError(f"Unsupported dimensionality: {ndim}")
    
    # Set overall title
    if title is None:
        title = f'{schema_name} - {scenario_name}'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def plot_error_distribution(numerical: np.ndarray,
                           analytical: np.ndarray,
                           coordinates: Union[np.ndarray, Tuple[np.ndarray, ...]],
                           schema_name: str,
                           scenario_name: str,
                           output_path: Optional[Union[str, Path]] = None,
                           relative: bool = False) -> Optional[plt.Figure]:
    """
    Plot error distribution heatmap/histogram.
    
    Parameters
    ----------
    numerical : np.ndarray
        Numerical solution.
    analytical : np.ndarray
        Analytical solution.
    coordinates : np.ndarray or tuple of np.ndarray
        Coordinate arrays.
    schema_name : str
        Name of the numerical schema.
    scenario_name : str
        Name of the test scenario.
    output_path : str or Path, optional
        If provided, save figure to this path.
    relative : bool, optional
        If True, plot relative error instead of absolute error.
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    ndim = numerical.ndim
    
    # Compute error
    if relative:
        epsilon = 1e-10
        error = np.abs(numerical - analytical) / (np.abs(analytical) + epsilon)
        error_label = 'Relative Error'
    else:
        error = np.abs(numerical - analytical)
        error_label = 'Absolute Error'
    
    if ndim == 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Handle coordinates - might be list/tuple containing single array
        if isinstance(coordinates, (list, tuple)):
            x = coordinates[0]
        else:
            x = coordinates
        
        # Ensure x is 1D
        if x.ndim > 1:
            x = x.flatten()
        
        axes[0].plot(x, error, 'k-', linewidth=2)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel(error_label)
        axes[0].set_title('Error vs Position')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(error.flatten(), bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel(error_label)
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Error Distribution')
        axes[1].grid(True, alpha=0.3, axis='y')
        
    elif ndim == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        x, y = coordinates
        
        im = axes[0].contourf(x, y, error, levels=20, cmap='hot')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_title(f'{error_label} Heatmap')
        axes[0].set_aspect('equal')
        plt.colorbar(im, ax=axes[0])
        
        axes[1].hist(error.flatten(), bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel(error_label)
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Error Distribution')
        axes[1].grid(True, alpha=0.3, axis='y')
        
    elif ndim == 3:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        x, y, z = coordinates
        center_idx = error.shape[2] // 2
        error_slice = error[:, :, center_idx]
        
        im = axes[0].contourf(x[:, :, center_idx], y[:, :, center_idx], error_slice, 
                             levels=20, cmap='hot')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_title(f'{error_label} Heatmap (center slice)')
        axes[0].set_aspect('equal')
        plt.colorbar(im, ax=axes[0])
        
        axes[1].hist(error.flatten(), bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel(error_label)
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Error Distribution')
        axes[1].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'{schema_name} - {scenario_name}: Error Analysis', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def plot_time_evolution(history: List[np.ndarray],
                       times: List[float],
                       golden_solution,
                       coordinates: Union[np.ndarray, Tuple[np.ndarray, ...]],
                       schema_name: str,
                       scenario_name: str,
                       output_path: Optional[Union[str, Path]] = None,
                       num_snapshots: int = 6) -> Optional[plt.Figure]:
    """
    Plot time evolution as grid of snapshots.
    
    Parameters
    ----------
    history : list of np.ndarray
        Time series of solution states.
    times : list of float
        Corresponding time values.
    golden_solution : GoldenSolution or callable
        Analytical solution object or evaluation function.
    coordinates : np.ndarray or tuple of np.ndarray
        Coordinate arrays.
    schema_name : str
        Name of the numerical schema.
    scenario_name : str
        Name of the test scenario.
    output_path : str or Path, optional
        If provided, save figure to this path.
    num_snapshots : int, optional
        Number of time snapshots to display (default 6).
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    ndim = history[0].ndim
    
    # Select time indices to display
    n_times = len(times)
    if n_times <= num_snapshots:
        indices = list(range(n_times))
    else:
        indices = [int(i * (n_times - 1) / (num_snapshots - 1)) for i in range(num_snapshots)]
    
    if ndim == 1:
        # 1D: stack plots vertically
        n_panels = len(indices)
        fig, axes = plt.subplots(n_panels, 1, figsize=(10, 2.5 * n_panels))
        if n_panels == 1:
            axes = [axes]
        
        # Handle coordinates - might be list/tuple containing single array
        if isinstance(coordinates, (list, tuple)):
            x = coordinates[0]
        else:
            x = coordinates
        
        # Ensure x is 1D
        if x.ndim > 1:
            x = x.flatten()
        
        for i, idx in enumerate(indices):
            t = times[idx]
            numerical = history[idx]
            
            # Evaluate analytical solution
            if hasattr(golden_solution, 'evaluate'):
                analytical = golden_solution.evaluate(coordinates, t)
            else:
                analytical = golden_solution(*coordinates, t)
            
            axes[i].plot(x, numerical, 'b-', label='Numerical', linewidth=2)
            axes[i].plot(x, analytical, 'r--', label='Analytical', linewidth=2)
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('Concentration')
            axes[i].set_title(f't = {t:.4f}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    elif ndim == 2:
        # 2D: grid of heatmaps
        n_panels = len(indices)
        ncols = min(3, n_panels)
        nrows = (n_panels + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        elif ncols == 1:
            axes = [[ax] for ax in axes]
        
        x, y = coordinates
        
        # Determine color scale from all snapshots
        all_values = []
        for idx in indices:
            all_values.append(history[idx].min())
            all_values.append(history[idx].max())
        vmin, vmax = min(all_values), max(all_values)

        # vmin, vmax = 0.0, 1.5
        
        for i, idx in enumerate(indices):
            row = i // ncols
            col = i % ncols
            ax = axes[row][col]
            
            t = times[idx]
            numerical = history[idx]
            
            # --- ADD CHECKS HERE ---
            nans = np.isnan(numerical).sum()
            infs = np.isinf(numerical).sum()
            if nans > 0 or infs > 0:
                print(f"Alert at t = {t:.4f}: Found {nans} NaNs and {infs} Infs")
            # -----------------------

            # countourf faces problems trying to plot results
            # white spots appear and colorbars do not obey vmin and vmax
            # im = ax.contourf(x, y, numerical, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
            im = ax.pcolormesh(x, y, numerical, vmin=vmin, vmax=vmax, shading='auto')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f't = {t:.4f}')
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax)
        
        # Hide unused subplots
        for i in range(len(indices), nrows * ncols):
            row = i // ncols
            col = i % ncols
            axes[row][col].axis('off')
    
    elif ndim == 3:
        # 3D: show center slices
        n_panels = len(indices)
        ncols = min(3, n_panels)
        nrows = (n_panels + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        elif ncols == 1:
            axes = [[ax] for ax in axes]
        
        x, y, z = coordinates
        center_idx = history[0].shape[2] // 2
        
        # Determine color scale
        all_values = []
        for idx in indices:
            slice_data = history[idx][:, :, center_idx]
            all_values.append(slice_data.min())
            all_values.append(slice_data.max())
        vmin, vmax = min(all_values), max(all_values)
        
        for i, idx in enumerate(indices):
            row = i // ncols
            col = i % ncols
            ax = axes[row][col]
            
            t = times[idx]
            numerical_slice = history[idx][:, :, center_idx]
            
            im = ax.contourf(x[:, :, center_idx], y[:, :, center_idx], numerical_slice, 
                           levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f't = {t:.4f} (center slice)')
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax)
        
        # Hide unused subplots
        for i in range(len(indices), nrows * ncols):
            row = i // ncols
            col = i % ncols
            axes[row][col].axis('off')
    
    fig.suptitle(f'{schema_name} - {scenario_name}: Time Evolution', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def plot_error_vs_time(times: List[float],
                      errors: Dict[str, List[float]],
                      schema_name: str,
                      scenario_name: str,
                      output_path: Optional[Union[str, Path]] = None) -> Optional[plt.Figure]:
    """
    Plot error metrics vs time.
    
    Parameters
    ----------
    times : list of float
        Time values.
    errors : dict
        Dictionary mapping error type to list of values over time.
        Keys should be like 'l2_relative', 'linf_relative', etc.
    schema_name : str
        Name of the numerical schema.
    scenario_name : str
        Name of the test scenario.
    output_path : str or Path, optional
        If provided, save figure to this path.
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # L2 error
    if 'l2_relative' in errors:
        axes[0].semilogy(times, errors['l2_relative'], 'b-', linewidth=2, label='L2 relative')
    if 'l2_absolute' in errors:
        axes[0].semilogy(times, errors['l2_absolute'], 'b--', linewidth=2, label='L2 absolute')
    
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('L2 Error')
    axes[0].set_title('L2 Error vs Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # L∞ error
    if 'linf_relative' in errors:
        axes[1].semilogy(times, errors['linf_relative'], 'r-', linewidth=2, label='L∞ relative')
    if 'linf_absolute' in errors:
        axes[1].semilogy(times, errors['linf_absolute'], 'r--', linewidth=2, label='L∞ absolute')
    
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('L∞ Error')
    axes[1].set_title('L∞ Error vs Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(f'{schema_name} - {scenario_name}: Error Evolution', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def plot_convergence_analysis(refinements: List[float],
                              errors: Dict[str, List[float]],
                              convergence_rates: Dict[str, float],
                              schema_name: str,
                              scenario_name: str,
                              output_path: Optional[Union[str, Path]] = None,
                              refinement_type: str = 'dt') -> Optional[plt.Figure]:
    """
    Plot convergence analysis (error vs refinement parameter).
    
    Parameters
    ----------
    refinements : list of float
        Refinement parameter values (e.g., dt or dx values).
    errors : dict
        Dictionary mapping error type to list of error values.
    convergence_rates : dict
        Dictionary mapping error type to computed convergence rate.
    schema_name : str
        Name of the numerical schema.
    scenario_name : str
        Name of the test scenario.
    output_path : str or Path, optional
        If provided, save figure to this path.
    refinement_type : str, optional
        Type of refinement ('dt' or 'dx'), for axis labeling.
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    for i, (error_type, error_values) in enumerate(errors.items()):
        rate = convergence_rates.get(error_type, None)
        marker = markers[i % len(markers)]
        
        if rate is not None:
            label = f'{error_type} (rate={rate:.2f})'
        else:
            label = error_type
        
        ax.loglog(refinements, error_values, marker=marker, linewidth=2, 
                 markersize=8, label=label)
    
    # Add reference lines for common orders
    if len(refinements) >= 2:
        h_ref = np.array([min(refinements), max(refinements)])
        for order in [1, 2]:
            ref_errors = (h_ref / h_ref[0])**order * errors[list(errors.keys())[0]][0]
            ax.loglog(h_ref, ref_errors, 'k--', alpha=0.3, linewidth=1)
            ax.text(h_ref[-1], ref_errors[-1], f'O({refinement_type}^{order})', 
                   fontsize=9, alpha=0.5)
    
    ax.set_xlabel(refinement_type)
    ax.set_ylabel('Error')
    ax.set_title(f'{schema_name} - {scenario_name}: Convergence Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig

def plot_method_comparison(scenario_name, schema_dict, output_path):
        
        # Grab analytical solution from the first result (it's the same for all)
        first_res = list(schema_dict.values())[0]
        analytical = first_res['analytical_final']
        ndim = analytical.ndim
        
        # Reconstruct x-axis (assuming generic domain [0, 1] if coords aren't saved)
        # If you saved 'coordinates' in result, use that instead.

        if ndim == 1:
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 8), sharex=True)

            nx = len(analytical)
            x = np.linspace(0, 1, nx) 
            
            # Plot Analytical
            ax1.plot(x, analytical, 'k--', label='Analytical', alpha=0.6)
            
            for schema_name, res in schema_dict.items():
                u_num = res['final_state']
                error = np.abs(u_num - analytical)
                
                ax1.plot(x, u_num, label=schema_name)
                ax2.semilogy(x, error, label=schema_name) # Log scale for error
                
            ax1.legend()
            ax1.set_ylabel("Concentration")
            ax1.set_xlabel("x")
            ax2.set_ylabel("Pointwise Error (Log)")
            ax2.legend()
            ax2.set_xlabel("x")

            plt.tight_layout()


        elif ndim == 2:
            # We plot n_schemas + 1 subplots (Analytical + each schema's error)
            n_plots = len(schema_dict) + 1
            fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), constrained_layout=True)
            
            if n_plots == 1: axes = [axes] # Handle edge case
            
            # 1. Plot Analytical Solution (First panel)
            im0 = axes[0].imshow(analytical.T, origin='lower', cmap='viridis')
            axes[0].set_title("Analytical Solution")
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("y")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            # Determine global max error for consistent color scaling across methods
            global_max_err = 0
            for res in schema_dict.values():
                err = np.abs(res['final_state'] - analytical)
                global_max_err = max(global_max_err, np.max(err))
                global_min_err = 1e-16
                if np.any(err > 0):
                    global_min_err = min(global_min_err, np.min(err[err > 0]))
                
            # 2. Plot Error Heatmaps for each method
            for ax, (schema_name, res) in zip(axes[1:], schema_dict.items()):
                u_num = res['final_state']
                error = np.abs(u_num - analytical)
                
                # Use 'inferno' for error to highlight hotspots
                # im = ax.imshow(error, origin='lower', cmap='inferno', 
                #             norm=colors.LogNorm(vmin=max(global_min_err, 1e-16), 
                #                                vmax=global_max_err))
                
                im = ax.imshow(error.T, origin='lower', cmap='inferno', 
                               vmin = 0, vmax=global_max_err)
                
                ax.set_title(f"Error: {schema_name}\nMax: {np.max(error):.2e}")
                ax.set_xlabel("x")
                ax.set_yticks([]) # Hide Y ticks for inner plots to save space
                
                # Individual colorbar or shared? Individual is safer if scales differ wildly,
                # but shared (using vmin/vmax) allows visual comparison.
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        else:
            print(f"Skipping plot: Dimensionality {ndim} not supported for comparison.")
            return None
        
        if fig is not None:
            fig.suptitle(f"Final state comparison: {scenario_name}", fontsize=14, fontweight='bold')
            plt.savefig(output_path)
            plt.close() 

        return fig

def plot_scenario(scenario: dict):
    """
    Visualize the scenario setup: domain, initial condition, agent/bulk positions.
    Only uses scenario dict (no simulation). Handles 1D, 2D, 3D.
    """
    import matplotlib.patches as mpatches

    # --- Extract domain and grid ---
    domain_size = scenario['domain_size']
    grid_points = scenario['grid_points']
    ic_spec = scenario['initial_condition']
    agents = scenario.get('agents', None)
    bulk = scenario.get('bulk', None)
    name = scenario.get('name', '')
    desc = scenario.get('description', '')

    # Normalize domain/grid to tuple
    if isinstance(domain_size, (int, float)):
        domain_size = (float(domain_size),)
    if isinstance(grid_points, (int, float)):
        grid_points = (int(grid_points),)
    ndim = len(domain_size)

    # Build grid
    axes = []
    for L, N in zip(domain_size, grid_points):
        axes.append(np.linspace(0, L, int(N)))
    mesh = np.meshgrid(*axes, indexing='ij')

    # Build initial condition (if possible)
    ic_func = None
    if isinstance(ic_spec, dict):
        ic_type = ic_spec.get('type', None)
        if ic_type == 'uniform':
            ic_func = lambda *args: np.full_like(args[0], ic_spec.get('value', 0.0), dtype=float)
        elif ic_type == 'gaussian':
            center = ic_spec.get('center', tuple(0.5 * np.array(domain_size)))
            amp = ic_spec.get('amplitude', 1.0)
            width = ic_spec.get('width', 0.1)
            def ic_func(*args):
                r2 = sum((np.asarray(a) - c) ** 2 for a, c in zip(args, np.atleast_1d(center)))
                return amp * np.exp(-r2 / (2 * width ** 2))
        elif ic_type == 'step_function':
            pos = ic_spec.get('position', 0.5)
            vL = ic_spec.get('value_left', 1.0)
            vR = ic_spec.get('value_right', 0.0)
            axis = ic_spec.get('axis', 0)
            def ic_func(*args):
                arr = np.where(args[axis] < pos, vL, vR)
                return arr
        elif ic_type == 'sine':
            w = ic_spec.get('wavenumber', 1.0)
            amp = ic_spec.get('amplitude', 1.0)
            def ic_func(*args):
                return amp * np.sin(w * args[0] * np.pi)
        else:
            ic_func = None
    elif callable(ic_spec):
        ic_func = ic_spec
    else:
        ic_func = None

    # Evaluate initial condition
    ic_vals = None
    if ic_func is not None:
        try:
            ic_vals = ic_func(*mesh)
        except Exception:
            ic_vals = None

    # --- Plotting ---
    # fig = plt.figure(figsize=(7 if ndim==1 else 8, 5 if ndim==1 else 7))
    fig = plt.figure(figsize=(5,5))
    border_lw = 1.0  # Tunable border linewidth
    if ndim == 1:
        ax = fig.add_subplot(1, 1, 1)
        x = axes[0]
        if ic_vals is not None:
            ax.plot(x, ic_vals, label='Initial condition', color='C0')
        # Agents
        if agents:
            for ag in agents:
                pos = ag.get('position', ag.get('center', None))
                if pos is not None:
                    xpos = pos[0] if isinstance(pos, (list, tuple)) else pos
                    ax.axvline(xpos, color='C3', linestyle='--', label='Agent', linewidth=border_lw)
                    # Mark agent with a black border (vertical line)
                    ax.axvline(xpos, color='k', linestyle='-', linewidth=border_lw/2)
        # Bulk regions
        if bulk and 'regions' in bulk:
            for reg in bulk['regions']:
                if reg.get('type') == 'sphere':
                    c = reg['center'][0] if isinstance(reg['center'], (list, tuple)) else reg['center']
                    r = reg['radius']
                    ax.axvspan(c - r, c + r, color='C2', alpha=0.2, label='Bulk region')
                    # Draw black border for bulk region
                    ax.plot([c - r, c - r], ax.get_ylim(), color='k', linewidth=border_lw/2, linestyle='-')
                    ax.plot([c + r, c + r], ax.get_ylim(), color='k', linewidth=border_lw/2, linestyle='-')
        ax.set_xlabel('x')
        ax.set_ylabel('Concentration')
        ax.set_title(f"{desc}")
        plt.suptitle(f"{name}")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    elif ndim == 2:
        ax = fig.add_subplot(1, 1, 1)
        x, y = mesh
        if ic_vals is not None:
            im = ax.pcolormesh(x, y, ic_vals, shading='auto', cmap='viridis')
            plt.colorbar(im, ax=ax, label='Initial concentration')
        # Agents
        if agents:
            for ag in agents:
                pos = ag.get('position', ag.get('center', None))
                if pos is not None:
                    # Agent marker with black edge
                    ax.plot(pos[0], pos[1], 'ro', markersize=7, label='Agent', markeredgecolor='k', markeredgewidth=border_lw)
        # Bulk regions
        if bulk and 'regions' in bulk:
            for reg in bulk['regions']:
                if reg.get('type') == 'sphere':
                    circ = mpatches.Circle(reg['center'], reg['radius'], color='C2', alpha=1, label='Bulk region', ec='k', lw=border_lw)
                    ax.add_patch(circ)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f"{desc}")
        plt.suptitle(f"{name}")
        # Avoid duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')
        ax.set_aspect('equal')
    elif ndim == 3:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        x, y, z = mesh
        # Plot a scatter of initial condition (center slice)
        if ic_vals is not None:
            idx = ic_vals.shape[2] // 2
            xs = x[:, :, idx].flatten()
            ys = y[:, :, idx].flatten()
            zs = z[:, :, idx].flatten()
            vals = ic_vals[:, :, idx].flatten()
            p = ax.scatter(xs, ys, zs, c=vals, cmap='viridis', marker='o', s=10)
            fig.colorbar(p, ax=ax, label='Initial concentration')
        # Agents
        if agents:
            for ag in agents:
                pos = ag.get('position', ag.get('center', None))
                if pos is not None:
                    ax.scatter(*pos, color='r', s=40, label='Agent', edgecolor='k', linewidth=border_lw)
        # Bulk regions (spheres only)
        if bulk and 'regions' in bulk:
            for reg in bulk['regions']:
                if reg.get('type') == 'sphere':
                    # Draw sphere wireframe (approximate)
                    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                    cx, cy, cz = reg['center']
                    r = reg['radius']
                    xs = cx + r * np.cos(u) * np.sin(v)
                    ys = cy + r * np.sin(u) * np.sin(v)
                    zs = cz + r * np.cos(v)
                    ax.plot_wireframe(xs, ys, zs, color='C2', alpha=0.2, linewidth=border_lw)
                    # Draw black border for sphere (approximate, just one circle)
                    ax.plot(xs[0], ys[0], zs[0], color='k', linewidth=border_lw/2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(f"{desc}")
        plt.suptitle(f"{name}")    
    plt.tight_layout()
    return fig