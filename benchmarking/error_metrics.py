"""
Error metrics for comparing numerical solutions to analytical (golden) solutions.

This module provides various error measures to quantify the accuracy of numerical
diffusion schema implementations.
"""

import numpy as np
from typing import Dict, List, Tuple, Union


def compute_l2_error(numerical: np.ndarray, analytical: np.ndarray, 
                     dx: Union[float, Tuple[float, ...]] = None) -> Dict[str, float]:
    """
    Compute relative L2 error between numerical and analytical solutions.
    
    The L2 error is computed as:
        ||u_num - u_ana||_2 / ||u_ana||_2
    
    where ||·||_2 is the discrete L2 norm:
        ||u||_2 = sqrt(sum(u^2) * dx^d)
    
    Parameters
    ----------
    numerical : np.ndarray
        Numerical solution.
    analytical : np.ndarray
        Analytical (golden) solution.
    dx : float or tuple of float, optional
        Grid spacing(s). If provided, computes weighted L2 norm.
        If None, computes unweighted norm (equivalent to dx=1).
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'l2_relative': Relative L2 error
        - 'l2_absolute': Absolute L2 norm of difference
        - 'l2_analytical': L2 norm of analytical solution
    """
    if numerical.shape != analytical.shape:
        raise ValueError(f"Shape mismatch: numerical {numerical.shape} vs analytical {analytical.shape}")
    
    difference = numerical - analytical
    
    # Compute grid volume element
    if dx is None:
        dV = 1.0
    elif isinstance(dx, (list, tuple)):
        dV = np.prod(dx)
    else:
        dV = dx**numerical.ndim
    
    # Compute L2 norms
    l2_diff = np.sqrt(np.sum(difference**2) * dV)
    l2_analytical = np.sqrt(np.sum(analytical**2) * dV)
    
    # Relative error (avoid division by zero)
    if l2_analytical > 1e-15:
        l2_relative = l2_diff / l2_analytical
    else:
        l2_relative = l2_diff
    
    return {
        'l2_relative': l2_relative,
        'l2_absolute': l2_diff,
        'l2_analytical': l2_analytical
    }


def compute_linf_error(numerical: np.ndarray, analytical: np.ndarray) -> Dict[str, float]:
    """
    Compute L-infinity (maximum pointwise) error.
    
    The L∞ error is:
        ||u_num - u_ana||_∞ = max|u_num - u_ana|
    
    Parameters
    ----------
    numerical : np.ndarray
        Numerical solution.
    analytical : np.ndarray
        Analytical (golden) solution.
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'linf_relative': Relative L∞ error (max|diff|/max|analytical|)
        - 'linf_absolute': Absolute L∞ error (max|diff|)
        - 'linf_analytical': L∞ norm of analytical solution
        - 'max_error_location': Index of maximum error location
    """
    if numerical.shape != analytical.shape:
        raise ValueError(f"Shape mismatch: numerical {numerical.shape} vs analytical {analytical.shape}")
    
    difference = np.abs(numerical - analytical)
    
    linf_diff = np.max(difference)
    linf_analytical = np.max(np.abs(analytical))
    max_error_idx = np.unravel_index(np.argmax(difference), difference.shape)
    
    # Relative error (avoid division by zero)
    if linf_analytical > 1e-15:
        linf_relative = linf_diff / linf_analytical
    else:
        linf_relative = linf_diff
    
    return {
        'linf_relative': linf_relative,
        'linf_absolute': linf_diff,
        'linf_analytical': linf_analytical,
        'max_error_location': max_error_idx
    }


def compute_mass_conservation_error(initial_mass: float, current_mass: float) -> Dict[str, float]:
    """
    Compute mass conservation error.
    
    For systems with zero-flux boundary conditions, total mass should be conserved.
    This metric quantifies the relative drift in total mass.
    
    Parameters
    ----------
    initial_mass : float
        Initial total mass (sum of concentration).
    current_mass : float
        Current total mass.
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'mass_conservation_relative': Relative mass change |(M_current - M_initial)/M_initial|
        - 'mass_conservation_absolute': Absolute mass change |M_current - M_initial|
        - 'initial_mass': Initial mass
        - 'current_mass': Current mass
    """
    mass_diff = abs(current_mass - initial_mass)
    
    if abs(initial_mass) > 1e-15:
        mass_relative = mass_diff / abs(initial_mass)
    else:
        mass_relative = mass_diff
    
    return {
        'mass_conservation_relative': mass_relative,
        'mass_conservation_absolute': mass_diff,
        'initial_mass': initial_mass,
        'current_mass': current_mass
    }


def compute_convergence_rate(errors: List[float], refinement_factors: List[float]) -> Dict[str, float]:
    """
    Compute convergence rate from errors at different resolutions.
    
    For a numerical method of order p, the error should scale as:
        error ∝ h^p
    
    where h is the grid spacing (or time step). Taking logarithms:
        log(error) ≈ p·log(h) + c
    
    This function fits a line to log(error) vs log(h) to estimate p.
    
    Parameters
    ----------
    errors : list of float
        Errors at different resolutions (must have at least 2 values).
    refinement_factors : list of float
        Corresponding refinement factors (h values). Should be same length as errors.
        For example, [0.1, 0.05, 0.025] for spatial refinement.
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'convergence_rate': Estimated order of convergence (slope of log-log plot)
        - 'intercept': Intercept of log-log fit
        - 'r_squared': R² goodness-of-fit measure
        - 'fitted_errors': Error values predicted by the fit
    """
    if len(errors) != len(refinement_factors):
        raise ValueError("errors and refinement_factors must have same length")
    
    if len(errors) < 2:
        raise ValueError("Need at least 2 data points to compute convergence rate")
    
    # Filter out any zero or negative errors (would cause log issues)
    valid_indices = [i for i, e in enumerate(errors) if e > 0]
    if len(valid_indices) < 2:
        raise ValueError("Need at least 2 positive errors to compute convergence rate")
    
    errors_valid = [errors[i] for i in valid_indices]
    h_valid = [refinement_factors[i] for i in valid_indices]
    
    # Compute log-log fit
    log_h = np.log(h_valid)
    log_errors = np.log(errors_valid)
    
    # Linear regression: log(error) = slope * log(h) + intercept
    coeffs = np.polyfit(log_h, log_errors, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    # Compute R²
    log_errors_fit = slope * log_h + intercept
    ss_res = np.sum((log_errors - log_errors_fit)**2)
    ss_tot = np.sum((log_errors - np.mean(log_errors))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Compute fitted errors in original scale
    fitted_errors = [np.exp(intercept) * h**slope for h in refinement_factors]
    
    return {
        'convergence_rate': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'fitted_errors': fitted_errors
    }


def compute_all_errors(numerical: np.ndarray, analytical: np.ndarray, 
                       dx: Union[float, Tuple[float, ...]] = None,
                       initial_mass: float = None) -> Dict[str, any]:
    """
    Compute all available error metrics.
    
    Parameters
    ----------
    numerical : np.ndarray
        Numerical solution.
    analytical : np.ndarray
        Analytical (golden) solution.
    dx : float or tuple of float, optional
        Grid spacing(s) for L2 norm computation.
    initial_mass : float, optional
        Initial mass for conservation check. If provided, also computes mass conservation error.
        
    Returns
    -------
    dict
        Combined dictionary with all error metrics.
    """
    errors = {}
    
    # L2 error
    errors.update(compute_l2_error(numerical, analytical, dx))
    
    # L∞ error
    errors.update(compute_linf_error(numerical, analytical))
    
    # Mass conservation (if initial mass provided)
    if initial_mass is not None:
        current_mass = np.sum(numerical)
        if dx is not None:
            if isinstance(dx, (list, tuple)):
                dV = np.prod(dx)
            else:
                dV = dx**numerical.ndim
            current_mass *= dV
        errors.update(compute_mass_conservation_error(initial_mass, current_mass))
    
    return errors


def compute_pointwise_error(numerical: np.ndarray, analytical: np.ndarray) -> np.ndarray:
    """
    Compute pointwise absolute error for visualization.
    
    Parameters
    ----------
    numerical : np.ndarray
        Numerical solution.
    analytical : np.ndarray
        Analytical (golden) solution.
        
    Returns
    -------
    np.ndarray
        Pointwise absolute error |u_num - u_ana|.
    """
    return np.abs(numerical - analytical)


def compute_relative_pointwise_error(numerical: np.ndarray, analytical: np.ndarray, 
                                     epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute pointwise relative error for visualization.
    
    Parameters
    ----------
    numerical : np.ndarray
        Numerical solution.
    analytical : np.ndarray
        Analytical (golden) solution.
    epsilon : float
        Small value to avoid division by zero.
        
    Returns
    -------
    np.ndarray
        Pointwise relative error |u_num - u_ana| / (|u_ana| + epsilon).
    """
    return np.abs(numerical - analytical) / (np.abs(analytical) + epsilon)
