"""
Initial condition helper functions for diffusion problems.

This module provides common initial condition patterns for diffusion
simulations in 1D, 2D, and 3D.
"""

import numpy as np
from typing import Tuple, Union


def gaussian(
    center: Union[float, Tuple[float, ...]],
    amplitude: float = 1.0,
    width: float = 1.0
):
    """
    Create a Gaussian initial condition.
    
    Returns a function that evaluates:
        u(x) = amplitude * exp(-||x - center||² / (2 * width²))
    
    Parameters
    ----------
    center : Union[float, Tuple[float, ...]]
        Center position of the Gaussian. Scalar for 1D, tuple for higher dimensions.
    amplitude : float, optional
        Maximum amplitude at the center. Default is 1.0.
    width : float, optional
        Standard deviation of the Gaussian. Default is 1.0.
        
    Returns
    -------
    Callable
        Function that takes coordinate arrays and returns the Gaussian distribution.
        
    Examples
    --------
    >>> # 1D Gaussian centered at x=0.5
    >>> ic = gaussian(center=0.5, amplitude=10.0, width=0.1)
    >>> schema.set_initial_condition(ic)
    
    >>> # 2D Gaussian centered at (0.5, 0.5)
    >>> ic = gaussian(center=(0.5, 0.5), amplitude=1.0, width=0.2)
    >>> schema.set_initial_condition(ic)
    """
    if isinstance(center, (int, float)):
        center = (float(center),)
    else:
        center = tuple(center)
    
    def _gaussian(*coords):
        # Compute squared distance from center
        r_squared = 0
        for coord, c in zip(coords, center):
            r_squared += (coord - c) ** 2
        
        return amplitude * np.exp(-r_squared / (2 * width**2))
    
    return _gaussian


def uniform(value: float = 1.0):
    """
    Create a uniform initial condition.
    
    Parameters
    ----------
    value : float, optional
        Constant value everywhere. Default is 1.0.
        
    Returns
    -------
    Callable
        Function that returns constant value.
        
    Examples
    --------
    >>> ic = uniform(value=5.0)
    >>> schema.set_initial_condition(ic)
    """
    def _uniform(*coords):
        return np.full_like(coords[0], value)
    
    return _uniform


def step_function(
    position: float,
    value_left: float = 1.0,
    value_right: float = 0.0,
    axis: int = 0
):
    """
    Create a step function initial condition.
    
    Creates a discontinuous step at the specified position along a given axis.
    
    Parameters
    ----------
    position : float
        Position where the step occurs.
    value_left : float, optional
        Value for coordinates < position. Default is 1.0.
    value_right : float, optional
        Value for coordinates >= position. Default is 0.0.
    axis : int, optional
        Axis along which the step occurs (0=x, 1=y, 2=z). Default is 0.
        
    Returns
    -------
    Callable
        Function that returns step function values.
        
    Examples
    --------
    >>> # Step at x=0.5: u=1 for x<0.5, u=0 for x>=0.5
    >>> ic = step_function(position=0.5, value_left=1.0, value_right=0.0)
    >>> schema.set_initial_condition(ic)
    """
    def _step(*coords):
        coord = coords[axis]
        result = np.where(coord < position, value_left, value_right)
        return result
    
    return _step


def checkerboard(
    spacing: float = 1.0,
    value_on: float = 1.0,
    value_off: float = 0.0
):
    """
    Create a checkerboard pattern (2D only).
    
    Creates alternating squares of two values.
    
    Parameters
    ----------
    spacing : float, optional
        Size of each square in the checkerboard. Default is 1.0.
    value_on : float, optional
        Value for "on" squares. Default is 1.0.
    value_off : float, optional
        Value for "off" squares. Default is 0.0.
        
    Returns
    -------
    Callable
        Function that returns checkerboard pattern.
        
    Examples
    --------
    >>> # Checkerboard with 0.5x0.5 squares
    >>> ic = checkerboard(spacing=0.5, value_on=10.0, value_off=0.0)
    >>> schema.set_initial_condition(ic)
    """
    def _checkerboard(*coords):
        if len(coords) < 2:
            raise ValueError("Checkerboard pattern requires at least 2D coordinates")
        
        x, y = coords[0], coords[1]
        
        # Determine checkerboard pattern
        x_idx = (x / spacing).astype(int)
        y_idx = (y / spacing).astype(int)
        
        # XOR to create alternating pattern
        pattern = (x_idx + y_idx) % 2
        
        result = np.where(pattern == 0, value_on, value_off)
        return result
    
    return _checkerboard


def sphere(
    center: Tuple[float, float, float],
    radius: float,
    value_inside: float = 1.0,
    value_outside: float = 0.0
):
    """
    Create a spherical region (3D).
    
    Creates a sphere with specified value inside and another value outside.
    Can also be used for circles in 2D.
    
    Parameters
    ----------
    center : Tuple[float, ...]
        Center coordinates of the sphere (x, y, z) for 3D or (x, y) for 2D.
    radius : float
        Radius of the sphere/circle.
    value_inside : float, optional
        Value inside the sphere. Default is 1.0.
    value_outside : float, optional
        Value outside the sphere. Default is 0.0.
        
    Returns
    -------
    Callable
        Function that returns spherical/circular pattern.
        
    Examples
    --------
    >>> # Sphere centered at (0.5, 0.5, 0.5) with radius 0.2
    >>> ic = sphere(center=(0.5, 0.5, 0.5), radius=0.2, value_inside=10.0)
    >>> schema.set_initial_condition(ic)
    
    >>> # Circle in 2D
    >>> ic = sphere(center=(0.5, 0.5), radius=0.3, value_inside=5.0)
    >>> schema.set_initial_condition(ic)
    """
    center = tuple(center)
    
    def _sphere(*coords):
        # Compute distance from center
        r_squared = 0
        for coord, c in zip(coords, center):
            r_squared += (coord - c) ** 2
        
        r = np.sqrt(r_squared)
        
        result = np.where(r <= radius, value_inside, value_outside)
        return result
    
    return _sphere


def radial_gradient(
    center: Union[float, Tuple[float, ...]],
    max_value: float = 1.0,
    max_radius: float = 1.0,
    decay_type: str = 'linear'
):
    """
    Create a radial gradient initial condition.
    
    Creates a radially symmetric gradient from a center point.
    
    Parameters
    ----------
    center : Union[float, Tuple[float, ...]]
        Center position of the gradient.
    max_value : float, optional
        Maximum value at the center. Default is 1.0.
    max_radius : float, optional
        Radius beyond which value is zero. Default is 1.0.
    decay_type : str, optional
        Type of decay: 'linear', 'quadratic', or 'exponential'.
        Default is 'linear'.
        
    Returns
    -------
    Callable
        Function that returns radial gradient.
        
    Examples
    --------
    >>> # Linear radial gradient
    >>> ic = radial_gradient(center=(0.5, 0.5), max_value=10.0, max_radius=0.5)
    >>> schema.set_initial_condition(ic)
    """
    if isinstance(center, (int, float)):
        center = (float(center),)
    else:
        center = tuple(center)
    
    def _radial_gradient(*coords):
        # Compute distance from center
        r_squared = 0
        for coord, c in zip(coords, center):
            r_squared += (coord - c) ** 2
        
        r = np.sqrt(r_squared)
        
        if decay_type == 'linear':
            value = max_value * np.maximum(0, 1 - r / max_radius)
        elif decay_type == 'quadratic':
            value = max_value * np.maximum(0, 1 - (r / max_radius)**2)
        elif decay_type == 'exponential':
            value = max_value * np.exp(-r / max_radius)
        else:
            raise ValueError(f"Unknown decay_type: {decay_type}")
        
        return value
    
    return _radial_gradient

def sine(
    wavenumber: Union[float, Tuple[float, ...]] = 1.0,
    amplitude: float = 1.0,
    phase: Union[float, Tuple[float, ...]] = 0.0
):
    """
    Create a sine wave initial condition.
    
    Returns a function that evaluates:
        u(x) = amplitude * product(sin(k_i * pi * x_i + phi_i))
    
    Parameters
    ----------
    wavenumber : Union[float, Tuple[float, ...]], optional
        Number of half-sine waves across the domain. Scalar for all dimensions 
        or a tuple for specific per-axis wavenumbers. Default is 1.0.
    amplitude : float, optional
        Maximum amplitude of the wave. Default is 1.0.
    phase : Union[float, Tuple[float, ...]], optional
        Phase shift in radians. Scalar or tuple. Default is 0.0.
        
    Returns
    -------
    Callable
        Function that takes coordinate arrays and returns the sine distribution.
        
    Examples
    --------
    >>> # 1D Sine wave (half-period)
    >>> ic = sine(wavenumber=1.0, amplitude=1.0)
    >>> schema.set_initial_condition(ic)
    
    >>> # 2D Sine wave with different frequencies
    >>> ic = sine(wavenumber=(2.0, 1.0), amplitude=1.0)
    >>> schema.set_initial_condition(ic)
    """
    
    def _sine(*coords):
        ndim = len(coords)
        
        # Ensure wavenumber and phase are tuples of length ndim
        if isinstance(wavenumber, (int, float)):
            k_vals = (float(wavenumber),) * ndim
        else:
            k_vals = tuple(wavenumber)
            
        if isinstance(phase, (int, float)):
            phi_vals = (float(phase),) * ndim
        else:
            phi_vals = tuple(phase)

        # Compute the product of sines for multi-dimensional waves
        result = np.full_like(coords[0], amplitude)
        for i in range(ndim):
            result *= np.sin(k_vals[i] * np.pi * coords[i] + phi_vals[i])
            
        return result
    
    return _sine

def sum_conditions(*conditions):
    """
    Sum multiple initial conditions.
    
    Useful for creating complex patterns by combining simpler ones.
    
    Parameters
    ----------
    *conditions : Callable
        Initial condition functions to sum.
        
    Returns
    -------
    Callable
        Function that returns the sum of all conditions.
        
    Examples
    --------
    >>> # Two Gaussians at different positions
    >>> ic1 = gaussian(center=(0.3, 0.5), amplitude=1.0, width=0.1)
    >>> ic2 = gaussian(center=(0.7, 0.5), amplitude=0.5, width=0.15)
    >>> ic_combined = sum_conditions(ic1, ic2)
    >>> schema.set_initial_condition(ic_combined)
    """
    def _sum(*coords):
        result = np.zeros_like(coords[0])
        for condition in conditions:
            result += condition(*coords)
        return result
    
    return _sum
