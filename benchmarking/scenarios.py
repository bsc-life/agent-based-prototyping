"""
Scenario specification for testing diffusion schemas.

This module provides a framework for defining test scenarios in a declarative,
dict-based manner. Scenarios specify all parameters needed to run a test:
domain, grid, physics parameters, initial/boundary conditions, agents, and
the analytical solution for validation.
"""

import numpy as np
from typing import Dict, Any, List, Union, Tuple, Callable
from diffusion_schemas import ImplicitEulerBCISchema
from diffusion_schemas.utils.initial_conditions import gaussian, uniform, step_function, checkerboard, sphere, sine
from diffusion_schemas.utils.boundary import (
    DirichletBC, NeumannBC, PeriodicBC, RobinBC, BoundaryCondition
)
from diffusion_schemas.utils.agents import Agent, CompleteAgent
from diffusion_schemas.utils.bulk import Bulk, NetRegion, LinearRegion, TargetRegion, RectangleRegion, SphereRegion
from benchmarking.golden_solutions import (
    GoldenSolution, GaussianDiffusion1D, GaussianDiffusion2D, GaussianDiffusion3D, StepFunctionDiffusion1D,
    create_golden_solution_from_dict,
    create_numerical_reference_cached
)
from diffusion_schemas import (ImplicitEulerBCISchema)

# ==============================================================================
# Functions to build
# ==============================================================================

def _build_initial_condition(ic_spec: Union[Dict[str, Any], Callable, np.ndarray, float]) -> Callable:
    """
    Build initial condition from specification.
    
    Parameters
    ----------
    ic_spec : dict, callable, ndarray, or float
        Initial condition specification.
        If dict, must have 'type' key with one of:
            - 'gaussian': center, amplitude, width
            - 'uniform': value
            - 'step_function': position, value_left, value_right, axis
            - 'checkerboard': spacing, value_on, value_off
            - 'sphere': center, radius, value_inside, value_outside
            - 'custom': function (callable accepting coordinates)
        If callable, ndarray, or float, returns as-is.
        
    Returns
    -------
    callable or ndarray or float
        Initial condition suitable for Schema.set_initial_condition().
    """
    if isinstance(ic_spec, dict):
        ic_type = ic_spec['type']
        
        if ic_type == 'gaussian':
            return gaussian(
                center=ic_spec.get('center', 0.5),
                amplitude=ic_spec.get('amplitude', 1.0),
                width=ic_spec.get('width', 0.1)
            )
        
        elif ic_type == 'uniform':
            return uniform(value=ic_spec.get('value', 1.0))
        
        elif ic_type == 'step_function':
            return step_function(
                position=ic_spec['position'],
                value_left=ic_spec.get('value_left', 1.0),
                value_right=ic_spec.get('value_right', 0.0),
                axis=ic_spec.get('axis', 0)
            )
        
        elif ic_type == 'checkerboard':
            return checkerboard(
                spacing=ic_spec.get('spacing', 1.0),
                value_on=ic_spec.get('value_on', 1.0),
                value_off=ic_spec.get('value_off', 0.0)
            )
        
        elif ic_type == 'sphere':
            return sphere(
                center=ic_spec['center'],
                radius=ic_spec['radius'],
                value_inside=ic_spec.get('value_inside', 1.0),
                value_outside=ic_spec.get('value_outside', 0.0)
            )
        elif ic_type == 'sine':
            return sine(
                wavenumber=ic_spec.get('wavenumber', 1.0),
                amplitude=ic_spec.get('amplitude', 1.0)
            )

        elif ic_type == 'custom':
            return ic_spec['function']
        
        else:
            raise ValueError(f"Unknown initial condition type: {ic_type}")
    
    else:
        # Already in suitable form (callable, ndarray, or scalar)
        return ic_spec

def _build_boundary_condition(bc_spec: Union[Dict[str, Any], BoundaryCondition, None]) -> BoundaryCondition:
    """
    Build boundary condition from specification.
    
    Parameters
    ----------
    bc_spec : dict, BoundaryCondition, or None
        Boundary condition specification.
        If dict, must have 'type' key with one of:
            - 'dirichlet': value (float or callable)
            - 'neumann': flux (float or callable)
            - 'periodic': no additional parameters
            - 'robin': alpha, beta, gamma
        If BoundaryCondition instance, returns as-is.
        If None, returns None (Schema will use default).
        
    Returns
    -------
    BoundaryCondition or None
        Boundary condition object.
    """
    if bc_spec is None:
        return None
    
    if isinstance(bc_spec, BoundaryCondition):
        return bc_spec
    
    if isinstance(bc_spec, dict):
        bc_type = bc_spec['type']
        
        if bc_type == 'dirichlet':
            return DirichletBC(value=bc_spec.get('value', 0.0))
        
        elif bc_type == 'neumann':
            return NeumannBC(flux=bc_spec.get('flux', 0.0))
        
        elif bc_type == 'periodic':
            return PeriodicBC()
        
        elif bc_type == 'robin':
            return RobinBC(
                alpha=bc_spec['alpha'],
                beta=bc_spec['beta'],
                gamma=bc_spec.get('gamma', 0.0)
            )
        
        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")
    
    raise TypeError(f"Invalid boundary condition specification type: {type(bc_spec)}")

def _build_agents(agents_spec: Union[List[Dict[str, Any]], List[Agent], List[CompleteAgent], None]) -> List[Agent]:
    """
    Build list of agents from specification.
    
    Parameters
    ----------
    agents_spec : list of dict, list of Agent, or None
        Agent specifications.
        If list of dict, each dict should have:
            - 'position': tuple of coordinates
            - For simple Agent: 'net_rate' (float or callable, optional, default 1.0)
            - For CompleteAgent (if 'saturation_density' is present):
                - 'secretion_rate': float or callable (optional, default 1.0)
                - 'uptake_rate': float or callable (optional, default 0.0)
                - 'saturation_density': float (required to trigger CompleteAgent, default 0.0)
            - 'kernel_width': float or None (optional, default None = point source)
            - 'name': str (optional)
        If list of Agent instances, returns as-is.
        If None, returns empty list.
        
    Returns
    -------
    list of Agent
        List of agent objects (Agent or CompleteAgent instances).
    """
    if agents_spec is None:
        return []
    
    if not agents_spec:  # Empty list
        return []
    
    # Check if already Agent instances
    if isinstance(agents_spec[0], (Agent, CompleteAgent)):
        return agents_spec
    
    # Build from dicts
    agents = []
    for agent_spec in agents_spec:
        use_complete_agent = (
            'saturation_density' in agent_spec
        )
        
        if use_complete_agent:
            # CompleteAgent uses secretion/uptake, not net_rate
            agent_kwargs = {
                'position': agent_spec['position'],
                'secretion_rate': agent_spec.get('secretion_rate', 1.0),
                'uptake_rate': agent_spec.get('uptake_rate', 0.0),
                'saturation_density': agent_spec.get('saturation_density', 0.0),
                'kernel_width': agent_spec.get('kernel_width', None),
                'name': agent_spec.get('name', '')
            }
            agent = CompleteAgent(**agent_kwargs)
        else:
            # Simple Agent uses net_rate
            agent_kwargs = {
                'position': agent_spec['position'],
                'net_rate': agent_spec.get('net_rate', 1.0),
                'kernel_width': agent_spec.get('kernel_width', None),
                'name': agent_spec.get('name', '')
            }
            agent = Agent(**agent_kwargs)
        
        agents.append(agent)
    
    return agents

def _build_bulk(bulk_spec: Union[Dict[str, Any], Bulk, None]) -> Union[Bulk, None]:
    """
    Build Bulk object from specification.
    
    Parameters
    ----------
    bulk_spec : dict, Bulk, or None
        Bulk specification.
        If dict, must have 'regions' key with list of region dicts:
            Each region dict should have:
                - 'type': 'rectangle' or 'sphere'
                - 'origin': tuple (for rectangle)
                - 'size': tuple (for rectangle)
                - 'center': tuple (for sphere)
                - 'radius': float (for sphere)
                - 'net_rate': float or callable (optional, default 0.0)
                - 'name': str (optional)
        If Bulk instance, returns as-is.
        If None, returns None.
        
    Returns
    -------
    Bulk or None
        Bulk object containing regions, or None if input is None.
    """
    if bulk_spec is None:
        return None
    
    if isinstance(bulk_spec, Bulk):
        return bulk_spec
    
    if isinstance(bulk_spec, dict):
        bulk = Bulk()
        
        regions = bulk_spec.get('regions', [])
        if not regions:
            return bulk
        
        for region_spec in regions:
            use_target_bulk = ('rho_target' in region_spec)
            use_linear_bulk = ('linear_rate' in region_spec and not use_target_bulk)

            region_type = region_spec.get('type')
            name = region_spec.get('name', '')
            
            if region_type == 'rectangle':
                domain = RectangleRegion(
                    origin=tuple(region_spec['origin']),
                    size=tuple(region_spec['size'])
                )
            elif region_type == 'sphere':
                domain = SphereRegion(
                    center=tuple(region_spec['center']),
                    radius=region_spec['radius']
                )
            else:
                raise ValueError(f"Unknown region type: {region_type}")
            
            if use_linear_bulk:
                linear_rate = region_spec.get('linear_rate', 0.0)
                region = LinearRegion(domain=domain, linear_rate=linear_rate, name=name)
            elif use_target_bulk:
                linear_rate = region_spec.get('linear_rate', 0.0)
                rho_target = region_spec.get('rho_target', 0.0)
                region = TargetRegion(domain=domain, linear_rate=linear_rate, rho_target=rho_target, name=name)
            else:
                net_rate = region_spec.get('net_rate', 0.0)
                region = NetRegion(domain=domain, net_rate=net_rate, name=name)
            bulk.add_region(region)
    
        return bulk
    
    raise TypeError(f"Invalid bulk specification type: {type(bulk_spec)}")

def _build_golden_solution(golden_spec: Union[Dict[str, Any], GoldenSolution, Callable],
                           scenario_params: Union[Dict[str, Any], None] = None,
                           store_history: bool = True
                           ) -> Union[GoldenSolution, Callable]:
    """
    Build golden solution from specification.
    
    Parameters
    ----------
    golden_spec : dict, GoldenSolution, or callable
        Golden solution specification.
        If dict, passed to create_golden_solution_from_dict().
        If GoldenSolution instance or callable, returns as-is.
        
    Returns
    -------
    GoldenSolution or callable
        Golden solution object or evaluation function.
    """

    # Previous implementation did not allow for numerical reference without specifying scenario_params in the golden_spec dict
    # if isinstance(golden_spec, dict):
    #     # Need to build it from dict specification
    #     # Numerical reference!
    #     return create_golden_solution_from_dict(golden_spec)
    # else:
    #     return golden_spec
    
    if isinstance(golden_spec, dict):
        if golden_spec.get('type') == 'numerical_reference' and 'scenario_params' not in golden_spec:
            if scenario_params is None:
                raise ValueError("numerical_reference requires scenario_params")
            golden_spec = {**golden_spec, 'scenario_params': scenario_params}
        return create_golden_solution_from_dict(golden_spec, store_history=store_history)
    return golden_spec

def create_scenario(name: str,
                   domain_size: Union[float, Tuple[float, ...]],
                   grid_points: Union[int, Tuple[int, ...]],
                   dt: float,
                   t_final: float,
                   initial_condition: Union[Dict[str, Any], Callable, np.ndarray, float],
                   golden_solution: Union[Dict[str, Any], GoldenSolution, Callable],
                   diffusion_coefficient: float = 1.0,
                   decay_rate: float = 0.0,
                   boundary_condition: Union[Dict[str, Any], BoundaryCondition, None] = None,
                   agents: Union[List[Dict[str, Any]], List[Agent], None] = None,
                   bulk: Union[Dict[str, Any], Bulk, None] = None,
                   **metadata) -> Dict[str, Any]:
    """
    Create a test scenario specification.
    
    This function creates a complete scenario specification as a dictionary that
    can be used by the test runner framework.
    
    Parameters
    ----------
    name : str
        Scenario name (used for reporting and file naming).
    domain_size : float or tuple of float
        Domain size (scalar for 1D, tuple for 2D/3D).
    grid_points : int or tuple of int
        Number of grid points (scalar for 1D, tuple for 2D/3D).
    dt : float
        Time step size.
    t_final : float
        Final simulation time.
    initial_condition : dict, callable, ndarray, or float
        Initial condition specification.
    golden_solution : dict, GoldenSolution, or callable
        Analytical solution for validation.
    diffusion_coefficient : float, optional
        Diffusion coefficient D (default 1.0).
    decay_rate : float, optional
        Decay rate λ (default 0.0).
    boundary_condition : dict, BoundaryCondition, or None, optional
        Boundary condition specification (default None = zero-flux).
    agents : list of dict, list of Agent, or None, optional
        Agent specifications (default None = no agents).
    bulk : dict, Bulk, or None, optional
        Bulk region specifications (default None = no bulk regions).
    **metadata : additional keyword arguments
        Additional metadata to store with scenario (e.g., description, tags).
        
    Returns
    -------
    dict
        Complete scenario specification.
    """
    scenario = {
        'name': name,
        'domain_size': domain_size,
        'grid_points': grid_points,
        'dt': dt,
        't_final': t_final,
        'diffusion_coefficient': diffusion_coefficient,
        'decay_rate': decay_rate,
        'initial_condition': initial_condition,
        'boundary_condition': boundary_condition,
        'agents': agents,
        'bulk': bulk,
        'golden_solution': golden_solution,
        **metadata
    }
    
    return scenario

def create_scenario_with_numerical_reference(
    name: str,
    schema_class,
    domain_size: Union[float, Tuple[float, ...]],
    grid_points: Union[int, Tuple[int, ...]],
    dt: float,
    t_final: float,
    initial_condition: Union[Dict[str, Any], Callable, np.ndarray, float],
    diffusion_coefficient: float = 1.0,
    decay_rate: float = 0.0,
    boundary_condition: Union[Dict[str, Any], BoundaryCondition, None] = None,
    agents: Union[List[Dict[str, Any]], List[Agent], None] = None,
    bulk: Union[Dict[str, Any], Bulk, None] = None,
    # dx_refinement_factor: int = 10,
    # dt_refinement_factor: int = 10,
    dx_ref: float = 1e-3,
    dt_ref: float = 1e-3,
    **metadata
) -> Dict[str, Any]:
    """
    Create a scenario that uses a high-resolution schema run as the golden solution.
    
    Parameters
    ----------
    name : str
        Scenario name.
    schema_class : class
        Schema class to use for reference solution (e.g., ImplicitLODSchema).
    domain_size : float or tuple
        Domain size.
    grid_points : int or tuple
        Number of grid points for the test (coarse grid).
    dt : float
        Time step for the test (coarse time step).
    t_final : float
        Final time.
    initial_condition : dict, callable, ndarray, or float
        Initial condition specification.
    diffusion_coefficient : float
        Diffusion coefficient.
    decay_rate : float
        Decay rate.
    boundary_condition : dict, BoundaryCondition, or None
        Boundary condition.
    agents : list or None
        Agent specifications.
    bulk : dict, Bulk, or None
        Bulk region specifications.
    dx_refinement_factor : int
        Factor to refine spatial grid for reference (default 10).
    dt_refinement_factor : int
        Factor to refine time step for reference (default 10).
    **metadata
        Additional metadata.
        
    Returns
    -------
    dict
        Scenario specification with numerical reference golden solution.
    """
    # Create base scenario parameters
    scenario_params = {
        'domain_size': domain_size,
        'grid_points': grid_points,
        'dt': dt,
        't_final': t_final,
        'diffusion_coefficient': diffusion_coefficient,
        'decay_rate': decay_rate,
        'initial_condition': initial_condition,
        'boundary_condition': boundary_condition,
        'agents': agents,
        'bulk': bulk
    }
    
    # Create the golden solution specification
    golden_solution_spec = {
        'type': 'numerical_reference',
        'schema_class': schema_class,
        'scenario_params': scenario_params,
        'dx_ref': dx_ref,
        'dt_ref': dt_ref
        # 'dx_refinement_factor': dx_refinement_factor,
        # 'dt_refinement_factor': dt_refinement_factor
    }
    
    # Create full scenario
    return create_scenario(
        name=name,
        domain_size=domain_size,
        grid_points=grid_points,
        dt=dt,
        t_final=t_final,
        initial_condition=initial_condition,
        golden_solution=golden_solution_spec,
        diffusion_coefficient=diffusion_coefficient,
        decay_rate=decay_rate,
        boundary_condition=boundary_condition,
        agents=agents,
        bulk=bulk,
        **metadata
    )


def create_scenario_with_numerical_reference_cached(
    name: str,
    schema_class,
    domain_size: Union[float, Tuple[float, ...]],
    grid_points: Union[int, Tuple[int, ...]],
    dt: float,
    t_final: float,
    initial_condition: Union[Dict[str, Any], Callable, np.ndarray, float],
    diffusion_coefficient: float = 1.0,
    decay_rate: float = 0.0,
    boundary_condition: Union[Dict[str, Any], BoundaryCondition, None] = None,
    agents: Union[List[Dict[str, Any]], List[Agent], None] = None,
    bulk: Union[Dict[str, Any], Bulk, None] = None,
    dx_ref: float = 1e-3,
    dt_ref: float = 1e-3,
    cache_dir: str = 'benchmark_results/.golden_cache',
    store_history: bool = True,
    **metadata
) -> Dict[str, Any]:
    """
    Same as create_scenario_with_numerical_reference, but the golden solution
    is computed once and cached to disk.  Subsequent calls with identical
    parameters load from cache in ~1 s instead of recomputing.

    All parameters are identical to create_scenario_with_numerical_reference,
    with the addition of *cache_dir* (default 'benchmark_results/.golden_cache').
    """
    # Build the same scenario_params dict used by the original function
    scenario_params = {
        'name': name,
        'domain_size': domain_size,
        'grid_points': grid_points,
        'dt': dt,
        't_final': t_final,
        'diffusion_coefficient': diffusion_coefficient,
        'decay_rate': decay_rate,
        'initial_condition': initial_condition,
        'boundary_condition': boundary_condition,
        'agents': agents,
        'bulk': bulk
    }

    # Compute or load the golden solution
    golden_solution = create_numerical_reference_cached(
        schema_class=schema_class,
        scenario_params=scenario_params,
        dx_ref=dx_ref,
        dt_ref=dt_ref,
        cache_dir=cache_dir,
        store_history=store_history
    )

    # Build the scenario with the already-resolved golden solution object
    return create_scenario(
        name=name,
        domain_size=domain_size,
        grid_points=grid_points,
        dt=dt,
        t_final=t_final,
        initial_condition=initial_condition,
        golden_solution=golden_solution,
        diffusion_coefficient=diffusion_coefficient,
        decay_rate=decay_rate,
        boundary_condition=boundary_condition,
        agents=agents,
        bulk=bulk,
        **metadata
    )


def build_scenario_components(scenario: Dict[str, Any], store_history: bool = True) -> Dict[str, Any]:
    """
    Build actual objects from scenario specification.
    
    This converts the dict-based specification into actual Python objects
    that can be used to configure a Schema instance.
    
    Parameters
    ----------
    scenario : dict
        Scenario specification (from create_scenario()).
        
    Returns
    -------
    dict
        Dictionary with built components:
        - 'initial_condition': Built IC
        - 'boundary_condition': Built BC
        - 'agents': List of Agent objects
        - 'bulk': Bulk object or None
        - 'golden_solution': Built golden solution
        Plus all original scenario fields.
    """
    built = scenario.copy()
    
    scenario_params = {
        k: v for k, v in scenario.items()
        if k not in ('name', 'description', 'golden_solution')
    }

    built['initial_condition'] = _build_initial_condition(scenario['initial_condition'])
    built['boundary_condition'] = _build_boundary_condition(scenario['boundary_condition'])
    built['agents'] = _build_agents(scenario.get('agents', None))
    built['golden_solution'] = _build_golden_solution(scenario['golden_solution'], 
                                                      scenario_params = scenario_params,
                                                      store_history = store_history)
    built['bulk'] = _build_bulk(scenario.get('bulk', None))

    return built

# ==============================================================================
# Default Scenarios
# ==============================================================================

# Default 2D Gaussian pulse scenario
GAUSSIAN_PULSE_2D = {
    'name': 'gaussian_pulse_2d',
    'description': '2D Gaussian pulse diffusion with zero-flux boundaries',
    
    # Domain and discretization
    'domain_size': (1.0, 1.0),
    'grid_points': (50, 50),
    'dt': 0.001,
    't_final': 0.5,
    
    # Physics parameters
    'diffusion_coefficient': 0.01,
    'decay_rate': 0.0,
    
    # Initial condition: Gaussian at center
    'initial_condition': {
        'type': 'gaussian',
        'center': (0.5, 0.5),
        'amplitude': 1.0,
        'width': 0.1
    },
    
    # Boundary condition: zero-flux (insulating)
    'boundary_condition': {
        'type': 'neumann',
        'flux': 0.0
    },
    
    # No agents
    'agents': None,
    
    # Analytical solution
    'golden_solution': {
        'type': 'gaussian_2d',
        'center': (0.5, 0.5),
        'amplitude': 1.0,
        'initial_width': 0.1,
        'diffusion_coefficient': 0.01
    }
}

# 1D Gaussian pulse for simpler testing
GAUSSIAN_PULSE_1D = {
    'name': 'gaussian_pulse_1d',
    'description': '1D Gaussian pulse diffusion with zero-flux boundaries',
    
    'domain_size': 1.0,
    'grid_points': 100,
    'dt': 0.0005,
    't_final': 0.5,
    
    'diffusion_coefficient': 0.01,
    'decay_rate': 0.0,
    
    'initial_condition': {
        'type': 'gaussian',
        'center': 0.5,
        'amplitude': 1.0,
        'width': 0.1
    },
    
    'boundary_condition': {
        'type': 'neumann',
        'flux': 0.0
    },
    
    'agents': None,
    
    'golden_solution': {
        'type': 'gaussian_1d',
        'center': 0.5,
        'amplitude': 1.0,
        'initial_width': 0.1,
        'diffusion_coefficient': 0.01
    }
}

# 3D Gaussian pulse for testing higher dimensions
GAUSSIAN_PULSE_3D = {
    'name': 'gaussian_pulse_3d',
    'description': '3D Gaussian pulse diffusion with zero-flux boundaries',
    
    'domain_size': (1.0, 1.0, 1.0),
    'grid_points': (30, 30, 30),
    'dt': 0.001,
    't_final': 0.2,
    
    'diffusion_coefficient': 0.01,
    'decay_rate': 0.0,
    
    'initial_condition': {
        'type': 'gaussian',
        'center': (0.5, 0.5, 0.5),
        'amplitude': 1.0,
        'width': 0.1
    },
    
    'boundary_condition': {
        'type': 'neumann',
        'flux': 0.0
    },
    
    'agents': None,
    
    'golden_solution': {
        'type': 'gaussian_3d',
        'center': (0.5, 0.5, 0.5),
        'amplitude': 1.0,
        'initial_width': 0.1,
        'diffusion_coefficient': 0.01
    }
}

# ==============================================================================
# Add-ons for additional testing
# ==============================================================================

STEP_FUNCTION_1D = {
    'name': 'step_function_1d',
    'description': '1D step diffusion with zero-flux boundaries',
    
    'domain_size': 1.0,
    'grid_points': 500,
    'dt': 0.00005,
    't_final': 0.16,
    
    'diffusion_coefficient': 0.01,
    'decay_rate': 0.0,
    
    'initial_condition': {
        'type': 'step_function',
        'position': 0.5,
        'value_left': 1.0,
        'value_right': 0.0,
        'axis' : 0
    },
    
    'boundary_condition': {
        'type': 'neumann',
        'flux': 0.0
    },
    
    'agents': None, 

    'golden_solution': {
            'type': 'step_function_1d',   # Matches the key in your factory
            'domain_length': 1.0,         # Should match domain_size above
            'position': 0.5,              # Where the step occurs (x0)
            'value_left': 1.0,            # u value for x < x0
            'value_right': 0.0,           # u value for x >= x0
            'axis': 0,                    # Axis (0=x)
            'diffusion_coefficient': 0.01, # Must match outer diffusion_coefficient
            'n_terms': 200                # Accuracy of analytical solution
        }
}

STEP_FUNCTION_2D = {
    'name': 'step_function_2d',
    'description': '2D step diffusion with zero-flux boundaries',
    
    'domain_size': (1.0, 1.0),
    'grid_points': (50, 50),
    'dt': 0.0005,
    't_final': 0.2,
    
    'diffusion_coefficient': 0.01,
    'decay_rate': 0.0,
    
    'initial_condition': {
        'type': 'step_function',
        'position': 0.5,
        'value_left': 1.0,
        'value_right': 0.0,
        'axis' : 0
    },
    
    'boundary_condition': {
        'type': 'neumann',
        'flux': 0.0
    },
    
    'agents': None, 

    'golden_solution': {
            'type': 'step_function_1d',   # Matches the key in your factory
            'domain_length': 1.0,         # Should match domain_size above
            'position': 0.5,              # Where the step occurs (x0)
            'value_left': 1.0,            # u value for x < x0
            'value_right': 0.0,           # u value for x >= x0
            'axis': 0,                    # Axis (0=x)
            'diffusion_coefficient': 0.01, # Must match outer diffusion_coefficient
            'n_terms': 200                # Accuracy of analytical solution
        }
}

EXPONENTIAL_DECAY_1D = {
    'name': 'exponential_decay_1d',
    'description': 'Pure exponential decay without spatial diffusion',
    
    'domain_size': 1.0,
    'grid_points': 100,
    'dt': 0.001,
    't_final': 2.0,
    
    'diffusion_coefficient': 0.0,  # Disabled to test purely decay
    'decay_rate': 1.5,             # λ value
    
    'initial_condition': {
        'type': 'gaussian',
        'center': 0.5,
        'amplitude': 10.0,
        'width': 0.1
    },
    
    'boundary_condition': {
        'type': 'neumann',
        'flux': 0.0
    },
    
    'agents': None,
    
    'golden_solution': {
        'type': 'exponential_decay',
        'initial_condition': gaussian(center=0.5, amplitude=10.0, width=0.1),
        'decay_rate': 1.5
    }
}

STEADY_STATE_AGENT_1D = {
    'name': 'steady_state_agent_1d',
    'description': '1D diffusion with decay and a continuous point source',
    
    'domain_size': 1.0,
    'grid_points': 51,  
    'dt': 0.005,
    't_final': 10.0, # Large enough to reach steady state (never reached anyway)
    
    'diffusion_coefficient': 0.01,
    'decay_rate': 0.1,
    
    'initial_condition': {
        'type': 'uniform',
        'value': 0.0
    },
    
    'boundary_condition': {
        'type': 'dirichlet',       # Approximating u -> 0 at infinite distance
        'value': [0.0, 0.0, 0.0, 0.0]
    },
    
    'agents': [
        {
            'position': (0.33,), # Must be an iterable
            'net_rate': 1.0 # Net rate of the point source agent
        }
    ],
    
    'golden_solution': {
        'type': 'steady_state_agent',
        'source_position': (0.33,),  # Should not be exactly at the center to avoid singularity
        'source_strength': 1.0,
        'diffusion_coefficient': 0.01,
        'decay_rate': 0.1,
        'ndim': 1
    }
}

STEADY_STATE_AGENT_2D = {
    'name': 'steady_state_agent_2d',
    'description': '2D diffusion with decay and a continuous point source',
    
    'domain_size': [1.0, 1.0],
    'grid_points': [51, 51],       # Odd number ensures a grid point lands exactly at [0.5, 0.5]
    'dt': 0.005,
    't_final': 20.0,               # Large enough to reach steady state
    
    'diffusion_coefficient': 0.01,
    'decay_rate': 0.1,
    
    'initial_condition': {
        'type': 'uniform',
        'value': 0.0
    },
    
    'boundary_condition': {
        'type': 'dirichlet',       # Approximating u -> 0 at infinite distance
        'value': [0.0, 0.0, 0.0, 0.0]
    },
    
    'agents': [
        {
            'position': [0.33, 0.33],
            'net_rate': 1.0        # Net rate of the point source agent
        }
    ],
    
    'golden_solution': {
        'type': 'steady_state_agent',
        'source_position': [0.33, 0.33],  # Should not be exactly at the center to avoid singularity
        'source_strength': 1.0,
        'diffusion_coefficient': 0.01,
        'decay_rate': 0.1,
        'ndim': 2
    }
}

SINE_DECAY_1D = {
    'name': 'sine_decay_1d',
    'description': '1D Sine wave decay with Dirichlet boundaries',
    
    'domain_size': 1.0,
    'grid_points': 100,
    'dt': 0.0005,
    't_final': 0.5,
    
    'diffusion_coefficient': 0.01,
    'decay_rate': 0.0,
    
    'initial_condition': {
        'type': 'sine',
        'wavenumber': 3.0,  # Number of half-sine waves
        'amplitude': 1.0
    },
    
    'boundary_condition': {
        'type': 'dirichlet',
        'value': [0.0, 0.0]
    },
    
    'agents': None,
    
    'golden_solution': {
        'type': 'sine_decay_1d',
        'wavenumber': 3.0,
        'amplitude': 1.0,
        'diffusion_coefficient': 0.01
    }
}



# ==============================================================================
# BioFVM CONVERGENCE TESTS
# ==============================================================================

# 1st test

CONVERGENCE_TEST_1 = {
    # BioFVM convergence test 1: 1D diffusion with cosine initial condition and zero-flux boundaries
    'name': 'convergence_test_1',
    'description': '1-D diffusion with analytical solution (BioFVM convergence test 1)',
    
    'domain_size': (1000,), # L0 = 500 um
    'grid_points': (int(1000 / 5),), # dx = 5 um
    'dt': 0.00001,
    't_final': 5,
    
    'diffusion_coefficient': 1e5,
    'decay_rate': 0.0,
    
    'initial_condition': {
        'type': 'custom',
        'function': lambda x: 1 + np.cos(np.pi * (x-500) / 500)
    },
    
    'boundary_condition': {
        'type': 'neumann',
        'flux': 0.0
    },
    
    'agents': None,

    'golden_solution': lambda x, t: (1.0 + np.cos(np.pi * (np.asarray(x).flatten()-500) / 500) * np.exp(- np.pi ** 2 * 1e5 / 500 ** 2 * t))
}

COSINE_DIFFUSION_2D = {
    # BioFVM convergence test 1: 1D diffusion with cosine initial condition and zero-flux boundaries
    'name': 'cosine_diffusion_2d',
    'description': 'First convergence test adapted to 2D',
    
    'domain_size': (1000.0, 1000.0),
    'grid_points': (int(1000 / 5), int(1000 / 5)),
    'dt': 0.00001,
    't_final': 2,
    
    'diffusion_coefficient': 1e5,
    'decay_rate': 0.0,
    
    'initial_condition': {
        'type': 'custom',
        'function': lambda x, y: (1 + np.cos(np.pi * (x-500) / 500)) * \
                                 (1 + np.cos(np.pi * (y-500) / 500))
    },
    
    'boundary_condition': {
        'type': 'neumann',
        'flux': 0.0
    },
    
    'agents': None,

    'golden_solution': lambda x, y, t: (
        (1.0 + np.cos(np.pi * (np.asarray(x)-500) / 500) * np.exp(- np.pi**2 * 1e5 / 500**2 * t)) *
        (1.0 + np.cos(np.pi * (np.asarray(y)-500) / 500) * np.exp(- np.pi**2 * 1e5 / 500**2 * t))
    )
}

# 2nd test

CONVERGENCE_TEST_2 = {
    'name': 'convergence_test_2',
    'description': '3-D diffusion-reaction with bulk sources (BioFVM convergence test 2)',
    
    # Domain is [-500, 500]^3, translated here to [0, 1000]^3 for positive coordinates
    'domain_size': (1000.0, 1000.0, 1000.0),
    'grid_points': (int(1000.0/10), int(1000.0/10), int(1000.0/10)), # Medium resolution dx = 10 μm 
    'dt': 0.01, # Example testing dt
    't_final': 4.0, # Simulated 4 minutes of diffusion 
    
    'diffusion_coefficient': 1e5, # D = 10^5 μm^2/min
    'decay_rate': 0.1, # λ = 0.1 min^-1 
    
    'initial_condition': {
        'type': 'uniform',
        'value': 38.0 # Typical starting point, bounded by saturation density of bulks ρ* = 38 mmHg 
    },
    
    'boundary_condition': {
        'type': 'neumann',
        'flux': 0.0 # D∇ρ·n = 0
    },
    
    'bulk': {
        'regions': [
            # X-faces (Full Y and Z spans)
            {
                'type': 'rectangle',
                'origin': (0.0, 0.0, 0.0),
                'size': (40.0, 1000.0, 1000.0),
                'linear_rate': 38.0,
                'rho_target': 38.0,
                'name': 'x_min_boundary'
            },
            {
                'type': 'rectangle',
                'origin': (960.0, 0.0, 0.0),
                'size': (40.0, 1000.0, 1000.0),
                'linear_rate': 38.0,
                'rho_target': 38.0,
                'name': 'x_max_boundary'
            },
            # Y-faces (Restricted X span to avoid overlapping with X-faces)
            {
                'type': 'rectangle',
                'origin': (40.0, 0.0, 0.0),
                'size': (920.0, 40.0, 1000.0),
                'linear_rate': 38.0,
                'rho_target': 38.0,
                'name': 'y_min_boundary'
            },
            {
                'type': 'rectangle',
                'origin': (40.0, 960.0, 0.0),
                'size': (920.0, 40.0, 1000.0),
                'linear_rate': 38.0,
                'rho_target': 38.0,
                'name': 'y_max_boundary'
            },
            # Z-faces (Restricted X and Y spans to avoid overlapping with X and Y faces)
            {
                'type': 'rectangle',
                'origin': (40.0, 40.0, 0.0),
                'size': (920.0, 920.0, 40.0),
                'linear_rate': 38.0,
                'rho_target': 38.0,
                'name': 'z_min_boundary'
            },
            {
                'type': 'rectangle',
                'origin': (40.0, 40.0, 960.0),
                'size': (920.0, 920.0, 40.0),
                'linear_rate': 38.0,
                'rho_target': 38.0,
                'name': 'z_max_boundary'
            }
        ]
    },
    
    # No analytical solution for this example
    'golden_solution': {
        'type': 'numerical_reference',
        'schema_class': ImplicitEulerBCISchema, 
        'dx_ref': 10.0, # High/fine spatial resolution for reference 
        'dt_ref': 0.0001 # Fine time resolution for reference 
    },

    'store_history': False
}

CONVERGENCE_TEST_2_2D = {
    'name': 'convergence_test_2_2D',
    'description': '2-D diffusion-reaction with bulk sources (BioFVM convergence test 2)',
    
    # Domain is [-500, 500]^2, translated here to [0, 1000]^2 for positive coordinates
    'domain_size': (1000.0, 1000.0),
    'grid_points': (int(1000.0/10), int(1000.0/10)), # Medium resolution dx = 10 μm 
    'dt': 0.001, # Example testing dt
    't_final': 4.0, # Simulated 4 minutes of diffusion 
    
    'diffusion_coefficient': 1e5, # D = 10^5 μm^2/min
    'decay_rate': 0.1, # λ = 0.1 min^-1 
    
    'initial_condition': {
        'type': 'uniform',
        'value': 38.0 # Typical starting point, bounded by saturation density of bulks ρ* = 38 mmHg 
    },
    
    'boundary_condition': {
        'type': 'neumann',
        'flux': 0.0 # D∇ρ·n = 0
    },
    # 'boundary_condition': {
    #     'type': 'dirichlet',
    #     'value': 38.0 # ρ = 38 mmHg at boundaries
    # },
    
    'bulk': {
        'regions': [
            # X-faces (Full Y and Z spans)
            {
                'type': 'rectangle',
                'origin': (0.0, 0.0),
                'size': (40.0, 1000.0),
                'linear_rate': 38.0,
                'rho_target': 38.0,
                'name': 'x_min_boundary'
            },
            {
                'type': 'rectangle',
                'origin': (960.0, 0.0),
                'size': (40.0, 1000.0),
                'linear_rate': 38.0,
                'rho_target': 38.0,
                'name': 'x_max_boundary'
            },
            # Y-faces (Restricted X span to avoid overlapping with X-faces)
            {
                'type': 'rectangle',
                'origin': (40.0, 0.0),
                'size': (920.0, 40.0),
                'linear_rate': 38.0,
                'rho_target': 38.0,
                'name': 'y_min_boundary'
            },
            {
                'type': 'rectangle',
                'origin': (40.0, 960.0),
                'size': (920.0, 40.0),
                'linear_rate': 38.0,
                'rho_target': 38.0,
                'name': 'y_max_boundary'
            }
        ]
    },
    
    # No analytical solution for this example
    'golden_solution': {
        'type': 'numerical_reference',
        'schema_class': ImplicitEulerBCISchema, 
        'dx_ref': 10.0, # High/fine spatial resolution for reference 
        'dt_ref': 0.0001 # Fine time resolution for reference 
    },

    'store_history': False
}

# 3rd test

cell_centers = []
tumor_center = np.array([500.0, 500.0, 500.0])
tumor_radius = 400.0 
# Grid aligned cells every 10 um (e.g., 5, 15, ..., 995)
coords = np.arange(5.0, 1000.0, 10.0)
for x in coords:
    for y in coords:
        for z in coords:
            pos = np.array([x, y, z])
            if np.linalg.norm(pos - tumor_center) <= tumor_radius:
                cell_centers.append((x, y, z))
# Cell radius derived from volume W_k = 1000 um^3 
cell_radius = (1000.0 * 0.75 / np.pi)**(1/3) 
cell_regions = [
    {
        'type': 'sphere',
        'center': pos,
        'radius': cell_radius,
        # Uptake rate U_k = 10 min^-1  
        # (Negative linear_rate acts as -U_k * rho in your compute_source method)
        'linear_rate': -10.0, 
        'name': f'cell_{i}'
    }
    for i, pos in enumerate(cell_centers)
]
boundary_regions = [
    {'type': 'rectangle', 'origin': (0.0, 0.0, 0.0), 'size': (40.0, 1000.0, 1000.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'x_min_boundary'},
    {'type': 'rectangle', 'origin': (960.0, 0.0, 0.0), 'size': (40.0, 1000.0, 1000.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'x_max_boundary'},
    {'type': 'rectangle', 'origin': (40.0, 0.0, 0.0), 'size': (920.0, 40.0, 1000.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'y_min_boundary'},
    {'type': 'rectangle', 'origin': (40.0, 960.0, 0.0), 'size': (920.0, 40.0, 1000.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'y_max_boundary'},
    {'type': 'rectangle', 'origin': (40.0, 40.0, 0.0), 'size': (920.0, 920.0, 40.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'z_min_boundary'},
    {'type': 'rectangle', 'origin': (40.0, 40.0, 960.0), 'size': (920.0, 920.0, 40.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'z_max_boundary'}
]
CONVERGENCE_TEST_3 = {
    'name': 'convergence_test_3',
    'description': '3-D diffusion-reaction with bulk sources and grid-aligned cell uptake (BioFVM convergence test 3)',
    
    'domain_size': (1000.0, 1000.0, 1000.0),
    'grid_points': (int(1000.0/20), int(1000.0/20), int(1000.0/20)),
    'dt': 0.01,
    't_final': 4.0,
    
    'diffusion_coefficient': 1e5,
    'decay_rate': 0.1,
    
    'initial_condition': {
        'type': 'uniform',
        'value': 38.0
    },
    
    'boundary_condition': {
        'type': 'neumann',
        'flux': 0.0
    },
    
    'bulk': {
        'regions': boundary_regions + cell_regions
    },
    
    'golden_solution': {
        'type': 'numerical_reference',
        'schema_class': None,
        'dx_ref': 10.0,
        'dt_ref': 0.0001
    },

    'store_history': False
}

ct3_2d_cell_centers = []
ct3_2d_tumor_center = np.array([500.0, 500.0])
ct3_2d_tumor_radius = 400.0
ct3_2d_coords = np.arange(5.0, 1000.0, 10.0)
for x in ct3_2d_coords:
    for y in ct3_2d_coords:
        pos = np.array([x, y])
        if np.linalg.norm(pos - ct3_2d_tumor_center) <= ct3_2d_tumor_radius:
            ct3_2d_cell_centers.append((x, y))
# Each cell occupies one 10um x 10um voxel (dx = dy = 10um).
ct3_2d_cell_size = (10.0, 10.0)
ct3_2d_cell_regions = [
    {
        'type': 'rectangle',
        'origin': (pos[0] - 5.0, pos[1] - 5.0),
        'size': ct3_2d_cell_size,
        'linear_rate': -10.0,
        'name': f'cell_{i}'
    }
    for i, pos in enumerate(ct3_2d_cell_centers)
]
ct3_2d_boundary_regions = [
    {'type': 'rectangle', 'origin': (0.0, 0.0), 'size': (40.0, 1000.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'x_min_boundary'},
    {'type': 'rectangle', 'origin': (960.0, 0.0), 'size': (40.0, 1000.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'x_max_boundary'},
    {'type': 'rectangle', 'origin': (40.0, 0.0), 'size': (920.0, 40.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'y_min_boundary'},
    {'type': 'rectangle', 'origin': (40.0, 960.0), 'size': (920.0, 40.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'y_max_boundary'}
]
CONVERGENCE_TEST_3_2D = {
    'name': 'convergence_test_3_2d',
    'description': '2-D diffusion-reaction with bulk sources and grid-aligned cell uptake (BioFVM convergence test 3 adaptation)',

    'domain_size': (1000.0, 1000.0),
    'grid_points': (int(1000.0 / 20), int(1000.0 / 20)),
    'dt': 0.091,
    't_final': 4.0,

    'diffusion_coefficient': 1e5,
    'decay_rate': 0.1,

    'initial_condition': {
        'type': 'uniform',
        'value': 38.0
    },

    'boundary_condition': {
        'type': 'neumann',
        'flux': 0.0
    },

    'bulk': {
        'regions': ct3_2d_boundary_regions + ct3_2d_cell_regions
    },

    'golden_solution': {
        'type': 'numerical_reference',
        'schema_class': ImplicitEulerBCISchema,
        'dx_ref': 10.0,
        'dt_ref': 0.0001
    },

    'store_history': False
}

# 4th test

def generate_hcp_cells(tumor_center, tumor_radius, cell_radius):
    """Generates Hexagonal Close Packed (HCP) cell centers within a sphere."""
    centers = []
    d = 2.0 * cell_radius
    i_max = int(tumor_radius / d) + 2
    j_max = int(tumor_radius / (d * np.sqrt(3)/2)) + 2
    k_max = int(tumor_radius / (d * np.sqrt(6)/3)) + 2
    for k in range(-k_max, k_max):
        for j in range(-j_max, j_max):
            for i in range(-i_max, i_max):
                x = i * d + (j % 2) * (d / 2.0) + (k % 2) * (d / 2.0)
                y = j * d * np.sqrt(3.0)/2.0 + (k % 2) * d * np.sqrt(3.0)/6.0
                z = k * d * np.sqrt(6.0)/3.0
                pos = np.array([x, y, z]) + tumor_center
                if np.linalg.norm(pos - tumor_center) <= tumor_radius:
                    centers.append(pos)
    return centers
tumor_center = np.array([500.0, 500.0, 500.0])
tumor_radius = 400.0
cell_radius = 5.0 # Radius is 5 μm for Example 4
cell_centers = generate_hcp_cells(tumor_center, tumor_radius, cell_radius)
cell_regions = [
    {
        'type': 'sphere',
        'center': tuple(pos),
        'radius': cell_radius,
        'linear_rate': -10.0, # Uptake rate U_k = 10 min^-1
        'name': f'cell_{i}'
    }
    for i, pos in enumerate(cell_centers)
]
boundary_regions = [
    {'type': 'rectangle', 'origin': (0.0, 0.0, 0.0), 'size': (40.0, 1000.0, 1000.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'x_min_boundary'},
    {'type': 'rectangle', 'origin': (960.0, 0.0, 0.0), 'size': (40.0, 1000.0, 1000.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'x_max_boundary'},
    {'type': 'rectangle', 'origin': (40.0, 0.0, 0.0), 'size': (920.0, 40.0, 1000.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'y_min_boundary'},
    {'type': 'rectangle', 'origin': (40.0, 960.0, 0.0), 'size': (920.0, 40.0, 1000.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'y_max_boundary'},
    {'type': 'rectangle', 'origin': (40.0, 40.0, 0.0), 'size': (920.0, 920.0, 40.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'z_min_boundary'},
    {'type': 'rectangle', 'origin': (40.0, 40.0, 960.0), 'size': (920.0, 920.0, 40.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'z_max_boundary'}
]
CONVERGENCE_TEST_4 = {
    'name': 'convergence_test_4',
    'description': '3-D diffusion-reaction with bulk sources, off-lattice cell uptake (BioFVM convergence test 4)',
    
    'domain_size': (1000.0, 1000.0, 1000.0),
    'grid_points': (int(1000.0/20), int(1000.0/20), int(1000.0/20)),
    'dt': 0.01,
    't_final': 4.0,
    
    'diffusion_coefficient': 1e5,
    'decay_rate': 0.1,
    
    'initial_condition': {
        'type': 'uniform',
        'value': 38.0
    },
    
    'boundary_condition': {
        'type': 'neumann',
        'flux': 0.0
    },
    
    'bulk': {
        'regions': boundary_regions + cell_regions
    },
    
    'golden_solution': {
        'type': 'numerical_reference',
        'schema_class': None,
        'dx_ref': 10.0,
        'dt_ref': 0.0001
    },

    'store_history': False
}

def generate_hex_cells_2d(tumor_center, tumor_radius, cell_radius):
    """Generates hexagonally packed 2D cell centers within a circle."""
    centers = []
    d = 2.0 * cell_radius
    i_max = int(tumor_radius / d) + 2
    j_max = int(tumor_radius / (d * np.sqrt(3) / 2)) + 2
    for j in range(-j_max, j_max):
        for i in range(-i_max, i_max):
            x = i * d + (j % 2) * (d / 2.0)
            y = j * d * np.sqrt(3.0) / 2.0
            pos = np.array([x, y]) + tumor_center
            if np.linalg.norm(pos - tumor_center) <= tumor_radius:
                centers.append(pos)
    return centers
ct4_2d_tumor_center = np.array([500.0, 500.0])
ct4_2d_tumor_radius = 400.0
ct4_2d_cell_radius = 10.0
ct4_2d_cell_centers = generate_hex_cells_2d(ct4_2d_tumor_center, ct4_2d_tumor_radius, ct4_2d_cell_radius)
ct4_2d_cell_regions = [
    {
        'type': 'sphere',
        'center': tuple(pos),
        'radius': ct4_2d_cell_radius,
        'linear_rate': -10.0,
        'name': f'cell_{i}'
    }
    for i, pos in enumerate(ct4_2d_cell_centers)
]
ct4_2d_boundary_regions = [
    {'type': 'rectangle', 'origin': (0.0, 0.0), 'size': (40.0, 1000.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'x_min_boundary'},
    {'type': 'rectangle', 'origin': (960.0, 0.0), 'size': (40.0, 1000.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'x_max_boundary'},
    {'type': 'rectangle', 'origin': (40.0, 0.0), 'size': (920.0, 40.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'y_min_boundary'},
    {'type': 'rectangle', 'origin': (40.0, 960.0), 'size': (920.0, 40.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'y_max_boundary'}
]
CONVERGENCE_TEST_4_2D = {
    'name': 'convergence_test_4_2d',
    'description': '2-D diffusion-reaction with bulk sources, off-lattice cell uptake (BioFVM convergence test 4 adaptation)',

    'domain_size': (1000.0, 1000.0),
    'grid_points': (int(1000.0 / 20), int(1000.0 / 20)),
    'dt': 0.01,
    't_final': 4.0,

    'diffusion_coefficient': 1e5,
    'decay_rate': 0.1,

    'initial_condition': {
        'type': 'uniform',
        'value': 38.0
    },

    'boundary_condition': {
        'type': 'neumann',
        'flux': 0.0
    },

    'bulk': {
        'regions': ct4_2d_boundary_regions + ct4_2d_cell_regions
    },

    'golden_solution': {
        'type': 'numerical_reference',
        'schema_class': None,
        'dx_ref': 10.0,
        'dt_ref': 0.0001
    },

    'store_history': False
}

# 5th test

tumor_center = np.array([500.0, 500.0, 500.0])
tumor_radius = 400.0
cell_radius = 5.0 
uptake_centers = generate_hcp_cells(tumor_center, tumor_radius, cell_radius)
k1_regions = [
    {
        'type': 'sphere',
        'center': tuple(pos),
        'radius': cell_radius,
        'linear_rate': -10.0, # Uptake rate U_k = 10 min^-1
        'name': f'uptake_cell_{i}'
    }
    for i, pos in enumerate(uptake_centers)
]
np.random.seed(42) # For reproducibility
source_centers = np.random.uniform(low=40.0, high=960.0, size=(200, 3))
k2_regions = [
    {
        'type': 'sphere',
        'center': tuple(pos),
        'radius': cell_radius,
        'linear_rate': 10.0, # Supply rate S_k = 10 min^-1
        'rho_target': 38.0,  # Target saturation
        'name': f'source_cell_{i}'
    }
    for i, pos in enumerate(source_centers)
]
boundary_regions = [
    {'type': 'rectangle', 'origin': (0.0, 0.0, 0.0), 'size': (40.0, 1000.0, 1000.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'x_min_boundary'},
    {'type': 'rectangle', 'origin': (960.0, 0.0, 0.0), 'size': (40.0, 1000.0, 1000.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'x_max_boundary'},
    {'type': 'rectangle', 'origin': (40.0, 0.0, 0.0), 'size': (920.0, 40.0, 1000.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'y_min_boundary'},
    {'type': 'rectangle', 'origin': (40.0, 960.0, 0.0), 'size': (920.0, 40.0, 1000.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'y_max_boundary'},
    {'type': 'rectangle', 'origin': (40.0, 40.0, 0.0), 'size': (920.0, 920.0, 40.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'z_min_boundary'},
    {'type': 'rectangle', 'origin': (40.0, 40.0, 960.0), 'size': (920.0, 920.0, 40.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'z_max_boundary'}
]
CONVERGENCE_TEST_5 = {
    'name': 'convergence_test_5',
    'description': '3-D diffusion-reaction with bulk sources, off-lattice cell uptake and supply (BioFVM convergence test 5)',
    
    'domain_size': (1000.0, 1000.0, 1000.0),
    'grid_points': (int(1000.0/20), int(1000.0/20), int(1000.0/20)),
    'dt': 0.01,
    't_final': 4.0,
    
    'diffusion_coefficient': 1e5,
    'decay_rate': 0.1,
    
    'initial_condition': {
        'type': 'uniform',
        'value': 38.0
    },
    
    'boundary_condition': {
        'type': 'neumann',
        'flux': 0.0
    },
    
    'bulk': {
        'regions': boundary_regions + k1_regions + k2_regions
    },
    
    'golden_solution': {
        'type': 'numerical_reference',
        'schema_class': None,
        'dx_ref': 10.0,
        'dt_ref': 0.0001
    },

    'store_history': False
}

ct5_2d_tumor_center = np.array([500.0, 500.0])
ct5_2d_tumor_radius = 400.0
ct5_2d_cell_radius = 5.0
ct5_2d_uptake_centers = generate_hex_cells_2d(
    ct5_2d_tumor_center,
    ct5_2d_tumor_radius,
    ct5_2d_cell_radius,
)
ct5_2d_k1_regions = [
    {
        'type': 'sphere',
        'center': tuple(pos),
        'radius': ct5_2d_cell_radius,
        'linear_rate': -10.0,
        'name': f'uptake_cell_{i}'
    }
    for i, pos in enumerate(ct5_2d_uptake_centers)
]
np.random.seed(8131)
ct5_2d_source_centers = np.random.uniform(low=40.0, high=960.0, size=(200, 2))
ct5_2d_k2_regions = [
    {
        'type': 'sphere',
        'center': tuple(pos),
        'radius': ct5_2d_cell_radius,
        'linear_rate': 10.0,
        'rho_target': 38.0,
        'name': f'source_cell_{i}'
    }
    for i, pos in enumerate(ct5_2d_source_centers)
]
ct5_2d_boundary_regions = [
    {'type': 'rectangle', 'origin': (0.0, 0.0), 'size': (40.0, 1000.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'x_min_boundary'},
    {'type': 'rectangle', 'origin': (960.0, 0.0), 'size': (40.0, 1000.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'x_max_boundary'},
    {'type': 'rectangle', 'origin': (40.0, 0.0), 'size': (920.0, 40.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'y_min_boundary'},
    {'type': 'rectangle', 'origin': (40.0, 960.0), 'size': (920.0, 40.0), 'linear_rate': 38.0, 'rho_target': 38.0, 'name': 'y_max_boundary'}
]
CONVERGENCE_TEST_5_2D = {
    'name': 'convergence_test_5_2d',
    'description': '2-D diffusion-reaction with bulk sources, off-lattice cell uptake and supply (BioFVM convergence test 5 adaptation)',

    'domain_size': (1000.0, 1000.0),
    'grid_points': (int(1000.0 / 20), int(1000.0 / 20)),
    'dt': 0.01,
    't_final': 4.0,

    'diffusion_coefficient': 1e5,
    'decay_rate': 0.1,

    'initial_condition': {
        'type': 'uniform',
        'value': 38.0
    },

    'boundary_condition': {
        'type': 'neumann',
        'flux': 0.0
    },

    'bulk': {
        'regions': ct5_2d_boundary_regions + ct5_2d_k1_regions + ct5_2d_k2_regions
    },

    'golden_solution': {
        'type': 'numerical_reference',
        'schema_class': None,
        'dx_ref': 10.0,
        'dt_ref': 0.0001
    },

    'store_history': False
}

# ==============================================================================
# Complex examples
# ==============================================================================

SINGLE_TUMOR_2D = {
    'name': 'single_tumor_2d',
    'description': '2D diffusion with decay and a single tumor region secreting continuously',
    
    # NOTE TIME UNITS ARE IN MINUTES
    'domain_size': (2000.0, 2000.0),
    'grid_points': (int(2000.0/20), int(2000.0/20)),
    'dt': 0.01,
    't_final': 64800,
    
    'diffusion_coefficient': float(1e5), # Diffusion coefficient in μm^2/min (typical for oxygen in tissue)
    'decay_rate': 0.1,
    
    'initial_condition': {
        'type': 'uniform',
        'value': 38.0
    },
    
    'boundary_condition': {
        'type': 'dirichlet',
        'value': 38.0
    },
        
    'bulk': {
        'regions': [
            {
                'type': 'sphere',
                'center': (500, 500),
                'radius': 250.0,
                'net_rate': -10.0, 
                'name': 'tumor_region'
            }
        ]
    },
    
    # No analytical solution for this complex scenario
    'golden_solution': {
        'type': 'numerical_reference',
        'schema_class': ImplicitEulerBCISchema, # Will use ImplicitLODBCSchema by default
        'dx_ref': 10.0, # Finer spatial resolution for reference
        'dt_ref': 0.0005
        # Maybe storing here store_history?
    },

    'store_history': False # Don't store history for this one to save memory, since the reference solution is large and we only care about final state
}

np.random.seed(20)  # for reproducibility
tumours = []
n = 30
radius = 100.0
for i in range(n):
    # while True:
    center = (np.random.uniform(250, 1750), np.random.uniform(250, 1750))
        # Check for overlap with existing tumors
        # if all(np.linalg.norm(np.array(center) - np.array(t['center'])) > 2*radius for t in tumours):
    tumours.append({
        'type': 'sphere',
        'center': center,
        'radius': radius,
        'net_rate': -10.0,
        'name': f'tumor_region_{i+1}'
        })
            # break

MULTIPLE_TUMOR_2D = {
    'name': 'multiple_tumor_2d',
    'description': '2D diffusion with decay and multiple tumor regions secreting continuously',
    
    # NOTE TIME UNITS ARE IN MINUTES
    'domain_size': (2000.0, 2000.0),
    'grid_points': (int(2000.0/20), int(2000.0/20)),
    'dt': 0.01,
    't_final': 64800,
    
    'diffusion_coefficient': float(1e5), # Diffusion coefficient in μm^2/min (typical for oxygen in tissue)
    'decay_rate': 0.1,
    
    'initial_condition': {
        'type': 'uniform',
        'value': 38.0
    },
    
    'boundary_condition': {
        'type': 'dirichlet',
        'value': 38.0
    },
        
    'bulk': {
        'regions': tumours
    },
    
    # No analytical solution for this complex scenario
    'golden_solution': {
        'type': 'numerical_reference',
        'schema_class': None, # Will use ImplicitLODBCSchema by default
        'dx_ref': 10.0, # Finer spatial resolution for reference
        'dt_ref': 0.0001 # Finer time step for reference
    },
    'store_history': False # Don't store history for this one to save memory, since the reference solution is large and we only care about final state
}

def generate_tumor_regions_by_density(
    density: float,
    domain_size: Union[float, Tuple[float, ...]],
    grid_points: Union[int, Tuple[int, ...]],
    tumor_radius: float,
    uptake_linear_rate: float = -10.0,
    secretion_linear_rate: float = 10.0,
    secretion_target_density: float = 38.0,
    secreting_fraction: float = 0.5,
    seed: int = 20,
    min_tumors: int = 0,
) -> List[Dict[str, Any]]:
    """
    Generate random tumor bulk regions from density for 1D/2D/3D scenarios.

    Density units are inferred from dimension (taken from grid_points length):
    - 1D: tumors per mm
    - 2D: tumors per mm^2
    - 3D: tumors per mm^3

        Domain coordinates are assumed to be in micrometers, so conversion uses
        1 mm = 1000 um.

        Bulk model mapping used by this helper:
        - Uptaking tumors are emitted as LinearRegion specs (linear_rate only).
        - Secreting tumors are emitted as TargetRegion specs
            (linear_rate + rho_target).
    """
    if density < 0:
        raise ValueError("density must be non-negative")
    if not 0.0 <= secreting_fraction <= 1.0:
        raise ValueError("secreting_fraction must be in [0, 1]")
    if min_tumors < 0:
        raise ValueError("min_tumors must be non-negative")

    if np.isscalar(grid_points):
        dim = 1
    else:
        dim = len(tuple(grid_points))

    if np.isscalar(domain_size):
        if dim != 1:
            raise ValueError("domain_size must be a tuple for 2D/3D scenarios")
        domain_dims = (float(domain_size),)
    else:
        domain_dims = tuple(float(v) for v in domain_size)
        if len(domain_dims) != dim:
            raise ValueError("domain_size and grid_points dimensions do not match")

    margin = float(tumor_radius)
    for size in domain_dims:
        if size <= 2.0 * margin:
            raise ValueError("Each domain dimension must be larger than 2 * tumor_radius")

    measure_um_dim = float(np.prod(domain_dims))
    measure_mm_dim = measure_um_dim / (1000.0 ** dim)

    n_tumors = int(round(density * measure_mm_dim))
    n_tumors = max(min_tumors, n_tumors)
    n_secreting = int(round(n_tumors * secreting_fraction))
    n_uptake = n_tumors - n_secreting

    rng = np.random.default_rng(seed)
    coords_by_axis = [
        rng.uniform(margin, size - margin, size=n_tumors)
        for size in domain_dims
    ]
    centers = list(zip(*coords_by_axis))

    shuffled_ids = rng.permutation(n_tumors)
    uptake_ids = set(shuffled_ids[:n_uptake])

    regions = []
    for idx, center in enumerate(centers):
        is_uptake = idx in uptake_ids
        region = {
            'type': 'sphere',
            'center': tuple(center),
            'radius': tumor_radius,
            'name': f"{'uptake' if is_uptake else 'secreting'}_tumor_region_{idx + 1}"
        }
        if is_uptake:
            region['linear_rate'] = uptake_linear_rate
        else:
            region['linear_rate'] = secretion_linear_rate
            region['rho_target'] = secretion_target_density
        regions.append(region)

    return regions

# ==============================================================================
# Functions to retrieve built scenarios
# ==============================================================================

def get_default_scenarios() -> List[Dict[str, Any]]:
    """
    Get list of default test scenarios.
    
    Returns
    -------
    list of dict
        Default scenario specifications.
    """
    return [
        GAUSSIAN_PULSE_1D,
        GAUSSIAN_PULSE_2D,
        GAUSSIAN_PULSE_3D
    ]

def get_scenario_by_name(name: str) -> Dict[str, Any]:
    """
    Retrieve a default scenario by name.
    
    Parameters
    ----------
    name : str
        Scenario name.
        
    Returns
    -------
    dict
        Scenario specification.
        
    Raises
    ------
    ValueError
        If scenario name not found.
    """
    scenarios = {
        'gaussian_pulse_1d': GAUSSIAN_PULSE_1D,
        'gaussian_pulse_2d': GAUSSIAN_PULSE_2D,
        'gaussian_pulse_3d': GAUSSIAN_PULSE_3D,
        'step_function_1d': STEP_FUNCTION_1D,
        'step_function_2d': STEP_FUNCTION_2D,
        'steady_state_agent_1d': STEADY_STATE_AGENT_1D,
        'steady_state_agent_2d': STEADY_STATE_AGENT_2D,
        'exponential_decay_1d': EXPONENTIAL_DECAY_1D,
        'sine_decay_1d': SINE_DECAY_1D,
        'convergence_test_1': CONVERGENCE_TEST_1,
        'cosine_diffusion_2d': COSINE_DIFFUSION_2D,
        'single_tumor_2d': SINGLE_TUMOR_2D,
        'multiple_tumor_2d': MULTIPLE_TUMOR_2D,
        'convergence_test_2': CONVERGENCE_TEST_2,
        'convergence_test_2_2d': CONVERGENCE_TEST_2_2D,
        'convergence_test_3': CONVERGENCE_TEST_3,
        'convergence_test_3_2d': CONVERGENCE_TEST_3_2D,
        'convergence_test_4': CONVERGENCE_TEST_4,
        'convergence_test_4_2d': CONVERGENCE_TEST_4_2D,
        'convergence_test_5': CONVERGENCE_TEST_5,
        'convergence_test_5_2d': CONVERGENCE_TEST_5_2D
    }
    
    if name not in scenarios:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(scenarios.keys())}")
    
    return scenarios[name]

