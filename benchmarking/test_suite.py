from abc import ABC, abstractmethod
import time
import numpy as np
from typing import Dict, Any, Union
from diffusion_schemas.utils import DirichletBC, NeumannBC, Agent
from benchmarking.scenarios import build_scenario_components
from benchmarking.error_metrics import compute_all_errors

class SimulationScenario(ABC):
    def __init__(self, name: str, schema_class, params: dict):
        self.name = name
        self.schema_class = schema_class # The class itself, not an instance
        self.params = params
        self.schema = None
        self.results = {}

    def initialize(self):
        """Common setup logic."""
        # 1. Instantiate the Schema using the passed params
        self.schema = self.schema_class(
            domain_size=self.params['L'],
            grid_points=self.params['N'],
            dt=self.params['dt'],
            diffusion_coefficient=self.params['D']
        )
        # 2. Call the custom setup defined by subclasses
        self.setup_physics()

    @abstractmethod
    def setup_physics(self):
        """
        Abstract method where subclasses must define:
        - Initial Conditions (IC)
        - Boundary Conditions (BC)
        - Agents
        """
        pass

    def run(self):
        """Standard execution loop for all tests."""
        print(f"\nRunning {self.name}...")
        start_time = time.time()
        
        # Execute the solve method on the schema
        # history is a list of np arrays
        history = self.schema.solve(
            t_final = self.params['t_final'], 
            store_history = self.params.get('store_history', False)
        )
        
        duration = time.time() - start_time
        
        # Auto-calculate standard metrics
        # Dictionary self.results can be extended with custom data
        final_state = self.schema.get_state()
        self.results = {
            'duration': duration,
            'max_concentration': final_state.max(),
            'min_concentration': final_state.min(),
            'final_state': final_state,
            'history': history
        }
        return self.results
    

# TEST CASES
# Must be defined as subclasses from the SimulationScenario abstract class
# Implementation of the setup_physics method is required to define the specific scenario

class FlexibleSimulation(SimulationScenario):
    def setup_physics(self):

        # Set initial condition (IC)
        if 'ic' in self.params and self.params['ic'] is not None:
            self.schema.set_initial_condition(self.params['ic'])
        else:
            self.schema.set_initial_condition(0.0) # Default to zero if no IC provided
            
        # Set boundary conditions (BC)
        # Assume we rely on operator splitting to handle BCs
        if 'bc' in self.params and self.params['bc'] is not None:
            self.schema.set_boundary_conditions(self.params['bc'])
        else:
            self.schema.set_boundary_conditions(NeumannBC(flux=0.0)) # Default to zero-flux if no BC provided

        # Add agents, if any
        if 'agents' in self.params and self.params['agents'] is not None: 
            for agent in self.params['agents']: 
                self.schema.add_agent(agent)

class TwoAgentInteraction(SimulationScenario): 
    def setup_physics(self): 

        # Set initial condition (IC)
        if 'ic' in self.params and self.params['ic'] is not None:
            self.schema.set_initial_condition(self.params['ic'])
        else:
            self.schema.set_initial_condition(0.0)

        # Set boundary conditions (BC)
        if 'bc' in self.params and self.params['bc'] is not None:
            self.schema.set_boundary_conditions(self.params['bc'])
        else:
            self.schema.set_boundary_conditions(NeumannBC(flux=0.0))

        # Add Agents (hardcoded for demonstration, can be extended to be more flexible)
        agents = []

        if self.schema.ndim == 1:
            agents.append(Agent((0.25,), secretion_rate=10.0))
            agents.append(Agent((0.75,), secretion_rate=15.0))
        elif self.schema.ndim == 2:
            agents.append(Agent((0.25, 0.25), secretion_rate=10.0))
            agents.append(Agent((0.75, 0.75), secretion_rate=15.0))
        elif self.schema.ndim == 3:
            agents.append(Agent((0.25, 0.25, 0.25), secretion_rate=10.0)) 
            agents.append(Agent((0.75, 0.75, 0.75), secretion_rate=15.0))
               
        for agent in agents:
            self.schema.add_agent(agent)


class ValidationScenario(SimulationScenario):
    """
    Scenario with analytical solution for validation.
    
    This extends SimulationScenario to include a golden solution and
    automatic error metric computation.
    """
    
    def __init__(self, name: str, schema_class, params: dict, golden_solution):
        """
        Initialize validation scenario.
        
        Parameters
        ----------
        name : str
            Scenario name.
        schema_class : type
            Schema class to test.
        params : dict
            Simulation parameters.
        golden_solution : GoldenSolution or callable
            Analytical solution for validation.
        """
        super().__init__(name, schema_class, params)
        self.golden_solution = golden_solution
    
    @classmethod
    def from_dict(cls, scenario_dict: Dict[str, Any], schema_class):
        """
        Create ValidationScenario from dictionary specification.
        
        This enables integration with the new scenario framework.
        
        Parameters
        ----------
        scenario_dict : dict
            Scenario specification from scenarios.create_scenario() or defaults.
        schema_class : type
            Schema class to test.
            
        Returns
        -------
        ValidationScenario
            Initialized scenario instance.
        """
        # Build scenario components
        built = build_scenario_components(scenario_dict)
        
        # Convert to params format expected by SimulationScenario
        params = {
            'L': scenario_dict['domain_size'],
            'N': scenario_dict['grid_points'],
            'dt': scenario_dict['dt'],
            'D': scenario_dict['diffusion_coefficient'],
            't_final': scenario_dict['t_final'],
            'ic': built['initial_condition'],
            'bc': built['boundary_condition'],
            'agents': built['agents'],
            'store_history': True
        }
        
        # Add decay rate if present
        if scenario_dict.get('decay_rate', 0.0) != 0.0:
            params['decay_rate'] = scenario_dict['decay_rate']
        
        # Create instance
        instance = cls(
            name=scenario_dict['name'],
            schema_class=schema_class,
            params=params,
            golden_solution=built['golden_solution']
        )
        
        return instance
    
    def initialize(self):
        """Initialize schema with decay rate support."""
        # Create schema with decay rate if provided
        init_params = {
            'domain_size': self.params['L'],
            'grid_points': self.params['N'],
            'dt': self.params['dt'],
            'diffusion_coefficient': self.params['D']
        }
        
        if 'decay_rate' in self.params:
            init_params['decay_rate'] = self.params['decay_rate']
        
        self.schema = self.schema_class(**init_params)
        self.setup_physics()
    
    def setup_physics(self):
        """Setup physics from params (IC, BC, agents)."""
        # Set initial condition
        if 'ic' in self.params and self.params['ic'] is not None:
            self.schema.set_initial_condition(self.params['ic'])
        else:
            self.schema.set_initial_condition(0.0)
        
        # Set boundary conditions
        if 'bc' in self.params and self.params['bc'] is not None:
            self.schema.set_boundary_conditions(self.params['bc'])
        else:
            self.schema.set_boundary_conditions(NeumannBC(flux=0.0))
        
        # Add agents
        if 'agents' in self.params and self.params['agents'] is not None:
            for agent in self.params['agents']:
                self.schema.add_agent(agent)
    
    def run(self):
        """Run simulation and compute validation errors."""
        # Run base simulation
        results = super().run()
        
        # Compute errors against golden solution
        final_state = results['final_state']
        coordinates = self.schema._create_coordinate_grids()
        t_final = self.params['t_final']
        
        # Evaluate analytical solution
        if hasattr(self.golden_solution, 'evaluate'):
            analytical = self.golden_solution.evaluate(coordinates, t_final)
        else:
            analytical = self.golden_solution(coordinates, t_final)
        
        # Compute initial mass for conservation check
        initial_mass = None
        if isinstance(self.params.get('bc'), NeumannBC):
            # Only check mass conservation for zero-flux BC
            if callable(self.params['bc'].flux):
                flux = self.params['bc'].flux(0)
            else:
                flux = self.params['bc'].flux
            
            if flux == 0.0:
                # Get initial state (re-evaluate IC)
                temp_schema = self.schema_class(
                    domain_size=self.params['L'],
                    grid_points=self.params['N'],
                    dt=self.params['dt'],
                    diffusion_coefficient=self.params['D']
                )
                temp_schema.set_initial_condition(self.params['ic'])
                initial_state = temp_schema.get_state()
                initial_mass = np.sum(initial_state)
                if self.schema.dx is not None:
                    if isinstance(self.schema.dx, (list, tuple)):
                        dV = np.prod(self.schema.dx)
                    else:
                        dV = self.schema.dx ** self.schema.ndim
                    initial_mass *= dV
        
        # Compute all errors
        errors = compute_all_errors(
            final_state, analytical,
            dx=self.schema.dx,
            initial_mass=initial_mass
        )
        
        # Add to results
        results['errors'] = errors
        results['analytical_solution'] = analytical
        
        self.results = results
        return results


# Maintain backward compatibility with existing code
SimulationScenario.from_dict = classmethod(
    lambda cls, scenario_dict, schema_class: 
    ValidationScenario.from_dict(scenario_dict, schema_class)
)
