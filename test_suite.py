from abc import ABC, abstractmethod
import time
from diffusion_schemas.utils import DirichletBC, NeumannBC, Agent

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


