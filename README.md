# Diffusion Schemas

A Python framework for solving diffusion equations (heat equation) with agent-based sources using multiple numerical methods in 1D, 2D, and 3D.

## Features

- **Multiple numerical methods**: Explicit Euler, Implicit Euler, and Crank-Nicolson schemes
- **Multi-dimensional support**: 1D, 2D, and 3D spatial domains
- **Flexible boundary conditions**: Dirichlet, Neumann, Periodic, and Robin (mixed) BCs
- **Agent-based sources**: Substrate-secreting agents with configurable positions and rates
- **Rich initial conditions**: Gaussian, uniform, step function, checkerboard, spherical, and more
- **Configurable parameters**: Diffusion coefficient, decay rate, time step, grid resolution

## Installation

### From source

```bash
git clone git@github.com:bsc-life/agent-based-prototyping.git
cd agent-based-prototyping
pip install -e .
```

### Development installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic 1D diffusion

```python
import numpy as np
import matplotlib.pyplot as plt
from diffusion_schemas import ExplicitEulerSchema
from diffusion_schemas.utils import gaussian

# Create a 1D diffusion schema
schema = ExplicitEulerSchema(
    domain_size=1.0,      # Domain: [0, 1]
    grid_points=100,      # 100 grid points
    dt=0.0001,           # Time step
    diffusion_coefficient=0.1,
    decay_rate=0.0
)

# Set initial condition: Gaussian at center
ic = gaussian(center=0.5, amplitude=1.0, width=0.05)
schema.set_initial_condition(ic)

# Solve for 0.1 time units
schema.solve(t_final=0.1)

# Get and plot result
u = schema.get_state()
x = np.linspace(0, 1, 100)
plt.plot(x, u)
plt.show()
```

### 2D diffusion with agents

```python
from diffusion_schemas import CrankNicolsonSchema
from diffusion_schemas.utils import Agent, DirichletBC

# Create 2D schema
schema = CrankNicolsonSchema(
    domain_size=(1.0, 1.0),
    grid_points=(50, 50),
    dt=0.001,
    diffusion_coefficient=0.05
)

# Set zero initial condition
schema.set_initial_condition(0.0)

# Add substrate-secreting agents
agent1 = Agent(position=(0.3, 0.3), secretion_rate=10.0, kernel_width=0.05)
agent2 = Agent(position=(0.7, 0.7), secretion_rate=5.0, kernel_width=0.05)
schema.add_agent(agent1)
schema.add_agent(agent2)

# Set boundary conditions (fixed at zero)
schema.set_boundary_conditions(DirichletBC(value=0.0))

# Solve
history = schema.solve(t_final=1.0, store_history=True)

# Visualize
import matplotlib.pyplot as plt
plt.imshow(schema.get_state(), origin='lower', extent=[0, 1, 0, 1])
plt.colorbar(label='Concentration')
plt.show()
```

### Comparing methods

```python
from diffusion_schemas import ExplicitEulerSchema, ImplicitEulerSchema, CrankNicolsonSchema

# Create same problem with different methods
common_params = {
    'domain_size': 1.0,
    'grid_points': 100,
    'dt': 0.001,
    'diffusion_coefficient': 0.1
}

explicit = ExplicitEulerSchema(**common_params, check_stability=False)
implicit = ImplicitEulerSchema(**common_params)
crank_nicolson = CrankNicolsonSchema(**common_params)

# Set identical initial conditions
ic = gaussian(center=0.5, amplitude=1.0, width=0.1)
for schema in [explicit, implicit, crank_nicolson]:
    schema.set_initial_condition(ic)

# Solve and compare
for schema in [explicit, implicit, crank_nicolson]:
    schema.solve(t_final=0.5)
    # Plot or analyze results...
```

## Mathematical Background

The framework solves the diffusion equation with decay and sources:

$$\frac{\partial u}{\partial t} = D\nabla^2 u - \lambda u + S(x, t)$$

where:
- $u$ is the concentration/temperature field
- $D$ is the diffusion coefficient
- $\lambda$ is the decay rate
- $S(x, t)$ is the source term from agents

### Numerical Methods

#### Explicit Euler (FTCS)
- **Stability**: Conditionally stable, requires $\Delta t \leq \Delta x^2 / (2dD)$
- **Accuracy**: First-order in time, second-order in space
- **Performance**: Fast per step, small time steps required

#### Implicit Euler
- **Stability**: Unconditionally stable
- **Accuracy**: First-order in time, second-order in space
- **Performance**: Requires solving linear system, allows larger time steps

#### Crank-Nicolson
- **Stability**: Unconditionally stable
- **Accuracy**: Second-order in time and space
- **Performance**: Best accuracy, requires solving linear system

## API Reference

### Schema Classes

All schema classes inherit from the abstract `Schema` base class and provide:

- `set_initial_condition(ic)`: Set initial concentration field
- `set_diffusion_coefficient(D)`: Set diffusion coefficient
- `set_decay_rate(λ)`: Set decay rate
- `set_boundary_conditions(bc)`: Set boundary conditions
- `add_agent(agent)`: Add substrate-secreting agent
- `step()`: Perform single time step
- `solve(t_final, store_history=False)`: Solve to final time
- `get_state()`: Get current concentration field
- `reset()`: Reset to zero initial condition

### Boundary Conditions

- `DirichletBC(value)`: Fixed value at boundaries
- `NeumannBC(flux)`: Fixed flux at boundaries (zero flux = insulating)
- `PeriodicBC()`: Periodic wrapping
- `RobinBC(alpha, beta, gamma)`: Mixed condition $\alpha u + \beta \frac{\partial u}{\partial n} = \gamma$

### Agents

```python
Agent(
    position=(x, y, z),      # Agent location
    secretion_rate=1.0,      # Secretion rate (or callable f(t))
    kernel_width=None,       # Gaussian smoothing width (None = point source)
    name="Agent_1"          # Optional name
)
```

### Initial Conditions

Helper functions in `diffusion_schemas.utils`:

- `gaussian(center, amplitude, width)`: Gaussian distribution
- `uniform(value)`: Constant value
- `step_function(position, value_left, value_right)`: Discontinuous step
- `checkerboard(spacing, value_on, value_off)`: 2D pattern
- `sphere(center, radius, value_inside, value_outside)`: Spherical/circular region
- `radial_gradient(center, max_value, max_radius, decay_type)`: Radial decay
- `sum_conditions(*conditions)`: Combine multiple conditions

## Examples

See the `examples/` directory for complete working examples:

- `example_1d_diffusion.py`: Basic 1D diffusion
- `example_2d_agents.py`: 2D with multiple agents
- `example_method_comparison.py`: Compare numerical methods
- `example_3d_sphere.py`: 3D diffusion from sphere

## Running Tests

```bash
pytest tests/
```

With coverage:

```bash
pytest tests/ --cov=diffusion_schemas --cov-report=html
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
