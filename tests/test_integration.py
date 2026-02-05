"""Integration tests combining all components."""

import pytest
import numpy as np
from diffusion_schemas import ExplicitEulerSchema, ImplicitEulerSchema, CrankNicolsonSchema
from diffusion_schemas.utils import (
    gaussian, uniform, Agent, 
    DirichletBC, NeumannBC
)


class TestAgentDiffusion:
    """Test diffusion with agents."""
    
    def test_1d_single_agent(self):
        """Test 1D diffusion with a single agent."""
        schema = CrankNicolsonSchema(
            domain_size=1.0,
            grid_points=50,
            dt=0.001,
            diffusion_coefficient=0.1,
            decay_rate=0.1
        )
        
        # Zero initial condition
        schema.set_initial_condition(0.0)
        
        # Add agent at center
        agent = Agent(position=(0.5,), secretion_rate=10.0, kernel_width=0.05)
        schema.add_agent(agent)
        
        # Solve to steady state
        schema.solve(t_final=5.0)
        
        u = schema.get_state()
        
        # Concentration should be highest near agent
        center_idx = 25
        assert u[center_idx] > u[0]
        assert u[center_idx] > u[-1]
        
        # Should have reached some steady value
        assert np.max(u) > 0
    
    def test_2d_multiple_agents(self):
        """Test 2D diffusion with multiple agents."""
        schema = ImplicitEulerSchema(
            domain_size=(1.0, 1.0),
            grid_points=(30, 30),
            dt=0.01,
            diffusion_coefficient=0.05,
            decay_rate=0.1
        )
        
        schema.set_initial_condition(0.0)
        
        # Add multiple agents
        agents = [
            Agent(position=(0.3, 0.3), secretion_rate=5.0, kernel_width=0.05),
            Agent(position=(0.7, 0.7), secretion_rate=8.0, kernel_width=0.05),
        ]
        
        for agent in agents:
            schema.add_agent(agent)
        
        # Set boundary conditions
        schema.set_boundary_conditions(DirichletBC(value=0.0))
        
        # Solve
        schema.solve(t_final=2.0)
        
        u = schema.get_state()
        
        # Should have concentration peaks near agents
        assert np.max(u) > 0
        assert np.mean(u) > 0
    
    def test_agent_clearing(self):
        """Test clearing agents."""
        schema = ExplicitEulerSchema(
            domain_size=1.0,
            grid_points=50,
            dt=0.0001,
            diffusion_coefficient=0.1
        )
        
        schema.set_initial_condition(0.0)
        
        # Add agent
        agent = Agent(position=(0.5,), secretion_rate=10.0)
        schema.add_agent(agent)
        
        # Solve
        schema.solve(t_final=0.01)
        u_with_agent = schema.get_state()
        
        # Clear agents and reset
        schema.clear_agents()
        schema.reset()
        
        # Solve again without agent
        schema.solve(t_final=0.01)
        u_without_agent = schema.get_state()
        
        # Should be different
        assert not np.allclose(u_with_agent, u_without_agent)
        # Without agent should remain zero
        assert np.allclose(u_without_agent, 0.0)


class TestBoundaryConditionIntegration:
    """Test integration with different boundary conditions."""
    
    def test_dirichlet_vs_neumann(self):
        """Test that Dirichlet and Neumann BCs give different results."""
        ic = gaussian(center=0.5, amplitude=1.0, width=0.1)
        
        # Dirichlet BC
        schema_dir = ExplicitEulerSchema(
            domain_size=1.0,
            grid_points=50,
            dt=0.0001,
            diffusion_coefficient=0.1
        )
        schema_dir.set_initial_condition(ic)
        schema_dir.set_boundary_conditions(DirichletBC(value=0.0))
        schema_dir.solve(t_final=0.1)
        u_dir = schema_dir.get_state()
        
        # Neumann BC
        schema_neu = ExplicitEulerSchema(
            domain_size=1.0,
            grid_points=50,
            dt=0.0001,
            diffusion_coefficient=0.1
        )
        schema_neu.set_initial_condition(ic)
        schema_neu.set_boundary_conditions(NeumannBC(flux=0.0))
        schema_neu.solve(t_final=0.1)
        u_neu = schema_neu.get_state()
        
        # Results should be different
        assert not np.allclose(u_dir, u_neu)
        
        # Dirichlet should have zero boundaries
        assert u_dir[0] == pytest.approx(0.0)
        assert u_dir[-1] == pytest.approx(0.0)
        
        # Neumann should have non-zero boundaries
        assert u_neu[0] > 0
        assert u_neu[-1] > 0


class TestComplexScenarios:
    """Test complex scenarios combining multiple features."""
    
    def test_agent_decay_equilibrium(self):
        """Test that agent secretion and decay reach equilibrium."""
        schema = CrankNicolsonSchema(
            domain_size=1.0,
            grid_points=50,
            dt=0.01,
            diffusion_coefficient=0.1,
            decay_rate=0.5
        )
        
        schema.set_initial_condition(0.0)
        
        agent = Agent(position=(0.5,), secretion_rate=10.0, kernel_width=0.1)
        schema.add_agent(agent)
        
        schema.set_boundary_conditions(DirichletBC(value=0.0))
        
        # Solve to steady state
        schema.solve(t_final=10.0)
        u_final = schema.get_state()
        
        # Take one more step - should be at equilibrium
        schema.step()
        u_next = schema.get_state()
        
        # Should be very close (at equilibrium)
        assert np.allclose(u_final, u_next, rtol=0.01)
    
    def test_time_dependent_source(self):
        """Test agent with time-dependent secretion rate."""
        schema = ImplicitEulerSchema(
            domain_size=1.0,
            grid_points=50,
            dt=0.01,
            diffusion_coefficient=0.1
        )
        
        schema.set_initial_condition(0.0)
        
        # Time-dependent secretion: ramps up then down
        def secretion(t):
            if t < 1.0:
                return 10.0 * t
            else:
                return 10.0 * (2.0 - t)
        
        agent = Agent(position=(0.5,), secretion_rate=secretion, kernel_width=0.1)
        schema.add_agent(agent)
        
        # Solve
        history = schema.solve(t_final=2.0, store_history=True)
        
        # Peak concentration should occur around t=1.0
        max_concentrations = [np.max(state) for state in history]
        peak_idx = np.argmax(max_concentrations)
        peak_time = peak_idx * schema.dt
        
        # Peak should be near t=1.0 (allowing for diffusion lag)
        assert 0.5 < peak_time < 1.5
    
    def test_3d_with_all_features(self):
        """Test 3D diffusion with agents, decay, and boundary conditions."""
        schema = ImplicitEulerSchema(
            domain_size=(1.0, 1.0, 1.0),
            grid_points=(15, 15, 15),
            dt=0.01,
            diffusion_coefficient=0.05,
            decay_rate=0.1
        )
        
        schema.set_initial_condition(0.0)
        
        # Add agents
        agents = [
            Agent(position=(0.3, 0.5, 0.5), secretion_rate=5.0, kernel_width=0.1),
            Agent(position=(0.7, 0.5, 0.5), secretion_rate=5.0, kernel_width=0.1),
        ]
        
        for agent in agents:
            schema.add_agent(agent)
        
        # Solve
        schema.solve(t_final=1.0)
        
        u = schema.get_state()
        
        # Should have reasonable solution
        assert np.all(np.isfinite(u))
        assert np.max(u) > 0
        assert np.min(u) >= 0


class TestNumericalProperties:
    """Test numerical properties like conservation and positivity."""
    
    def test_positivity_preservation(self):
        """Test that non-negative IC stays non-negative."""
        schema = CrankNicolsonSchema(
            domain_size=1.0,
            grid_points=50,
            dt=0.01,
            diffusion_coefficient=0.1
        )
        
        # Non-negative IC
        ic = gaussian(center=0.5, amplitude=1.0, width=0.1)
        schema.set_initial_condition(ic)
        
        schema.solve(t_final=1.0)
        
        # Solution should remain non-negative
        assert np.all(schema.state >= -1e-10)  # Allow small numerical error
    
    def test_maximum_principle(self):
        """Test discrete maximum principle for Dirichlet problem."""
        schema = ImplicitEulerSchema(
            domain_size=1.0,
            grid_points=50,
            dt=0.01,
            diffusion_coefficient=0.1,
            decay_rate=0.0  # No decay
        )
        
        ic = gaussian(center=0.5, amplitude=1.0, width=0.1)
        schema.set_initial_condition(ic)
        
        initial_max = np.max(schema.state)
        
        # Dirichlet BC at 0
        schema.set_boundary_conditions(DirichletBC(value=0.0))
        
        schema.solve(t_final=1.0)
        
        # Maximum should not increase (may decrease due to diffusion to boundaries)
        final_max = np.max(schema.state)
        assert final_max <= initial_max + 1e-10
