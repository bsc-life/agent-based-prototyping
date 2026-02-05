"""Tests for the base Schema class."""

import pytest
import numpy as np
from diffusion_schemas.base import Schema


class DummySchema(Schema):
    """Dummy implementation of Schema for testing."""
    
    def step(self):
        """Simple forward Euler for testing."""
        self.state += self.dt * self.diffusion_coefficient
        self.t += self.dt


class TestSchemaInitialization:
    """Test Schema initialization and parameter setting."""
    
    def test_1d_initialization(self):
        """Test 1D schema initialization."""
        schema = DummySchema(
            domain_size=1.0,
            grid_points=100,
            dt=0.01,
            diffusion_coefficient=0.1,
            decay_rate=0.05
        )
        
        assert schema.ndim == 1
        assert schema.domain_size == (1.0,)
        assert schema.grid_points == (100,)
        assert schema.dt == 0.01
        assert schema.diffusion_coefficient == 0.1
        assert schema.decay_rate == 0.05
        assert schema.t == 0.0
        assert schema.state.shape == (100,)
    
    def test_2d_initialization(self):
        """Test 2D schema initialization."""
        schema = DummySchema(
            domain_size=(1.0, 2.0),
            grid_points=(50, 100),
            dt=0.001,
        )
        
        assert schema.ndim == 2
        assert schema.domain_size == (1.0, 2.0)
        assert schema.grid_points == (50, 100)
        assert schema.state.shape == (50, 100)
    
    def test_3d_initialization(self):
        """Test 3D schema initialization."""
        schema = DummySchema(
            domain_size=(1.0, 1.0, 1.0),
            grid_points=(20, 20, 20),
            dt=0.01,
        )
        
        assert schema.ndim == 3
        assert schema.state.shape == (20, 20, 20)
    
    def test_grid_spacing(self):
        """Test grid spacing calculation."""
        schema = DummySchema(
            domain_size=(1.0, 2.0),
            grid_points=(11, 21),
            dt=0.01,
        )
        
        assert schema.dx[0] == pytest.approx(0.1)  # 1.0 / (11-1)
        assert schema.dx[1] == pytest.approx(0.1)  # 2.0 / (21-1)
    
    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        with pytest.raises(ValueError, match="Mismatch"):
            DummySchema(
                domain_size=(1.0, 2.0),
                grid_points=(50,),  # Wrong dimension
                dt=0.01,
            )


class TestInitialConditions:
    """Test setting initial conditions."""
    
    def test_array_initial_condition(self):
        """Test setting IC from array."""
        schema = DummySchema(
            domain_size=1.0,
            grid_points=10,
            dt=0.01,
        )
        
        ic = np.arange(10)
        schema.set_initial_condition(ic)
        
        assert np.allclose(schema.state, ic)
        assert schema.t == 0.0
    
    def test_scalar_initial_condition(self):
        """Test setting IC from scalar."""
        schema = DummySchema(
            domain_size=1.0,
            grid_points=10,
            dt=0.01,
        )
        
        schema.set_initial_condition(5.0)
        
        assert np.allclose(schema.state, 5.0)
    
    def test_callable_initial_condition(self):
        """Test setting IC from function."""
        schema = DummySchema(
            domain_size=1.0,
            grid_points=11,
            dt=0.01,
        )
        
        def ic_func(x):
            return x**2
        
        schema.set_initial_condition(ic_func)
        
        x = np.linspace(0, 1, 11)
        expected = x**2
        
        assert np.allclose(schema.state, expected)
    
    def test_wrong_shape_initial_condition(self):
        """Test that wrong shape raises error."""
        schema = DummySchema(
            domain_size=1.0,
            grid_points=10,
            dt=0.01,
        )
        
        wrong_ic = np.ones(20)  # Wrong size
        
        with pytest.raises(ValueError, match="shape"):
            schema.set_initial_condition(wrong_ic)


class TestParameters:
    """Test parameter setters."""
    
    def test_set_diffusion_coefficient(self):
        """Test setting diffusion coefficient."""
        schema = DummySchema(
            domain_size=1.0,
            grid_points=10,
            dt=0.01,
        )
        
        schema.set_diffusion_coefficient(0.5)
        assert schema.diffusion_coefficient == 0.5
    
    def test_negative_diffusion_coefficient(self):
        """Test that negative D raises error."""
        schema = DummySchema(
            domain_size=1.0,
            grid_points=10,
            dt=0.01,
        )
        
        with pytest.raises(ValueError, match="non-negative"):
            schema.set_diffusion_coefficient(-0.1)
    
    def test_set_decay_rate(self):
        """Test setting decay rate."""
        schema = DummySchema(
            domain_size=1.0,
            grid_points=10,
            dt=0.01,
        )
        
        schema.set_decay_rate(0.1)
        assert schema.decay_rate == 0.1
    
    def test_negative_decay_rate(self):
        """Test that negative decay raises error."""
        schema = DummySchema(
            domain_size=1.0,
            grid_points=10,
            dt=0.01,
        )
        
        with pytest.raises(ValueError, match="non-negative"):
            schema.set_decay_rate(-0.1)


class TestSolve:
    """Test the solve method."""
    
    def test_solve_basic(self):
        """Test basic solve functionality."""
        schema = DummySchema(
            domain_size=1.0,
            grid_points=10,
            dt=0.1,
            diffusion_coefficient=1.0,
        )
        
        schema.set_initial_condition(0.0)
        schema.solve(t_final=1.0)
        
        assert schema.t == pytest.approx(1.0)
    
    def test_solve_with_history(self):
        """Test solve with history storage."""
        schema = DummySchema(
            domain_size=1.0,
            grid_points=10,
            dt=0.1,
        )
        
        schema.set_initial_condition(0.0)
        history = schema.solve(t_final=0.5, store_history=True)
        
        assert len(history) == 6  # Initial + 5 steps
        assert schema.t == pytest.approx(0.5)
    
    def test_solve_invalid_time(self):
        """Test that solving backwards raises error."""
        schema = DummySchema(
            domain_size=1.0,
            grid_points=10,
            dt=0.1,
        )
        
        schema.set_initial_condition(0.0)
        schema.t = 1.0
        
        with pytest.raises(ValueError, match="must be greater"):
            schema.solve(t_final=0.5)


class TestReset:
    """Test reset functionality."""
    
    def test_reset(self):
        """Test that reset works correctly."""
        schema = DummySchema(
            domain_size=1.0,
            grid_points=10,
            dt=0.1,
        )
        
        schema.set_initial_condition(5.0)
        schema.solve(t_final=1.0)
        
        # State should have changed
        assert not np.allclose(schema.state, 5.0)
        assert schema.t > 0
        
        # Reset
        schema.reset()
        
        assert np.allclose(schema.state, 0.0)
        assert schema.t == 0.0


class TestGetState:
    """Test get_state method."""
    
    def test_get_state_returns_copy(self):
        """Test that get_state returns a copy, not reference."""
        schema = DummySchema(
            domain_size=1.0,
            grid_points=10,
            dt=0.1,
        )
        
        schema.set_initial_condition(1.0)
        state = schema.get_state()
        
        # Modify returned state
        state[:] = 999.0
        
        # Original should be unchanged
        assert np.allclose(schema.state, 1.0)
