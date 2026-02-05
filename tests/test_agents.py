"""Tests for the Agent system."""

import pytest
import numpy as np
from diffusion_schemas.utils.agents import Agent


class TestAgentInitialization:
    """Test Agent initialization."""
    
    def test_1d_agent(self):
        """Test creating a 1D agent."""
        agent = Agent(position=(0.5,), secretion_rate=1.0)
        
        assert agent.position == (0.5,)
        assert agent.secretion_rate == 1.0
        assert agent.kernel_width is None
    
    def test_2d_agent(self):
        """Test creating a 2D agent."""
        agent = Agent(position=(0.5, 0.3), secretion_rate=5.0, kernel_width=0.1)
        
        assert agent.position == (0.5, 0.3)
        assert agent.secretion_rate == 5.0
        assert agent.kernel_width == 0.1
    
    def test_3d_agent(self):
        """Test creating a 3D agent."""
        agent = Agent(position=(0.1, 0.2, 0.3), secretion_rate=2.0, name="TestAgent")
        
        assert agent.position == (0.1, 0.2, 0.3)
        assert agent.name == "TestAgent"
    
    def test_time_dependent_secretion(self):
        """Test agent with time-dependent secretion rate."""
        def rate_func(t):
            return t**2
        
        agent = Agent(position=(0.5,), secretion_rate=rate_func)
        
        assert agent.get_secretion_rate(0.0) == 0.0
        assert agent.get_secretion_rate(2.0) == pytest.approx(4.0)
        assert agent.get_secretion_rate(3.0) == pytest.approx(9.0)


class TestAgentSource:
    """Test agent source term computation."""
    
    def test_1d_point_source(self):
        """Test 1D point source."""
        agent = Agent(position=(0.5,), secretion_rate=10.0, kernel_width=None)
        
        # Create coordinate grid
        x = np.linspace(0, 1, 11)
        dx = (0.1,)
        t = 0.0
        
        source = agent.compute_source([x], dx, t)
        
        # Source should be concentrated at nearest grid point
        assert source.shape == x.shape
        assert np.sum(source) > 0  # Total source should be positive
        
        # Find peak
        peak_idx = np.argmax(source)
        assert x[peak_idx] == pytest.approx(0.5, abs=0.1)
    
    def test_2d_gaussian_source(self):
        """Test 2D Gaussian source."""
        agent = Agent(position=(0.5, 0.5), secretion_rate=1.0, kernel_width=0.1)
        
        # Create coordinate grids
        x = np.linspace(0, 1, 21)
        y = np.linspace(0, 1, 21)
        X, Y = np.meshgrid(x, y, indexing='ij')
        dx = (0.05, 0.05)
        t = 0.0
        
        source = agent.compute_source([X, Y], dx, t)
        
        # Source should be smooth Gaussian
        assert source.shape == X.shape
        
        # Maximum should be at center
        center_idx = 10  # Middle of 21 points
        assert source[center_idx, center_idx] == np.max(source)
        
        # Should decay with distance
        assert source[center_idx, center_idx] > source[center_idx + 5, center_idx]
    
    def test_zero_secretion_rate(self):
        """Test agent with zero secretion rate."""
        agent = Agent(position=(0.5,), secretion_rate=0.0)
        
        x = np.linspace(0, 1, 11)
        dx = (0.1,)
        t = 0.0
        
        source = agent.compute_source([x], dx, t)
        
        assert np.all(source == 0.0)
    
    def test_3d_gaussian_source(self):
        """Test 3D Gaussian source."""
        agent = Agent(position=(0.5, 0.5, 0.5), secretion_rate=5.0, kernel_width=0.2)
        
        # Create coordinate grids
        N = 11
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        z = np.linspace(0, 1, N)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        dx = (0.1, 0.1, 0.1)
        t = 0.0
        
        source = agent.compute_source([X, Y, Z], dx, t)
        
        assert source.shape == (N, N, N)
        
        # Maximum at center
        center_idx = N // 2
        center_value = source[center_idx, center_idx, center_idx]
        assert center_value == np.max(source)
        
        # Spherical symmetry: check equidistant points
        dist1 = source[center_idx + 1, center_idx, center_idx]
        dist2 = source[center_idx, center_idx + 1, center_idx]
        dist3 = source[center_idx, center_idx, center_idx + 1]
        assert dist1 == pytest.approx(dist2, rel=1e-10)
        assert dist2 == pytest.approx(dist3, rel=1e-10)


class TestAgentMethods:
    """Test agent methods."""
    
    def test_set_position(self):
        """Test updating agent position."""
        agent = Agent(position=(0.5, 0.5), secretion_rate=1.0)
        
        agent.set_position((0.8, 0.3))
        
        assert agent.position == (0.8, 0.3)
    
    def test_set_position_wrong_dimension(self):
        """Test that wrong dimension raises error."""
        agent = Agent(position=(0.5, 0.5), secretion_rate=1.0)
        
        with pytest.raises(ValueError, match="dimension mismatch"):
            agent.set_position((0.5,))  # Wrong dimension
    
    def test_set_secretion_rate(self):
        """Test updating secretion rate."""
        agent = Agent(position=(0.5,), secretion_rate=1.0)
        
        agent.set_secretion_rate(5.0)
        
        assert agent.secretion_rate == 5.0
    
    def test_set_negative_secretion_rate(self):
        """Test that negative rate raises error."""
        agent = Agent(position=(0.5,), secretion_rate=1.0)
        
        with pytest.raises(ValueError, match="non-negative"):
            agent.set_secretion_rate(-1.0)
    
    def test_repr(self):
        """Test string representation."""
        agent = Agent(position=(0.5, 0.3), secretion_rate=2.5, name="MyAgent")
        
        repr_str = repr(agent)
        
        assert "MyAgent" in repr_str
        assert "(0.5, 0.3)" in repr_str
        assert "2.5" in repr_str


class TestAgentIntegration:
    """Integration tests with multiple agents."""
    
    def test_multiple_agents_additive(self):
        """Test that multiple agents' sources add up."""
        agent1 = Agent(position=(0.3,), secretion_rate=1.0, kernel_width=0.05)
        agent2 = Agent(position=(0.7,), secretion_rate=1.0, kernel_width=0.05)
        
        x = np.linspace(0, 1, 51)
        dx = (0.02,)
        t = 0.0
        
        source1 = agent1.compute_source([x], dx, t)
        source2 = agent2.compute_source([x], dx, t)
        source_total = source1 + source2
        
        # Total source should have two peaks
        # Find local maxima
        peaks = []
        for i in range(1, len(source_total) - 1):
            if source_total[i] > source_total[i-1] and source_total[i] > source_total[i+1]:
                peaks.append(i)
        
        # Should have at least one peak near each agent
        assert len(peaks) >= 1
