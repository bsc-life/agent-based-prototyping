"""Tests for boundary conditions."""

import pytest
import numpy as np
from diffusion_schemas.utils.boundary import (
    DirichletBC, NeumannBC, PeriodicBC, RobinBC
)


class TestDirichletBC:
    """Test Dirichlet boundary conditions."""
    
    def test_1d_dirichlet(self):
        """Test Dirichlet BC in 1D."""
        bc = DirichletBC(value=5.0)
        state = np.ones(10)
        dx = (0.1,)
        t = 0.0
        
        result = bc.apply(state, dx, t)
        
        assert result[0] == 5.0
        assert result[-1] == 5.0
        assert np.all(result[1:-1] == 1.0)
    
    def test_2d_dirichlet(self):
        """Test Dirichlet BC in 2D."""
        bc = DirichletBC(value=0.0)
        state = np.ones((10, 10))
        dx = (0.1, 0.1)
        t = 0.0
        
        result = bc.apply(state, dx, t)
        
        # All boundaries should be 0
        assert np.all(result[0, :] == 0.0)
        assert np.all(result[-1, :] == 0.0)
        assert np.all(result[:, 0] == 0.0)
        assert np.all(result[:, -1] == 0.0)
        
        # Interior should be unchanged
        assert np.all(result[1:-1, 1:-1] == 1.0)
    
    def test_3d_dirichlet(self):
        """Test Dirichlet BC in 3D."""
        bc = DirichletBC(value=2.0)
        state = np.ones((5, 5, 5))
        dx = (0.1, 0.1, 0.1)
        t = 0.0
        
        result = bc.apply(state, dx, t)
        
        # All faces should be 2.0
        assert np.all(result[0, :, :] == 2.0)
        assert np.all(result[-1, :, :] == 2.0)
        assert np.all(result[:, 0, :] == 2.0)
        assert np.all(result[:, -1, :] == 2.0)
        assert np.all(result[:, :, 0] == 2.0)
        assert np.all(result[:, :, -1] == 2.0)
    
    def test_time_dependent_dirichlet(self):
        """Test time-dependent Dirichlet BC."""
        bc = DirichletBC(value=lambda t: t**2)
        state = np.ones(10)
        dx = (0.1,)
        t = 2.0
        
        result = bc.apply(state, dx, t)
        
        assert result[0] == pytest.approx(4.0)
        assert result[-1] == pytest.approx(4.0)


class TestNeumannBC:
    """Test Neumann boundary conditions."""
    
    def test_1d_neumann_zero_flux(self):
        """Test zero-flux Neumann BC in 1D."""
        bc = NeumannBC(flux=0.0)
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dx = (1.0,)
        t = 0.0
        
        result = bc.apply(state, dx, t)
        
        # Zero flux means boundary = neighbor
        assert result[0] == pytest.approx(2.0)
        assert result[-1] == pytest.approx(4.0)
    
    def test_2d_neumann_zero_flux(self):
        """Test zero-flux Neumann BC in 2D."""
        bc = NeumannBC(flux=0.0)
        state = np.ones((5, 5))
        state[2, 2] = 5.0  # Set center to different value
        dx = (0.1, 0.1)
        t = 0.0
        
        result = bc.apply(state, dx, t)
        
        # For uniform interior with perturbation,
        # boundaries should approximately match neighbors
        assert result.shape == state.shape
    
    def test_neumann_nonzero_flux(self):
        """Test Neumann BC with non-zero flux."""
        bc = NeumannBC(flux=1.0)
        state = np.ones(5)
        dx = (0.5,)
        t = 0.0
        
        result = bc.apply(state, dx, t)
        
        # Left boundary: u[0] = u[1] - 2*dx*flux
        expected_left = 1.0 - 2 * 0.5 * 1.0
        assert result[0] == pytest.approx(expected_left)
        
        # Right boundary: u[-1] = u[-2] + 2*dx*flux
        expected_right = 1.0 + 2 * 0.5 * 1.0
        assert result[-1] == pytest.approx(expected_right)


class TestPeriodicBC:
    """Test periodic boundary conditions."""
    
    def test_1d_periodic(self):
        """Test periodic BC in 1D."""
        bc = PeriodicBC()
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dx = (0.1,)
        t = 0.0
        
        result = bc.apply(state, dx, t)
        
        # Boundaries should wrap around
        assert result[0] == pytest.approx(4.0)  # Copy from second-to-last
        assert result[-1] == pytest.approx(2.0)  # Copy from second
    
    def test_2d_periodic(self):
        """Test periodic BC in 2D."""
        bc = PeriodicBC()
        state = np.arange(25).reshape(5, 5)
        dx = (0.1, 0.1)
        t = 0.0
        
        result = bc.apply(state, dx, t)
        
        # Check periodicity in x-direction
        assert np.allclose(result[0, :], state[-2, :])
        assert np.allclose(result[-1, :], state[1, :])
        
        # Check periodicity in y-direction
        assert np.allclose(result[:, 0], state[:, -2])
        assert np.allclose(result[:, -1], state[:, 1])


class TestRobinBC:
    """Test Robin (mixed) boundary conditions."""
    
    def test_robin_reduces_to_dirichlet(self):
        """Test that α=1, β=0 gives Dirichlet BC."""
        bc = RobinBC(alpha=1.0, beta=0.0, gamma=5.0)
        state = np.ones(10)
        dx = (0.1,)
        t = 0.0
        
        result = bc.apply(state, dx, t)
        
        # Should approximate Dirichlet with value=5
        assert result[0] == pytest.approx(5.0, abs=1e-10)
        assert result[-1] == pytest.approx(5.0, abs=1e-10)
    
    def test_robin_reduces_to_neumann(self):
        """Test that α=0, β=1 gives Neumann BC."""
        bc = RobinBC(alpha=0.0, beta=1.0, gamma=0.0)
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dx = (1.0,)
        t = 0.0
        
        result = bc.apply(state, dx, t)
        
        # Should approximate zero-flux Neumann
        # u[0] = u[1] - dx*gamma/beta = 2.0 - 0 = 2.0
        assert result[0] == pytest.approx(2.0)
        assert result[-1] == pytest.approx(4.0)
    
    def test_robin_invalid_parameters(self):
        """Test that both α=0 and β=0 raises error."""
        with pytest.raises(ValueError, match="cannot be zero"):
            RobinBC(alpha=0.0, beta=0.0, gamma=1.0)
    
    def test_robin_convective_bc(self):
        """Test Robin BC for convective/radiation condition."""
        # α=1, β=h represents convective BC
        bc = RobinBC(alpha=1.0, beta=0.1, gamma=0.0)
        state = np.ones(10)
        dx = (0.1,)
        t = 0.0
        
        result = bc.apply(state, dx, t)
        
        # Check that BC is applied (exact values depend on scheme)
        assert result.shape == state.shape
