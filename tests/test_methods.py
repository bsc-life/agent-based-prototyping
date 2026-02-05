"""Tests for numerical methods."""

import pytest
import numpy as np
from diffusion_schemas import ExplicitEulerSchema, ImplicitEulerSchema, CrankNicolsonSchema
from diffusion_schemas.utils import gaussian, DirichletBC, NeumannBC


class TestExplicitEuler:
    """Test Explicit Euler method."""
    
    def test_1d_diffusion_conserves_mass(self):
        """Test that mass is conserved with zero-flux BC."""
        schema = ExplicitEulerSchema(
            domain_size=1.0,
            grid_points=50,
            dt=0.00001,
            diffusion_coefficient=0.1,
            check_stability=False
        )
        
        # Gaussian initial condition
        ic = gaussian(center=0.5, amplitude=1.0, width=0.1)
        schema.set_initial_condition(ic)
        schema.set_boundary_conditions(NeumannBC(flux=0.0))
        
        initial_mass = np.sum(schema.state)
        
        # Solve
        schema.solve(t_final=0.01)
        
        final_mass = np.sum(schema.state)
        
        # Mass should be approximately conserved
        assert final_mass == pytest.approx(initial_mass, rel=0.01)
    
    def test_stability_warning(self):
        """Test that stability warning is issued for large dt."""
        with pytest.warns(UserWarning, match="Stability condition violated"):
            schema = ExplicitEulerSchema(
                domain_size=1.0,
                grid_points=100,
                dt=0.1,  # Too large!
                diffusion_coefficient=1.0,
                check_stability=True
            )
    
    def test_no_warning_with_check_disabled(self):
        """Test that no warning when check is disabled."""
        with pytest.warns(None) as warnings:
            schema = ExplicitEulerSchema(
                domain_size=1.0,
                grid_points=100,
                dt=0.1,
                diffusion_coefficient=1.0,
                check_stability=False
            )
        
        # Filter out pytest internal warnings
        user_warnings = [w for w in warnings if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0
    
    def test_2d_diffusion(self):
        """Test 2D diffusion."""
        schema = ExplicitEulerSchema(
            domain_size=(1.0, 1.0),
            grid_points=(20, 20),
            dt=0.0001,
            diffusion_coefficient=0.1
        )
        
        # Gaussian at center
        ic = gaussian(center=(0.5, 0.5), amplitude=1.0, width=0.1)
        schema.set_initial_condition(ic)
        
        initial_max = np.max(schema.state)
        
        # Solve
        schema.solve(t_final=0.1)
        
        # Maximum should decrease (spreading)
        final_max = np.max(schema.state)
        assert final_max < initial_max
    
    def test_decay(self):
        """Test that decay reduces concentration."""
        schema = ExplicitEulerSchema(
            domain_size=1.0,
            grid_points=50,
            dt=0.0001,
            diffusion_coefficient=0.0,  # No diffusion
            decay_rate=1.0
        )
        
        schema.set_initial_condition(1.0)
        
        # Solve for time t
        t_final = 0.1
        schema.solve(t_final=t_final)
        
        # Analytical solution: u(t) = u0 * exp(-λ*t)
        expected = np.exp(-schema.decay_rate * t_final)
        
        assert np.allclose(schema.state, expected, rtol=0.01)


class TestImplicitEuler:
    """Test Implicit Euler method."""
    
    def test_1d_diffusion(self):
        """Test 1D implicit diffusion."""
        schema = ImplicitEulerSchema(
            domain_size=1.0,
            grid_points=50,
            dt=0.01,  # Can use larger dt than explicit
            diffusion_coefficient=0.1
        )
        
        ic = gaussian(center=0.5, amplitude=1.0, width=0.1)
        schema.set_initial_condition(ic)
        
        initial_max = np.max(schema.state)
        
        schema.solve(t_final=0.1)
        
        # Maximum should decrease
        final_max = np.max(schema.state)
        assert final_max < initial_max
    
    def test_unconditional_stability(self):
        """Test that implicit method is stable with large dt."""
        schema = ImplicitEulerSchema(
            domain_size=1.0,
            grid_points=50,
            dt=0.1,  # Large time step
            diffusion_coefficient=1.0
        )
        
        schema.set_initial_condition(1.0)
        
        # Should not blow up even with large dt
        schema.solve(t_final=1.0)
        
        # Solution should be bounded
        assert np.all(np.isfinite(schema.state))
        assert np.all(schema.state >= 0)
    
    def test_2d_diffusion(self):
        """Test 2D implicit diffusion."""
        schema = ImplicitEulerSchema(
            domain_size=(1.0, 1.0),
            grid_points=(20, 20),
            dt=0.01,
            diffusion_coefficient=0.1
        )
        
        ic = gaussian(center=(0.5, 0.5), amplitude=1.0, width=0.1)
        schema.set_initial_condition(ic)
        schema.set_boundary_conditions(DirichletBC(value=0.0))
        
        schema.solve(t_final=0.5)
        
        # Check that solution is reasonable
        assert np.all(np.isfinite(schema.state))
        assert np.max(schema.state) <= 1.0
    
    def test_parameter_change_rebuilds_matrix(self):
        """Test that changing parameters rebuilds system matrix."""
        schema = ImplicitEulerSchema(
            domain_size=1.0,
            grid_points=10,
            dt=0.01,
            diffusion_coefficient=0.1
        )
        
        old_matrix = schema.system_matrix.copy()
        
        # Change diffusion coefficient
        schema.set_diffusion_coefficient(0.5)
        
        # Matrix should be different
        assert not np.allclose(old_matrix.toarray(), schema.system_matrix.toarray())


class TestCrankNicolson:
    """Test Crank-Nicolson method."""
    
    def test_1d_diffusion(self):
        """Test 1D Crank-Nicolson diffusion."""
        schema = CrankNicolsonSchema(
            domain_size=1.0,
            grid_points=50,
            dt=0.01,
            diffusion_coefficient=0.1
        )
        
        ic = gaussian(center=0.5, amplitude=1.0, width=0.1)
        schema.set_initial_condition(ic)
        
        schema.solve(t_final=0.1)
        
        # Solution should be smooth and bounded
        assert np.all(np.isfinite(schema.state))
    
    def test_theta_parameter(self):
        """Test that theta parameter works."""
        # theta = 0 -> explicit
        # theta = 0.5 -> Crank-Nicolson
        # theta = 1 -> implicit
        
        schema_cn = CrankNicolsonSchema(
            domain_size=1.0,
            grid_points=50,
            dt=0.001,
            diffusion_coefficient=0.1,
            theta=0.5
        )
        
        assert schema_cn.theta == 0.5
        
        # Test with different theta
        schema_impl = CrankNicolsonSchema(
            domain_size=1.0,
            grid_points=50,
            dt=0.001,
            diffusion_coefficient=0.1,
            theta=1.0
        )
        
        assert schema_impl.theta == 1.0
    
    def test_invalid_theta(self):
        """Test that invalid theta raises error."""
        with pytest.raises(ValueError, match="theta must be in"):
            CrankNicolsonSchema(
                domain_size=1.0,
                grid_points=50,
                dt=0.01,
                theta=1.5  # Invalid!
            )
    
    def test_second_order_accuracy(self):
        """Test that Crank-Nicolson is more accurate than first-order methods."""
        # This is a qualitative test - CN should give smaller error
        
        t_final = 0.1
        ic = gaussian(center=0.5, amplitude=1.0, width=0.1)
        
        # Coarse time step
        dt = 0.01
        
        # Crank-Nicolson
        schema_cn = CrankNicolsonSchema(
            domain_size=1.0,
            grid_points=100,
            dt=dt,
            diffusion_coefficient=0.1
        )
        schema_cn.set_initial_condition(ic)
        schema_cn.solve(t_final=t_final)
        u_cn = schema_cn.get_state()
        
        # Implicit Euler (first-order)
        schema_impl = ImplicitEulerSchema(
            domain_size=1.0,
            grid_points=100,
            dt=dt,
            diffusion_coefficient=0.1
        )
        schema_impl.set_initial_condition(ic)
        schema_impl.solve(t_final=t_final)
        u_impl = schema_impl.get_state()
        
        # Reference solution with very fine time step
        schema_ref = CrankNicolsonSchema(
            domain_size=1.0,
            grid_points=100,
            dt=dt/10,
            diffusion_coefficient=0.1
        )
        schema_ref.set_initial_condition(ic)
        schema_ref.solve(t_final=t_final)
        u_ref = schema_ref.get_state()
        
        # CN should be closer to reference
        error_cn = np.linalg.norm(u_cn - u_ref)
        error_impl = np.linalg.norm(u_impl - u_ref)
        
        assert error_cn < error_impl


class TestMethodComparison:
    """Compare different methods on the same problem."""
    
    def test_all_methods_converge_to_similar_solution(self):
        """Test that all methods give similar results."""
        
        ic = gaussian(center=0.5, amplitude=1.0, width=0.1)
        
        # Use small dt for explicit to ensure stability
        explicit = ExplicitEulerSchema(
            domain_size=1.0,
            grid_points=50,
            dt=0.0001,
            diffusion_coefficient=0.1,
            check_stability=False
        )
        explicit.set_initial_condition(ic)
        explicit.solve(t_final=0.01)
        u_explicit = explicit.get_state()
        
        # Implicit and CN can use larger dt
        implicit = ImplicitEulerSchema(
            domain_size=1.0,
            grid_points=50,
            dt=0.001,
            diffusion_coefficient=0.1
        )
        implicit.set_initial_condition(ic)
        implicit.solve(t_final=0.01)
        u_implicit = implicit.get_state()
        
        crank = CrankNicolsonSchema(
            domain_size=1.0,
            grid_points=50,
            dt=0.001,
            diffusion_coefficient=0.1
        )
        crank.set_initial_condition(ic)
        crank.solve(t_final=0.01)
        u_crank = crank.get_state()
        
        # All should be reasonably close
        assert np.allclose(u_explicit, u_implicit, rtol=0.1)
        assert np.allclose(u_explicit, u_crank, rtol=0.1)
        assert np.allclose(u_implicit, u_crank, rtol=0.1)
