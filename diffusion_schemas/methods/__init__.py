"""Numerical methods for solving diffusion equations."""

from diffusion_schemas.methods.explicit_euler import ExplicitEulerSchema
from diffusion_schemas.methods.implicit import ImplicitEulerSchema
from diffusion_schemas.methods.crank_nicolson import CrankNicolsonSchema
from diffusion_schemas.methods.implicit_ADI import ADISchema
from diffusion_schemas.methods.crank_nicolson_ADI import CrankNicolsonADISchema

__all__ = [
    "ExplicitEulerSchema",
    "ImplicitEulerSchema",
    "CrankNicolsonSchema",
    "ADISchema",
    "CrankNicolsonADISchema",
]
