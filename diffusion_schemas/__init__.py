"""
Diffusion Schemas: A framework for solving diffusion equations with agent-based sources.

This package provides abstract interfaces and concrete implementations for solving
the heat/diffusion equation using various numerical methods in 1D, 2D, and 3D.
"""

from diffusion_schemas.base import Schema
from diffusion_schemas.methods.explicit_euler import ExplicitEulerSchema
from diffusion_schemas.methods.implicit import ImplicitEulerSchema
from diffusion_schemas.methods.crank_nicolson import CrankNicolsonSchema
from diffusion_schemas.methods.implicit_ADI import ADISchema
from diffusion_schemas.methods.crank_nicolson_ADI import CrankNicolsonADISchema

__version__ = "0.1.0"

__all__ = [
    "Schema",
    "ExplicitEulerSchema",
    "ImplicitEulerSchema",
    "CrankNicolsonSchema",
    "ADISchema",
    "CrankNicolsonADISchema"
]
