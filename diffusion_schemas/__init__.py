"""
Diffusion Schemas: A framework for solving diffusion equations with agent-based sources.

This package provides abstract interfaces and concrete implementations for solving
the heat/diffusion equation using various numerical methods in 1D, 2D, and 3D.
"""

from diffusion_schemas.base import Schema
from diffusion_schemas.methods.explicit_euler import ExplicitEulerSchema
from diffusion_schemas.methods.implicit import ImplicitEulerSchema
from diffusion_schemas.methods.crank_nicolson import CrankNicolsonSchema
from diffusion_schemas.methods.implicit_LOD import ImplicitLODSchema
from diffusion_schemas.methods.crank_nicolson_LOD import CrankNicolsonLODSchema
from diffusion_schemas.methods_BC.explicit_euler import ExplicitEulerBCSchema
from diffusion_schemas.methods_BC.implicit import ImplicitEulerBCSchema
from diffusion_schemas.methods_BC.crank_nicolson import CrankNicolsonBCSchema
from diffusion_schemas.methods_BC.implicit_LOD import ImplicitLODBCSchema
from diffusion_schemas.methods_BC.crank_nicolson_LOD import CrankNicolsonLODBCSchema
from diffusion_schemas.methods_BC.ADI import ADIBCSchema
from diffusion_schemas.methods_BC_I.ADI import ADIBCISchema


__version__ = "0.1.0"

__all__ = [
    "Schema",
    "ExplicitEulerSchema",
    "ImplicitEulerSchema",
    "CrankNicolsonSchema",
    "ImplicitLODSchema",
    "CrankNicolsonLODSchema",
    
    "ExplicitEulerBCSchema",
    "ImplicitEulerBCSchema",
    "CrankNicolsonBCSchema",
    "ImplicitLODBCSchema",
    "CrankNicolsonLODBCSchema",
    "ADIBCSchema",

    "ADIBCISchema"
]
