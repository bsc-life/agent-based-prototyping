"""Numerical methods for solving diffusion equations."""

from diffusion_schemas.methods_BC_I.implicit import ImplicitEulerBCISchema
from diffusion_schemas.methods_BC_I.crank_nicolson import CrankNicolsonBCISchema
from diffusion_schemas.methods_BC_I.implicit_LOD import ImplicitLODBCISchema
from diffusion_schemas.methods_BC_I.crank_nicolson_LOD import CrankNicolsonLODBCISchema
from diffusion_schemas.methods_BC_I.ADI import ADIBCISchema

__all__ = [
    "ImplicitEulerBCISchema",
    "CrankNicolsonBCISchema",
    "ImplicitLODBCISchema",
    "CrankNicolsonLODBCISchema",
    "ADIBCISchema",
]

