"""
Diffusion Schemas: A framework for solving diffusion equations with agent-based sources.

This package provides abstract interfaces and concrete implementations for solving
the heat/diffusion equation using various numerical methods in 1D, 2D, and 3D.
"""

from diffusion_schemas.base import Schema

from diffusion_schemas.methods.explicit_euler import ExplicitEulerSchema
# from diffusion_schemas.methods.implicit import ImplicitEulerSchema
# from diffusion_schemas.methods.crank_nicolson import CrankNicolsonSchema
# from diffusion_schemas.methods.implicit_LOD import ImplicitLODSchema
# from diffusion_schemas.methods.crank_nicolson_LOD import CrankNicolsonLODSchema
from diffusion_schemas.methods_BC.explicit_euler import ExplicitEulerBCSchema
# from diffusion_schemas.methods_BC.implicit import ImplicitEulerBCSchema
# from diffusion_schemas.methods_BC.crank_nicolson import CrankNicolsonBCSchema
# from diffusion_schemas.methods_BC.implicit_LOD import ImplicitLODBCSchema
# from diffusion_schemas.methods_BC.crank_nicolson_LOD import CrankNicolsonLODBCSchema
# from diffusion_schemas.methods_BC.ADI import ADIBCSchema
# from diffusion_schemas.methods_BC_I.implicit import ImplicitEulerBCISchema
# from diffusion_schemas.methods_BC_I.crank_nicolson import CrankNicolsonBCISchema
# from diffusion_schemas.methods_BC_I.implicit_LOD import ImplicitLODBCISchema
# from diffusion_schemas.methods_BC_I.crank_nicolson_LOD import CrankNicolsonLODBCISchema
# from diffusion_schemas.methods_BC_I.ADI import ADIBCISchema
# from diffusion_schemas.methods_BC_OS.implicit import ImplicitEulerBCOSSchema
# from diffusion_schemas.methods_BC_OS.implicit_LOD import ImplicitLODBCOSSchema
# from diffusion_schemas.methods_BC_OS.ADI import ADIBCOSSchema
# from diffusion_schemas.methods_BC_OS.crank_nicolson import CrankNicolsonBCOSSchema
# from diffusion_schemas.methods_BC_OS.crank_nicolson_LOD import CrankNicolsonLODBCOSSchema

# from diffusion_schemas.methods_BC_I.implicit_LOD_opt import ImplicitLODBCISchema as ImplicitLODBCISchemaOpt
# from diffusion_schemas.methods_BC_I.crank_nicolson_LOD_opt import CrankNicolsonLODBCISchema as CrankNicolsonLODBCISchemaOpt
# from diffusion_schemas.methods_BC_I.ADI_opt import ADIBCISchema as ADIBCISchemaOpt

from diffusion_schemas.methods_unified import (
    ADISchema,ADIBCSchema,ADIBCISchema,ADIBCOSSchema,ADIBCIOptSchema,
    ImplicitEulerSchema,ImplicitEulerBCSchema,ImplicitEulerBCISchema,ImplicitEulerBCOSSchema,
    ImplicitLODSchema,ImplicitLODBCSchema,ImplicitLODBCISchema,ImplicitLODBCOSSchema,ImplicitLODBCIOptSchema,
    CrankNicolsonSchema,CrankNicolsonBCSchema,CrankNicolsonBCISchema,CrankNicolsonBCOSSchema,
    CrankNicolsonLODSchema,CrankNicolsonLODBCSchema,CrankNicolsonLODBCISchema,CrankNicolsonLODBCOSSchema,CrankNicolsonLODBCIOptSchema
)


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

    "ImplicitEulerBCISchema",
    "CrankNicolsonBCISchema",
    "ImplicitLODBCISchema",
    "CrankNicolsonLODBCISchema",
    "ADIBCISchema",

    "ImplicitEulerBCOSSchema",
    "CrankNicolsonBCOSSchema",
    "ImplicitLODBCOSSchema",
    "CrankNicolsonLODBCOSSchema",
    "ADIBCOSSchema",

    "ImplicitLODBCIOptSchema",
    "CrankNicolsonLODBCIOptSchema",
    "ADIBCIOptSchema",
]

