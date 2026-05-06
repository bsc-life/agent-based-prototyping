"""Unified entrypoints for diffusion schemas.

These classes wrap the existing implementations without changing behavior.
"""

from diffusion_schemas.methods_unified.ADI import (
    ADISchema,
    ADIBCSchema,
    ADIBCISchema,
    ADIBCOSSchema,
    ADIBCIOptSchema
)
from diffusion_schemas.methods_unified.implicit import (
    ImplicitEulerSchema,
    ImplicitEulerBCSchema,
    ImplicitEulerBCISchema,
    ImplicitEulerBCOSSchema,
)
from diffusion_schemas.methods_unified.implicit_LOD import (
    ImplicitLODSchema,
    ImplicitLODBCSchema,
    ImplicitLODBCISchema,
    ImplicitLODBCOSSchema,
    ImplicitLODBCIOptSchema
)
from diffusion_schemas.methods_unified.crank_nicolson import (
    CrankNicolsonSchema,
    CrankNicolsonBCSchema,
    CrankNicolsonBCISchema,
    CrankNicolsonBCOSSchema,
)
from diffusion_schemas.methods_unified.crank_nicolson_LOD import (
    CrankNicolsonLODSchema,
    CrankNicolsonLODBCSchema,
    CrankNicolsonLODBCISchema,
    CrankNicolsonLODBCOSSchema,
    CrankNicolsonLODBCIOptSchema
)

__all__ = [
    "ADISchema",
    "ADIBCSchema",
    "ADIBCISchema",
    "ADIBCOSSchema",
    "ADIBCIOptSchema",
    "ImplicitEulerSchema",
    "ImplicitEulerBCSchema",
    "ImplicitEulerBCISchema",
    "ImplicitEulerBCOSSchema",
    "ImplicitLODSchema",
    "ImplicitLODBCSchema",
    "ImplicitLODBCISchema",
    "ImplicitLODBCOSSchema",
    "ImplicitLODBCIOptSchema",
    "CrankNicolsonSchema",
    "CrankNicolsonBCSchema",
    "CrankNicolsonBCISchema",
    "CrankNicolsonBCOSSchema",
    "CrankNicolsonLODSchema",
    "CrankNicolsonLODBCSchema",
    "CrankNicolsonLODBCISchema",
    "CrankNicolsonLODBCOSSchema",
    "CrankNicolsonLODBCIOptSchema",
]
