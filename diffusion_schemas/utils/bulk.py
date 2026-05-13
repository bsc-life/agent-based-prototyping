"""
Bulk source/sink system for volumetric supply and uptake.

This module implements bulk regions that contribute source terms to the
diffusion equation over extended spatial domains (rectangles, spheres),
as opposed to agent-based point/Gaussian sources.

Inspired by BioFVM's bulk supply/uptake appendix. Each region defines a
geometric domain and a net export rate. Boundary voxels receive a
fractional contribution proportional to their overlap with the region.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union, List, Optional, Callable
import numpy as np

# =============================================================================
# Region domains
# =============================================================================

class RegionDomain(ABC):
    """
    Abstract base for region geometry definitions.
    
    A region domain describes a geometric shape that can be rasterized
    onto a discrete grid, producing a mask of overlap fractions in [0, 1].
    """

    @abstractmethod # so that every subclass must implement this method
    def rasterize(
        self,
        coords: List[np.ndarray],
        dx: Tuple[float, ...]
    ) -> np.ndarray:
        """
        Rasterize this region onto the grid.
        
        Parameters
        ----------
        coords : List[np.ndarray]
            Coordinate grids for each dimension (1D arrays for 1D,
            meshgrids for 2D/3D).
        dx : Tuple[float, ...]
            Grid spacing in each dimension.
            
        Returns
        -------
        np.ndarray
            Overlap fraction array with values in [0, 1], same shape
            as the coordinate grids.
        """
        pass


class RectangleRegion(RegionDomain):
    """
    Axis-aligned rectangular (hyper-rectangular) region domain.
    
    Defined by an origin (lower-left corner) and extents in each dimension.
    Works for 1D intervals, 2D rectangles, and 3D boxes.
    
    Parameters
    ----------
    origin : Tuple[float, ...]
        Lower-left corner of the rectangle, e.g. (x0,) for 1D,
        (x0, y0) for 2D, (x0, y0, z0) for 3D.
    size : Tuple[float, ...]
        Extents in each dimension, e.g. (width,) for 1D,
        (width, height) for 2D, (width, height, depth) for 3D.
    """

    def __init__(
        self,
        origin: Tuple[float, ...],
        size: Tuple[float, ...]
    ):
        if len(origin) != len(size):
            raise ValueError(
                f"origin has {len(origin)} dimensions but size has {len(size)}"
            )
        self.origin = tuple(origin)
        self.size = tuple(size)
        self.ndim = len(origin)

    # -----------------------------------------------------------------
    # def rasterize(
    #     self,
    #     coords: List[np.ndarray],
    #     dx: Tuple[float, ...]
    # ) -> np.ndarray:
    #     """
    #     Rasterize the rectangle onto the grid.
        
    #     Interior voxels receive 1.0; boundary voxels receive a fraction
    #     equal to the 1-D overlap along each axis (product rule for nD).
        
    #     Parameters
    #     ----------
    #     coords : List[np.ndarray]
    #         Coordinate grids.
    #     dx : Tuple[float, ...]
    #         Grid spacing in each dimension.
            
    #     Returns
    #     -------
    #     np.ndarray
    #         Overlap fraction array in [0, 1].
    #     """
    #     overlap = None

    #     for dim in range(self.ndim):
    #         # 1D coordinate values along this axis
    #         if self.ndim == 1:
    #             c = coords[0]
    #         else:
    #             c = coords[dim]

    #         lo = self.origin[dim]
    #         hi = self.origin[dim] + self.size[dim]
    #         half_dx = dx[dim] / 2.0

    #         # Voxel boundaries
    #         voxel_lo = c - half_dx
    #         voxel_hi = c + half_dx

    #         # Overlap length along this axis for each voxel
    #         overlap_lo = np.maximum(voxel_lo, lo)
    #         overlap_hi = np.minimum(voxel_hi, hi)
    #         axis_overlap = np.maximum(overlap_hi - overlap_lo, 0.0) / dx[dim]

    #         if overlap is None:
    #             overlap = axis_overlap
    #         else:
    #             overlap = overlap * axis_overlap

    #     return overlap

    def rasterize(
            self,
            coords: List[np.ndarray],
            dx: Tuple[float, ...]
        ) -> np.ndarray:
        overlap = np.ones_like(coords[0], dtype=float)

        for dim in range(self.ndim):
            c = coords[0] if self.ndim == 1 else coords[dim]
            lo = self.origin[dim]
            hi = self.origin[dim] + self.size[dim]
            half_dx = dx[dim] / 2.0

            overlap_lo = np.maximum(c - half_dx, lo)
            overlap_hi = np.minimum(c + half_dx, hi)
            
            # In-place multiplication avoids redundant array allocations
            overlap *= np.maximum(overlap_hi - overlap_lo, 0.0) / dx[dim]

        return overlap

    def __repr__(self) -> str:
        return f"RectangleRegion(origin={self.origin}, size={self.size})"


class SphereRegion(RegionDomain):
    """
    Spherical (or circular in 2D, point-interval in 1D) region domain.
    
    Parameters
    ----------
    center : Tuple[float, ...]
        Center of the sphere, e.g. (x,) for 1D, (x, y) for 2D,
        (x, y, z) for 3D.
    radius : float
        Radius of the sphere.
    """

    def __init__(
        self,
        center: Tuple[float, ...],
        radius: float
    ):
        if radius <= 0:
            raise ValueError("Radius must be positive")
        self.center = tuple(center)
        self.radius = radius
        self.ndim = len(center)

    # -----------------------------------------------------------------
    # def rasterize(
    #     self,
    #     coords: List[np.ndarray],
    #     dx: Tuple[float, ...]
    # ) -> np.ndarray:
    #     """
    #     Rasterize the sphere onto the grid.
        
    #     Interior voxels (whose center is well inside the sphere) receive
    #     1.0. Boundary voxels are assigned a fraction estimated by
    #     sub-sampling the voxel volume.
        
    #     Parameters
    #     ----------
    #     coords : List[np.ndarray]
    #         Coordinate grids.
    #     dx : Tuple[float, ...]
    #         Grid spacing in each dimension.
            
    #     Returns
    #     -------
    #     np.ndarray
    #         Overlap fraction array in [0, 1].
    #     """
    #     # Squared distance from each grid point to the sphere center
    #     if self.ndim == 1:
    #         r_sq = (coords[0] - self.center[0]) ** 2
    #     else:
    #         r_sq = sum(
    #             (coords[dim] - self.center[dim]) ** 2
    #             for dim in range(self.ndim)
    #         )

    #     # Maximum possible half-diagonal of a voxel
    #     half_diag = np.sqrt(sum((d / 2.0) ** 2 for d in dx))
    #     R = self.radius

    #     # Classify voxels
    #     dist = np.sqrt(r_sq)

    #     # Fully inside: entire voxel fits inside the sphere
    #     # Will be set to 1.0
    #     fully_inside = dist + half_diag <= R

    #     # Fully outside: no part of the voxel touches the sphere
    #     # Will be left as 0.0
    #     fully_outside = dist - half_diag >= R

    #     # Boundary voxels need sub-sampling
    #     boundary = ~fully_inside & ~fully_outside

    #     overlap = np.zeros_like(dist)
    #     overlap[fully_inside] = 1.0

    #     if np.any(boundary):
    #         overlap[boundary] = self._subsample_overlap(
    #             coords, dx, boundary
    #         )

    #     return overlap

    # # Method to compute boundary voxel overlap by sub-sampling
    # def _subsample_overlap(
    #     self,
    #     coords: List[np.ndarray],
    #     dx: Tuple[float, ...],
    #     mask: np.ndarray,
    #     n_samples: int = 4
    # ) -> np.ndarray:
    #     """
    #     Estimate overlap fraction for boundary voxels via sub-sampling.
        
    #     Splits each boundary voxel into n_samples^ndim sub-points and
    #     counts how many fall inside the sphere.
        
    #     Parameters
    #     ----------
    #     coords : List[np.ndarray]
    #         Coordinate grids.
    #     dx : Tuple[float, ...]
    #         Grid spacing.
    #     mask : np.ndarray of bool
    #         Boolean mask selecting boundary voxels.
    #     n_samples : int
    #         Number of sub-samples per dimension (default 4).
            
    #     Returns
    #     -------
    #     np.ndarray
    #         Overlap fractions for the masked voxels.
    #     """
    #     # Sub-sample offsets along one axis: centred in the voxel
    #     offsets_1d = np.linspace(-0.5, 0.5, n_samples, endpoint=False) + 0.5 / n_samples

    #     # Build all sub-sample offset combinations
    #     if self.ndim == 1:
    #         offset_combos = offsets_1d[:, None] * dx[0]          # (n, 1)
    #     else:
    #         grids = np.meshgrid(
    #             *[offsets_1d * dx[d] for d in range(self.ndim)],
    #             indexing='ij'
    #         )
    #         offset_combos = np.stack(
    #             [g.ravel() for g in grids], axis=-1
    #         )                                                     # (n^ndim, ndim)

    #     n_total = offset_combos.shape[0]

    #     # Centre coordinates of the boundary voxels
    #     if self.ndim == 1:
    #         centres = coords[0][mask][:, None]                    # (M, 1)
    #     else:
    #         centres = np.stack(
    #             [coords[d][mask] for d in range(self.ndim)], axis=-1
    #         )                                                     # (M, ndim)

    #     # (M, n_total, ndim) = (M,1,ndim) + (1, n_total, ndim)
    #     sample_pts = centres[:, None, :] + offset_combos[None, :, :]

    #     # Squared distance from each sample to sphere centre
    #     center_arr = np.array(self.center)
    #     r_sq = np.sum((sample_pts - center_arr) ** 2, axis=-1)   # (M, n_total)

    #     inside_count = np.sum(r_sq <= self.radius ** 2, axis=1)
    #     return inside_count / n_total

    def rasterize(
        self,
        coords: List[np.ndarray],
        dx: Tuple[float, ...]
    ) -> np.ndarray:
        if self.ndim == 1:
            r_sq = (coords[0] - self.center[0]) ** 2
        else:
            r_sq = sum(
                (coords[dim] - self.center[dim]) ** 2
                for dim in range(self.ndim)
            )

        half_diag = np.sqrt(sum((d / 2.0) ** 2 for d in dx))
        R = self.radius

        # OPTIMIZATION: Avoid expensive full-grid np.sqrt() by squaring the bounds
        inner_bound = max(0.0, R - half_diag) ** 2
        outer_bound = (R + half_diag) ** 2

        fully_inside = r_sq <= inner_bound
        fully_outside = r_sq >= outer_bound
        boundary = ~fully_inside & ~fully_outside

        overlap = np.zeros_like(r_sq, dtype=float)
        overlap[fully_inside] = 1.0

        if np.any(boundary):
            overlap[boundary] = self._subsample_overlap(
                coords, dx, boundary
            )

        return overlap

    def _subsample_overlap(
        self,
        coords: List[np.ndarray],
        dx: Tuple[float, ...],
        mask: np.ndarray,
        n_samples: int = 4
    ) -> np.ndarray:
        offsets_1d = np.linspace(-0.5, 0.5, n_samples, endpoint=False) + 0.5 / n_samples

        if self.ndim == 1:
            offset_combos = offsets_1d[:, None] * dx[0]          
        else:
            grids = np.meshgrid(
                *[offsets_1d * dx[d] for d in range(self.ndim)],
                indexing='ij'
            )
            offset_combos = np.stack(
                [g.ravel() for g in grids], axis=-1
            )                                                     

        n_total = offset_combos.shape[0]

        if self.ndim == 1:
            centres = coords[0][mask][:, None]                    
        else:
            centres = np.stack(
                [coords[d][mask] for d in range(self.ndim)], axis=-1
            )                                                     

        # OPTIMIZATION: Use distance expansion (v+o)^2 = v^2 + 2vo + o^2 
        # to avoid massive 3D array allocations and use fast BLAS matrix dot products
        vec = centres - np.array(self.center)
        vec_sq = np.sum(vec ** 2, axis=-1)
        off_sq = np.sum(offset_combos ** 2, axis=-1)
        
        cross = np.dot(vec, offset_combos.T)
        r_sq_samples = vec_sq[:, None] + 2 * cross + off_sq[None, :]

        inside_count = np.sum(r_sq_samples <= self.radius ** 2, axis=1)
        return inside_count / n_total

    def __repr__(self) -> str:
        return f"SphereRegion(center={self.center}, radius={self.radius})"


# =============================================================================
# Region and Bulk classes
# =============================================================================

class Region:
    """
    A single bulk region combining a geometric domain with a source term.
    
    Parameters
    ----------
    domain : RegionDomain
        Geometric shape of the region (RectangleRegion or SphereRegion).
    net_rate : Union[float, Callable]
        Net export rate (constant or time-dependent callable).
        Positive values inject substrate; negative values consume it.
    name : str, optional
        Optional human-readable label.
    """

    def __init__(
        self,
        domain: RegionDomain,
        name: str = ""
    ):
        self.domain = domain
        self.name = name or f"Region_{id(self)}"


class NetRegion(Region):
    def __init__(
        self,
        domain: RegionDomain,
        net_rate: Union[float, Callable] = 0.0,
        name: str = ""
    ):
        super().__init__(domain, name=name)
        self.domain = domain
        self.net_rate = net_rate
        self.name = name or f"Region_{id(self)}"

    # -----------------------------------------------------------------
    def get_net_rate(self, t: float) -> float:
        """
        Get the net rate at time *t*.
        
        Parameters
        ----------
        t : float
            Current simulation time.
            
        Returns
        -------
        float
            Net export rate.
        """
        if callable(self.net_rate):
            return self.net_rate(t)
        return self.net_rate

    # -----------------------------------------------------------------
    def set_net_rate(self, rate: Union[float, Callable]) -> None:
        """
        Update the net rate.
        
        Parameters
        ----------
        rate : Union[float, Callable]
            New constant rate or time-dependent callable.
        """
        self.net_rate = rate

    def __repr__(self) -> str:
        rate = self.net_rate if not callable(self.net_rate) else "f(t)"
        return f"Region(name='{self.name}', domain={self.domain}, net_rate={rate})"
    
class LinearRegion(Region):
    def __init__(self, domain: RegionDomain, linear_rate: Union[float, Callable], name: str = ""):
        super().__init__(domain, name=name)
        self.linear_rate = linear_rate

    def set_linear_rate(self, rate: Union[float, Callable] = 0.0) -> None:
        self.linear_rate = rate 

    def get_linear_rate(self, t: float) -> float:
        if callable(self.linear_rate):
            return self.linear_rate(t)
        return self.linear_rate

class TargetRegion(LinearRegion):
    def __init__(self, domain: RegionDomain, linear_rate: Union[float, Callable], rho_target: Union[float, Callable], name: str = ""):
        super().__init__(domain, linear_rate=linear_rate, name=name)
        self.rho_target = rho_target

    def set_rho_target(self, target: Union[float, Callable]) -> None:
        self.rho_target = target

    def get_rho_target(self, t: float) -> float:
        if callable(self.rho_target):
            return self.rho_target(t)
        return self.rho_target

class Bulk:
    """
    Container for bulk source/sink regions.
    
    A Bulk object holds a list of :class:`Region` instances and computes the
    aggregate source term by rasterizing every region onto the schema's grid
    and summing contributions (overlapping regions are additive).
    
    Parameters
    ----------
    regions : List[Region], optional
        Initial list of regions. Default is an empty list.
        
    Examples
    --------
    >>> rect = RectangleRegion(origin=(0.2, 0.2), size=(0.3, 0.3))
    >>> region = Region(domain=rect, net_rate=5.0, name="source_patch")
    >>> bulk = Bulk(regions=[region])
    >>> source = bulk.compute_source(coords, dx, t=0.0)
    """

    def __init__(self, regions: Optional[List[Region]] = None):
        self._precomputed = False
        self._regions: List[Region] = list(regions) if regions else []

    # -- region management ------------------------------------------------

    def add_region(self, region: Region) -> None:
        """
        Add a region.
        
        Parameters
        ----------
        region : Region
            Region to add.
        """
        self._regions.append(region)
        self._precomputed = False

    def remove_region(self, name: str) -> None:
        """
        Remove a region by name.
        
        Parameters
        ----------
        name : str
            Name of the region to remove.
            
        Raises
        ------
        ValueError
            If no region with that name exists.
        """
        for i, r in enumerate(self._regions):
            if r.name == name:
                self._regions.pop(i)
                self._precomputed = False
                return
        raise ValueError(f"No region named '{name}'")

    def clear_regions(self) -> None:
        """Remove all regions."""
        self._regions = []
        self._precomputed = False

    @property
    def regions(self) -> List[Region]:
        """Return the current list of regions (read-only copy)."""
        return list(self._regions)

    def __len__(self) -> int:
        return len(self._regions)

    # source term computation

    def compute_source(
        self,
        field: np.ndarray,
        coords: List[np.ndarray],
        dx: Tuple[float, ...],
        dt: float,
        t: float
    ) -> np.ndarray:
        """
        Compute the aggregate source term from all regions.
        
        Each region is rasterized onto the grid to produce an overlap
        mask (values in [0, 1]).  The source contribution of a region is
        ``overlap * net_rate(t)``.  Overlapping regions are summed.
        
        Parameters
        ----------
        field. np.ndarray
            Current substrate concentration field (used to avoid negative sources).
        coords : List[np.ndarray]
            Coordinate grids (1D arrays for 1D, meshgrids for 2D/3D).
        dx : Tuple[float, ...]
            Grid spacing in each dimension.
        t : float
            Current simulation time.
            
        Returns
        -------
        np.ndarray
            Source term array, same shape as the coordinate grids.
        """

        self._precompute_rates(coords, dx, t)
        source = (self._net_cached_rates + 
                  field * self._linear_cached_rates - 
                  field * self._target_cached_rates + self._target_cached_bias)

        # Negative fields problem (revisit this logic)
        # If there exists any negative values
        # AND we are about to remove more substrate than present
        idx = (source * dt < -field) & (source < 0.0)
        source[idx] = -field[idx] / dt

        # NOTE
        # Not adding division of rate by grid volume as in Agent implementation
        # Each voxel already receives a fraction of the total rate based on the overlap, so no need to divide by voxel volume here. The net_rate is effectively a per-voxel contribution already after multiplying by the overlap fraction.
        # Rate is already given by unit volume 

        return source
    
    def compute_source_implicit(self, coords, dx, t) -> None:

        self._precompute_rates(coords, dx, t)  # precompute rates at the next time step for implicit contribution
        self.rhs_contribution = np.zeros_like(coords[0])
        self.lhs_contribution = np.zeros_like(coords[0])    

        # Cached arrays are already sums over all regions.
        self.rhs_contribution = self._net_cached_rates + self._target_cached_bias
        self.lhs_contribution = self._target_cached_rates - self._linear_cached_rates

    # def _precompute_rates(self, coords, dx, t) -> None:
    #     """
    #     Precompute net rates for all regions at the current time step.
        
    #     This can be used to optimize performance if there are many regions
    #     with time-dependent rates, by avoiding repeated calls to get_net_rate
    #     during source computation.
    #     """
        
    #     # Check one by one if any region has a callable rate. 
    #     # If all are constant, we can skip precomputation after the first time.
    #     all_constant = True
    #     for region in self._regions:
    #         if isinstance(region, NetRegion) and callable(region.net_rate):
    #             all_constant = False
    #             break
    #         elif isinstance(region, TargetRegion):
    #             if callable(region.linear_rate) or callable(region.rho_target):
    #                 all_constant = False
    #                 break
    #         elif isinstance(region, LinearRegion) and callable(region.linear_rate):
    #             all_constant = False
    #             break
    #     if self._precomputed and all_constant:
    #         return
    #     self._precomputed = True

    #     self._net_cached_rates = np.zeros_like(coords[0])
    #     self._linear_cached_rates = np.zeros_like(coords[0])
    #     self._target_cached_rates = np.zeros_like(coords[0])
    #     self._target_cached_bias = np.zeros_like(coords[0])
    #     counter = 0
    #     import time
    #     starttime = time.time()
    #     print("\nStarting rasterization of bulk regions...")
    #     for region in self._regions: 
    #         if counter > 0 and counter % 1000 == 0:
    #             endtime = time.time()
    #             print(f"Rasterizing region: {counter}. Time taken: {endtime - starttime:.2f} seconds.")
    #             starttime = time.time()
    #         counter += 1
    #         if isinstance(region, TargetRegion):
    #             rasterization = region.domain.rasterize(coords, dx)
    #             self._target_cached_rates += rasterization * region.get_linear_rate(t)
    #             self._target_cached_bias += rasterization * region.get_linear_rate(t) * region.get_rho_target(t)
    #         elif isinstance(region, LinearRegion):
    #             self._linear_cached_rates += region.domain.rasterize(coords, dx) * region.get_linear_rate(t)
    #         elif isinstance(region, NetRegion): 
    #             self._net_cached_rates += region.domain.rasterize(coords, dx) * region.get_net_rate(t)

    def _precompute_rates(self, coords, dx, t) -> None:
        all_constant = True
        for region in self._regions:
            if isinstance(region, NetRegion) and callable(region.net_rate):
                all_constant = False
                break
            elif isinstance(region, TargetRegion):
                if callable(region.linear_rate) or callable(region.rho_target):
                    all_constant = False
                    break
            elif isinstance(region, LinearRegion) and callable(region.linear_rate):
                all_constant = False
                break
                
        if self._precomputed and all_constant:
            return
        self._precomputed = True

        self._net_cached_rates = np.zeros_like(coords[0], dtype=float)
        self._linear_cached_rates = np.zeros_like(coords[0], dtype=float)
        self._target_cached_rates = np.zeros_like(coords[0], dtype=float)
        self._target_cached_bias = np.zeros_like(coords[0], dtype=float)

        import time
        starttime = time.time()
        print(f"\nGrouping {len(self._regions)} bulk regions for batch processing...")

        # 1. Group regions by shape to batch process rectangles instantly
        rectangles = []
        spheres = []

        for r in self._regions:
            net_rate = r.get_net_rate(t) if isinstance(r, NetRegion) else 0.0
            lin_rate = r.get_linear_rate(t) if isinstance(r, (LinearRegion, TargetRegion)) else 0.0
            tgt_rate = r.get_linear_rate(t) if isinstance(r, TargetRegion) else 0.0
            tgt_bias = tgt_rate * r.get_rho_target(t) if isinstance(r, TargetRegion) else 0.0
            
            if isinstance(r.domain, RectangleRegion):
                rectangles.append((r.domain.origin, r.domain.size, net_rate, lin_rate, tgt_rate, tgt_bias))
            else:
                spheres.append((r, net_rate, lin_rate, tgt_rate, tgt_bias))

        # 2. Process rectangles in massive chunks (Vectorized)
        if rectangles:
            self._process_rectangle_chunks(rectangles, coords, dx)
            print(f"Rasterized {len(rectangles)} rectangles in chunks. Total time: {time.time() - starttime:.2f} seconds.")

        # 3. Process spheres individually (Fallback to preserve sub-sampling accuracy)
        if spheres:
            print(f"Rasterizing {len(spheres)} spheres individually...")
            for s_data in spheres:
                r_obj, net, lin, tgt, bias = s_data
                mask = r_obj.domain.rasterize(coords, dx)
                self._net_cached_rates += mask * net
                self._linear_cached_rates += mask * lin
                self._target_cached_rates += mask * tgt
                self._target_cached_bias += mask * bias

    def _process_rectangle_chunks(self, rect_data, coords, dx, chunk_size=1000):
        """
        Calculates overlaps for thousands of rectangular regions simultaneously
        using NumPy broadcasting, drastically reducing Python loop overhead.
        """
        # Unpack the list of tuples into fast NumPy arrays
        origins = np.array([d[0] for d in rect_data], dtype=float)
        sizes = np.array([d[1] for d in rect_data], dtype=float)
        nets = np.array([d[2] for d in rect_data], dtype=float)
        lins = np.array([d[3] for d in rect_data], dtype=float)
        tgts = np.array([d[4] for d in rect_data], dtype=float)
        biases = np.array([d[5] for d in rect_data], dtype=float)

        ndim = len(dx)
        n_regions = len(origins)
        
        # Reshape tuple dynamically fits 1D, 2D, or 3D grids
        # For 3D, this turns a chunk of shape (1000,) into (1000, 1, 1, 1) to broadcast against the grid
        bc_shape = (-1,) + (1,) * ndim 

        for i in range(0, n_regions, chunk_size):
            end = min(i + chunk_size, n_regions)
            chunk_len = end - i
            
            # Pre-allocate chunk overlap array: shape (chunk_size, Nx, Ny, Nz)
            overlap = np.ones((chunk_len,) + coords[0].shape, dtype=float)
            
            for dim in range(ndim):
                c = coords[0] if ndim == 1 else coords[dim]
                c_exp = c[None, ...] # Expand grid by 1 axis for broadcasting
                
                lo = origins[i:end, dim].reshape(bc_shape)
                hi = (origins[i:end, dim] + sizes[i:end, dim]).reshape(bc_shape)
                half_dx = dx[dim] / 2.0
                
                overlap_lo = np.maximum(c_exp - half_dx, lo)
                overlap_hi = np.minimum(c_exp + half_dx, hi)
                
                overlap *= np.maximum(overlap_hi - overlap_lo, 0.0) / dx[dim]
                
            # Collapse the chunk axis (axis=0) and add directly to the global arrays
            self._net_cached_rates += np.sum(overlap * nets[i:end].reshape(bc_shape), axis=0)
            self._linear_cached_rates += np.sum(overlap * lins[i:end].reshape(bc_shape), axis=0)
            self._target_cached_rates += np.sum(overlap * tgts[i:end].reshape(bc_shape), axis=0)
            self._target_cached_bias += np.sum(overlap * biases[i:end].reshape(bc_shape), axis=0)
               

    def __repr__(self) -> str:
        return f"Bulk(n_regions={len(self._regions)})"

