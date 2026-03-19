"""
BoundaryConditions: apply no-slip walls and inflow/outflow conditions.

MAC staggered grid conventions:
  - u[j, i]: x-velocity at face x=i*dx, y=(j+0.5)*dy  →  shape (ny, nx+1)
  - v[j, i]: y-velocity at face x=(i+0.5)*dx, y=j*dy  →  shape (ny+1, nx)
  - p[j, i]: pressure at cell center                   →  shape (ny, nx)

The left/right u-faces and top/bottom v-faces sit *exactly* on the walls, so
setting them is an exact condition. The tangential components (u at top/bottom
rows, v at left/right columns) are set at the nearest interior face — a
first-order approximation that avoids ghost-cell divergence blowup.

Supported types per wall:
  - "no-slip"  : zero normal + zero tangential velocity
  - "inflow"   : prescribed velocity (differentiable parameter)
  - "outflow"  : zero-gradient (Neumann)
  - "moving"   : tangential velocity prescribed (e.g. lid)
"""

from __future__ import annotations
import torch
from typing import Optional, Tuple
from .fluid_state import FluidState


class BoundaryConditions:
    """
    Boundary conditions for a 2D rectangular domain on a MAC staggered grid.

    Args:
        nx, ny:   grid dimensions (cell count)
        top:      BC type for top wall    ("no-slip" | "moving" | "outflow")
        bottom:   BC type for bottom wall
        left:     BC type for left wall   ("no-slip" | "inflow" | "outflow")
        right:    BC type for right wall  ("no-slip" | "outflow")
        lid_vel:  x-velocity of moving top lid
        inflow_u: prescribed u-profile at left inflow, shape (B, ny) or scalar
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        top: str = "no-slip",
        bottom: str = "no-slip",
        left: str = "no-slip",
        right: str = "no-slip",
        lid_vel: float = 1.0,
        inflow_u: Optional[torch.Tensor] = None,
        inflow_v: Optional[torch.Tensor] = None,
    ):
        self.nx = nx
        self.ny = ny
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.lid_vel = lid_vel
        self.inflow_u = inflow_u
        self.inflow_v = inflow_v

    def apply(self, state: FluidState) -> FluidState:
        """Apply boundary conditions, returning a new FluidState."""
        u = state.u.clone()
        v = state.v.clone()

        # --- Top wall ---
        # v at top wall face (exact no-penetration)
        if self.top in ("no-slip", "moving"):
            v[:, -1, :] = 0.0
        elif self.top == "outflow":
            v[:, -1, :] = v[:, -2, :]

        # u at top row (tangential, approximate)
        if self.top == "no-slip":
            u[:, -1, :] = 0.0
        elif self.top == "moving":
            # Smoothed lid: sin² profile → zero at corners, avoids the
            # velocity-discontinuity singularity at lid-wall junctions.
            # u = lid_vel * sin²(π * i / (nx+1))  for face i in [0, nx]
            lid_val = self._lid_val(state)
            nx1 = u.shape[-1]  # nx+1 faces
            pi = torch.acos(torch.zeros(1, device=u.device)).item() * 2
            idx = torch.arange(nx1, dtype=u.dtype, device=u.device)
            smooth = torch.sin(pi * idx / (nx1 - 1)) ** 2  # (nx+1,)
            u[:, -1, :] = lid_val * smooth
        elif self.top == "outflow":
            u[:, -1, :] = u[:, -2, :]

        # --- Bottom wall ---
        v[:, 0, :] = 0.0  # no-penetration (exact)
        if self.bottom == "no-slip":
            u[:, 0, :] = 0.0
        elif self.bottom == "outflow":
            u[:, 0, :] = u[:, 1, :]

        # --- Left wall ---
        if self.left == "no-slip":
            u[:, :, 0] = 0.0          # no-penetration (exact face)
            v[:, :, 0] = 0.0          # tangential (approx)
        elif self.left == "inflow":
            if self.inflow_u is not None:
                u[:, :, 0] = self.inflow_u
            else:
                u[:, :, 0] = 1.0
            v[:, :, 0] = 0.0
        elif self.left == "outflow":
            u[:, :, 0] = u[:, :, 1]
            v[:, :, 0] = v[:, :, 1]

        # --- Right wall ---
        if self.right == "no-slip":
            u[:, :, -1] = 0.0         # no-penetration (exact)
            v[:, :, -1] = 0.0         # tangential (approx)
        elif self.right == "outflow":
            u[:, :, -1] = u[:, :, -2]
            v[:, :, -1] = v[:, :, -2]

        return FluidState(u=u, v=v, p=state.p, dx=state.dx, dy=state.dy)

    def apply_pressure_neumann(self, p: torch.Tensor) -> torch.Tensor:
        """Zero-gradient (Neumann) pressure BC on all walls. Shape (B, ny, nx)."""
        p = p.clone()
        p[:, :, 0]  = p[:, :, 1]    # left
        p[:, :, -1] = p[:, :, -2]   # right
        p[:, 0, :]  = p[:, 1, :]    # bottom
        p[:, -1, :] = p[:, -2, :]   # top
        return p

    def _lid_val(self, state: FluidState) -> torch.Tensor:
        """Return lid velocity as a tensor compatible with u's top row."""
        if isinstance(self.lid_vel, torch.Tensor):
            return self.lid_vel.expand_as(state.u[:, -1, :])
        return torch.tensor(float(self.lid_vel), dtype=state.u.dtype, device=state.u.device)
