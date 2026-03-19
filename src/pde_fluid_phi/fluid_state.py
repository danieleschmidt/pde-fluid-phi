"""
FluidState: 2D velocity and pressure fields on a staggered (MAC) grid.

Staggered grid layout (MAC grid):
  - u (x-velocity): defined at horizontal cell faces, shape (ny, nx+1)
  - v (y-velocity): defined at vertical cell faces,  shape (ny+1, nx)
  - p (pressure):   defined at cell centers,          shape (ny, nx)

All tensors are PyTorch — gradients flow through all operations.
"""

from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class FluidState:
    """Holds the complete fluid state on a 2D staggered MAC grid.

    Args:
        u:  x-velocity at horizontal faces, shape (batch, ny, nx+1)
        v:  y-velocity at vertical faces,   shape (batch, ny+1, nx)
        p:  pressure at cell centers,        shape (batch, ny, nx)
        dx: grid spacing in x
        dy: grid spacing in y
    """
    u: torch.Tensor   # (B, ny, nx+1)
    v: torch.Tensor   # (B, ny+1, nx)
    p: torch.Tensor   # (B, ny, nx)
    dx: float = 1.0
    dy: float = 1.0

    @staticmethod
    def zeros(
        nx: int,
        ny: int,
        dx: float = 1.0,
        dy: float = 1.0,
        batch: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "FluidState":
        """Create a zero-initialised fluid state."""
        kw = dict(device=device, dtype=dtype)
        return FluidState(
            u=torch.zeros(batch, ny, nx + 1, **kw),
            v=torch.zeros(batch, ny + 1, nx, **kw),
            p=torch.zeros(batch, ny, nx, **kw),
            dx=dx,
            dy=dy,
        )

    @property
    def device(self) -> torch.device:
        return self.u.device

    @property
    def dtype(self) -> torch.dtype:
        return self.u.dtype

    @property
    def batch(self) -> int:
        return self.u.shape[0]

    @property
    def ny(self) -> int:
        return self.p.shape[1]

    @property
    def nx(self) -> int:
        return self.p.shape[2]

    def u_at_centers(self) -> torch.Tensor:
        """Interpolate u from faces to cell centers. Shape (B, ny, nx)."""
        return 0.5 * (self.u[:, :, :-1] + self.u[:, :, 1:])

    def v_at_centers(self) -> torch.Tensor:
        """Interpolate v from faces to cell centers. Shape (B, ny, nx)."""
        return 0.5 * (self.v[:, :-1, :] + self.v[:, 1:, :])

    def velocity_magnitude(self) -> torch.Tensor:
        """Speed at cell centers. Shape (B, ny, nx)."""
        uc = self.u_at_centers()
        vc = self.v_at_centers()
        return torch.sqrt(uc ** 2 + vc ** 2 + 1e-12)

    def divergence(self) -> torch.Tensor:
        """Velocity divergence at cell centers (should → 0 after projection).
        Shape (B, ny, nx)."""
        du = (self.u[:, :, 1:] - self.u[:, :, :-1]) / self.dx
        dv = (self.v[:, 1:, :] - self.v[:, :-1, :]) / self.dy
        return du + dv

    def detach(self) -> "FluidState":
        """Return a copy with tensors detached from the computation graph."""
        return FluidState(
            u=self.u.detach(),
            v=self.v.detach(),
            p=self.p.detach(),
            dx=self.dx,
            dy=self.dy,
        )

    def clone(self) -> "FluidState":
        return FluidState(
            u=self.u.clone(),
            v=self.v.clone(),
            p=self.p.clone(),
            dx=self.dx,
            dy=self.dy,
        )
