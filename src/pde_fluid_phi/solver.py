"""
NavierStokesSolver: fully-differentiable 2D incompressible Navier-Stokes solver.

Pipeline per timestep:
  1. Semi-Lagrangian advection  (differentiable via grid_sample)
  2. Explicit viscous diffusion  (differentiable finite differences)
  3. Pressure projection         (Jacobi iteration — differentiable)
  4. Boundary condition enforcement

All PyTorch operations — gradients flow from any output back through
the entire timestepping sequence.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Optional, List

from .fluid_state import FluidState
from .boundary import BoundaryConditions


class NavierStokesSolver:
    """
    2D incompressible Navier-Stokes solver on a MAC staggered grid.

    Governing equations (dimensionless):
        ∂u/∂t + (u·∇)u = -∇p + (1/Re)∇²u
        ∇·u = 0

    Args:
        nx:           number of cells in x
        ny:           number of cells in y
        dx:           cell width
        dy:           cell height
        dt:           timestep
        Re:           Reynolds number
        bc:           BoundaryConditions instance
        n_jacobi:     Jacobi iterations for pressure solve
        device:       torch device
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float = 1.0,
        dy: float = 1.0,
        dt: float = 0.01,
        Re: float = 100.0,
        bc: Optional[BoundaryConditions] = None,
        n_jacobi: int = 50,
        device: Optional[torch.device] = None,
    ):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.Re = Re
        self.nu = 1.0 / Re          # kinematic viscosity
        self.n_jacobi = n_jacobi
        self.device = device or torch.device("cpu")
        self.bc = bc or BoundaryConditions(nx, ny)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, state: FluidState) -> FluidState:
        """Advance the fluid state by one timestep. Fully differentiable."""
        # 1. Advection
        u_adv, v_adv = self._advect(state)

        # 2. Diffusion
        u_diff = self._diffuse(u_adv, is_u=True)
        v_diff = self._diffuse(v_adv, is_u=False)

        # 3. Build intermediate state and enforce BCs before projection
        state_int = FluidState(u=u_diff, v=v_diff, p=state.p, dx=self.dx, dy=self.dy)
        state_int = self.bc.apply(state_int)

        # 4. Pressure projection (make divergence-free)
        state_out = self._project(state_int)

        # 5. Final BC enforcement
        state_out = self.bc.apply(state_out)

        return state_out

    def rollout(self, state: FluidState, n_steps: int) -> List[FluidState]:
        """Simulate n_steps timesteps, returning a list of states."""
        states = [state]
        cur = state
        for _ in range(n_steps):
            cur = self.step(cur)
            states.append(cur)
        return states

    # ------------------------------------------------------------------
    # Private: Advection (semi-Lagrangian)
    # ------------------------------------------------------------------

    def _advect(self, state: FluidState):
        """Semi-Lagrangian advection — trace backwards along characteristics.

        Uses torch.nn.functional.grid_sample for differentiable bilinear
        interpolation of the velocity field.
        """
        u, v = state.u, state.v
        dt = self.dt
        nx, ny = self.nx, self.ny

        # Interpolate velocities to a common cell-center grid first
        # u: (B, ny, nx+1) → centers (B, ny, nx)
        uc = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
        vc = 0.5 * (v[:, :-1, :] + v[:, 1:, :])

        # Build normalised coordinates for grid_sample: range [-1, 1]
        B = u.shape[0]
        # Cell-centre positions
        xs = torch.linspace(-1, 1, nx, device=self.device, dtype=u.dtype)
        ys = torch.linspace(-1, 1, ny, device=self.device, dtype=u.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (ny, nx)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Departure point: x_dep = x - dt * u(x)
        # Normalise velocity: factor = dt * 2 / (domain_size)
        norm_x = dt * 2.0 / (nx * self.dx)
        norm_y = dt * 2.0 / (ny * self.dy)

        dep_x = grid[..., 0] - norm_x * uc
        dep_y = grid[..., 1] - norm_y * vc
        dep_grid = torch.stack([dep_x, dep_y], dim=-1)  # (B, ny, nx, 2)

        # Advect u (interpolate u_c at departure point, then re-stagger)
        uc_adv = F.grid_sample(
            uc.unsqueeze(1), dep_grid, mode="bilinear",
            padding_mode="border", align_corners=True,
        ).squeeze(1)  # (B, ny, nx)

        vc_adv = F.grid_sample(
            vc.unsqueeze(1), dep_grid, mode="bilinear",
            padding_mode="border", align_corners=True,
        ).squeeze(1)  # (B, ny, nx)

        # Re-stagger: interpolate back to face positions
        u_adv = torch.zeros_like(u)
        u_adv[:, :, 1:-1] = 0.5 * (uc_adv[:, :, :-1] + uc_adv[:, :, 1:])
        u_adv[:, :, 0] = uc_adv[:, :, 0]
        u_adv[:, :, -1] = uc_adv[:, :, -1]

        v_adv = torch.zeros_like(v)
        v_adv[:, 1:-1, :] = 0.5 * (vc_adv[:, :-1, :] + vc_adv[:, 1:, :])
        v_adv[:, 0, :] = vc_adv[:, 0, :]
        v_adv[:, -1, :] = vc_adv[:, -1, :]

        return u_adv, v_adv

    # ------------------------------------------------------------------
    # Private: Diffusion (explicit viscosity)
    # ------------------------------------------------------------------

    def _diffuse(self, q: torch.Tensor, is_u: bool) -> torch.Tensor:
        """Explicit central-difference diffusion: q += dt * nu * ∇²q.

        Uses F.conv2d with a Laplacian kernel — differentiable.
        """
        laplacian_kernel = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]],
            dtype=q.dtype, device=self.device,
        ).view(1, 1, 3, 3) / (self.dx * self.dy)

        # Pad with reflect to respect walls approximately
        q_pad = F.pad(q.unsqueeze(1), (1, 1, 1, 1), mode="reflect")
        lap = F.conv2d(q_pad, laplacian_kernel).squeeze(1)  # (B, ny/ny+1, nx+1/nx)

        return q + self.dt * self.nu * lap

    # ------------------------------------------------------------------
    # Private: Pressure projection (Jacobi iteration)
    # ------------------------------------------------------------------

    def _project(self, state: FluidState) -> FluidState:
        """Helmholtz-Hodge projection to enforce ∇·u = 0.

        Solves:  ∇²p = (1/dt) ∇·u*   (Poisson equation)
        Then:    u = u* - dt ∇p

        Jacobi iteration — each iteration is a differentiable op.
        """
        u, v, p = state.u, state.v, state.p
        dt, dx, dy = self.dt, self.dx, self.dy

        # Divergence of intermediate velocity (RHS of Poisson eq)
        div = (u[:, :, 1:] - u[:, :, :-1]) / dx + (v[:, 1:, :] - v[:, :-1, :]) / dy
        rhs = div / dt  # (B, ny, nx)

        # Jacobi iterations for ∇²p = rhs
        for _ in range(self.n_jacobi):
            # Neighbour sum with zero-Neumann BC (padding)
            p_pad = F.pad(p, (1, 1, 1, 1), mode="replicate")
            p_left  = p_pad[:, 1:-1, :-2]
            p_right = p_pad[:, 1:-1, 2:]
            p_down  = p_pad[:, :-2, 1:-1]
            p_up    = p_pad[:, 2:, 1:-1]
            p = (p_left / dx**2 + p_right / dx**2
                 + p_down / dy**2 + p_up / dy**2
                 - rhs) / (2.0 / dx**2 + 2.0 / dy**2)

        # Apply Neumann pressure BCs
        p = self.bc.apply_pressure_neumann(p)

        # Subtract pressure gradient to obtain divergence-free velocity
        dp_dx = (p[:, :, 1:] - p[:, :, :-1]) / dx   # (B, ny, nx) → faces (nx+1)
        # Pad to match u shape
        dp_dx_padded = F.pad(dp_dx, (1, 1), mode="replicate")  # (B, ny, nx+1) approx
        # Trim to correct face count: only interior faces (1:nx)
        u_new = u.clone()
        u_new[:, :, 1:-1] = u[:, :, 1:-1] - dt * dp_dx

        dp_dy = (p[:, 1:, :] - p[:, :-1, :]) / dy   # (B, ny-1, nx) → faces (ny+1)
        v_new = v.clone()
        v_new[:, 1:-1, :] = v[:, 1:-1, :] - dt * dp_dy

        return FluidState(u=u_new, v=v_new, p=p, dx=dx, dy=dy)
