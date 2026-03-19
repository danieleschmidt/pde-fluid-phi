"""
FlowOptimizer: gradient-based optimization of inflow conditions.

Because NavierStokesSolver is fully differentiable, we can compute
∂L/∂θ where θ are inflow parameters and L is a loss on the flow state.
This enables PDE-constrained optimization entirely via autograd.
"""

from __future__ import annotations
import torch
import torch.optim as optim
from typing import Callable, List, Optional

from .fluid_state import FluidState
from .solver import NavierStokesSolver
from .boundary import BoundaryConditions


class FlowOptimizer:
    """
    Optimize inflow parameters to drive flow toward a target pattern.

    Example:
        optimizer = FlowOptimizer(solver, n_steps=10, lr=0.01)
        inflow = optimizer.optimize(initial_state, target_state, n_iter=100)

    Args:
        solver:      NavierStokesSolver instance
        n_steps:     how many simulation steps to unroll per optimization iteration
        lr:          learning rate for Adam
        device:      torch device
    """

    def __init__(
        self,
        solver: NavierStokesSolver,
        n_steps: int = 5,
        lr: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        self.solver = solver
        self.n_steps = n_steps
        self.lr = lr
        self.device = device or torch.device("cpu")

    def optimize(
        self,
        initial_state: FluidState,
        target_fn: Callable[[FluidState], torch.Tensor],
        inflow_param: Optional[torch.Tensor] = None,
        n_iter: int = 100,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Optimize inflow velocity to minimise target_fn applied to the
        final simulated state.

        Args:
            initial_state:  starting FluidState (detached)
            target_fn:      callable(FluidState) → scalar loss tensor
            inflow_param:   initial inflow u values (B, ny); if None, zeros
            n_iter:         number of gradient steps
            verbose:        print loss every 10 steps

        Returns:
            Optimized inflow_param tensor.
        """
        B = initial_state.batch
        ny = initial_state.ny

        if inflow_param is None:
            inflow_param = torch.zeros(
                B, ny, device=self.device, dtype=initial_state.dtype
            )

        param = inflow_param.clone().requires_grad_(True)
        adam = optim.Adam([param], lr=self.lr)

        losses: List[float] = []

        for i in range(n_iter):
            adam.zero_grad()

            # Inject current inflow param into the BC
            self.solver.bc.inflow_u = param
            self.solver.bc.left = "inflow"

            # Unroll n_steps through the differentiable solver
            state = FluidState(
                u=initial_state.u.clone(),
                v=initial_state.v.clone(),
                p=initial_state.p.clone(),
                dx=initial_state.dx,
                dy=initial_state.dy,
            )
            for _ in range(self.n_steps):
                state = self.solver.step(state)

            loss = target_fn(state)
            loss.backward()
            adam.step()

            losses.append(loss.item())
            if verbose and (i % 10 == 0 or i == n_iter - 1):
                print(f"  iter {i:4d}  loss={loss.item():.6f}")

        self.losses = losses
        return param.detach()

    @staticmethod
    def vorticity_loss(state: FluidState, target_vorticity: torch.Tensor) -> torch.Tensor:
        """MSE between simulated and target vorticity fields."""
        uc = state.u_at_centers()
        vc = state.v_at_centers()
        # Central-difference curl: ∂v/∂x - ∂u/∂y
        dvdx = (vc[:, :, 1:] - vc[:, :, :-1]) / state.dx
        dudy = (uc[:, 1:, :] - uc[:, :-1, :]) / state.dy
        # Trim to common size
        h = min(dvdx.shape[1], dudy.shape[1])
        w = min(dvdx.shape[2], dudy.shape[2])
        omega = dvdx[:, :h, :w] - dudy[:, :h, :w]
        t = target_vorticity[:, :h, :w]
        return ((omega - t) ** 2).mean()

    @staticmethod
    def velocity_loss(state: FluidState, target_u: torch.Tensor, target_v: torch.Tensor) -> torch.Tensor:
        """MSE between simulated and target velocity at cell centers."""
        uc = state.u_at_centers()
        vc = state.v_at_centers()
        return ((uc - target_u) ** 2 + (vc - target_v) ** 2).mean()
