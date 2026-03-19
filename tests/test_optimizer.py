"""Tests for FlowOptimizer."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pytest
from pde_fluid_phi.fluid_state import FluidState
from pde_fluid_phi.boundary import BoundaryConditions
from pde_fluid_phi.solver import NavierStokesSolver
from pde_fluid_phi.optimizer import FlowOptimizer


def make_channel_solver(nx=16, ny=16):
    bc = BoundaryConditions(nx, ny, top="no-slip", bottom="no-slip",
                             left="inflow", right="outflow")
    return NavierStokesSolver(nx=nx, ny=ny, dx=1/nx, dy=1/ny,
                               dt=0.005, Re=50, bc=bc, n_jacobi=30)


def test_optimizer_returns_tensor():
    solver = make_channel_solver()
    state = FluidState.zeros(16, 16, dx=1/16, dy=1/16)

    opt = FlowOptimizer(solver, n_steps=3, lr=0.05)
    result = opt.optimize(state, lambda s: -s.u_at_centers().mean(),
                          n_iter=5, verbose=False)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 16)


def test_optimizer_reduces_loss():
    nx, ny = 16, 16
    solver = make_channel_solver(nx, ny)
    state = FluidState.zeros(nx, ny, dx=1/nx, dy=1/ny)
    cy, cy_end = ny//4, 3*ny//4

    def target_fn(s):
        return -s.u_at_centers()[:, cy:cy_end, :].mean()

    opt = FlowOptimizer(solver, n_steps=4, lr=0.05)
    opt.optimize(state, target_fn, n_iter=20, verbose=False)
    # Loss should decrease over iterations
    assert opt.losses[-1] < opt.losses[0], (
        f"Loss did not decrease: {opt.losses[0]:.4f} → {opt.losses[-1]:.4f}"
    )


def test_velocity_loss_static():
    state = FluidState.zeros(8, 8)
    target_u = torch.zeros(1, 8, 8)
    target_v = torch.zeros(1, 8, 8)
    loss = FlowOptimizer.velocity_loss(state, target_u, target_v)
    assert loss.item() == 0.0
