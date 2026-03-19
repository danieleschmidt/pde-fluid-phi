"""Tests for BoundaryConditions."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pytest
from pde_fluid_phi.fluid_state import FluidState
from pde_fluid_phi.boundary import BoundaryConditions


def make_state(nx=8, ny=8, fill=1.0):
    s = FluidState.zeros(nx, ny)
    s.u[:] = fill
    s.v[:] = fill
    return s


def test_noslip_walls_zero_velocity():
    nx, ny = 8, 8
    bc = BoundaryConditions(nx, ny, top="no-slip", bottom="no-slip",
                             left="no-slip", right="no-slip")
    state = make_state(nx, ny)
    out = bc.apply(state)
    # left u-face should be 0
    assert out.u[:, :, 0].abs().max().item() == 0.0
    # top v-face should be 0
    assert out.v[:, -1, :].abs().max().item() == 0.0
    # bottom v-face should be 0
    assert out.v[:, 0, :].abs().max().item() == 0.0


def test_moving_lid_sets_top_u():
    nx, ny = 8, 8
    bc = BoundaryConditions(nx, ny, top="moving", lid_vel=2.0,
                             bottom="no-slip", left="no-slip", right="no-slip")
    state = make_state(nx, ny, fill=0.0)
    out = bc.apply(state)
    # Moving lid: top row set to lid_vel * sin²(π*i/(nx)).
    # Corners (i=0 and i=nx) must be exactly 0 (no velocity singularity).
    assert out.u[:, -1, 0].abs().max().item() == 0.0,  "Left corner should be 0"
    assert out.u[:, -1, -1].abs().max().item() == 0.0, "Right corner should be 0"
    # Middle face (i = nx//2) should be close to lid_vel
    mid = nx // 2
    assert out.u[:, -1, mid].item() > 1.5, "Mid-lid velocity should be near lid_vel=2"


def test_inflow_left():
    nx, ny = 8, 8
    inflow = torch.ones(1, ny) * 0.5
    bc = BoundaryConditions(nx, ny, left="inflow", right="outflow",
                             top="no-slip", bottom="no-slip", inflow_u=inflow)
    state = make_state(nx, ny, fill=0.0)
    out = bc.apply(state)
    assert torch.allclose(out.u[:, :, 0], inflow)


def test_pressure_neumann():
    nx, ny = 8, 8
    bc = BoundaryConditions(nx, ny)
    p = torch.rand(1, ny, nx)
    p_bc = bc.apply_pressure_neumann(p)
    # Left wall: p[:,:,0] == p[:,:,1]
    assert torch.allclose(p_bc[:, :, 0], p_bc[:, :, 1])
    # Right wall
    assert torch.allclose(p_bc[:, :, -1], p_bc[:, :, -2])
    # Top
    assert torch.allclose(p_bc[:, -1, :], p_bc[:, -2, :])
    # Bottom
    assert torch.allclose(p_bc[:, 0, :], p_bc[:, 1, :])
