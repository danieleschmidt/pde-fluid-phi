"""Tests for FluidState."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pytest
from pde_fluid_phi.fluid_state import FluidState


def test_zeros_shape():
    s = FluidState.zeros(nx=8, ny=6)
    assert s.u.shape == (1, 6, 9)   # (B, ny, nx+1)
    assert s.v.shape == (1, 7, 8)   # (B, ny+1, nx)
    assert s.p.shape == (1, 6, 8)   # (B, ny, nx)


def test_batch_shape():
    s = FluidState.zeros(nx=4, ny=4, batch=3)
    assert s.batch == 3
    assert s.u.shape[0] == 3


def test_u_at_centers():
    s = FluidState.zeros(nx=4, ny=4)
    s.u[:] = 2.0
    uc = s.u_at_centers()
    assert uc.shape == (1, 4, 4)
    assert torch.allclose(uc, torch.tensor(2.0))


def test_v_at_centers():
    s = FluidState.zeros(nx=4, ny=4)
    s.v[:] = 3.0
    vc = s.v_at_centers()
    assert vc.shape == (1, 4, 4)
    assert torch.allclose(vc, torch.tensor(3.0))


def test_divergence_zero_field():
    s = FluidState.zeros(nx=8, ny=8)
    div = s.divergence()
    assert div.abs().max().item() == 0.0


def test_velocity_magnitude():
    s = FluidState.zeros(nx=4, ny=4)
    s.u[:] = 3.0
    s.v[:] = 4.0
    mag = s.velocity_magnitude()
    # sqrt(3^2 + 4^2) = 5
    assert torch.allclose(mag, torch.tensor(5.0), atol=1e-4)


def test_clone_independence():
    s = FluidState.zeros(nx=4, ny=4)
    s2 = s.clone()
    s2.u[:] = 999.0
    assert s.u.max().item() == 0.0


def test_detach_no_grad():
    s = FluidState.zeros(nx=4, ny=4)
    s.u.requires_grad_(True)
    sd = s.detach()
    assert not sd.u.requires_grad
