"""
Lid-driven cavity flow demo.

Classic benchmark for 2D incompressible flow solvers.  A square domain
with no-slip walls; the top wall moves at unit velocity in +x.

Run:
    python examples/lid_driven_cavity.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np

from pde_fluid_phi import FluidState, NavierStokesSolver, BoundaryConditions


def run(nx=32, ny=32, Re=100, n_steps=50, dt=0.005):
    print(f"Lid-driven cavity: {nx}x{ny} grid, Re={Re}, {n_steps} steps, dt={dt}")
    device = torch.device("cpu")

    dx = 1.0 / nx
    dy = 1.0 / ny

    # Lid-driven BCs: moving top lid, no-slip on other walls
    bc = BoundaryConditions(
        nx=nx, ny=ny,
        top="moving",
        bottom="no-slip",
        left="no-slip",
        right="no-slip",
        lid_vel=1.0,
    )

    solver = NavierStokesSolver(
        nx=nx, ny=ny, dx=dx, dy=dy, dt=dt, Re=Re,
        bc=bc, n_jacobi=60, device=device,
    )

    state = FluidState.zeros(nx, ny, dx=dx, dy=dy, batch=1, device=device)

    # Simulate
    max_speeds = []
    divergences = []

    for step_i in range(n_steps):
        state = solver.step(state)

        speed_max = state.velocity_magnitude().max().item()
        div_max = state.divergence().abs().max().item()
        max_speeds.append(speed_max)
        divergences.append(div_max)

        if (step_i + 1) % 10 == 0:
            uc = state.u_at_centers()[0]  # (ny, nx)
            vc = state.v_at_centers()[0]
            print(
                f"  step {step_i+1:3d} | "
                f"max speed={speed_max:.4f} | "
                f"max |div|={div_max:.2e} | "
                f"mean u={uc.mean().item():+.4f} | "
                f"mean v={vc.mean().item():+.4f}"
            )

    # Final state summary
    print("\n=== Final state ===")
    uc = state.u_at_centers()[0].numpy()
    vc = state.v_at_centers()[0].numpy()
    p = state.p[0].numpy()
    print(f"  u: min={uc.min():.4f}  max={uc.max():.4f}  mean={uc.mean():.4f}")
    print(f"  v: min={vc.min():.4f}  max={vc.max():.4f}  mean={vc.mean():.4f}")
    print(f"  p: min={p.min():.4f}  max={p.max():.4f}  mean={p.mean():.4f}")
    print(f"  Divergence max (final): {divergences[-1]:.2e}")
    print(f"  Divergence max (avg):   {np.mean(divergences):.2e}")

    # Gradient test — verify backprop through solver
    print("\n=== Gradient test ===")
    state0 = FluidState.zeros(nx, ny, dx=dx, dy=dy, batch=1, device=device)
    # Make top-lid velocity a learnable scalar
    lid_vel_param = torch.tensor(1.0, requires_grad=True)

    bc_grad = BoundaryConditions(nx=nx, ny=ny, top="moving", bottom="no-slip",
                                  left="no-slip", right="no-slip")
    bc_grad.lid_vel = lid_vel_param
    solver_grad = NavierStokesSolver(
        nx=nx, ny=ny, dx=dx, dy=dy, dt=dt, Re=Re,
        bc=bc_grad, n_jacobi=30, device=device,
    )

    s = state0
    for _ in range(5):
        s = solver_grad.step(s)

    loss = s.velocity_magnitude().mean()
    loss.backward()
    grad = lid_vel_param.grad
    print(f"  d(mean_speed)/d(lid_vel) = {grad.item():.6f}")
    assert grad is not None and abs(grad.item()) > 0, "Gradient is zero!"
    print("  ✓ Gradient flows correctly through the solver.")

    return state


if __name__ == "__main__":
    state = run(nx=32, ny=32, Re=100, n_steps=50, dt=0.005)
    print("\nDone.")
