"""
Flow optimization demo.

Uses FlowOptimizer to find inflow velocity profiles that produce
a target flow pattern (e.g., maximum average x-velocity in the center).

Run:
    python examples/flow_optimization.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from pde_fluid_phi import FluidState, NavierStokesSolver, BoundaryConditions, FlowOptimizer


def run(nx=24, ny=24, Re=50, n_steps=8, n_iter=50, lr=0.05):
    print(f"Flow optimization: {nx}x{ny} grid, Re={Re}")
    print(f"Goal: maximize u-velocity in the center of the domain\n")

    device = torch.device("cpu")
    dx = 1.0 / nx
    dy = 1.0 / ny

    bc = BoundaryConditions(
        nx=nx, ny=ny,
        top="no-slip",
        bottom="no-slip",
        left="inflow",
        right="outflow",
    )

    solver = NavierStokesSolver(
        nx=nx, ny=ny, dx=dx, dy=dy, dt=0.005, Re=Re,
        bc=bc, n_jacobi=40, device=device,
    )

    initial_state = FluidState.zeros(nx, ny, dx=dx, dy=dy, batch=1, device=device)

    # Target: maximize u-velocity in the center strip
    cy = ny // 4
    cy_end = 3 * ny // 4

    def target_fn(state: FluidState) -> torch.Tensor:
        uc = state.u_at_centers()  # (B, ny, nx)
        center_u = uc[:, cy:cy_end, :]
        # Minimise negative of center u (maximise it)
        return -center_u.mean()

    opt = FlowOptimizer(solver, n_steps=n_steps, lr=lr, device=device)

    print("Optimizing inflow profile...")
    inflow_opt = opt.optimize(
        initial_state=initial_state,
        target_fn=target_fn,
        n_iter=n_iter,
        verbose=True,
    )

    print(f"\nOptimized inflow u profile (mean={inflow_opt.mean().item():.4f}):")
    print(f"  min={inflow_opt.min().item():.4f}  max={inflow_opt.max().item():.4f}")
    print(f"  shape: {tuple(inflow_opt.shape)}")

    # Compare baseline vs optimized
    def eval_flow(inflow_u):
        bc_eval = BoundaryConditions(
            nx=nx, ny=ny, top="no-slip", bottom="no-slip",
            left="inflow", right="outflow", inflow_u=inflow_u,
        )
        solver_eval = NavierStokesSolver(
            nx=nx, ny=ny, dx=dx, dy=dy, dt=0.005, Re=Re,
            bc=bc_eval, n_jacobi=40,
        )
        s = FluidState.zeros(nx, ny, dx=dx, dy=dy, batch=1)
        for _ in range(n_steps):
            s = solver_eval.step(s)
        uc = s.u_at_centers()
        return uc[:, cy:cy_end, :].mean().item()

    baseline_u = torch.zeros(1, ny)
    baseline_center = eval_flow(baseline_u)

    opt_inflow = inflow_opt.detach().unsqueeze(0) if inflow_opt.dim() == 1 else inflow_opt.detach()
    optimized_center = eval_flow(opt_inflow)

    print(f"\nCenter u-velocity: baseline={baseline_center:.4f}  optimized={optimized_center:.4f}")
    improvement = optimized_center - baseline_center
    print(f"Improvement: {improvement:+.4f}")
    assert optimized_center > baseline_center, "Optimization did not improve the objective!"
    print("✓ Optimizer successfully improved center flow.")


if __name__ == "__main__":
    run()
    print("\nDone.")
