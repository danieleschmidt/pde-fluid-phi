# pde-fluid-phi

**Differentiable 2D Navier-Stokes solver in PyTorch.**

Inspired by [PhiFlow](https://github.com/tum-pbs/PhiFlow), this library
implements a fully differentiable incompressible fluid simulation on a
staggered MAC grid. Every operation is a PyTorch tensor op — gradients
flow backwards through the entire physics pipeline, enabling
PDE-constrained optimization via autograd.

---

## What it is

A clean implementation of the 2D incompressible Navier-Stokes equations:

```
∂u/∂t + (u·∇)u = −∇p + (1/Re)∇²u
∇·u = 0
```

Solved on a MAC (marker-and-cell) staggered grid using:
- **Semi-Lagrangian advection** — stable for large timesteps
- **Explicit viscous diffusion** — central differences
- **Pressure projection** — Jacobi iteration on the Poisson equation
- **Helmholtz decomposition** — projects velocity onto divergence-free space

---

## Architecture

```
FluidState          — 2D velocity (u, v) + pressure on a staggered grid
NavierStokesSolver  — differentiable NS timestepper
BoundaryConditions  — no-slip walls, inflow/outflow, moving lid
FlowOptimizer       — gradient-based inflow optimization via autograd
```

### Staggered MAC grid

```
  ┌────v────┬────v────┐
  │         │         │
  u    p    u    p    u
  │         │         │
  └────v────┴────v────┘

u[j,i] at x = i·dx,     y = (j+0.5)·dy   shape (ny, nx+1)
v[j,i] at x = (i+0.5)·dx, y = j·dy       shape (ny+1, nx)
p[j,i] at cell centers                     shape (ny, nx)
```

---

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.0.

---

## Quick start

### Lid-driven cavity flow

```python
from pde_fluid_phi import FluidState, NavierStokesSolver, BoundaryConditions

nx, ny = 32, 32
dx = dy = 1.0 / nx

bc = BoundaryConditions(
    nx=nx, ny=ny,
    top="moving",      # lid moves at unit velocity
    bottom="no-slip",
    left="no-slip",
    right="no-slip",
    lid_vel=1.0,
)

solver = NavierStokesSolver(
    nx=nx, ny=ny, dx=dx, dy=dy,
    dt=0.005, Re=100,
    bc=bc, n_jacobi=60,
)

state = FluidState.zeros(nx, ny, dx=dx, dy=dy)

for step in range(50):
    state = solver.step(state)

print("Max speed:", state.velocity_magnitude().max().item())
```

### Gradient-based flow optimization

Because the solver is fully differentiable, you can optimize inflow
conditions to match a target flow pattern:

```python
from pde_fluid_phi import FluidState, NavierStokesSolver, BoundaryConditions, FlowOptimizer

bc = BoundaryConditions(nx=24, ny=24, left="inflow", right="outflow",
                         top="no-slip", bottom="no-slip")
solver = NavierStokesSolver(nx=24, ny=24, dx=1/24, dy=1/24, dt=0.005, Re=50, bc=bc)

# Maximize average u-velocity in the center strip
def target_fn(state):
    return -state.u_at_centers()[:, 6:18, :].mean()

opt = FlowOptimizer(solver, n_steps=8, lr=0.05)
inflow_opt = opt.optimize(
    initial_state=FluidState.zeros(24, 24, dx=1/24, dy=1/24),
    target_fn=target_fn,
    n_iter=50,
)
```

---

## Demos

```bash
# Lid-driven cavity — 50 timesteps with gradient verification
python examples/lid_driven_cavity.py

# Flow optimization — maximize center-channel velocity
python examples/flow_optimization.py
```

---

## Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

21 tests covering FluidState, BoundaryConditions, NavierStokesSolver, and FlowOptimizer.

---

## Physics notes

**Why differentiable?**

Traditional CFD solvers treat physics as a black box — you run them
forward, then hand-design feedback controllers. A differentiable solver
lets you ask: *given the final flow state, what inflow would have
produced it?* Autograd answers that question directly, enabling:

- Inverse problems (infer boundary conditions from observations)
- PDE-constrained shape optimization
- Physics-informed neural network training data generation
- Differentiable simulation-in-the-loop for learned controllers

**MAC grid choice**

The marker-and-cell (MAC/staggered) grid co-locates each velocity
component with the face it's normal to. This avoids pressure-velocity
decoupling (checkerboard instability) that plagues collocated grids,
at the cost of slightly more complex interpolation.

**Jacobi pressure solve**

The Poisson equation `∇²p = (1/dt)∇·u*` is solved by Jacobi iteration.
Jacobi is simple and differentiable (unlike multigrid), but converges
slowly — use `n_jacobi=50–200` depending on grid size and tolerance
requirements. For production use, replace with a differentiable
conjugate gradient (e.g., via `torch.linalg.solve` on coarse grids).

---

## License

MIT
