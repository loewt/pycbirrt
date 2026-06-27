# Gafro backend: drive IK/planning off the task space's controllable DOF

**Date:** 2026-06-26
**Scope:** `pycbirrt/src/pycbirrt/backends/gafro.py` and `pycbirrt/examples/franka_tsr_interactive.py` only.

## Problem

Planning a `geodude.xml` `left_ur5e/endeffector_link` chain crashes in the gafro IK
solver:

```
ValueError: operands could not be broadcast together with shapes (7,) (6,)
  q = self._clamp_to_limits(q + dq)
```

### Root cause (verified empirically)

The backend conflates three numbers that are equal for a plain revolute arm but
diverge for a chain that contains a non-revolute joint:

| Quantity | Source | Value for this chain |
|---|---|---|
| Reported DOF | `manipulator.get_dof()` | **7** |
| Joint-limit length | `get_joint_limits_min/max()` | **7** |
| Geometric Jacobian columns | `compute_ee_geometric_jacobian(q).shape[1]` | **6** |
| EE error dimension | `Motor.log()` | 6 |

`solve()` builds `q` and the joint-limit clamp from `get_dof()` (7) but builds
`dq = J.T @ ...` from the geometric Jacobian (6 columns). `q + dq` is then
`(7,) + (6,)` → broadcast error.

The chain's first joint (index 0) is a **prismatic torso joint**, range `[0, 0.5]`.
All 7 joints move the end-effector, but gafro's **geometric (rotor-derivative)
Jacobian models only the 6 revolute joints (indices 1..6) and omits the prismatic
joint 0.** Verified:

- `J_geo @ dq == log(M_new · M_old⁻¹)` exactly, for a `dq` applied to chain joints 1..6.
- `J_geo`'s linear rows equal the finite-difference Jacobian's columns 1..6.

So the geometric Jacobian's **column count is the true controllable DOF** for the
CGA-log error convention the backend already uses, and those columns map to the
**trailing** chain joints (1..6); the leading joint(s) are not part of this
task space's controllable set.

## Design principle

**The task space's geometric Jacobian is the single source of truth for the
controllable joint set.** The backend derives `dof`, joint limits, the IK update
`dq`, and the clamp all from the Jacobian's column count — never from
`get_dof()`. This is what "generic over the task space, not the kinematic chain"
means concretely: any task space (revolute arm, chain with a passive/prismatic
joint, and in principle cooperative/primitive task spaces) drives planning at the
DOF its own Jacobian exposes, with no per-robot configuration.

The CBiRRT planner, the `RobotModel`/`IKSolver` Protocols, and `tsr/` are already
DOF-agnostic — they consume only `dof`, `joint_limits`, `forward_kinematics`, and
`solve_valid`. They need **no changes**; they follow automatically once the
backend reports a self-consistent DOF.

## Components

### 1. Controllable-DOF helper (shared)

A small helper derives, from a `SingleArmTaskSpace`, the consistent triple used by
both the model and the solver:

- `ctrl_dof = compute_ee_geometric_jacobian(q_probe).shape[1]` (probe once with a
  zero or midpoint config).
- `full_dof = get_dof()`.
- `n_fixed = full_dof - ctrl_dof` — the number of leading joints the Jacobian
  omits (held fixed). The geometric Jacobian omits the **leading** joints, so the
  controllable joints are the **trailing** `ctrl_dof` entries; verified for this
  chain (`J_geo` columns == chain joints 1..6). The helper asserts
  `n_fixed >= 0` and, when `n_fixed == 0`, is a no-op (the common revolute-arm
  case is unaffected).
- Controllable joint limits = the **last `ctrl_dof`** entries of
  `get_joint_limits_min/max()`.

A pair of converters bridges controllable-space and full-chain space:

- `to_full(q_ctrl, fixed)` → length-`full_dof` vector with the leading `n_fixed`
  entries taken from `fixed` and the trailing `ctrl_dof` from `q_ctrl`.
- `to_ctrl(q_full)` → the trailing `ctrl_dof` entries.

`compute_ee_motor` and `compute_ee_geometric_jacobian` are always called with the
full-chain vector.

### 2. `GafroIKSolver` — operate in controllable space

- `self._dof` = `ctrl_dof` (from the Jacobian), not `get_dof()`.
- `self.joint_limits` = controllable limits (trailing `ctrl_dof`).
- The non-controlled (leading) joints are **held fixed** at the value supplied via
  `q_init` (if `q_init` has full length, its leading entries are the hold values;
  if it has controllable length or is `None`, the hold values default to the
  controllable-limit-midpoint-padded base / zeros). This matches the agreed
  behavior: IK only moves the task space's joints; the prismatic torso is a fixed
  mounting offset for this task space.
- Iteration: `q` is length `ctrl_dof`; each step rebuilds the full vector via
  `to_full` for FK and Jacobian, computes the 6-vector `error` and the
  `ctrl_dof`-length `dq`, and `q + dq` now matches. `_clamp_to_limits` clamps in
  controllable space.
- Return values stay length `ctrl_dof` (the DOF the planner plans in). `solve_valid`
  limit-checks against the controllable limits.

### 3. `GafroRobotModel` — consistent adapter (kept, not deleted)

`GafroRobotModel` remains a thin `RobotModel` adapter because the example uses
`.system` / `.mesh_root` / `.manipulator` for the viewer, and the planner expects
the Protocol shape. It is made consistent with the solver:

- `.dof` → `ctrl_dof` (via the shared helper), not `get_dof()`.
- `.joint_limits` → controllable limits.
- `.forward_kinematics(q)` accepts a controllable-length `q`, rebuilds the
  full-chain vector with the fixed leading joints held at their base value, and
  returns the `Motor`.

The fixed-joint base value for the model defaults to the controllable picture's
implied zeros for the leading joints (the torso at its lower limit / supplied
base); FK is unaffected for `n_fixed == 0` robots.

### 4. Example (`franka_tsr_interactive.py`)

- `START_Q` and `angular_joints` are sized to `robot.dof` (now `ctrl_dof`) instead
  of hardcoded 7. `START_Q` becomes the controllable-DOF start (the trailing
  revolute joints).
- `_pad(q, system.get_dof())` continues to expand a controllable config to the
  full **system** vector for the visualizer; it already zero-pads, so the only
  change is that it now receives a `ctrl_dof`-length `q`. The mapping of
  controllable joints into the right system slots for the viewer is handled by the
  model's full-chain reconstruction where the viewer needs FK; the raw
  `robot_viz.update` path keeps using `_pad` against the system DOF.

## Error handling

- The helper asserts `ctrl_dof <= full_dof`. If a future task space exposes a
  Jacobian with *more* columns than `get_dof()` (not expected), it raises a clear
  `ValueError` rather than silently mis-slicing.
- `n_fixed == 0` (every revolute arm today) is an explicit fast path: `to_full` /
  `to_ctrl` are identities, so existing single-arm robots behave exactly as before.

## Testing

1. **Regression (the crash):** plan the `geodude.xml` `left_ur5e/endeffector_link`
   chain to a TSR; assert no broadcast error and that a path of `ctrl_dof`-width
   waypoints is returned (`--no-viz` path of the example).
2. **DOF consistency:** assert `solver._dof == model.dof ==
   compute_ee_geometric_jacobian(probe).shape[1]` and that `len(joint_limits[0])`
   matches.
3. **`J @ dq == error`:** unit-check that one damped-LS step's `J @ dq` reproduces
   the CGA log error twist (guards the convention the fix relies on).
4. **No regression for revolute arms:** a plain 6/7-DOF arm chain
   (`n_fixed == 0`) still solves IK and plans, with identity controllable/full
   mapping.
5. **Fixed-joint hold:** after IK, the reconstructed full config's leading joints
   equal the supplied base (the torso did not move).

## Out of scope

- No changes to `planner.py`, the Protocols, `tsr/`, or other backends
  (`mujoco.py`, `eaik.py`).
- No gafro-side change to make the geometric Jacobian include the prismatic joint;
  the prismatic joint is intentionally held fixed for this task space.
- No new task-space-as-explicit-API-parameter surface across pycbirrt/tsr.
