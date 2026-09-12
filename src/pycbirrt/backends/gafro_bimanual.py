# SPDX-License-Identifier: MIT
"""Native gafro bimanual backend: DualArmTaskSpace kinematics + IK.

Forward kinematics returns a :class:`~tsr.bimanual.BimanualPose` — the
(absolute, relative) pose pair a :class:`~tsr.bimanual.BimanualTSR` speaks in.
The IK solver (see :class:`GafroBimanualIKSolver`) stacks only the Jacobians of
the pose components that are present in the target.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gafro import SystemSerialization
from scipy.optimize import least_squares
from tsr.bimanual import BimanualPose

from pycbirrt.backends.gafro import _as_vector

if TYPE_CHECKING:
    from gafro import System


class GafroBimanualModel:
    """Dual-arm kinematics over a gafro DualArmTaskSpace."""

    def __init__(self, system: "System", task_space_name):
        self.system = system
        self.cooperative = system.get_task_space(task_space_name)

        self._ctrl_idx = np.asarray(
            self.cooperative.get_controlled_joint_indices(), dtype=int)

        # gafro returns joint limits as a JointPosition wrapper rather than an
        # array; _as_vector reads its coefficients (gafropy returned an array).
        self._lower = _as_vector(system.get_joint_limits_min())
        self._upper = _as_vector(system.get_joint_limits_max())

        # The URDF/YAML-declared rest pose can sit fractions of a radian outside its
        # own declared limits (rounding in the robot description, e.g. gripper
        # joints on this rig) -- clamp so anything seeded from it (IK's default
        # init, a planner root) is actually valid, not silently poisoned from the start.
        self.default_system_configuration = np.clip(
            _as_vector(system.get_default_configuration()),
            self._lower, self._upper,
        )

    @classmethod
    def from_file(cls, path: str, task_space_name) -> "GafroBimanualModel":
        return cls(SystemSerialization.load(path), task_space_name)

    @property
    def dof(self) -> int:
        return len(self._ctrl_idx)

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lower, self._upper

    def forward_kinematics(self, q: np.ndarray) -> BimanualPose:
        # gafropy returned both task-space motors from one call; gafro exposes
        # them as two accessors.
        return BimanualPose(
            absolute=self.cooperative.compute_absolute_motor(q),
            relative=self.cooperative.compute_relative_motor(q),
        )

    def normalize_pose(self, x) -> BimanualPose:
        """Pass-through: bimanual poses are already :class:`BimanualPose`."""
        if not isinstance(x, BimanualPose):
            raise TypeError(f"expected BimanualPose, got {type(x)}")
        return x


class GafroBimanualIKSolver:
    """Differential IK for a bimanual target, solved with SciPy's trust-region
    least squares (``scipy.optimize.least_squares``, method ``"trf"``).

    The residual stacks the CGA error twist(s) of whichever pose components
    are present in the target (absolute, relative, or both); the Jacobian is
    the matching stack of gafro's analytic geometric Jacobians, supplied to
    the solver directly rather than finite-differenced. ``"trf"`` also
    respects the joint limits natively, so solutions never need clamping.
    """

    def __init__(self, model: "GafroBimanualModel", collision_checker=None,
                 max_iterations: int = 200, tolerance: float = 1e-3):
        self.model = model
        self.coop = model.cooperative
        self._ctrl_idx = model._ctrl_idx
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.collision_checker = collision_checker

    def _residual_and_jacobian(self, target: BimanualPose, q: np.ndarray):
        errors = []
        jacobians = []
        if target.absolute is not None:
            current = self.coop.compute_absolute_motor(q)
            errors.append(np.asarray(
                current.inverse().multiply(target.absolute).log(), dtype=float))
            jacobians.append(np.asarray(
                self.coop.compute_absolute_geometric_jacobian(q), dtype=float))

        if target.relative is not None:
            current = self.coop.compute_relative_motor(q)
            errors.append(np.asarray(
                current.inverse().multiply(target.relative).log(), dtype=float))
            jacobians.append(np.asarray(
                self.coop.compute_relative_geometric_jacobian(q), dtype=float))

        return np.concatenate(errors), np.vstack(jacobians)

    def solve(self, target: BimanualPose, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        if target.absolute is None and target.relative is None:
            raise ValueError("bimanual IK target has no active component")

        lower, upper = self.model.joint_limits
        if q_init is not None:
            q = np.asarray(q_init, dtype=float).copy()
        else:
            # The joint-limit midpoint (all-zero for this rig) is a poor seed for a
            # coupled 12-DOF bimanual target -- it's often near-singular for one arm
            # or the other. The model's declared rest pose is a real, reachable
            # configuration, so it converges far more reliably as a default.
            q = self.model.default_system_configuration.copy()

        # error(q) = current(q)^-1 * target, so d(error)/dq = -geometric_jacobian(q).
        def residual(q_ctrl):
            q[self._ctrl_idx] = q_ctrl
            error, _ = self._residual_and_jacobian(target, q)
            return error

        def jacobian(q_ctrl):
            q[self._ctrl_idx] = q_ctrl
            _, J = self._residual_and_jacobian(target, q)
            return -J

        result = least_squares(
            residual, q[self._ctrl_idx], jac=jacobian,
            bounds=(lower[self._ctrl_idx], upper[self._ctrl_idx]),
            method="trf", max_nfev=self.max_iterations,
        )

        if np.linalg.norm(result.fun) >= self.tolerance:
            return []

        q[self._ctrl_idx] = result.x
        return [q]

    def solve_valid(self, target: BimanualPose, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        solutions = self.solve(target, q_init)
        valid = []
        lower, upper = self.model.joint_limits
        ctrl = self._ctrl_idx
        for q in solutions:
            # Only the controlled joints are IK's to answer for; the rest carry
            # whatever the seed configuration held (e.g. gripper joints), which
            # `solve()` never touches and shouldn't be re-validated here.
            if not (np.all(q[ctrl] >= lower[ctrl] - 1e-6) and np.all(q[ctrl] <= upper[ctrl] + 1e-6)):
                continue
            if self.collision_checker is not None and not self.collision_checker.is_valid(q):
                continue
            valid.append(q)
        return valid
