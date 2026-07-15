# SPDX-License-Identifier: MIT
"""Native gafropy bimanual backend: CooperativeTaskSpace kinematics + IK.

Forward kinematics returns a :class:`~tsr.bimanual.BimanualPose` — the
(absolute, relative) pose pair a :class:`~tsr.bimanual.BimanualTSR` speaks in.
The IK solver (see :class:`GafroBimanualIKSolver`) stacks only the Jacobians of
the pose components that are present in the target.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gafropy import CooperativeTaskSpace, SystemSerialization

from pycbirrt.backends.gafro import _ee_error_twist
from tsr.bimanual import BimanualPose

if TYPE_CHECKING:
    from gafropy import System


def _discover_chains(system) -> tuple[str, str]:
    names = system.get_kinematic_chain_names()
    left = next((c for c in names if "left" in c.lower()), None)
    right = next((c for c in names if "right" in c.lower()), None)
    if left is None or right is None:
        raise ValueError(
            f"could not auto-discover left/right chains from {names}; "
            "pass left_chain / right_chain explicitly")
    return left, right


class GafroBimanualModel:
    """Dual-arm kinematics over a gafro CooperativeTaskSpace."""

    def __init__(self, system: "System", left_chain: str | None = None, right_chain: str | None = None):
        if left_chain is None or right_chain is None:
            dl, dr = _discover_chains(system)
            left_chain = left_chain or dl
            right_chain = right_chain or dr
        self.system = system
        self.left_chain = left_chain
        self.right_chain = right_chain
        self.cooperative = CooperativeTaskSpace(system, "coop", [left_chain, right_chain])

        self._ctrl_idx = np.asarray(self.cooperative.get_controlled_joints(), dtype=int)
        ts = self.cooperative
        full_lower = np.asarray(ts.extract_configuration(system.get_joint_limits_min()), dtype=float)
        full_upper = np.asarray(ts.extract_configuration(system.get_joint_limits_max()), dtype=float)
        self._lower = full_lower[self._ctrl_idx]
        self._upper = full_upper[self._ctrl_idx]
        self._base_full = 0.5 * (full_lower + full_upper)
        self.default_system_configuration = np.asarray(
            system.get_default_configuration(), dtype=float)

    @classmethod
    def from_file(cls, path: str, left_chain: str | None = None, right_chain: str | None = None) -> "GafroBimanualModel":
        return cls(SystemSerialization.load(path), left_chain, right_chain)

    @property
    def dof(self) -> int:
        return len(self._ctrl_idx)

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lower, self._upper

    def _to_task_full(self, q: np.ndarray) -> np.ndarray:
        """Controlled-width ``q`` -> full task-width config (non-controlled held)."""
        q_full = self._base_full.copy()
        q_full[self._ctrl_idx] = np.asarray(q, dtype=float)
        return q_full

    def forward_kinematics(self, q: np.ndarray) -> BimanualPose:
        q_full = self._to_task_full(q)
        return BimanualPose(
            absolute=self.cooperative.compute_absolute_motor(q_full),
            relative=self.cooperative.compute_relative_motor(q_full),
        )

    def normalize_pose(self, x) -> BimanualPose:
        """Pass-through: bimanual poses are already :class:`BimanualPose`."""
        if not isinstance(x, BimanualPose):
            raise TypeError(f"expected BimanualPose, got {type(x)}")
        return x


class GafroBimanualIKSolver:
    """Adaptive differential IK for a bimanual target.

    Each iteration stacks only the geometric Jacobians of the pose components
    present in the target (absolute, relative, or both), and drives the CGA
    error twist(s) to zero with damped least squares over the joined dual-arm
    joint vector.
    """

    def __init__(self, model: "GafroBimanualModel", collision_checker=None,
                 damping: float = 0.1, max_iterations: int = 200, tolerance: float = 1e-3):
        self.model = model
        self.coop = model.cooperative
        self._ctrl_idx = model._ctrl_idx
        self.damping = damping
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.collision_checker = collision_checker

    def _clamp(self, q: np.ndarray) -> np.ndarray:
        lower, upper = self.model.joint_limits
        return np.clip(q, lower, upper)

    def solve(self, target: BimanualPose, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        if target.absolute is None and target.relative is None:
            raise ValueError("bimanual IK target has no active component")

        lower, upper = self.model.joint_limits
        if q_init is not None:
            q = np.asarray(q_init, dtype=float).copy()
        else:
            q = 0.5 * (lower + upper)

        for _ in range(self.max_iterations):
            q_full = self.model._to_task_full(q)

            errors = []
            jacobians = []
            if target.absolute is not None:
                current = self.coop.compute_absolute_motor(q_full)
                errors.append(_ee_error_twist(target.absolute, current))
                J = np.asarray(self.coop.compute_absolute_geometric_jacobian(q_full), dtype=float)
                jacobians.append(J[:, self._ctrl_idx])
            if target.relative is not None:
                current = self.coop.compute_relative_motor(q_full)
                errors.append(_ee_error_twist(target.relative, current))
                J = np.asarray(self.coop.compute_relative_geometric_jacobian(q_full), dtype=float)
                jacobians.append(J[:, self._ctrl_idx])

            error = np.concatenate(errors)
            if np.linalg.norm(error) < self.tolerance:
                return [q]

            J = np.vstack(jacobians)  # (6 or 12) x controlled-dof
            JJT = J @ J.T
            damped = JJT + self.damping**2 * np.eye(JJT.shape[0])
            dq = J.T @ np.linalg.solve(damped, error)
            q = self._clamp(q + dq)

        return []

    def solve_valid(self, target: BimanualPose, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        solutions = self.solve(target, q_init)
        valid = []
        lower, upper = self.model.joint_limits
        for q in solutions:
            if not (np.all(q >= lower - 1e-6) and np.all(q <= upper + 1e-6)):
                continue
            if self.collision_checker is not None and not self.collision_checker.is_valid(q):
                continue
            valid.append(q)
        return valid
