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
