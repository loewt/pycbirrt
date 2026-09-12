#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The gafro three-/four-arm cooperative backend.

These need a real multi-chain robot description; they skip cleanly if absent.
"""

import os

import numpy as np
import pytest

ROBOT = os.environ.get(
    "MULTIARM_TEST_ROBOT",
    "/home/tobi/tmp/vvv/gafro-robot-descriptions/assets/robots/aloha/aloha/aloha.yaml")
CHAINS = [
    "right/right_finger_link",
    "left/left_finger_link",
    "left/right_finger_link",
    "right/left_finger_link",
]
requires_robot = pytest.mark.skipif(not os.path.exists(ROBOT), reason=f"missing {ROBOT}")


@pytest.fixture
def model_factory():
    from pycbirrt.backends.gafro_multiarm import GafroMultiArmModel

    def build(arm_count):
        return GafroMultiArmModel.from_file(ROBOT, CHAINS[:arm_count])

    return build


@requires_robot
def test_rejects_unsupported_arm_counts(model_factory):
    from pycbirrt.backends.gafro_multiarm import GafroMultiArmModel

    too_few = [CHAINS[:1], CHAINS[:2]]
    too_many = [CHAINS + CHAINS[:1]]  # 5 chains
    for chains in too_few + too_many:
        with pytest.raises(ValueError, match="3 or 4 chains"):
            GafroMultiArmModel.from_file(ROBOT, chains)


@requires_robot
@pytest.mark.parametrize("arm_count,tsr_name,dof", [(3, "CircleTSR", 6), (4, "SphereTSR", 4)])
def test_model_selects_task_space_for_arm_count(model_factory, arm_count, tsr_name, dof):
    model = model_factory(arm_count)
    assert model.arm_count == arm_count
    assert model.tsr_class.__name__ == tsr_name
    assert model.tsr_class._DOF == dof


@requires_robot
@pytest.mark.parametrize("arm_count", [3, 4])
def test_forward_kinematics_returns_similarity(model_factory, arm_count):
    from gafro import SimilarityTransformation

    model = model_factory(arm_count)
    pose = model.forward_kinematics(np.full(model.dof, 0.1))
    assert isinstance(pose, SimilarityTransformation)
    # And it reads out as this arm count's coordinate vector.
    zero = model.tsr_class(Bw=np.zeros((model.tsr_class._DOF, 2)))
    assert zero.to_bw(pose).shape == (model.tsr_class._DOF,)


@requires_robot
@pytest.mark.parametrize("arm_count", [3, 4])
def test_ik_reaches_an_exact_reachable_pose(model_factory, arm_count):
    from pycbirrt.backends.gafro_multiarm import GafroMultiArmIKSolver

    model = model_factory(arm_count)
    solver = GafroMultiArmIKSolver(model, max_iterations=400, tolerance=1e-5)
    zero = model.tsr_class(Bw=np.zeros((model.tsr_class._DOF, 2)))

    target = model.forward_kinematics(np.full(model.dof, 0.15))
    solutions = solver.solve(target, q_init=np.full(model.dof, 0.05))
    assert solutions, "IK did not converge on a self-generated target"

    error = zero.to_bw(model.forward_kinematics(solutions[0])) - zero.to_bw(target)
    assert np.linalg.norm(error) < 1e-4


@requires_robot
@pytest.mark.parametrize("arm_count", [3, 4])
def test_ik_solves_into_a_region(model_factory, arm_count):
    from pycbirrt.backends.gafro_multiarm import GafroMultiArmIKSolver

    model = model_factory(arm_count)
    solver = GafroMultiArmIKSolver(model, max_iterations=400, tolerance=1e-5)
    zero = model.tsr_class(Bw=np.zeros((model.tsr_class._DOF, 2)))

    seed = zero.to_bw(model.forward_kinematics(np.full(model.dof, 0.15)))
    width = np.full(model.tsr_class._DOF, 0.05)
    region = model.tsr_class(Bw=np.column_stack([seed - width, seed + width]))

    solutions = solver.solve(region, q_init=np.full(model.dof, 0.05))
    assert solutions, "IK did not converge into the region"
    assert region.contains(model.forward_kinematics(solutions[0]), tolerance=1e-4)


@requires_robot
def test_dilation_constraint_changes_the_configuration():
    """Constraining dilation must actually resize the spanned sphere.

    This is the point of the extra coordinate: with the centre pinned, asking
    for a different scale has to move the arms.
    """
    from tsr.multiarm import SphereTSR

    from pycbirrt.backends.gafro_multiarm import GafroMultiArmIKSolver, GafroMultiArmModel

    model = GafroMultiArmModel.from_file(ROBOT, CHAINS)
    solver = GafroMultiArmIKSolver(model, max_iterations=500, tolerance=1e-5)
    zero = SphereTSR(Bw=np.zeros((4, 2)))
    base = zero.to_bw(model.forward_kinematics(np.full(model.dof, 0.15)))

    achieved = []
    for offset in (-0.3, 0.0, 0.3):
        goal = base.copy()
        goal[3] += offset
        width = np.array([0.01, 0.01, 0.01, 0.002])
        region = SphereTSR(Bw=np.column_stack([goal - width, goal + width]))
        solutions = solver.solve(region, q_init=np.full(model.dof, 0.05))
        assert solutions, f"no solution for dilation offset {offset}"
        achieved.append(zero.to_bw(model.forward_kinematics(solutions[0]))[3])

    # Monotone in the requested dilation, and actually distinct.
    assert achieved[0] < achieved[1] < achieved[2]
    assert achieved[2] - achieved[0] > 0.4


@requires_robot
def test_base_configuration_holds_non_controlled_joints(model_factory):
    model = model_factory(4)
    base = model.base_configuration
    assert base.shape == (model.cooperative.get_dof(),)
    # Scattering a controlled vector leaves the held joints untouched.
    q = np.full(model.dof, 0.2)
    full = model._to_task_full(q)
    held = np.setdiff1d(np.arange(len(full)), model._ctrl_idx)
    np.testing.assert_allclose(full[held], base[held])


@requires_robot
@pytest.mark.parametrize("arm_count", [3, 4])
def test_ik_escapes_a_degenerate_seed(model_factory, arm_count):
    """A symmetric seed sits on the degenerate surface; IK must jitter off it.

    The limit midpoint has every joint equal, which for a symmetric array puts
    the end-effectors where the spanned primitive has no finite scale. Solving
    with no seed starts exactly there, so it must still succeed.
    """
    from pycbirrt.backends.gafro_multiarm import GafroMultiArmIKSolver

    model = model_factory(arm_count)
    solver = GafroMultiArmIKSolver(model, max_iterations=300, tolerance=1e-4)
    zero = model.tsr_class(Bw=np.zeros((model.tsr_class._DOF, 2)))
    seed = zero.to_bw(model.forward_kinematics(np.full(model.dof, 0.15)))
    width = np.full(model.tsr_class._DOF, 0.05)
    region = model.tsr_class(Bw=np.column_stack([seed - width, seed + width]))

    assert solver.solve(region), "IK failed from the default (symmetric) seed"


@requires_robot
def test_degenerate_pose_is_infinitely_far_not_an_error():
    """A planner probing a degenerate configuration must get inf, not an exception."""
    from tsr.multiarm import SphereTSR

    class _DegenerateSimilarity:
        def get_canonical_decomposition(self):
            raise AssertionError("should not be reached")

    tsr = SphereTSR(Bw=np.zeros((4, 2)))
    # is_degenerate short-circuits before the decomposition is used.
    import tsr.multiarm as multiarm

    original = multiarm.is_degenerate
    multiarm.is_degenerate = lambda _pose: True
    try:
        distance, witness = tsr.distance(_DegenerateSimilarity())
    finally:
        multiarm.is_degenerate = original
    assert distance == float("inf")
    assert witness.shape == (4,)
