# SPDX-License-Identifier: MIT
import os
import numpy as np
import pytest

ROBOT = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "robots", "geodude", "geodude.xml")
ROBOT = os.path.abspath(ROBOT)
requires_robot = pytest.mark.skipif(not os.path.exists(ROBOT), reason=f"missing {ROBOT}")


@pytest.fixture
def model():
    from pycbirrt.backends.gafro_bimanual import GafroBimanualModel
    return GafroBimanualModel.from_file(ROBOT)


@requires_robot
def test_fk_returns_pose_pair(model):
    from tsr.bimanual import BimanualPose
    from gafropy import Motor
    q = np.zeros(model.dof)
    pose = model.forward_kinematics(q)
    assert isinstance(pose, BimanualPose)
    assert isinstance(pose.absolute, Motor)
    assert isinstance(pose.relative, Motor)


@requires_robot
def test_dof_is_two_arms_controlled(model):
    # geodude is two 6-DOF UR5e arms; controlled dof should be the joined width.
    assert model.dof == model.cooperative.get_controlled_dof()
    lower, upper = model.joint_limits
    assert len(lower) == model.dof == len(upper)


@requires_robot
def test_normalize_pose_passthrough(model):
    from tsr.bimanual import BimanualPose
    from gafropy import Motor
    p = BimanualPose(absolute=Motor(), relative=Motor())
    assert model.normalize_pose(p) is p
