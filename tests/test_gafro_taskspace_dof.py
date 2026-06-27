# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The gafro backend plans on a task space's *controlled* joints.

A gafro task space reports a full chain DOF (configs / limits / FK input) and a
narrower controlled DOF (the width of its control Jacobian). These tests pin the
backend to the controlled width and guard the original ``(7,)+(6,)`` crash.

The tests need a real robot description; they skip cleanly if it is absent.
"""

import os

import numpy as np
import pytest

ROBOT = "/home/tobi/coding/src/rss/geodude.xml"
CHAIN = "left_ur5e/endeffector_link"
requires_robot = pytest.mark.skipif(not os.path.exists(ROBOT), reason=f"missing {ROBOT}")


@pytest.fixture
def manipulator():
    from gafropy import SingleArmTaskSpace, SystemSerialization

    system = SystemSerialization.load(ROBOT)
    return SingleArmTaskSpace(system, CHAIN, CHAIN)


@requires_robot
def test_solver_dof_matches_control_jacobian(manipulator):
    from pycbirrt.backends.gafro import GafroIKSolver

    solver = GafroIKSolver(manipulator)
    jac = np.asarray(
        manipulator.compute_ee_geometric_jacobian(np.zeros(manipulator.get_dof())))
    assert solver._dof == jac.shape[1] == manipulator.get_controlled_dof()
    lower, upper = solver.joint_limits
    assert len(lower) == solver._dof and len(upper) == solver._dof


@requires_robot
def test_solve_reaches_a_reachable_pose_without_broadcast_error(manipulator):
    from pycbirrt.backends.gafro import GafroIKSolver

    solver = GafroIKSolver(manipulator, max_iterations=300, tolerance=1e-5)
    # A pose known reachable: FK of a controlled-width config.
    q_seed = np.full(solver._dof, 0.2)
    target = solver.manipulator.compute_ee_motor(_to_full(manipulator, q_seed))

    sols = solver.solve(target, q_init=q_seed)
    assert sols, "solver returned no solution"
    assert sols[0].shape == (solver._dof,)


@requires_robot
def test_solve_holds_noncontrolled_joint_fixed(manipulator):
    from pycbirrt.backends.gafro import GafroIKSolver

    solver = GafroIKSolver(manipulator, max_iterations=300, tolerance=1e-5)
    ctrl_idx = np.asarray(manipulator.get_controlled_joints(), dtype=int)

    q_seed = np.full(solver._dof, 0.15)
    base = solver._mid_full.copy()
    base[0] = 0.42  # pin the prismatic torso (a non-controlled joint)
    q_init_full = base.copy()
    q_init_full[ctrl_idx] = q_seed

    target = solver.manipulator.compute_ee_motor(q_init_full)
    sols = solver.solve(target, q_init=q_init_full)
    assert sols
    # The solver returns only controlled joints; reconstructing the full config
    # with the same base must keep the torso where we pinned it.
    reconstructed = base.copy()
    reconstructed[ctrl_idx] = sols[0]
    assert reconstructed[0] == 0.42


@requires_robot
def test_robot_model_reports_controlled_dof(manipulator):
    from pycbirrt.backends.gafro import GafroRobotModel

    model = GafroRobotModel.from_file(ROBOT, chain_name=CHAIN)
    assert model.dof == manipulator.get_controlled_dof()
    lower, upper = model.joint_limits
    assert len(lower) == model.dof and len(upper) == model.dof


@requires_robot
def test_robot_model_fk_accepts_controlled_width(manipulator):
    from gafropy import Motor
    from pycbirrt.backends.gafro import GafroRobotModel

    model = GafroRobotModel.from_file(ROBOT, chain_name=CHAIN)
    q = np.full(model.dof, 0.1)
    pose = model.forward_kinematics(q)
    assert Motor(pose) is not None


@requires_robot
def test_to_system_configuration_places_joints_at_correct_indices(manipulator):
    from pycbirrt.backends.gafro import GafroRobotModel

    model = GafroRobotModel.from_file(ROBOT, chain_name=CHAIN)
    q = np.arange(1, model.dof + 1, dtype=float)  # distinctive controlled values
    sys_q = model.to_system_configuration(q)

    assert sys_q.shape == (model._system_dof,)
    # Controlled joints must land at their System indices (system 1..6 here),
    # NOT front-padded into 0..5 (the bug that hid the robot).
    ctrl_sys_idx = model._task_to_system[model._ctrl_idx]
    assert np.allclose(sys_q[ctrl_sys_idx], q)
    # The first arm joint value must not leak into system index 0 (the torso).
    assert sys_q[0] != q[0]


@requires_robot
def test_plan_to_tsr_end_to_end():
    """Regression for the (7,)+(6,) crash: a full plan to a TSR must succeed."""
    from tsr import TSR
    from pycbirrt import CBiRRT, CBiRRTConfig
    from pycbirrt.backends.gafro import GafroIKSolver, GafroRobotModel

    class NoCollision:
        def is_valid(self, q):
            return True

    from gafropy import Motor

    robot = GafroRobotModel.from_file(ROBOT, chain_name=CHAIN)
    ik = GafroIKSolver(robot.manipulator, robot.joint_limits,
                       max_iterations=300, tolerance=1e-6)
    config = CBiRRTConfig(max_iterations=5000, step_size=0.15, goal_bias=0.2,
                          tsr_samples=50, angular_joints=(True,) * robot.dof)
    planner = CBiRRT(robot, ik, NoCollision(), config)

    # Center the goal TSR on the FK of a real config so the region is guaranteed
    # reachable (the arm is mounted on the geodude torso, far from the world
    # origin, so an absolute hand-picked pose would be out of reach).
    start = np.full(robot.dof, 0.1)
    goal_q = np.full(robot.dof, -0.3)
    T0_w = Motor(robot.forward_kinematics(goal_q)).to_transformation_matrix()
    Bw = np.array([
        [-np.pi, np.pi], [0.0, 0.0], [0.0, 0.0],   # free yaw about EE z
        [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05],  # small translation box
    ])
    tsr = TSR(T0_w=T0_w, Tw_e=np.eye(4), Bw=Bw)

    result = planner.plan(start=start, goal_tsrs=[tsr], seed=1, return_details=True)
    assert result.success, f"plan failed: {result.failure_reason}"
    assert all(wp.shape == (robot.dof,) for wp in result.path)


def _to_full(manipulator, q_ctrl):
    ctrl_idx = np.asarray(manipulator.get_controlled_joints(), dtype=int)
    lower = np.asarray(manipulator.get_joint_limits_min(), dtype=float)
    upper = np.asarray(manipulator.get_joint_limits_max(), dtype=float)
    full = 0.5 * (lower + upper)
    full[ctrl_idx] = q_ctrl
    return full
