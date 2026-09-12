# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Example: one manipulator, arrayed 1-4 times on a circle, under a TSR.

The *same* UR5e is instanced N times, evenly spaced on a circle and all facing
inward, so their workspaces overlap near the centre. The point is to show how
the task-space description -- and therefore the TSR -- changes with the number
of cooperating arms, on an otherwise identical robot:

    arms  task space                      TSR          constrained coordinates
    ----  -----------------------------   ----------   ----------------------------
    1     SingleArmTaskSpace              TSR          [b12,b13,b23, tx,ty,tz]   (6)
    2     DualArmTaskSpace                BimanualTSR  absolute + relative Motor (6+6)
    3     TripleCooperativeTaskSpace      CircleTSR    [tx,ty,tz, dilation, n1,n2] (6)
    4     QuadrupleCooperativeTaskSpace   SphereTSR    [tx,ty,tz, dilation]        (4)

Three and four arms span a *circle* and a *sphere*, and gafro describes them by
a ``SimilarityTransformation`` -- which carries a **dilation** as well as a
rotation and translation, because the arms can grow or shrink the shape they
hold. The symmetry of the spanned primitive is why not every rotation is
constrained: a sphere is unchanged by any rotation about its centre (so
``SphereTSR`` has none), and a circle is unchanged by spin about its own axis
(so ``CircleTSR`` keeps only the two that tilt its plane).

The visualization shows the arms, the spanned primitive, and -- for 3 and 4
arms -- a dilation slider that resizes the held shape and re-solves IK, which
is the clearest way to see the extra coordinate doing real work.

Install the viz extras first:  pip install gafro[viz]

Run with:
    python examples/circle_array_tsr.py --arms 4
    python examples/circle_array_tsr.py --arms 3 --no-viz
    python examples/circle_array_tsr.py --arms 1 --no-viz
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import gafro as ga
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _circle_array import COOPERATIVE_TASK_SPACE, chain_names, write_circle_array  # noqa: E402

# Per-arm-count colours for the spanned primitive.
PRIMITIVE_COLOR = (255, 170, 0)


def _patch_visualizer_joint_limits() -> None:
    """Work around a gafro Visualizer bug when reading System joint limits.

    ``gafro.visualization.robot.Robot._actuated_joints`` does
    ``np.asarray(system.get_joint_limits_min())``,
    but that returns a ``JointPosition`` wrapper rather than an array, so numpy
    produces a 0-d array and the subsequent ``lo[idx]`` raises IndexError. This
    affects *every* robot, not just the composed arrays here. Patch the accessor
    to go through ``coefficients()`` when present.
    """
    try:
        from gafro.visualization import robot as robot_viz
    except ImportError:  # pragma: no cover - viz extras not installed
        return
    visual = getattr(robot_viz, "Robot", None)
    original = getattr(visual, "_actuated_joints", None)
    if visual is None or original is None or getattr(original, "_limits_patched", False):
        return

    def _vector(value):
        coefficients = getattr(value, "coefficients", None)
        if callable(coefficients):
            value = coefficients()
        return np.asarray(value, dtype=float).ravel()

    def _actuated_joints(self):
        low = _vector(self.system.get_joint_limits_min())
        high = _vector(self.system.get_joint_limits_max())
        joints = []
        for name in self.system.get_joint_names():
            joint = self.system.get_joint(name)
            if not joint.is_actuated():
                continue
            index = joint.get_index()
            joints.append((name, index, float(low[index]), float(high[index])))
        joints.sort(key=lambda entry: entry[1])
        return joints

    _actuated_joints._limits_patched = True
    visual._actuated_joints = _actuated_joints


def build_system(arm_count: int, radius: float, out_dir: Path | None = None):
    """Compose and load the N-arm circular array."""
    directory = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="circle_array_"))
    path = directory / f"circle_array_{arm_count}.yaml"
    write_circle_array(arm_count, path, radius=radius)
    return ga.SystemSerialization.load(str(path)), path


def build_model(arm_count: int, system):
    """The pycbirrt model matching this arm count, plus a label for it."""
    chains = chain_names(arm_count)
    if arm_count == 1:
        from pycbirrt.backends.gafro import GafroRobotModel

        return GafroRobotModel(system, chain_name=chains[0]), "TSR (single arm)"
    if arm_count == 2:
        from pycbirrt.backends.gafro_bimanual import GafroBimanualModel

        return (GafroBimanualModel(system, COOPERATIVE_TASK_SPACE),
                "BimanualTSR (absolute + relative)")
    from pycbirrt.backends.gafro_multiarm import GafroMultiArmModel

    model = GafroMultiArmModel(system, chains)
    return model, f"{model.tsr_class.__name__} ({model.tsr_class._DOF} DOF, incl. dilation)"


# Alternate arms are lifted by this much to break the array's symmetry.
SEED_STAGGER = 0.25


def seed_configuration(model, arm_count: int) -> np.ndarray:
    """A configuration with every arm reaching inward toward the circle centre.

    Alternating arms are staggered in shoulder-lift. This matters for the
    four-arm case: with every arm at an identical angle the array is perfectly
    symmetric and the four end-effectors come out **exactly coplanar**, and four
    coplanar points do not define a finite sphere -- the spanned sphere degenerates
    (its radius runs off to ~1e9 and the dilation coordinate diverges, since
    ``atanh`` blows up as the dilator ratio approaches 1). Breaking the symmetry
    gives the four points a real circumsphere and makes the dilation coordinate
    well-conditioned.

    Three arms never have this problem: any three non-collinear points define a
    circle. The stagger is applied anyway to keep the arm counts comparable.
    """
    q = np.full(model.dof, 0.2)
    joints_per_arm = model.dof // max(arm_count, 1)
    if joints_per_arm >= 2:
        for arm in range(1, arm_count, 2):
            q[arm * joints_per_arm + 1] += SEED_STAGGER
    return q


def describe_pose(model, arm_count: int, q: np.ndarray) -> str:
    """One-line readout of the task-space pose for this arm count."""
    pose = model.forward_kinematics(q)
    if arm_count == 1:
        translator = pose.get_translator()
        return f"EE position [{translator.x():+.3f} {translator.y():+.3f} {translator.z():+.3f}]"
    if arm_count == 2:
        absolute = pose.absolute.get_translator()
        return (f"absolute [{absolute.x():+.3f} {absolute.y():+.3f} {absolute.z():+.3f}]"
                f"  relative |log| {np.linalg.norm(np.asarray(pose.relative.log())):.3f}")
    zero = model.tsr_class(Bw=np.zeros((model.tsr_class._DOF, 2)))
    coordinates = zero.to_bw(pose)
    labels = model.tsr_class._LABELS
    return "  ".join(f"{name}={value:+.3f}" for name, value in zip(labels, coordinates))


def plan_region(model, arm_count: int, q_seed: np.ndarray, half_width: float = 0.05):
    """A TSR centred on the seed pose, of the shape this arm count calls for."""
    if arm_count in (1, 2):
        return None  # single/dual regions are built by their own demos
    zero = model.tsr_class(Bw=np.zeros((model.tsr_class._DOF, 2)))
    centre = zero.to_bw(model.forward_kinematics(q_seed))
    width = np.full(model.tsr_class._DOF, half_width)
    return model.tsr_class(Bw=np.column_stack([centre - width, centre + width]))


def spanned_primitive(model, arm_count: int, q: np.ndarray):
    """The circle / sphere the end-effectors currently span, for drawing."""
    if arm_count not in (3, 4):
        return None
    zero = model.tsr_class(Bw=np.zeros((model.tsr_class._DOF, 2)))
    coordinates = zero.to_bw(model.forward_kinematics(q))
    return zero._primitive_from_bw(coordinates)


class NoCollision:
    """Free space; the array's arms are far enough apart for this demo."""

    def is_valid(self, q) -> bool:  # noqa: D102
        return True


def plan_bigger_shape(model, arm_count: int, q_seed: np.ndarray, growth: float = 0.25):
    """Plan a path that grows the held circle / sphere by ``growth`` in log scale.

    Returns ``(path, region)``, or ``(None, region)`` if no path was found.
    """
    from pycbirrt import CBiRRT, CBiRRTConfig
    from pycbirrt.backends.gafro_multiarm import GafroMultiArmIKSolver

    dof = model.tsr_class._DOF
    zero = model.tsr_class(Bw=np.zeros((dof, 2)))
    base = zero.to_bw(model.forward_kinematics(q_seed))
    goal = base.copy()
    goal[3] += growth
    width = np.full(dof, 0.03)
    width[3] = 0.02
    region = model.tsr_class(Bw=np.column_stack([goal - width, goal + width]))

    solver = GafroMultiArmIKSolver(model, max_iterations=150, tolerance=1e-4,
                                   collision_checker=NoCollision())
    config = CBiRRTConfig(max_iterations=800, step_size=0.25, goal_bias=0.3,
                          tsr_samples=15, angular_joints=(True,) * model.dof)
    planner = CBiRRT(model, solver, NoCollision(), config)
    result = planner.plan(start=q_seed, goal_tsrs=[region], seed=1, return_details=True)
    return (result.path if result.success else None), region


def visualize(model, system, arm_count: int, q_seed: np.ndarray, label: str, port: int,
              path=None):
    """Serve a viser scene: the arm array, the spanned primitive, a dilation slider."""
    _patch_visualizer_joint_limits()
    viz = ga.Visualizer(port=port)
    robot_viz = viz.add_robot(system, joint_sliders=False)
    robot_viz.update(to_system(model, q_seed))

    state = {"q": q_seed.copy(), "node": None}

    def redraw():
        primitive = spanned_primitive(model, arm_count, state["q"])
        if primitive is None:
            return
        if state["node"] is not None:
            state["node"].remove()
        add = viz.add_sphere if arm_count == 4 else viz.add_circle
        state["node"] = add(primitive, name="/spanned", color=PRIMITIVE_COLOR, opacity=0.3)

    redraw()
    viz.add_label(f"{arm_count} arm(s) - {label}", position=(0.0, 0.0, 0.9), name="/title")

    if path:
        def on_frame(_index, q):
            state["q"] = np.asarray(q, dtype=float)
            robot_viz.update(to_system(model, state["q"]))
            redraw()

        viz.add_playback(list(path), on_frame)

    if arm_count in (3, 4):
        from pycbirrt.backends.gafro_multiarm import GafroMultiArmIKSolver

        solver = GafroMultiArmIKSolver(model, max_iterations=400, tolerance=1e-5)
        zero = model.tsr_class(Bw=np.zeros((model.tsr_class._DOF, 2)))
        base = zero.to_bw(model.forward_kinematics(q_seed))

        with viz.gui.add_folder("Held shape"):
            slider = viz.gui.add_slider("dilation (log scale)", min=-0.6, max=0.6,
                                        step=0.02, initial_value=0.0)
            status = viz.gui.add_text("status", initial_value="-", disabled=True)

        def on_dilation(_event=None):
            goal = base.copy()
            goal[3] = base[3] + slider.value
            width = np.array([0.02] * 3 + [0.005]
                             + [0.05] * (model.tsr_class._DOF - 4))
            region = model.tsr_class(Bw=np.column_stack([goal - width, goal + width]))
            solutions = solver.solve(region, q_init=state["q"])
            if not solutions:
                status.value = f"no IK at dilation {slider.value:+.2f}"
                return
            state["q"] = solutions[0]
            robot_viz.update(to_system(model, state["q"]))
            redraw()
            achieved = zero.to_bw(model.forward_kinematics(state["q"]))[3]
            status.value = f"scale x{np.exp(achieved - base[3]):.3f}"

        slider.on_update(on_dilation)

    viz.show()


def to_system(model, q: np.ndarray) -> np.ndarray:
    """Controlled-width q -> System-width config for the visualizer.

    Models differ in how much of the System they cover: a single-arm model
    scatters its chain into the full config, the multi-arm models hold their
    non-controlled joints, and the bimanual model already plans in System width.
    """
    if hasattr(model, "to_system_configuration"):
        return model.to_system_configuration(q)
    if hasattr(model, "_to_task_full"):
        return model._to_task_full(q)
    return np.asarray(q, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", type=int, default=4, choices=(1, 2, 3, 4),
                        help="number of manipulators on the circle (default: 4)")
    parser.add_argument("--radius", type=float, default=0.9,
                        help="circle radius the arms are mounted on (default: 0.9)")
    parser.add_argument("--port", type=int, default=8080, help="viser port (default: 8080)")
    parser.add_argument("--no-viz", action="store_true", help="print only, no viewer")
    parser.add_argument("--out-dir", default=None,
                        help="where to write the composed array (default: a temp dir)")
    parser.add_argument("--plan", action="store_true",
                        help="plan a path that grows the held shape (3 and 4 arms)")
    args = parser.parse_args()

    system, path = build_system(args.arms, args.radius, args.out_dir)
    model, label = build_model(args.arms, system)
    q_seed = seed_configuration(model, args.arms)

    print(f"array:      {args.arms} x UR5e on a circle of radius {args.radius} m")
    print(f"written to: {path}")
    print(f"system dof: {system.get_dof()}   planning dof: {model.dof}")
    print(f"task space: {label}")
    print(f"pose:       {describe_pose(model, args.arms, q_seed)}")

    region = plan_region(model, args.arms, q_seed)
    if region is not None:
        pose = model.forward_kinematics(q_seed)
        print(f"region:     {region}")
        print(f"            contains seed pose: {region.contains(pose, tolerance=1e-6)}")

    path = None
    if args.plan:
        if args.arms < 3:
            print("plan:       --plan covers the cooperative (3/4 arm) cases")
        else:
            path, _region = plan_bigger_shape(model, args.arms, q_seed)
            if path is None:
                print("plan:       no path found")
            else:
                zero = model.tsr_class(Bw=np.zeros((model.tsr_class._DOF, 2)))
                start_dilation = zero.to_bw(model.forward_kinematics(q_seed))[3]
                end_dilation = zero.to_bw(model.forward_kinematics(path[-1]))[3]
                print(f"plan:       {len(path)} waypoints, "
                      f"dilation {start_dilation:+.3f} -> {end_dilation:+.3f} "
                      f"(x{np.exp(end_dilation - start_dilation):.2f} bigger)")

    if args.no_viz:
        return 0

    visualize(model, system, args.arms, q_seed, label, args.port, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
