# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import clone, configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_assets.robots.unitree import G1_CFG

ROBOT = clone(G1_CFG)
ROBOT.prim_path = "{ENV_REGEX_NS}/Robot"
ROBOT.spawn.fix_root_link = True
# Throwing must respect arm/torso contacts in both supported backends.
for articulation_props in ROBOT.spawn.articulation_props:
    if isinstance(articulation_props, NewtonArticulationCfg):
        articulation_props.self_collision_enabled = True
    elif isinstance(articulation_props, PhysxArticulationCfg):
        articulation_props.enabled_self_collisions = True
# Replace the asset's generic manipulation effort limit with a conservative arm limit.
ROBOT.actuators["arms"].joint_effort_limit = {
    ".*_shoulder_.*_joint": 25.0,
    ".*_elbow_.*_joint": 25.0,
    ".*_(zero|one|two|three|four|five|six)_joint": 5.0,
}
ROBOT.actuators["arms"].stiffness = {
    ".*_shoulder_.*_joint": 100.0,
    ".*_elbow_.*_joint": 80.0,
    ".*_(zero|one|two|three|four|five|six)_joint": 20.0,
}
ROBOT.actuators["arms"].damping = {
    ".*_shoulder_.*_joint": 5.0,
    ".*_elbow_.*_joint": 5.0,
    ".*_(zero|one|two|three|four|five|six)_joint": 1.0,
}
ROBOT.init_state.joint_pos["right_shoulder_pitch_joint"] = -1.2
ROBOT.init_state.joint_pos.pop(".*_elbow_pitch_joint")
ROBOT.init_state.joint_pos["left_elbow_pitch_joint"] = 0.87
ROBOT.init_state.joint_pos["right_elbow_pitch_joint"] = 1.2
ROBOT.init_state.joint_pos["right_elbow_roll_joint"] = 1.57
ROBOT.init_state.joint_pos["right_one_joint"] = -1.0
ROBOT.init_state.joint_pos["right_two_joint"] = -1.0
ROBOT.init_state.joint_pos.update(
    {"right_three_joint": 0.8, "right_five_joint": 0.8, "right_four_joint": 0.8, "right_six_joint": 0.8}
)


@configclass
class TargetedShotPutSceneCfg(InteractiveSceneCfg):
    """Fixed-base G1 and a freely simulated shot with physical hand contacts."""

    robot: ArticulationCfg = ROBOT
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.035,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=1.5),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.2, 1.0)),
    )


@configclass
class TargetedShotPutEnvCfg(DirectRLEnvCfg):
    """Contact throwing prototype; target coordinates and distances are in [m]."""

    decimation = 4
    episode_length_s = 3.0
    action_space = 12
    observation_space = 53
    state_space = 0
    sim = SimulationCfg(
        dt=1 / 240,
        render_interval=decimation,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=256, nconmax=128, integrator="implicitfast"),
            num_substeps=1,
            use_cuda_graph=True,
        ),
    )
    scene = TargetedShotPutSceneCfg(num_envs=4096, env_spacing=6.0, replicate_physics=True)
    controlled_joints = ["right_(shoulder|elbow|zero|one|two|three|four|five|six).*_joint"]
    palm_body = "right_palm_link"
    ball_radius = 0.035
    palm_offset = (0.085, 0.05, 0.0)
    target_min_radius = 0.5
    target_max_radius = 2.0
    # Front sector relative to the reset heading; targets stay fixed while stepping.
    target_angle_range = (-math.pi / 3, math.pi / 3)
    success_radius = 0.15
    minimum_flight_s = 0.12
    minimum_release_speed = 0.4
    mass_range = (1.5, 2.5)
    # Widen reset randomization over 1000 PPO iterations (16 rollout steps each).
    mass_curriculum_steps = 16_000
    mass_curriculum_initial_steps = 0
    accuracy_scale = 0.3

    def __post_init__(self):
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(3.0, -3.0, 2.0), lookat=(0.0, 0.0, 0.8))
