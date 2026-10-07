# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Free-base balance and recovery stages for contact-driven throwing."""

import math

import torch

from isaaclab.utils import clone, configclass
from isaaclab.utils.math import quat_apply

from .targeted_shot_put_env import TargetedShotPutEnv
from .targeted_shot_put_env_cfg import ROBOT, TargetedShotPutEnvCfg


@configclass
class StandingShotPutEnvCfg(TargetedShotPutEnvCfg):
    action_space = 30
    observation_space = 116
    episode_length_s = 6.0
    controlled_joints = [
        "right_(shoulder|elbow|zero|one|two|three|four|five|six).*_joint",
        "left_(shoulder|elbow).*_joint",
        ".*_(hip|knee|ankle).*_joint",
        "torso_joint",
    ]
    mass_range = (1.5, 1.5)
    target_min_radius = 0.5
    target_max_radius = 0.75
    target_angle_range = (-math.pi / 6, math.pi / 6)
    hold_only = True
    recovery_s = 2.0
    minimum_root_height = 0.5
    maximum_tilt = 0.6
    leg_action_scale = 0.6
    torso_action_scale = 0.35
    moving_speed_penalty = 0.05

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = clone(ROBOT)
        self.scene.robot.spawn.fix_root_link = False
        # Provisional bounded limits; hardware calibration remains a transfer gate.
        self.scene.robot.actuators["legs"].joint_effort_limit = 60.0


class StandingShotPutEnv(TargetedShotPutEnv):
    """Keep the free robot upright during holding, throwing and recovery."""

    cfg: StandingShotPutEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.impact_seen = torch.zeros_like(self.released)
        self.new_impact = torch.zeros_like(self.released)
        self.recovery_time = torch.zeros_like(self.flight_time)
        self.fallen = torch.zeros_like(self.released)
        self.balanced = torch.zeros_like(self.released)
        names = [self.robot.joint_names[i] for i in self.joints.tolist()]
        self.action_scale = torch.tensor(
            [
                self.cfg.leg_action_scale
                if any(x in name for x in ("hip", "knee", "ankle"))
                else self.cfg.torso_action_scale
                if "torso" in name
                else 1.0
                for name in names
            ],
            device=self.device,
        )
        # Cache the initial palm pose: reset must not reuse the last thrown-arm pose.
        self.reset_palm_position = (
            self.robot.data.body_link_pos_w.torch[:, self.palm] - self.scene.env_origins
        ).clone()
        self.reset_palm_quaternion = self.robot.data.body_link_quat_w.torch[:, self.palm].clone()

    def _apply_action(self):
        target = self.robot.data.default_joint_pos.torch.clone()
        target[:, self.joints] += self.actions * self.action_scale
        limits = self.robot.data.soft_joint_pos_limits.torch
        target = target.clamp(limits[..., 0], limits[..., 1])
        self.robot.actuators.target_command.set_position_index(value=target)

    def _get_observations(self):
        obs = super()._get_observations()["policy"]
        return {
            "policy": torch.cat(
                (
                    obs,
                    self.robot.data.root_lin_vel_b.torch,
                    self.robot.data.root_ang_vel_b.torch,
                    self.robot.data.projected_gravity_b.torch,
                ),
                dim=1,
            )
        }

    def _get_dones(self):
        old_error = self.impact_error.clone()
        old_flight = self.flight_time.clone()
        _, timeout = super()._get_dones()
        self.new_impact = self.landed & ~self.impact_seen
        self.impact_error = torch.where(self.impact_seen, old_error, self.impact_error)
        self.flight_time = torch.where(self.impact_seen, old_flight, self.flight_time)
        self.impact_seen |= self.new_impact
        # First impact remains latched while the robot continues recovering.
        self.landed = self.impact_seen.clone()
        gravity = self.robot.data.projected_gravity_b.torch
        height = self.robot.data.root_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        self.fallen = (height < self.cfg.minimum_root_height) | (gravity[:, 2] > -math.cos(self.cfg.maximum_tilt))
        self.balanced = (
            ~self.fallen
            & (gravity[:, :2].norm(dim=1) < 0.25)
            & (self.robot.data.root_lin_vel_w.torch.norm(dim=1) < 0.3)
            & (self.robot.data.root_ang_vel_w.torch.norm(dim=1) < 0.5)
        )
        self.recovery_time = torch.where(self.impact_seen & self.balanced, self.recovery_time + self.step_dt, 0.0)
        escaped = (self.ball.data.root_pos_w.torch - self.scene.env_origins).norm(dim=1) > 4.0
        if self.cfg.hold_only:
            return self.fallen | self.released | escaped, timeout
        recovered = (self.recovery_time >= self.cfg.recovery_s) & self.balanced
        return self.fallen | escaped | recovered, timeout

    def _get_rewards(self):
        gravity = self.robot.data.projected_gravity_b.torch
        upright = torch.exp(-10 * gravity[:, :2].square().sum(dim=1))
        speed = self.robot.data.root_lin_vel_w.torch.square().sum(dim=1)
        angular = self.robot.data.root_ang_vel_w.torch.square().sum(dim=1)
        pose = (self.robot.data.joint_pos.torch - self.robot.data.default_joint_pos.torch).square().mean(dim=1)
        smooth = (self.actions - self.previous_actions).square().mean(dim=1)
        # Allow repositioning before impact, then reward settling for recovery.
        speed_penalty = torch.where(self.impact_seen | self.cfg.hold_only, 0.5, self.cfg.moving_speed_penalty)
        balance_reward = (
            2 * upright - speed_penalty * speed - 0.1 * angular - 0.1 * pose - 0.1 * smooth
        ) * self.step_dt
        if self.cfg.hold_only:
            holding = ~self.released
            return balance_reward + holding.float() * self.step_dt - 5 * (self.fallen | self.released).float()
        # Parent scores landing only once, rather than every recovery tick.
        self.landed = self.new_impact
        throw_reward = super()._get_rewards()
        self.landed = self.impact_seen.clone()
        recovered = (self.recovery_time >= self.cfg.recovery_s) & self.balanced
        accurate = (
            self.release_upward
            & (self.release_speed >= self.cfg.minimum_release_speed)
            & (self.flight_time >= self.cfg.minimum_flight_s)
            & (self.impact_error < self.cfg.success_radius)
        )
        return throw_reward + balance_reward - 10 * self.fallen.float() + 10 * (recovered & accurate).float()

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        recovered = (self.recovery_time[env_ids] >= self.cfg.recovery_s) & self.balanced[env_ids]
        valid = (
            self.release_upward[env_ids]
            & (self.release_speed[env_ids] >= self.cfg.minimum_release_speed)
            & (self.flight_time[env_ids] >= self.cfg.minimum_flight_s)
        )
        self.extras.setdefault("log", {}).update(
            {
                "Metrics/fall_rate": self.fallen[env_ids].float().mean(),
                "Metrics/balanced_rate": self.balanced[env_ids].float().mean(),
                "Metrics/recovered_rate": (
                    (self.recovery_time[env_ids] >= self.cfg.recovery_s) & self.balanced[env_ids]
                )
                .float()
                .mean(),
                "Metrics/balanced_hit_rate": (
                    recovered & valid & (self.impact_error[env_ids] < self.cfg.success_radius)
                )
                .float()
                .mean(),
                "Metrics/hold_success_rate": (
                    (self.elapsed_steps[env_ids] >= self.max_episode_length)
                    & ~self.fallen[env_ids]
                    & ~self.released[env_ids]
                )
                .float()
                .mean(),
            }
        )
        pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
        )
        super()._reset_idx(env_ids)
        ball_pose = self.ball.data.default_root_pose.torch[env_ids].clone()
        offset = torch.tensor(self.cfg.palm_offset, device=self.device).expand(len(env_ids), -1)
        ball_pose[:, :3] = (
            self.scene.env_origins[env_ids]
            + self.reset_palm_position[env_ids]
            + quat_apply(self.reset_palm_quaternion[env_ids], offset)
        )
        self.ball.write_root_pose_to_sim_index(root_pose=ball_pose, env_ids=env_ids)
        self.impact_seen[env_ids] = False
        self.new_impact[env_ids] = False
        self.recovery_time[env_ids] = 0
        self.fallen[env_ids] = False
        self.balanced[env_ids] = False
