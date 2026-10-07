# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply

from .targeted_shot_put_env_cfg import TargetedShotPutEnvCfg


class TargetedShotPutEnv(DirectRLEnv):
    """Learn contact-driven release and first-impact accuracy with a fixed-base G1."""

    cfg: TargetedShotPutEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.ball = self.scene["ball"]
        joints, names = self.robot.find_joints(self.cfg.controlled_joints)
        if len(joints) != self.cfg.action_space:
            raise ValueError(f"Expected {self.cfg.action_space} throwing joints, found {names}")
        self.joints = torch.tensor(joints, device=self.device)
        bodies, _ = self.robot.find_bodies(self.cfg.palm_body)
        if len(bodies) != 1:
            raise ValueError(f"Palm body missing: {self.robot.body_names}")
        self.palm = bodies[0]
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.previous_actions = self.actions.clone()
        self.target = torch.zeros((self.num_envs, 2), device=self.device)
        self.mass = torch.ones((self.num_envs, 1), device=self.device)
        self.released = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.release_speed = torch.zeros(self.num_envs, device=self.device)
        self.release_upward = self.released.clone()
        self.flight_time = torch.zeros(self.num_envs, device=self.device)
        self.elapsed_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.landed = self.released.clone()
        self.impact_error = torch.full((self.num_envs,), 4.0, device=self.device)
        self.previous_ball = self.ball.data.root_pos_w.torch.clone()
        self.previous_potential = torch.zeros(self.num_envs, device=self.device)

    def _pre_physics_step(self, actions):
        self.elapsed_steps += 1
        self.previous_actions.copy_(self.actions)
        self.actions = actions.clamp(-1.0, 1.0)
        self.previous_ball.copy_(self.ball.data.root_pos_w.torch)

    def _apply_action(self):
        target = self.robot.data.default_joint_pos.torch.clone()
        target[:, self.joints] += self.actions * 1.0
        limits = self.robot.data.soft_joint_pos_limits.torch
        target = target.clamp(limits[..., 0], limits[..., 1])
        self.robot.actuators.target_command.set_position_index(value=target)

    def _get_observations(self):
        q = self.robot.data.joint_pos.torch[:, self.joints]
        v = self.robot.data.joint_vel.torch[:, self.joints]
        p = self.ball.data.root_pos_w.torch - self.scene.env_origins
        palm = self.robot.data.body_link_pos_w.torch[:, self.palm] - self.scene.env_origins
        obs = torch.cat(
            (
                q,
                0.1 * v,
                self.actions,
                self.target,
                p,
                self.ball.data.root_lin_vel_w.torch,
                p - palm,
                self.mass,
                (self.elapsed_steps / self.max_episode_length).unsqueeze(1),
                self.released.float().unsqueeze(1),
                self.flight_time.unsqueeze(1),
                self.release_speed.unsqueeze(1),
                self.release_upward.float().unsqueeze(1),
            ),
            dim=1,
        )
        return {"policy": obs}

    def _get_dones(self):
        p = self.ball.data.root_pos_w.torch
        palm = self.robot.data.body_link_pos_w.torch[:, self.palm]
        velocity = self.ball.data.root_lin_vel_w.torch
        speed = velocity[:, :2].norm(dim=1)
        new_release = (~self.released) & ((p - palm).norm(dim=1) > 0.16)
        self.release_speed[new_release] = speed[new_release]
        self.release_upward[new_release] = (velocity[new_release, 2] > 0.2) & (self.elapsed_steps[new_release] >= 6)
        self.released |= new_release
        self.flight_time += self.released.float() * self.step_dt
        self.landed = (p[:, 2] <= self.cfg.ball_radius + 0.005) & (self.elapsed_steps > 2)
        # Interpolate the final flight segment, so bouncing/rolling cannot improve accuracy.
        frac = (
            (self.previous_ball[:, 2] - self.cfg.ball_radius) / (self.previous_ball[:, 2] - p[:, 2]).clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        impact = self.previous_ball[:, :2] + frac[:, None] * (p[:, :2] - self.previous_ball[:, :2])
        error = (impact - self.scene.env_origins[:, :2] - self.target).norm(dim=1)
        self.impact_error = torch.where(self.landed, error, self.impact_error)
        escaped = (p - self.scene.env_origins).norm(dim=1) > 4.0
        return self.landed | escaped, self.elapsed_steps >= self.max_episode_length

    def _get_rewards(self):
        p = self.ball.data.root_pos_w.torch - self.scene.env_origins
        v = self.ball.data.root_lin_vel_w.torch
        t = (v[:, 2] + torch.sqrt(v[:, 2].square() + 2 * 9.81 * (p[:, 2] - self.cfg.ball_radius).clamp_min(0))) / 9.81
        predicted = p[:, :2] + t[:, None] * v[:, :2]
        error = (predicted - self.target).norm(dim=1)
        potential = torch.exp(-error.square() / 0.5)
        shaping = 0.99 * potential - self.previous_potential
        self.previous_potential.copy_(potential)
        valid = (
            self.released
            & self.release_upward
            & (self.flight_time >= self.cfg.minimum_flight_s)
            & (self.release_speed >= self.cfg.minimum_release_speed)
        )
        hit = valid & self.landed & (self.impact_error < self.cfg.success_radius)
        terminal = (
            self.landed.float()
            * torch.where(
                valid, 10 * torch.exp(-self.impact_error.square() / self.cfg.accuracy_scale**2), -torch.ones_like(error)
            )
            + hit.float() * 10
        )
        penalty = 0.01 * self.actions.square().mean(dim=1) + 0.02 * (
            self.actions - self.previous_actions
        ).square().mean(dim=1)
        return shaping + terminal - penalty * self.step_dt

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        finished = self.landed[env_ids]
        valid = (
            self.released[env_ids]
            & self.release_upward[env_ids]
            & (self.flight_time[env_ids] >= self.cfg.minimum_flight_s)
            & (self.release_speed[env_ids] >= self.cfg.minimum_release_speed)
        )
        self.extras.setdefault("log", {}).update(
            {
                "Metrics/hit_rate": (finished & valid & (self.impact_error[env_ids] < self.cfg.success_radius))
                .float()
                .mean(),
                "Metrics/valid_throw_rate": (finished & valid).float().mean(),
                "Metrics/impact_error_m": self.impact_error[env_ids].mean(),
            }
        )
        super()._reset_idx(env_ids)
        q = self.robot.data.default_joint_pos.torch[env_ids].clone()
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
        self.robot.actuators.target_command.set_position_index(value=q, env_ids=env_ids)
        count = len(env_ids)
        radius = torch.sqrt(
            torch.rand(count, device=self.device) * (self.cfg.target_max_radius**2 - self.cfg.target_min_radius**2)
            + self.cfg.target_min_radius**2
        )
        angle = torch.empty(count, device=self.device).uniform_(*self.cfg.target_angle_range)
        self.target[env_ids] = torch.stack((radius * angle.cos(), radius * angle.sin()), dim=1)
        progress = min(
            (self.common_step_counter + self.cfg.mass_curriculum_initial_steps)
            / max(self.cfg.mass_curriculum_steps, 1),
            1.0,
        )
        mass_min, mass_final_max = self.cfg.mass_range
        mass_max = mass_min + progress * (mass_final_max - mass_min)
        self.mass[env_ids] = torch.empty((count, 1), device=self.device).uniform_(mass_min, mass_max)
        self.extras["log"].update(
            {
                "Curriculum/mass_max_kg": mass_max,
                "Curriculum/mass_progress": progress,
                "Metrics/ball_mass_kg": self.mass[env_ids].mean(),
            }
        )
        self.ball.set_masses_index(masses=self.mass[env_ids], env_ids=env_ids)
        inertia = torch.eye(3, device=self.device).reshape(1, 1, 9).repeat(count, 1, 1)
        inertia *= 0.4 * self.mass[env_ids, :, None] * self.cfg.ball_radius**2
        self.ball.set_inertias_index(inertias=inertia, env_ids=env_ids)
        pose = self.ball.data.default_root_pose.torch[env_ids].clone()
        offset = torch.tensor(self.cfg.palm_offset, device=self.device).expand(count, -1)
        pose[:, :3] = self.robot.data.body_link_pos_w.torch[env_ids, self.palm] + quat_apply(
            self.robot.data.body_link_quat_w.torch[env_ids, self.palm], offset
        )
        self.ball.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.ball.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((count, 6), device=self.device), env_ids=env_ids
        )
        self.actions[env_ids] = 0
        self.previous_actions[env_ids] = 0
        self.released[env_ids] = False
        self.release_speed[env_ids] = 0
        self.release_upward[env_ids] = False
        self.flight_time[env_ids] = 0
        self.elapsed_steps[env_ids] = 0
        self.landed[env_ids] = False
        self.impact_error[env_ids] = 4
        self.previous_potential[env_ids] = 0
