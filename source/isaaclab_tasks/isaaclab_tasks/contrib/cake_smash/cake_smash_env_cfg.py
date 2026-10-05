# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a rigid cherry crushing an authored layered MPM cake."""

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg


@configclass
class CakeSmashEnvCfg(DirectRLEnvCfg):
    """Single-workcell demo using unmodified Newton implicit MPM and rigid coupling.

    ``physics_asset_path`` identifies an external, authored USD asset with
    ``/World/Cake`` particle layers, ``/World/Stand`` and ``/World/Cherry``.
    Geometry and per-layer material values are imported rather than regenerated.
    """

    physics_asset_path: str = os.environ.get("ISAACLAB_CAKE_PHYSICS_USD_PATH", MISSING)
    cherry_mass: float = 2.0
    drop_gap: float = 0.03
    drop_offset: tuple[float, float] = (0.055, 0.020)
    gravity_only: bool = False
    decimation = 4
    episode_length_s = 10.0
    action_space = 1
    observation_space = 13
    state_space = 0
    seed = 42
    ui_window_class_type = None
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.5, replicate_physics=True)
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=4,
        physics=NewtonCfg(
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name="rigid",
                        solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False, njmax=128, nconmax=64),
                        bodies=[r"/World/envs/env_.*/CakeAsset/Cherry"],
                        include_static_shapes=True,
                        substeps=2,
                    ),
                    CouplerEntryCfg(
                        name="cake",
                        solver_cfg=MPMSolverCfg(
                            voxel_size=0.026,
                            grid_type="sparse",
                            grid_padding=0,
                            max_active_cell_count=1024,
                            max_iterations=60,
                            tolerance=1.0e-4,
                            strain_basis="P0",
                            velocity_basis="Q1",
                            collider_basis="S2",
                            transfer_scheme="pic",
                            integration_scheme="pic",
                            collider_velocity_mode="forward",
                            air_drag=1.0,
                            critical_fraction=0.0,
                        ),
                        all_particles=True,
                        in_place=True,
                    ),
                ],
                proxies=[
                    CouplerProxyMappingCfg(
                        source="rigid",
                        destination="cake",
                        bodies=[r"/World/envs/env_.*/CakeAsset/Cherry"],
                        mass_scale=1.0,
                        mode="lagged",
                        collision_pipeline=None,
                    )
                ],
                iterations=1,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
            use_cuda_graph=True,
        ),
    )
