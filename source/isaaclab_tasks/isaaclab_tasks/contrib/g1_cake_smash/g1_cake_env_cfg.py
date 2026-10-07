# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a trained fixed-base cherry throw into cake."""

import os
from dataclasses import MISSING

from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

from isaaclab.utils import configclass

from .targeted_shot_put_env_cfg import TargetedShotPutEnvCfg


@configclass
class G1CakeEnvCfg(TargetedShotPutEnvCfg):
    physics_asset_path: str = os.environ.get("ISAACLAB_CAKE_PHYSICS_USD_PATH", MISSING)
    gaussian_asset_path: str | None = os.environ.get("ISAACLAB_CAKE_GAUSSIAN_USD_PATH")
    policy_params_path: str = os.environ.get("ISAACLAB_SHOT_POLICY_PARAMS_PATH", MISSING)
    cherry_mass: float = 2.0
    cake_offset: tuple[float, float, float] = (1.08, -0.025, 0.0)
    ground_target: tuple[float, float] = (1.2, 0.0)
    render_samples: int = 1
    seed = 9731
    ui_window_class_type = None

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.sim.default_visualizer_cfg = NewtonRTXVisualizerCfg(
            eye=(2.25, -2.8, 1.7),
            lookat=(0.60, 0.0, 0.60),
            window_width=1280,
            window_height=960,
            show_static=False,
            distant_light_rotation=(45, -30, 0),
            render_settings={
                "omni:rtx:dlss:frameGeneration": ("Bool", False),
                "omni:rtx:post:aa:op": ("Token", "dlss"),
                "omni:rtx:post:dlss:execMode": ("Token", "quality"),
            },
        )
