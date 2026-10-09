# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical learned G1 throw into a fracture-capable Gaussian cake."""

import numpy as np
from isaaclab_newton.physics import NewtonManager

from isaaclab.utils import clone, configclass

from ..cake_fracture.cake_fracture_env import CakeFractureEnvCfg
from ..cake_fracture.gaussian_binding import FractureGaussianStream
from ..cake_fracture.paste_surface import paste_groups
from .g1_cake_env import G1CakeEnv
from .g1_cake_env_cfg import G1CakeEnvCfg


@configclass
class G1CakeFractureEnvCfg(G1CakeEnvCfg):
    """Keep the trained robot mechanics and configure the cake independently."""

    cherry_mass: float = 1.5
    cake_offset: tuple[float, float, float] = (1.13, 0.01, 0.0)
    cake: CakeFractureEnvCfg = CakeFractureEnvCfg(gaussian_paste_flow=True)

    def __post_init__(self):
        super().__post_init__()
        visualizer = self.sim.default_visualizer_cfg
        visualizer.eye = (1.60, -1.55, 1.10)
        visualizer.lookat = (0.50, -0.02, 0.65)
        visualizer.rtx_environment = "studio"
        visualizer.enable_sky = False
        visualizer.background_color = (0.12, 0.14, 0.17)


class G1CakeFractureEnv(G1CakeEnv):
    """One physical projectile from hand contact through cohesive MPM impact."""

    cfg: G1CakeFractureEnvCfg

    def __init__(self, cfg: G1CakeFractureEnvCfg, render_mode=None, **kwargs):
        if cfg.cake.hybrid_paste_rendering:
            raise ValueError("G1 fracture cake currently supports pure Gaussian rendering only.")
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        if self.cfg.cake.paste_viscosity > 0:
            NewtonManager.get_solver().solver("cake").configure_paste(paste_groups(self), self.cfg.cake.paste_viscosity)

    def _create_cake_config(self, cfg):
        cake = clone(cfg.cake)
        cake.physics_asset_path = cfg.physics_asset_path
        cake.cherry_mass = cfg.cherry_mass
        cake.configure_solver()
        explicit = cake.sim.physics.solver_cfg.entries[1].solver_cfg.solver_config
        explicit.impactor_body_suffix = "/Ball"
        explicit.cherry_radius = cfg.ball_radius
        explicit.coupling_rate = round(1.0 / cfg.sim.dt)
        offset = np.asarray(cfg.cake_offset)
        explicit.grid_origin = tuple(np.asarray(explicit.grid_origin) + offset)
        explicit.vessels = tuple((*tuple(np.asarray(vessel[:3]) + offset), *vessel[3:]) for vessel in explicit.vessels)
        cfg.cake = cake
        return cake

    def _create_gaussian_stream(self):
        offset = tuple(np.asarray(self.cfg.cake_offset) + np.asarray(self.cfg.cake.cake_position_offset))
        return FractureGaussianStream(
            self, self.cfg.gaussian_asset_path, position_offset=offset, appearance_cfg=self.cfg.cake
        )

    def _reset_solver(self):
        NewtonManager.get_solver().reset(NewtonManager.get_state_0(), world_mask=None, flags=None)

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        solver = NewtonManager.get_solver().solver("cake")
        solver.update_fields()
        solver.check()
