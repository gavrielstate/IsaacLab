# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A single physical cherry travels from trained hand contact into native MPM cake."""

import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import warp as wp
import yaml
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import NewtonCfg, NewtonManager, NewtonMPMManager
from isaaclab_visualizers.newton import NewtonRTXVisualizer, NewtonRTXVisualizerCfg

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import UsdFileCfg
from isaaclab.utils import clone, replace, update_from_dict

from isaaclab_tasks.contrib.cake_smash.cake_smash_env import _prepare_asset
from isaaclab_tasks.contrib.cake_smash.cake_smash_env_cfg import CakeSmashEnvCfg
from isaaclab_tasks.contrib.cake_smash.gaussian_stream import CakeGaussianStream

from .g1_cake_env_cfg import G1CakeEnvCfg
from .targeted_shot_put_env import TargetedShotPutEnv


def _restore_policy_mechanics(cfg: G1CakeEnvCfg) -> None:
    """Restore recorded mechanics without replacing the current task or visualizers."""
    training = yaml.load((Path(cfg.policy_params_path).expanduser() / "env.yaml").read_text(), Loader=yaml.FullLoader)
    if training["action_space"] != 12 or training["observation_space"] != 53:
        raise ValueError("This task requires the fixed-base 53-observation / 12-action policy.")
    for field in (
        "decimation",
        "episode_length_s",
        "controlled_joints",
        "palm_body",
        "ball_radius",
        "palm_offset",
        "target_min_radius",
        "target_max_radius",
        "target_angle_range",
        "minimum_flight_s",
        "minimum_release_speed",
    ):
        setattr(cfg, field, training[field])
    for field in ("dt", "gravity", "render_interval", "use_newton_actuators"):
        setattr(cfg.sim, field, training["sim"][field])
    solver = training["sim"]["physics"]["solver_cfg"]
    update_from_dict(
        cfg.sim.physics.solver_cfg,
        {key: value for key, value in solver.items() if key not in ("class_type", "solver_type")},
    )
    robot = training["scene"]["robot"]
    cfg.scene.robot.spawn.usd_path = robot["spawn"]["usd_path"]
    cfg.scene.ground.spawn.usd_path = training["scene"]["ground"]["spawn"]["usd_path"]
    update_from_dict(cfg.scene.robot.init_state, robot["init_state"])
    for name, values in robot["actuators"].items():
        update_from_dict(
            cfg.scene.robot.actuators[name], {key: value for key, value in values.items() if key != "class_type"}
        )


def _prepare_cherry(stage: Usd.Stage, output: Path, radius: float, mass: float) -> None:
    """Keep the trained collision sphere and scale procedural cherry decoration to it."""
    cherry = Usd.Stage.CreateNew(str(output))
    Sdf.CopySpec(stage.GetRootLayer(), "/World/Cherry", cherry.GetRootLayer(), "/Cherry")
    Sdf.CopySpec(stage.GetRootLayer(), "/World/Looks", cherry.GetRootLayer(), "/Cherry/Looks")
    root = cherry.GetPrimAtPath("/Cherry")
    root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    UsdGeom.Xformable(root).GetOrderedXformOps()[0].Set(Gf.Vec3d(0.0))
    sphere = UsdGeom.Sphere(cherry.GetPrimAtPath("/Cherry/Collider"))
    scale = radius / float(sphere.GetRadiusAttr().Get())
    sphere.GetRadiusAttr().Set(radius)
    sphere.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-radius), Gf.Vec3f(radius)]))
    for prim in cherry.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), np.float32) * scale
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
            mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.asarray([points.min(0), points.max(0)])))
        for relation in prim.GetRelationships():
            targets = relation.GetTargets()
            if targets:
                relation.SetTargets(
                    [target.ReplacePrefix(Sdf.Path("/World/Looks"), Sdf.Path("/Cherry/Looks")) for target in targets]
                )
    UsdPhysics.MassAPI(root).GetMassAttr().Set(mass)
    UsdPhysics.MassAPI(root).GetDiagonalInertiaAttr().Set(Gf.Vec3f(0.4 * mass * radius**2))
    material = UsdShade.Material.Define(cherry, "/Cherry/Looks/ShotPhysics")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(1.0)
    physics.CreateDynamicFrictionAttr(0.8)
    physics.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material, materialPurpose="physics")
    cherry.SetDefaultPrim(root)
    cherry.GetRootLayer().Save()


class G1CakeEnv(TargetedShotPutEnv):
    """Fixed-base trained throw, with manual reset and standard Lab RTX publication."""

    cfg: G1CakeEnvCfg

    def __init__(self, cfg: G1CakeEnvCfg, render_mode=None, **kwargs):
        if cfg.scene.num_envs != 1:
            raise ValueError("G1 Cake currently supports one workcell.")
        if type(cfg.render_samples) is not int or cfg.render_samples < 1:
            raise ValueError("render_samples must be a positive integer.")
        if not math.isfinite(cfg.cherry_mass) or not 1.5 <= cfg.cherry_mass <= 2.5:
            raise ValueError("Use a cherry mass in the trained range [1.5, 2.5] kg.")
        if not all(math.isfinite(value) for value in (*cfg.cake_offset, *cfg.ground_target)):
            raise ValueError("Cake and target positions must be finite [m].")
        cfg = replace(cfg)
        _restore_policy_mechanics(cfg)
        cfg.mass_range = (cfg.cherry_mass, cfg.cherry_mass)
        cfg.mass_curriculum_initial_steps = cfg.mass_curriculum_steps
        cake_cfg = self._create_cake_config(cfg)
        coupled = clone(cake_cfg.sim.physics.solver_cfg)
        coupled.entries[0].solver_cfg = cfg.sim.physics.solver_cfg
        coupled.entries[0].bodies = [r"/World/envs/env_.*/Robot/.*", r"/World/envs/env_.*/Ball"]
        coupled.entries[0].substeps = 1
        coupled.proxies[0].bodies = [r"/World/envs/env_.*/Ball"]
        cfg.sim.physics = NewtonCfg(
            solver_cfg=coupled, collision_cfg=cake_cfg.sim.physics.collision_cfg, use_cuda_graph=True
        )
        self.gaussian_stream = None
        self._gaussian_visualizers = []
        self._asset_directory = tempfile.TemporaryDirectory(prefix="isaaclab-g1-cake-")
        adapter = Path(self._asset_directory.name) / "cake.usdc"
        self.component_names, _ = _prepare_asset(cake_cfg, adapter)
        stage = Usd.Stage.Open(str(adapter))
        cherry = adapter.with_name("cherry.usdc")
        _prepare_cherry(stage, cherry, cfg.ball_radius, cfg.cherry_mass)
        stage.RemovePrim("/World/Cherry")
        for name in self.component_names:
            points = UsdGeom.Points(stage.GetPrimAtPath(f"/World/CakeLayers/{name}/geometry/points")).GetPointsAttr()
            points.Set(
                Vt.Vec3fArray.FromNumpy(np.asarray(points.Get(), np.float32) + np.asarray(cfg.cake_offset, np.float32))
            )
        UsdGeom.Xform.Define(stage, "/World/Stand").AddTranslateOp(opSuffix="cake_offset").Set(
            Gf.Vec3d(*cfg.cake_offset)
        )
        stage.GetRootLayer().Save()
        cfg.scene.ball.spawn = UsdFileCfg(usd_path=str(cherry))
        cfg.scene.cake_asset = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/CakeAsset", spawn=UsdFileCfg(usd_path=str(adapter))
        )
        for name in self.component_names:
            setattr(
                cfg.scene,
                name,
                MPMObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/CakeAsset/CakeLayers/{name}", spawn=None, cloning_contexts=None
                ),
            )
        visualizers = cfg.sim.visualizer_cfgs
        if not isinstance(visualizers, list):
            visualizers = [visualizers]
        for viz in [cfg.sim.default_visualizer_cfg, *visualizers]:
            if isinstance(viz, NewtonRTXVisualizerCfg):
                viz.render_settings = {**(viz.render_settings or {}), "omni:rtx:rtpt:spp": ("Int", cfg.render_samples)}
        super().__init__(cfg, render_mode, **kwargs)
        self._gaussian_visualizers = [viz for viz in self.sim.visualizers if isinstance(viz, NewtonRTXVisualizer)]
        if self._gaussian_visualizers:
            if len(self._gaussian_visualizers) != 1 or cfg.gaussian_asset_path is None:
                self.close()
                raise ValueError("G1 Cake requires one RTX visualizer and a cake Gaussian asset.")
            self.gaussian_stream = self._create_gaussian_stream()
            self._gaussian_visualizers[0].add_scene_stream(self.gaussian_stream)

    def _create_cake_config(self, cfg):
        """Select cake physics before constructing the coupled robot scene."""
        return CakeSmashEnvCfg(physics_asset_path=cfg.physics_asset_path, cherry_mass=cfg.cherry_mass)

    def _create_gaussian_stream(self):
        """Bind the native material frames to the translated Gaussian asset."""
        return CakeGaussianStream(self, self.cfg.gaussian_asset_path, position_offset=self.cfg.cake_offset)

    @property
    def particle_positions(self) -> wp.array:
        """Native particle positions [m], without CPU readback."""
        return NewtonManager.get_state_0().particle_q

    def _get_dones(self):
        super()._get_dones()
        invalid = ~torch.isfinite(self.ball.data.root_pos_w.torch).all(dim=1)
        invalid |= ~torch.isfinite(wp.to_torch(self.particle_positions)).all()
        return invalid, torch.zeros_like(invalid)

    def _pre_physics_step(self, actions):
        finished = self.landed | (self.elapsed_steps >= self.max_episode_length)
        super()._pre_physics_step(torch.where(finished[:, None], self.actions, actions))

    def _reset_idx(self, env_ids):
        for viz in self._gaussian_visualizers:
            viz.finish_frame()
        self._reset_solver()
        super()._reset_idx(env_ids)
        self.target[env_ids] = torch.tensor(self.cfg.ground_target, device=self.device)
        if self.gaussian_stream is not None:
            self.gaussian_stream.invalidate()
        for viz in self._gaussian_visualizers:
            viz.reset_render_history()

    def _reset_solver(self):
        """Restore material history before placing the trained shot in the palm."""
        NewtonMPMManager.reset_solver_state(world_mask=None)

    def close(self):
        try:
            for viz in self._gaussian_visualizers:
                viz.finish_frame()
            super().close()
        finally:
            if self.gaussian_stream is not None:
                self.gaussian_stream.close()
            self._asset_directory.cleanup()
