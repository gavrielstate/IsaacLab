# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A normal Gym environment whose cake motion comes entirely from native Newton."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import warp as wp
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import NewtonManager, NewtonMPMManager
from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.utils import replace

from .cake_smash_env_cfg import CakeSmashEnvCfg


def _prepare_asset(cfg: CakeSmashEnvCfg, output: Path) -> tuple[list[str], tuple[float, float, float]]:
    """Adapt authored layer paths to MPMObject's geometry/points convention.

    The adapter copies USD opinions, including heterogeneous masses, widths,
    physical materials and initial plastic strain. An explicit strength scale
    adjusts elastic stiffness, damping and yield limits without changing mass
    or particle geometry. The generated file stays outside the source repository.
    """
    source = Path(cfg.physics_asset_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Cake physics USD not found: {source}")
    stage = Usd.Stage.Open(Usd.Stage.Open(str(source)).Flatten())
    if not math.isfinite(cfg.material_strength_scale) or cfg.material_strength_scale <= 0.0:
        raise ValueError("material_strength_scale must be finite and positive.")
    if cfg.material_strength_scale != 1.0:
        for prim in stage.Traverse():
            if "NewtonMPMMaterialAPI" not in prim.GetAppliedSchemas():
                continue
            for name in ("youngsModulus", "elasticDamping", "yieldPressure", "yieldStress"):
                attr = prim.GetAttribute(f"newton:mpm:{name}")
                if not attr or attr.Get() is None:
                    raise ValueError(f"Cake material {prim.GetPath()} must author newton:mpm:{name} to scale strength.")
                attr.Set(float(attr.Get()) * cfg.material_strength_scale)
    cake = stage.GetPrimAtPath("/World/Cake")
    if not cake:
        raise ValueError("Cake asset must contain /World/Cake particle layers.")
    components = [prim for prim in cake.GetChildren() if prim.IsA(UsdGeom.Points)]
    names = []
    top = -math.inf
    for prim in components:
        names.append(prim.GetName())
        points = np.asarray(UsdGeom.Points(prim).GetPointsAttr().Get())
        top = max(top, float(points[:, 2].max()))
        root = f"/World/CakeLayers/{prim.GetName()}"
        UsdGeom.Xform.Define(stage, root)
        UsdGeom.Scope.Define(stage, root + "/geometry")
        Sdf.CopySpec(stage.GetRootLayer(), prim.GetPath(), stage.GetRootLayer(), root + "/geometry/points")
        copied = stage.GetPrimAtPath(root + "/geometry/points")
        copied.CreateRelationship("physics:simulationOwner").SetTargets([Sdf.Path("/World/PhysicsScene")])
        UsdGeom.Imageable(copied).MakeInvisible()
    cake.SetActive(False)
    if not names:
        raise ValueError("Cake asset contains no authored MPM particle layers.")
    cherry = stage.GetPrimAtPath("/World/Cherry")
    radius = float(UsdGeom.Sphere(stage.GetPrimAtPath("/World/Cherry/Collider")).GetRadiusAttr().Get())
    if not math.isfinite(cfg.cherry_mass) or cfg.cherry_mass <= 0.0:
        raise ValueError("cherry_mass must be finite and positive [kg].")
    if not math.isfinite(cfg.drop_gap) or cfg.drop_gap < 0.0:
        raise ValueError("drop_gap must be finite and nonnegative [m].")
    if not all(math.isfinite(value) for value in cfg.drop_offset):
        raise ValueError("drop_offset must contain finite positions [m].")
    if cfg.drop_height is not None and (not math.isfinite(cfg.drop_height) or cfg.drop_height < 0.0):
        raise ValueError("drop_height must be finite and nonnegative [m].")
    # Use the authored physical cake top, including the original top surface,
    # rather than the highest interior particle center.
    authored_pos = UsdGeom.Xformable(cherry).GetLocalTransformation().ExtractTranslation()
    original_gap = stage.GetPrimAtPath("/World").GetAttribute("cake:dropGap").Get()
    top = float(authored_pos[2]) - radius - float(original_gap)
    height = top + cfg.drop_gap + radius if cfg.drop_height is None else cfg.drop_height
    pos = (cfg.drop_offset[0], cfg.drop_offset[1], height)
    if cfg.gravity_only:
        pos = (2.0, 0.0, pos[2])
    UsdPhysics.ArticulationRootAPI.Apply(cherry)
    UsdPhysics.MassAPI(cherry).GetMassAttr().Set(cfg.cherry_mass)
    inertia = 0.4 * cfg.cherry_mass * radius * radius
    UsdPhysics.MassAPI(cherry).GetDiagonalInertiaAttr().Set(Gf.Vec3f(inertia))
    UsdGeom.Xformable(cherry).GetOrderedXformOps()[0].Set(Gf.Vec3d(*pos))
    stage.GetRootLayer().Export(str(output))
    return names, pos


class CakeSmashEnv(DirectRLEnv):
    """Passive drop demo with a standard GPU Gym step/reset boundary.

    The single action is reserved and has no effect: dropping is an initial
    condition. A future robot task can replace this action with robot controls
    without modifying cake physics. Reward is zero because this is a physical
    demonstration rather than an invented cake-height optimization objective.
    """

    cfg: CakeSmashEnvCfg

    def __init__(self, cfg: CakeSmashEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.scene.num_envs != 1:
            raise ValueError("CakeSmash currently validates one workcell; multi-world isolation is not verified.")
        if type(cfg.render_samples) is not int or cfg.render_samples < 1:
            raise ValueError("render_samples must be a positive integer.")
        self.gaussian_stream = None
        self._gaussian_visualizers = []
        self._asset_directory = tempfile.TemporaryDirectory(prefix="isaaclab-cake-")
        adapter_path = Path(self._asset_directory.name) / "cake.usdc"
        self.component_names, initial_pos = _prepare_asset(cfg, adapter_path)
        self._initial_drop_height = initial_pos[2]
        cfg = replace(cfg)
        visualizer_cfgs = cfg.sim.visualizer_cfgs
        if not isinstance(visualizer_cfgs, list):
            visualizer_cfgs = [visualizer_cfgs]
        for visualizer_cfg in [cfg.sim.default_visualizer_cfg, *visualizer_cfgs]:
            if isinstance(visualizer_cfg, NewtonRTXVisualizerCfg):
                visualizer_cfg.render_settings = {
                    **(visualizer_cfg.render_settings or {}),
                    "omni:rtx:rtpt:spp": ("Int", cfg.render_samples),
                }
        # One root owns USD physics import. Runtime views do not reimport their
        # overlapping descendant subtrees into Newton.
        cfg.scene.cake_asset = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/CakeAsset", spawn=sim_utils.UsdFileCfg(usd_path=str(adapter_path))
        )
        cfg.scene.cherry = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/CakeAsset/Cherry",
            spawn=None,
            cloning_contexts=None,
            init_state=RigidObjectCfg.InitialStateCfg(pos=initial_pos),
        )
        for name in self.component_names:
            setattr(
                cfg.scene,
                name,
                MPMObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/CakeAsset/CakeLayers/{name}", spawn=None, cloning_contexts=None
                ),
            )
        super().__init__(cfg, render_mode, **kwargs)
        self.cherry = self.scene["cherry"]
        self.layers = [self.scene[name] for name in self.component_names]
        self._zero_reward = torch.zeros(self.num_envs, device=self.device)
        if self.sim.visualizers:
            from isaaclab_visualizers.newton import NewtonRTXVisualizer

            self._gaussian_visualizers = [viz for viz in self.sim.visualizers if isinstance(viz, NewtonRTXVisualizer)]
        if self._gaussian_visualizers:
            try:
                if len(self._gaussian_visualizers) != 1:
                    raise ValueError("Cake Gaussian publication supports one RTX visualizer.")
                if cfg.gaussian_asset_path is None:
                    raise ValueError("Set gaussian_asset_path or ISAACLAB_CAKE_GAUSSIAN_USD_PATH for newton_rtx.")
                from .gaussian_stream import CakeGaussianStream

                self.gaussian_stream = CakeGaussianStream(self, cfg.gaussian_asset_path)
                self._gaussian_visualizers[0].add_scene_stream(self.gaussian_stream)
                self._gaussian_visualizers[0].register_ui_callback(self._render_drop_controls, position="panel")
            except Exception:
                self.close()
                raise

    @property
    def particle_positions(self) -> wp.array:
        """Native CUDA particle positions [m], without a CPU copy."""
        return NewtonManager.get_state_0().particle_q

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        pass

    def _apply_action(self) -> None:
        pass

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return {"policy": self.cherry.data.root_state_w.torch}

    def _get_rewards(self) -> torch.Tensor:
        return self._zero_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        positions = wp.to_torch(self.particle_positions)
        invalid = ~torch.isfinite(positions).all()
        time_out = self.episode_length_buf >= self.max_episode_length
        if not self.cfg.reset_on_timeout:
            time_out = torch.zeros_like(time_out)
        return invalid.expand(self.num_envs), time_out

    def set_drop_offset(self, offset: tuple[float, float]) -> None:
        """Select the next drop's horizontal position [m]; apply on episode reset.

        Editing the selection leaves the current physical state and camera intact.
        """
        if len(offset) != 2 or not all(math.isfinite(value) for value in offset):
            raise ValueError("Drop offset must contain two finite positions [m].")
        self.cfg.drop_offset = tuple(float(value) for value in offset)

    def set_drop_pose(self, position: tuple[float, float, float]) -> None:
        """Place a stationary upright cherry immediately and remember its next reset pose.

        Position is in workcell coordinates [m]. Windowed simulation pauses so
        the cherry stays at the selected position until Resume. Cake state and
        camera pose are preserved. Safe to call from the standard viewer UI.
        """
        if len(position) != 3 or not all(math.isfinite(value) for value in position) or position[2] < 0.0:
            raise ValueError("Cherry position must contain finite X/Y/Z values with nonnegative Z [m].")
        for visualizer in self._gaussian_visualizers:
            if not visualizer.cfg.headless:
                visualizer.set_training_paused(True)
            visualizer.finish_frame()
        self.cfg.drop_offset = tuple(float(value) for value in position[:2])
        self.cfg.drop_height = float(position[2])
        self.cfg.gravity_only = False
        pose = self.cherry.data.default_root_pose.torch.clone()
        pose[:, :3] = torch.tensor(position, device=self.device) + self.scene.env_origins
        self.cherry.write_root_pose_to_sim_index(root_pose=pose)
        self.cherry.write_root_velocity_to_sim_index(root_velocity=torch.zeros((self.num_envs, 6), device=self.device))
        self.sim.forward()
        for visualizer in self._gaussian_visualizers:
            # Clearing history during ImGui drawing could invalidate the mapped
            # output being displayed. Defer it to the next rendering boundary.
            visualizer.request_render_history_reset()

    def _render_drop_controls(self, imgui) -> None:
        """Add task controls to the standard Newton viewer sidebar."""
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Cake drop"):
            return
        presets = {
            "Center": (0.0, 0.0),
            "Left": (-0.055, 0.020),
            "Right": (0.055, 0.020),
            "Rim": (0.100, 0.020),
            "Cut face": (0.040, -0.060),
        }
        labels = ["Custom", *presets]
        current = next(
            (
                labels.index(name)
                for name, offset in presets.items()
                if all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-6) for a, b in zip(offset, self.cfg.drop_offset))
            ),
            0,
        )
        changed, selected = imgui.combo("Drop preset", current, labels)
        x, y = self.cfg.drop_offset
        if changed and selected:
            x, y = presets[labels[selected]]
        z = self._initial_drop_height if self.cfg.drop_height is None else self.cfg.drop_height
        changed_x, x = imgui.slider_float("Cherry X", x, -0.12, 0.12, "%.3f m")
        changed_y, y = imgui.slider_float("Cherry Y", y, -0.12, 0.12, "%.3f m")
        changed_z, z = imgui.slider_float("Cherry center Z", z, 0.0, 1.0, "%.3f m")
        if (changed and selected) or changed_x or changed_y or changed_z:
            self.set_drop_pose((x, y, z))
        imgui.text_wrapped(
            "Position changes move the cherry and pause simulation. Resume to drop; Reset restores cake."
        )

    def _reset_idx(self, env_ids) -> None:
        for visualizer in self._gaussian_visualizers:
            visualizer.finish_frame()
        super()._reset_idx(env_ids)
        # The coupled solver reset restores rigid state as well as MPM state.
        # Apply the user-selected cherry pose after that restoration.
        NewtonMPMManager.reset_solver_state(world_mask=None)
        if self.gaussian_stream is not None:
            self.gaussian_stream.invalidate()
        pose = self.cherry.data.default_root_pose.torch[env_ids].clone()
        if not self.cfg.gravity_only:
            pose[:, 0] = self.cfg.drop_offset[0]
            pose[:, 1] = self.cfg.drop_offset[1]
        if self.cfg.drop_height is not None:
            pose[:, 2] = self.cfg.drop_height
        pose[:, :3] += self.scene.env_origins[env_ids]
        self.cherry.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.cherry.write_root_velocity_to_sim_index(
            root_velocity=self.cherry.data.default_root_vel.torch[env_ids], env_ids=env_ids
        )
        for visualizer in self._gaussian_visualizers:
            visualizer.reset_render_history()

    def close(self) -> None:
        try:
            for visualizer in self._gaussian_visualizers:
                visualizer.finish_frame()
            super().close()
        finally:
            if self.gaussian_stream is not None:
                self.gaussian_stream.close()
            self._asset_directory.cleanup()
