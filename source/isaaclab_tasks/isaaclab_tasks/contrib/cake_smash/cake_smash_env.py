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

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.utils import replace

from .cake_smash_env_cfg import CakeSmashEnvCfg


def _prepare_asset(cfg: CakeSmashEnvCfg, output: Path) -> tuple[list[str], tuple[float, float, float]]:
    """Adapt authored layer paths to MPMObject's geometry/points convention.

    The adapter copies USD opinions, including heterogeneous masses, widths,
    physical materials and initial plastic strain. It does not alter particles
    or invent physics. The generated file stays outside the source repository.
    """
    source = Path(cfg.physics_asset_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Cake physics USD not found: {source}")
    stage = Usd.Stage.Open(Usd.Stage.Open(str(source)).Flatten())
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
    # Use the authored physical cake top, including the original top surface,
    # rather than the highest interior particle center.
    authored_pos = UsdGeom.Xformable(cherry).GetLocalTransformation().ExtractTranslation()
    original_gap = stage.GetPrimAtPath("/World").GetAttribute("cake:dropGap").Get()
    top = float(authored_pos[2]) - radius - float(original_gap)
    pos = (cfg.drop_offset[0], cfg.drop_offset[1], top + cfg.drop_gap + radius)
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
        self._asset_directory = tempfile.TemporaryDirectory(prefix="isaaclab-cake-")
        adapter_path = Path(self._asset_directory.name) / "cake.usdc"
        self.component_names, initial_pos = _prepare_asset(cfg, adapter_path)
        cfg = replace(cfg)
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
        return invalid.expand(self.num_envs), self.episode_length_buf >= self.max_episode_length

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        pose = self.cherry.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        self.cherry.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.cherry.write_root_velocity_to_sim_index(
            root_velocity=self.cherry.data.default_root_vel.torch[env_ids], env_ids=env_ids
        )
        NewtonMPMManager.reset_solver_state(world_mask=None)

    def close(self) -> None:
        super().close()
        self._asset_directory.cleanup()
