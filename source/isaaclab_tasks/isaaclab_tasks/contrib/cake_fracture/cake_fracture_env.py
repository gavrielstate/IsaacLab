# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standard Lab cake task with experimental fracture-driven material fields."""

import math

import numpy as np
from isaaclab_newton.physics import NewtonManager

from pxr import Usd, UsdGeom

from isaaclab.utils import configclass

from ..cake_smash.cake_smash_env import CakeSmashEnv
from ..cake_smash.cake_smash_env_cfg import CakeSmashEnvCfg
from .gaussian_binding import FractureGaussianStream
from .solver import CakeFractureSolverCfg, SolverCakeFracture


@configclass
class CakeFractureEnvCfg(CakeSmashEnvCfg):
    contact_broadphase: bool = False
    """Reject disjoint swept particle-extent boxes before field contact."""
    bond_graph: str = "nearest"
    """Rest-neighbor sampling: nearest or signed-axis directional."""
    bond_neighbors: int = 8
    crush_start: float = 0.0
    """Plastic log-volume collapse at onset of sponge cohesive damage."""
    crush_final: float = 0.0
    """Plastic log-volume collapse at complete damage; zero disables."""
    grain_threshold: int = 6
    bond_strength: float = 400.0
    bond_peak: float = 0.001
    bond_final: float = 0.004
    mpm_grid_half_extent: float = 0.52
    """Horizontal grid half extent [m]; enlarge to retain widely scattered debris."""
    fracture_fields: int = 0
    """Zero reserves one slot per particle; contact work uses only active fragments."""
    coupling_fracture: bool = False
    """Update fragment velocity fields at 120 Hz rather than the 30 Hz task boundary."""
    field_separation_damage: float = 0.9999
    """Damage threshold for separating velocity fields while retaining cohesive bond traction."""
    cherry_contact_restitution: float = 0.0
    """Aggregate particle/cherry rebound, bounded by the incoming contact kinetic energy."""
    cherry_contact_stiffness: float = 0.0
    """Finite contact spring stiffness per represented area [N/m^3]; zero uses velocity contact."""
    cherry_contact_damping_ratio: float = 0.75
    node_local_fracture: bool = False
    """Allow local crack opening even while intact bonds elsewhere connect the cake."""
    compression_pressure: float = 1500.0
    compression_hardening: float = 10000.0
    hybrid_paste_rendering: bool = False
    """Render cream/frosting with Newton particle surfaces and retain sponge Gaussians."""
    paste_viscosity: float = 0.0
    """Fracture-activated cream/frosting plastic-flow viscosity [Pa s]; zero disables."""
    paste_surface_voxel_size: float = 0.003
    paste_surface_threshold: float = 0.15
    explicit_substep_rate: int = 4800

    def __post_init__(self):
        super().__post_init__()
        viz = self.sim.default_visualizer_cfg
        viz.eye = (0.52, -0.68, 0.58)
        viz.lookat = (0.0, 0.0, 0.31)
        viz.focal_length = 17.0
        viz.window_width = 1280
        viz.window_height = 960
        viz.show_static = True
        viz.rtx_environment = "studio"
        viz.enable_sky = False
        viz.background_color = (0.12, 0.14, 0.17)

    def configure_solver(self):
        stage = Usd.Stage.Open(self.physics_asset_path)
        tray = UsdGeom.Cylinder(stage.GetPrimAtPath("/World/Stand/TrayCollider"))
        center = UsdGeom.Xformable(tray).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        radius = float(tray.GetRadiusAttr().Get())
        half_height = 0.5 * float(tray.GetHeightAttr().Get())
        lower = min(
            float(
                np.min(
                    np.asarray(UsdGeom.Points(prim).GetPointsAttr().Get())[:, 2]
                    - 0.5 * np.cbrt(4.0 / 3.0 * np.pi) * 0.5 * np.asarray(UsdGeom.Points(prim).GetWidthsAttr().Get())
                )
            )
            for prim in stage.GetPrimAtPath("/World/Cake").GetChildren()
            if prim.IsA(UsdGeom.Points)
        )
        # Author a nonpenetrating rest pose instead of damaging bonds through
        # a first-step collision projection of the copied asset.
        x, y, z = self.cake_position_offset
        self.cake_position_offset = (x, y, z + max(float(center[2]) + half_height - lower - z, 0.0))
        if not math.isfinite(self.mpm_grid_half_extent) or self.mpm_grid_half_extent < 0.052:
            raise ValueError("mpm_grid_half_extent must be finite and at least two grid cells (0.052 m)")
        cherry = UsdGeom.Sphere(stage.GetPrimAtPath("/World/Cherry/Collider"))
        self.sim.physics.solver_cfg.entries[1].solver_cfg = CakeFractureSolverCfg(
            solver_config=SolverCakeFracture.Config(
                grid_origin=(-self.mpm_grid_half_extent, -self.mpm_grid_half_extent, -0.104),
                grid_resolution=(math.ceil(2 * self.mpm_grid_half_extent / 0.026) + 1,) * 2 + (31,),
                grid_spacing=0.026,
                max_active_nodes=8192,
                fields=self.fracture_fields,
                paste_field_count=2 if self.paste_viscosity > 0 else 0,
                substep_rate=self.explicit_substep_rate,
                coupling_rate=120,
                ground_height=0.0,
                ground_friction=0.35,
                field_friction=0.25,
                contact_block_dim=1,
                contact_broadphase=self.contact_broadphase,
                # Finite cylindrical pedestal: unlike the diagnostic's infinite shelf,
                # debris beyond its rim can fall to the floor.
                vessels=((*tuple(center), 0.0, radius, half_height),),
                softening=0.0,
                bruise_rate=0.0,
                adhesion_strength=0.0,
                crush_start=self.crush_start,
                crush_final=self.crush_final,
                bond_graph=self.bond_graph,
                neighbor_count=self.bond_neighbors,
                grain_threshold=self.grain_threshold,
                bond_strength=self.bond_strength,
                bond_peak=self.bond_peak,
                bond_final=self.bond_final,
                cherry_mass=self.cherry_mass,
                cherry_radius=float(cherry.GetRadiusAttr().Get()),
                compression_pressure=self.compression_pressure,
                compression_hardening=self.compression_hardening,
                coupling_fracture=self.coupling_fracture,
                field_separation_damage=self.field_separation_damage,
                node_local_fracture=self.node_local_fracture,
                contact_restitution=self.cherry_contact_restitution,
                contact_stiffness=self.cherry_contact_stiffness,
                contact_damping_ratio=self.cherry_contact_damping_ratio,
            )
        )


class CakeFractureEnv(CakeSmashEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        cfg.configure_solver()
        self.paste_surfaces = None
        if not math.isfinite(cfg.paste_viscosity) or cfg.paste_viscosity < 0:
            raise ValueError("paste_viscosity must be finite and nonnegative")
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        if cfg.paste_viscosity > 0:
            from .paste_surface import paste_groups

            NewtonManager.get_solver().solver("cake").configure_paste(paste_groups(self), cfg.paste_viscosity)
        if cfg.hybrid_paste_rendering and self._gaussian_visualizers:
            from .paste_surface import PasteSurfaces

            self.paste_surfaces = PasteSurfaces(self)
            self._gaussian_visualizers[0].add_scene_stream(self.paste_surfaces)

    def _create_gaussian_stream(self, asset_path):
        if self.cfg.hybrid_paste_rendering:
            from .paste_surface import HybridGaussianStream

            return HybridGaussianStream(self, asset_path, position_offset=self.cfg.cake_position_offset)
        return FractureGaussianStream(self, asset_path, position_offset=self.cfg.cake_position_offset)

    def _reset_solver(self):
        NewtonManager.get_solver().reset(NewtonManager.get_state_0(), world_mask=None, flags=None)
        if getattr(self, "paste_surfaces", None) is not None:
            self.paste_surfaces.invalidate()

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        solver = NewtonManager.get_solver().solver("cake")
        solver.update_fields()
        solver.check()

    def close(self):
        if getattr(self, "paste_surfaces", None) is not None:
            self.paste_surfaces.close()
            self.paste_surfaces = None
        super().close()
