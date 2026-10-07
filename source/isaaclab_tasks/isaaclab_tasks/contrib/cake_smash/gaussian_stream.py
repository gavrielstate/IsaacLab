# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render-only skinning of exterior and interior cake Gaussians."""

from pathlib import Path

import numpy as np
import warp as wp
from isaaclab_newton.physics import NewtonManager
from scipy.spatial import cKDTree

from pxr import Sdf, Usd, UsdGeom, Vt

from isaaclab_contrib.mpm_gaussians import Binding, GaussianArrayStream, GaussianLocalFrame


class CakeGaussianStream:
    """Consume native Newton material frames with Nicolas' Gaussian binding.

    Construct before stepping the environment. A physics-tick callback
    integrates display frames at the actual MPM rate, including when robot
    actuator stepping folds several physics ticks into one manager step.
    """

    def __init__(self, env, asset_path: str, position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.env = env
        self._array_stream = None
        self._renderer = None
        self.prepared = None
        self._published_step = None
        if env.common_step_counter != 0 or NewtonManager.has_captured_cuda_graph():
            raise ValueError("Construct the cake Gaussian stream before advancing physics.")
        cfg = env.cfg.sim.physics
        if cfg.num_substeps != 1:
            raise ValueError("Cake display frames require one MPM substep per physics tick.")
        entry = next(entry for entry in cfg.solver_cfg.entries if entry.name == "cake")
        if entry.substeps != 1:
            raise ValueError("Cake display frames require exactly one MPM entry substep.")
        self.source = Usd.Stage.Open(str(Path(asset_path).expanduser().resolve()))
        if self.source is None:
            raise ValueError(f"Cannot open cake Gaussian USD: {asset_path}")
        prim = self.source.GetPrimAtPath("/World/Cake")
        if not prim or prim.GetTypeName() != "ParticleField3DGaussianSplat":
            raise ValueError("Cake Gaussian USD must contain /World/Cake as ParticleField3DGaussianSplat.")
        self.asset = {
            key: np.asarray(prim.GetAttribute(attribute).Get(), np.float32).copy()
            for key, attribute in (("xyz", "positions"), ("scales", "scales"), ("rotations", "orientations"))
        }
        self.asset["xyz"] += np.asarray(position_offset, np.float32)
        model = NewtonManager.get_model()
        rest = model.particle_q.numpy()
        physical_regions = np.full(len(rest), -1, np.int32)
        ranges = NewtonManager.backend.particle_ranges
        for component, name in enumerate(env.component_names):
            matches = [interval for path, interval in ranges.items() if f"/CakeLayers/{name}/" in path]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one imported particle range for cake component {name}.")
            start, count = matches[0]
            physical_regions[start : start + count] = component
        if np.any(physical_regions < 0):
            raise ValueError("Every imported particle must belong to a cake layer.")
        visual_regions = np.asarray(prim.GetAttribute("cake:component_ids").Get(), np.int32).copy()
        if len(visual_regions) != len(self.asset["xyz"]):
            raise ValueError("Expected one material component per Gaussian.")
        # Decorative frosting curls inherit the nearest physical layer. Named
        # layer Gaussians retain their explicit component, even across interfaces.
        decoration = visual_regions < 0
        nearest = cKDTree(rest).query(self.asset["xyz"][decoration])[1]
        visual_regions[decoration] = physical_regions[nearest]
        self.asset["regions"] = visual_regions
        coupled = NewtonManager.get_solver()
        if coupled is None:
            raise RuntimeError("Initialize the native Newton solver before constructing a Gaussian stream.")
        self.solver = coupled.solver("cake")
        self.state = coupled.entry_state("cake")
        self.path = "/World/CakeGaussians"
        with wp.ScopedDevice(model.device):
            self.binding = self.create_binding(rest, physical_regions)
            self.frame = GaussianLocalFrame(self.asset["xyz"], self.asset["scales"], model.device)
            self.binding.deform_gpu(env.particle_positions, self.deformation_frames, host=False)
        NewtonManager.register_post_physics_step_callback(self.advance_frames)

    @property
    def deformation_frames(self):
        return self.state.mpm.particle_transform

    def create_binding(self, rest, physical_regions):
        return Binding(self.asset, rest, physical_regions)

    def advance_frames(self) -> None:
        """Integrate native display frames at each MPM tick; leave dynamics alone."""
        self.solver.update_particle_frames(self.state, self.state, self.env.cfg.sim.dt)

    def author(self, stage: Usd.Stage) -> None:
        """Copy the external field and its authored radiance materials."""
        layer = self.source.Flatten()
        Sdf.CreatePrimInLayer(stage.GetRootLayer(), "/World")
        Sdf.CopySpec(layer, "/World/Looks", stage.GetRootLayer(), "/World/Looks")
        Sdf.CopySpec(layer, "/World/Cake", stage.GetRootLayer(), self.path)
        prim = stage.GetPrimAtPath(self.path)
        for name in ("positions", "scales", "orientations"):
            attr = prim.GetAttribute(name)
            value = Vt.Vec3fArray.FromNumpy(self.asset["xyz"]) if name == "positions" else attr.Get()
            attr.Set(value, 0)
            attr.Set(value, 1)
        xyz, scales = self.asset["xyz"], self.asset["scales"]
        extent = np.array([(xyz - 3 * scales).min(0), (xyz + 3 * scales).max(0)], np.float32)
        UsdGeom.Boundable(prim).CreateExtentAttr(Vt.Vec3fArray.FromNumpy(extent))

    def bind(self, renderer) -> None:
        self._renderer = renderer
        self._array_stream = GaussianArrayStream(
            renderer, [self.path], {"positions": 3, "scales": 3, "orientations": 4}
        )

    def prepare(self):
        with wp.ScopedDevice(self.env.particle_positions.device):
            xyz, scales, rotations = self.binding.deform_gpu(
                self.env.particle_positions, self.deformation_frames, host=False
            )
            xyz, scales, transform = self.frame.evaluate(xyz, scales)
        return transform, {"positions": [xyz], "scales": [scales], "orientations": [rotations]}

    def publish(self, renderer) -> None:
        """Prepare and retain geometry for Lab's standard RTX visualizer."""
        step = self.env.sim.get_physics_step_count()
        if self._published_step == step:
            # Rewriting unchanged fields prevents RTX temporal convergence.
            # Camera-only redraws must keep the existing geometry bindings.
            return
        self.prepared = self.prepare()
        wp.synchronize_device(self.env.particle_positions.device)
        self.update(renderer, self.prepared)
        self._published_step = step

    def invalidate(self) -> None:
        """Republish after state restoration without a physics step."""
        self._published_step = None

    def update(self, renderer, prepared) -> None:
        from ovrtx import Semantic  # noqa: PLC0415

        transform, values = prepared
        renderer.write_attribute(
            prim_paths=[self.path], attribute_name="omni:xform", tensor=transform[None], semantic=Semantic.XFORM_MAT4x4
        )
        self._array_stream.write(values)

    def verify(self, renderer=None, prepared=None) -> None:
        """Read back published arrays outside timing; visible frames still need inspection."""
        if renderer is None:
            renderer = self._renderer
        if prepared is None:
            prepared = self.prepared
        if self._array_stream is None or prepared is None:
            raise RuntimeError("Publish Gaussian geometry before verification.")
        _, values = prepared
        self._array_stream.verify(
            renderer, {name: [array.numpy() for array in arrays] for name, arrays in values.items()}
        )

    def release_renderer(self) -> None:
        """Release renderer bindings while keeping captured physics frames alive."""
        if self._array_stream is not None:
            self._array_stream.close()
            self._array_stream = None
        self._renderer = None
        self.prepared = None
        self.invalidate()

    def close(self) -> None:
        self.release_renderer()
        NewtonManager.unregister_post_physics_step_callback(self.advance_frames)
