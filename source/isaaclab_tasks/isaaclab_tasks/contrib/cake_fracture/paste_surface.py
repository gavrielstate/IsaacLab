# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mix sponge Gaussians with Newton particle-surface meshes in the Lab RTX visualizer."""

import numpy as np
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton.geometry import ParticleSurface

from pxr import Sdf, Usd, UsdGeom, UsdShade

from .gaussian_binding import FractureGaussianStream, FragmentBinding


def paste_groups(env):
    """Return cream and frosting particle IDs from imported named component ranges."""
    groups = [[], []]
    for name in env.component_names:
        kind = 0 if name.startswith("Cream") else 1 if name.startswith("Frosting") else -1
        if kind < 0:
            continue
        matches = [r for path, r in NewtonManager.backend.particle_ranges.items() if f"/CakeLayers/{name}/" in path]
        if len(matches) != 1:
            raise ValueError(f"Expected one particle range for paste component {name}")
        start, count = matches[0]
        groups[kind].extend(range(start, start + count))
    return [np.asarray(ids, dtype=np.int32) for ids in groups]


class HybridGaussianStream(FractureGaussianStream):
    """Keep sponge radiance attachments; replace cream/frosting with material surfaces."""

    def create_binding(self, rest, physical_regions):
        excluded = [i for i, name in enumerate(self.env.component_names) if name.startswith(("Cream", "Frosting"))]
        self._original_count = len(self.asset["xyz"])
        self._keep = ~np.isin(self.asset["regions"], excluded)
        for key, value in self.asset.items():
            self.asset[key] = value[self._keep]
        return FragmentBinding(self.asset, rest, physical_regions, self.solver)

    def author(self, stage):
        super().author(stage)
        prim = stage.GetPrimAtPath(self.path)
        # Keep every per-Gaussian radiance/SH attribute aligned with the filtered geometry.
        for attr in prim.GetAttributes():
            for time in [Usd.TimeCode.Default(), *map(Usd.TimeCode, attr.GetTimeSamples())]:
                value = attr.Get(time)
                if value is not None and hasattr(value, "__len__") and len(value) == self._original_count:
                    array = np.ascontiguousarray(np.asarray(value)[self._keep])
                    attr.Set(type(value).FromNumpy(array), time)
        for name, key in (("positions", "xyz"), ("scales", "scales"), ("orientations", "rotations")):
            for time in (Usd.TimeCode.Default(), Usd.TimeCode(0), Usd.TimeCode(1)):
                attr = prim.GetAttribute(name)
                attr.Set(type(attr.Get()).FromNumpy(self.asset[key]), time)


@wp.kernel
def volume_radius(volume: wp.array[float], radii: wp.array[float]):
    i = wp.tid()
    radii[i] = wp.pow(3.0 * volume[i] / (4.0 * wp.pi), 1.0 / 3.0)


class PasteSurfaces:
    """Publish Newton particle surfaces through the standard Lab RTX scene-stream protocol."""

    def __init__(self, env):
        self.env = env
        self.last_step = None
        self.meshes = []
        self.contexts = []
        self.flags = []
        self.bindings = []
        self.renderer = None
        self.triangle_counts = [0, 0]
        self.paths = ["/World/CakePaste/Cream", "/World/CakePaste/Frosting"]
        count = NewtonManager.get_model().particle_count
        self.radii = wp.empty(count, dtype=float, device=env.device)
        for ids in paste_groups(env):
            flags = np.zeros(count, np.int32)
            flags[ids] = 1
            self.flags.append(wp.array(flags, dtype=int, device=env.device))
            self.contexts.append(
                ParticleSurface(
                    voxel_size=env.cfg.paste_surface_voxel_size,
                    kernel_radius=0.04,
                    kernel_scale=0.5,
                    threshold=env.cfg.paste_surface_threshold,
                    smooth_lambda=0.0,
                    anisotropic=True,
                    anisotropy_ratio=4.0,
                    anisotropy_min_neighbors=4,
                    mesh_smooth_iterations=1,
                    mesh_smooth_lambda=0.5,
                    device=env.device,
                )
            )

    def author(self, stage):
        UsdGeom.Xform.Define(stage, "/World/CakePaste")
        for kind, path in enumerate(self.paths):
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            mesh.SetNormalsInterpolation("vertex")
            for attr in (
                mesh.CreatePointsAttr(),
                mesh.CreateNormalsAttr(),
                mesh.CreateFaceVertexCountsAttr(),
                mesh.CreateFaceVertexIndicesAttr(),
            ):
                for time in (Usd.TimeCode.Default(), Usd.TimeCode(0), Usd.TimeCode(1)):
                    attr.Set([], time)
            material = UsdShade.Material.Define(stage, path + "Material")
            shader = UsdShade.Shader.Define(stage, path + "Material/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                (0.93, 0.85, 0.67) if kind == 0 else (0.78, 0.59, 0.43)
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35 if kind == 0 else 0.55)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    def bind(self, renderer):
        from ovrtx import BindingFlag

        self.renderer = renderer
        try:
            for path in self.paths:
                self.bindings.append({})
                for name, dtype, shape in (
                    ("points", "float32", (3,)),
                    ("normals", "float32", (3,)),
                    ("faceVertexIndices", "int32", ()),
                    ("faceVertexCounts", "int32", ()),
                ):
                    self.bindings[-1][name] = renderer.bind_array_attribute(
                        [path], name, dtype=dtype, shape=shape, flags=BindingFlag.OPTIMIZE
                    )
        except Exception:
            self.release_renderer()
            raise

    def publish(self, renderer):
        from ovrtx import DataAccess

        step = self.env.sim.get_physics_step_count()
        if step == self.last_step:
            return
        solver = NewtonManager.get_solver().solver("cake")
        wp.launch(volume_radius, len(self.radii), inputs=[solver.volume], outputs=[self.radii], device=self.env.device)
        meshes = []
        for kind, (surface, flags) in enumerate(zip(self.contexts, self.flags)):
            mesh = surface.extract(self.env.particle_positions, self.radii, particle_flags=flags)
            points, indices, normals = mesh.to_arrays()
            if points is None:
                points = wp.empty(0, dtype=wp.vec3, device=self.env.device)
                indices = wp.empty(0, dtype=int, device=self.env.device)
                normals = wp.empty(0, dtype=wp.vec3, device=self.env.device)
            counts = wp.full(len(indices) // 3, 3, dtype=int, device=self.env.device)
            self.triangle_counts[kind] = len(counts)
            wp.synchronize_device(self.env.device)
            for name, value in (
                ("points", points),
                ("normals", normals),
                ("faceVertexIndices", indices),
                ("faceVertexCounts", counts),
            ):
                self.bindings[kind][name].write([value], data_access=DataAccess.ASYNC)
            meshes.append((mesh, points, indices, normals, counts))
        self.meshes = meshes
        self.last_step = step

    def verify(self):
        """Read back retained mesh arrays after finishing the renderer frame."""
        for path, mesh in zip(self.paths, self.meshes, strict=True):
            for name, expected in zip(
                ("points", "faceVertexIndices", "normals", "faceVertexCounts"), mesh[1:], strict=True
            ):
                restored = self.renderer.read_array_attribute(attribute_name=name, prim_paths=[path])
                actual = np.from_dlpack(restored[path]).reshape(expected.numpy().shape)
                np.testing.assert_allclose(actual, expected.numpy(), atol=1e-7)
                if not np.isfinite(actual).all():
                    raise AssertionError(f"Non-finite surface attribute {name} on {path}")

    def invalidate(self):
        self.last_step = None

    def release_renderer(self):
        for bindings in self.bindings:
            for binding in bindings.values():
                binding.unbind()
        self.bindings.clear()
        self.renderer = None
        self.meshes.clear()
        self.invalidate()

    def close(self):
        for viz in self.env._gaussian_visualizers:
            viz.finish_frame()
        self.release_renderer()
