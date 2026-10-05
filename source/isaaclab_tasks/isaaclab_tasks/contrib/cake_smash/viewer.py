# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native Newton RTX viewer with Nicolas' persistent Gaussian publication."""

import math
import time

import numpy as np
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton.viewer import ViewerRTX

from pxr import Gf, Sdf, UsdGeom, UsdLux

from isaaclab_contrib.mpm_gaussians.settings import apply_sampling_settings, require_live_gaussian_renderer

from .gaussian_stream import CakeGaussianStream


class CakeViewer(ViewerRTX):
    """Render cake Gaussians and the Newton model's solid stand/cherry geometry."""

    def __init__(
        self,
        env,
        gaussian_asset: str,
        samples: int = 1,
        asynchronous: bool = False,
        antialiasing: str = "default",
        lighting: str = "studio",
        **kwargs,
    ):
        require_live_gaussian_renderer()
        self.env = env
        self.samples = samples
        self.antialiasing = antialiasing
        self.lighting = lighting
        self.sampling_overrides = {}
        self.last_timings = {}
        self._cpu_presentation = False
        self.stream = CakeGaussianStream(env, gaussian_asset)
        try:
            with wp.ScopedDevice(env.particle_positions.device):
                super().__init__(
                    environment="default" if lighting == "front" else lighting,
                    fps=round(1 / env.step_dt),
                    async_rendering=asynchronous,
                    **kwargs,
                )
                self.show_particles = False
                self.show_static = True
                self.set_model(NewtonManager.get_model())
            target = np.array([0, 0, 0.20])
            eye = np.array([0.4666667, -0.7066667, 0.4933333])
            delta = target - eye
            self.set_camera(
                wp.vec3(*eye),
                math.degrees(math.asin(delta[2] / np.linalg.norm(delta))),
                math.degrees(math.atan2(delta[1], delta[0])),
            )
            self.camera.near = 0.001
            self.camera.pivot = type(self.camera.pos)(*target)
        except Exception:
            # Native RTX initialization is deferred until the first draw, but
            # the stream has already registered its native frame callback.
            self.stream.close()
            raise

    def _init_window(self):
        try:
            super()._init_window()
        except RuntimeError as error:
            if "Failed to register GL texture resource" not in str(error):
                raise
            # The native window was partially constructed before CUDA/GL
            # registration failed. Replace only its image presentation path.
            if self._window is not None:
                self._window.close()
                self._window = None
            self._tex_resource = None
            from .cpu_window import initialize  # noqa: PLC0415

            initialize(self)
            self._cpu_presentation = True
            print("CUDA/OpenGL interop unavailable: using CPU image presentation; Newton and RTX remain on GPU.")

    def _init_ovrtx(self):
        self.stream.author(self.stage)
        super()._init_ovrtx()
        self.stream.bind(self._rtx)

    def _add_default_lights(self):
        super()._add_default_lights()
        if self.lighting == "front":
            light = self.stage.GetPrimAtPath("/root/_RTXDistantLight")
            UsdGeom.Xform(light).GetOrderedXformOps()[0].Set(Gf.Vec3f(45, -30, 0))
            UsdLux.DistantLight(light).GetAngleAttr().Set(6.0)

    def _add_camera_lights_and_render_product(self):
        super()._add_camera_lights_and_render_product()
        for prim in self.stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                UsdGeom.Camera(prim).CreateClippingRangeAttr(Gf.Vec2f(0.001, 100))
        product = self.stage.GetPrimAtPath(self._render_product_path)
        product.CreateAttribute("omni:rtx:dlss:frameGeneration", Sdf.ValueTypeNames.Bool).Set(False)
        self.sampling_overrides = apply_sampling_settings(product, self.antialiasing, self.samples)

    def _render_and_display(self):
        started = time.perf_counter()
        self.stream.update(self._rtx, self.prepared)
        published = time.perf_counter()
        super()._render_and_display()
        if not self._async:
            self._present_frame()
        self.last_timings["publication_seconds"] = published - started
        self.last_timings["ovrtx_render_seconds"] = time.perf_counter() - published

    def _log_particles(self, state):
        # Gaussians supply cake appearance. Skipping the debug point cloud also
        # avoids ViewerRTX's otherwise unnecessary particle readback.
        pass

    def draw(self, time_s: float) -> None:
        with wp.ScopedDevice(self.env.particle_positions.device):
            # The prior render can overlap physics, but must complete before
            # skinning reuses its referenced Gaussian arrays and transforms.
            self.last_timings["prior_render_completion_wait_seconds"] = self.finish_frame()
            started = time.perf_counter()
            self.prepared = self.stream.prepare()
            # Persistent publication receives completed caller-owned CUDA
            # arrays. This fence transfers no particle or Gaussian data to CPU.
            wp.synchronize_device(self.env.particle_positions.device)
            self.last_timings["skinning_bounds_fence_seconds"] = time.perf_counter() - started
            self.begin_frame(time_s)
            self.log_state(NewtonManager.get_state_0())
            self.end_frame()

    def finish_frame(self) -> float:
        """Complete a native asynchronous render before reusing its inputs."""
        if self._render_result is None:
            return 0.0
        started = time.perf_counter()
        self._render_products = self._render_result.wait().fetch()
        self._render_result = None
        self._present_frame()
        return time.perf_counter() - started

    def _present_frame(self) -> None:
        """Present OVRTX 0.6 output with the native GPU/OpenGL blitter.

        Its output keys are prim paths. Find LdrColor by semantic source name;
        the stock viewer's older literal-key lookup misses these frames.
        """
        if self._window is None or self._window.context is None or self._should_close:
            return
        from ovrtx import Device  # noqa: PLC0415

        for product in (self._render_products or {}).values():
            for frame in product.frames:
                for variable in frame.render_vars.values():
                    if variable.source_name == "LdrColor":
                        if self._cpu_presentation:
                            from .cpu_window import present  # noqa: PLC0415

                            with variable.map(device=Device.CPU) as mapping:
                                present(self, np.from_dlpack(mapping))
                            return
                        with variable.map(device=Device.CUDA) as mapping:
                            pixels = wp.from_dlpack(mapping, dtype=wp.vec4ub)
                            self._blit_to_window(pixels)
                            mapping.unmap(stream=pixels.device.stream.cuda_stream)
                        return
        raise RuntimeError("No completed OVRTX LdrColor output for window presentation.")

    def capture_image(self) -> np.ndarray:
        """Read OVRTX 0.6 color output, whose keys are render-var prim paths."""
        from ovrtx import Device  # noqa: PLC0415

        self.finish_frame()
        for product in (self._render_products or {}).values():
            for frame in product.frames:
                for variable in frame.render_vars.values():
                    if variable.source_name == "LdrColor":
                        with variable.map(device=Device.CPU) as mapping:
                            return np.from_dlpack(mapping).copy()
        raise RuntimeError("No completed OVRTX LdrColor output.")

    def close(self) -> None:
        self.finish_frame()
        renderer = self._rtx
        self.stream.close()
        super().close()
        if renderer is not None:
            renderer.destroy()
