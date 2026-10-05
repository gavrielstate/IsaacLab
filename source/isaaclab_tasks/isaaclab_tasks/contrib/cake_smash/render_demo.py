# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the native Isaac Lab cake with OVRTX Gaussian appearance and optional video."""

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from PIL import Image

from isaaclab_contrib.mpm_gaussians.settings import require_live_gaussian_renderer

from .cake_smash_env_cfg import CakeSmashEnvCfg
from .viewer import CakeViewer


def _run_window(env, viewer, action, args, frame_count):
    """Use native viewer controls; retain the final state until reset or close."""
    reset_requested = False
    frame = 0

    def request_reset():
        nonlocal reset_requested
        # GUI callbacks can execute during presentation. Reset between frames.
        reset_requested = True

    viewer.set_reset_callback(request_reset)
    print("RTX viewer: Space pauses, period steps, Reset restarts. Close the window to exit.")
    print(f"Physics stops after {args.seconds:g} seconds; the final state remains available for inspection.")
    while viewer.is_running():
        started = time.perf_counter()
        if reset_requested:
            viewer.finish_frame()
            env.reset(seed=42)
            viewer._rtx.reset(time=0.0)
            frame = 0
            reset_requested = False
        if frame < frame_count and viewer.should_step():
            env.step(action)
            wp.synchronize_device(args.device)
            frame += 1
        viewer.draw(frame / args.render_fps)
        viewer.finish_frame()
        time.sleep(max(0.0, 1.0 / args.render_fps - (time.perf_counter() - started)))
    viewer.set_reset_callback(None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-asset", required=True)
    parser.add_argument("--gaussian-asset", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--antialiasing", choices=("default", "dlaa", "quality"), default="default")
    parser.add_argument("--lighting", choices=("studio", "default", "front"), default="studio")
    parser.add_argument("--render-fps", type=int, choices=(15, 30), default=30)
    parser.add_argument("--physics-hz", type=int, choices=(30, 60, 90, 120), default=120)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--collider-basis", choices=("Q1", "S2"), default="S2")
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--mass", type=float, default=2.0)
    parser.add_argument("--gap", type=float, default=0.03)
    parser.add_argument("--offset", type=float, nargs=2, default=(0.055, 0.020))
    parser.add_argument(
        "--async-render", action="store_true", help="Overlap native rendering with the next physics step."
    )
    parser.add_argument("--output", type=Path, help="Required for headless rendering; use a fresh directory.")
    parser.add_argument("--window", action="store_true", help="Open the native RTX viewer and retain the final state.")
    parser.add_argument("--paused", action="store_true", help="Start the interactive viewer paused.")
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--verify-render", action="store_true", help="Read back final native arrays outside loop timing."
    )
    args = parser.parse_args()
    require_live_gaussian_renderer()
    if args.seconds <= 0 or args.width <= 0 or args.height <= 0 or args.samples <= 0:
        parser.error("Duration, resolution and samples must be positive.")
    frame_count = round(args.seconds * args.render_fps)
    if frame_count <= 0 or args.iterations <= 0 or args.capacity <= 0:
        parser.error("At least one frame and positive iterations/capacity are required.")
    if args.paused and not args.window:
        parser.error("--paused requires --window.")
    if args.window and (args.video or args.verify_render):
        parser.error("Use headless rendering for --video and --verify-render.")
    if args.output is None and not args.window:
        parser.error("Headless rendering requires --output.")
    if args.output is not None and args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a fresh output directory to preserve earlier results.")
    if args.video and (shutil.which("ffmpeg") is None or args.width % 2 or args.height % 2):
        parser.error("Video needs ffmpeg and even image dimensions.")
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
    cfg = CakeSmashEnvCfg(
        physics_asset_path=args.physics_asset,
        episode_length_s=args.seconds + 1,
        cherry_mass=args.mass,
        drop_gap=args.gap,
        drop_offset=tuple(args.offset),
        decimation=args.physics_hz // args.render_fps,
    )
    cfg.sim.device = args.device
    cfg.sim.dt = 1 / args.physics_hz
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.physics.load_visual_shapes = True
    cfg.sim.physics.solver_cfg.entries[1].solver_cfg.max_iterations = args.iterations
    cfg.sim.physics.solver_cfg.entries[1].solver_cfg.collider_basis = args.collider_basis
    cfg.sim.physics.solver_cfg.entries[1].solver_cfg.max_active_cell_count = args.capacity
    env = gym.make("IsaacContrib-Cake-Smash-Direct", cfg=cfg).unwrapped
    viewer, video = None, None
    try:
        env.reset(seed=42)
        viewer = CakeViewer(
            env,
            args.gaussian_asset,
            args.samples,
            args.async_render,
            antialiasing=args.antialiasing,
            lighting=args.lighting,
            headless=not args.window,
            paused=args.paused,
            width=args.width,
            height=args.height,
        )
        if args.video:
            video = subprocess.Popen(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-n",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{args.width}x{args.height}",
                    "-r",
                    str(args.render_fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    str(args.output / "demo.mp4"),
                ],
                stdin=subprocess.PIPE,
            )
        action = torch.zeros((1, 1), device=args.device)
        # Capture native physics and warm geometry publication before timing.
        # No asynchronous renderer may be active during CUDA graph capture.
        # Reset restores constitutive history and native display frames.
        env.step(action)
        viewer.draw(env.step_dt)
        viewer.finish_frame()
        env.reset(seed=42)
        viewer._rtx.reset(time=0.0)
        for _ in range(3):
            viewer.draw(0.0)
        viewer.finish_frame()
        if args.window:
            _run_window(env, viewer, action, args, frame_count)
            return
        frame_times, capture_times = [], []
        phase_times = []
        for frame in range(frame_count):
            started = time.perf_counter()
            env.step(action)
            wp.synchronize_device(args.device)
            simulated = time.perf_counter()
            viewer.draw((frame + 1) / args.render_fps)
            # Captured frames correspond to this exact simulation tick. Count
            # renderer completion as rendering, never hide it in capture cost.
            current_wait = viewer.finish_frame() if args.video else 0.0
            completed = time.perf_counter()
            frame_times.append(completed - started)
            phase_times.append(
                {
                    "physics_seconds": simulated - started,
                    "current_render_completion_wait_seconds": current_wait,
                    **viewer.last_timings,
                }
            )
            if video is not None:
                video.stdin.write(np.ascontiguousarray(viewer.capture_image()[..., :3]).tobytes())
            capture_times.append(time.perf_counter() - completed)
        final_completion = viewer.finish_frame()
        frame_times[-1] += final_completion
        reset_error = None
        if args.verify_render:
            viewer.stream.verify(viewer._rtx, viewer.prepared)
            # Verification is outside timing. Republish the reset geometry,
            # clearing OVRTX history so the old crushed image cannot ghost.
            env.reset(seed=42)
            viewer._rtx.reset(time=(frame_count + 1) / args.render_fps)
            for _ in range(3):
                viewer.draw((frame_count + 1) / args.render_fps)
                viewer.finish_frame()
            viewer.stream.verify(viewer._rtx, viewer.prepared)
            transform, values = viewer.prepared
            xyz = wp.to_torch(values["positions"][0])
            world = xyz * float(transform[0, 0]) + torch.tensor(transform[3, :3], device=xyz.device)
            rest = torch.tensor(viewer.stream.asset["xyz"], device=xyz.device)
            reset_error = float((world - rest).abs().max())
            if reset_error >= 2e-6:
                raise RuntimeError(f"Reset Gaussian rest-position error: {reset_error} m")
            Image.fromarray(viewer.capture_image()[..., :3]).save(args.output / "reset.png")
        report = {
            "physics": "Isaac Lab native Newton implicit MPM with reciprocal rigid coupling",
            "newton_version": importlib.metadata.version("newton"),
            "ovrtx_version": importlib.metadata.version("ovrtx"),
            "ovstage_version": importlib.metadata.version("ovstage"),
            "stand_representation": "solid mesh",
            "skinning": "Nicolas affine4 render-only binding",
            "native_frame_updates": "at each MPM tick",
            "physics_asset": str(Path(args.physics_asset).expanduser().resolve()),
            "gaussian_asset": str(Path(args.gaussian_asset).expanduser().resolve()),
            "particle_count": len(env.particle_positions),
            "physics_device": args.device,
            "physics_gpu_name": torch.cuda.get_device_name(args.device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "renderer_requested_active_cuda_gpus": viewer._rtx.config.active_cuda_gpus,
            "cherry_mass_kg": args.mass,
            "drop_surface_gap_m": args.gap,
            "drop_offset_m": list(args.offset),
            "gaussian_count": len(viewer.stream.asset["xyz"]),
            "physics_hz": args.physics_hz,
            "render_fps": args.render_fps,
            "iterations": args.iterations,
            "iteration_budget_semantics": "native maximum; solver may converge earlier",
            "collider_basis": args.collider_basis,
            "sparse_capacity": args.capacity,
            "asynchronous_rendering": args.async_render,
            "resolution": [args.width, args.height],
            "camera_eye_m": [float(value) for value in viewer.camera.pos],
            "camera_target_m": [float(value) for value in viewer.camera.pivot],
            "camera_vertical_fov_degrees": float(viewer.camera.fov),
            "samples": args.samples,
            "sampling_overrides": viewer.sampling_overrides,
            "lighting": args.lighting,
            "simulated_seconds": len(frame_times) / args.render_fps,
            "physics_skin_render_seconds": sum(frame_times),
            "capture_encode_seconds": sum(capture_times),
            "realtime_factor_excluding_capture": len(frame_times) / args.render_fps / sum(frame_times),
            "final_renderer_completion_seconds_included": final_completion,
            "runtime_particle_cpu_readback": False,
            "runtime_gaussian_cpu_readback": False,
            "local_bounds_cpu_readback": "six scalars per render frame",
            "native_sparse_status_cpu_readback": "latest and accumulated uint32 arrays per physics tick",
            "rigid_pose_submission": "Newton ViewerRTX's existing transform path",
            "timing_excludes_first_render": True,
            "timing_excludes_physics_capture_and_warmup": True,
            "phase_wall_seconds": {name: sum(p[name] for p in phase_times) for name in phase_times[0]},
            "final_native_array_readback_verified": args.verify_render,
            "reset_native_array_readback_verified": args.verify_render,
            "reset_gaussian_rest_position_error_m": reset_error,
            "reset_verification_excluded_from_timing": True,
            "renderer_history_reset_between_episodes": True,
            "native_array_verification_excluded_from_timing": True,
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        try:
            if video is not None:
                video.stdin.close()
                if video.wait() != 0:
                    raise RuntimeError("ffmpeg failed to encode the recording.")
        finally:
            try:
                if viewer is not None:
                    viewer.close()
            finally:
                env.close()


if __name__ == "__main__":
    main()
