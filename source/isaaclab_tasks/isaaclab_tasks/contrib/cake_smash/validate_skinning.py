# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Validate native cake Gaussian skinning and resets without invoking a renderer.

This checks finite arrays, rest geometry, one-way particle state ownership and
reset frames. It does not verify OVRTX publication or rendered visibility.
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import warp as wp

from pxr import Usd

from isaaclab_tasks.contrib.cake_smash.cake_smash_env_cfg import CakeSmashEnvCfg
from isaaclab_tasks.contrib.cake_smash.gaussian_stream import CakeGaussianStream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-asset", required=True)
    parser.add_argument("--gaussian-asset", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = CakeSmashEnvCfg(physics_asset_path=args.physics_asset)
    cfg.sim.device = args.device
    env = gym.make("IsaacContrib-Cake-Smash-Direct", cfg=cfg).unwrapped
    stream = None
    try:
        env.reset()
        stream = CakeGaussianStream(env, args.gaussian_asset)
        initial_q = wp.to_torch(env.particle_positions).clone()
        stage = Usd.Stage.CreateInMemory()
        stream.author(stage)
        n = len(stream.asset["xyz"])
        before = wp.to_torch(env.particle_positions).clone()
        transform, values = stream.prepare()
        xyz = wp.to_torch(values["positions"][0])
        world = xyz * float(transform[0, 0]) + torch.tensor(transform[3, :3], device=xyz.device)
        expected = torch.tensor(stream.asset["xyz"], device=xyz.device)
        error = float((world - expected).abs().max())
        assert error < 2e-6, error
        assert torch.equal(before, wp.to_torch(env.particle_positions))
        assert all(bool(torch.isfinite(wp.to_torch(a[0])).all()) for a in values.values())
        action = torch.zeros((1, 1), device=args.device)
        for _ in range(30):
            env.step(action)
        transform, values = stream.prepare()
        after_q = wp.to_torch(env.particle_positions).clone()
        assert all(bool(torch.isfinite(wp.to_torch(a[0])).all()) for a in values.values())
        assert torch.equal(after_q, wp.to_torch(env.particle_positions))
        f = wp.to_torch(stream.state.mpm.particle_transform)
        deformation = float((f - torch.eye(3, device=f.device)).abs().max())
        assert deformation > 1e-3, deformation
        env.reset()
        reset = float((wp.to_torch(env.particle_positions) - initial_q).abs().max())
        reset_frames = float(
            (wp.to_torch(stream.state.mpm.particle_transform) - torch.eye(3, device=f.device)).abs().max()
        )
        assert reset == 0 and reset_frames == 0, (reset, reset_frames)
        report = dict(
            gaussian_count=n,
            interior_gaussian_count=int(
                (
                    np.asarray(stream.source.GetPrimAtPath("/World/Cake").GetAttribute("cake:is_surface").Get()) == 0
                ).sum()
            ),
            rest_world_position_error_m=error,
            native_frames_changed=deformation,
            skinning_does_not_write_particle_positions=True,
            all_arrays_finite=True,
            reset_particle_error_m=reset,
            reset_frame_error=reset_frames,
            renderer_exercised=False,
            source="Nicolas affine4 binding and local frame; native Newton update_particle_frames callback at 120 Hz",
        )
    finally:
        if stream:
            stream.close()
        env.close()

    # The display-frame callback adds work to the captured physics graph.
    # Compare its trajectory with an independent native-only environment.
    baseline = gym.make("IsaacContrib-Cake-Smash-Direct", cfg=cfg).unwrapped
    try:
        baseline.reset()
        for _ in range(30):
            baseline.step(action)
        difference = float((wp.to_torch(baseline.particle_positions) - after_q).abs().max())
        assert difference < 2e-5, difference
        report["frame_callback_vs_native_only_position_error_m"] = difference
    finally:
        baseline.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("CAKE_STREAM_VERIFIED", json.dumps(report))


if __name__ == "__main__":
    main()
