# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""External Cake Gaussian field through the standard Lab RTX lifecycle."""

import os

import gymnasium as gym
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from isaaclab_visualizers.newton import NewtonRTXVisualizer, NewtonRTXVisualizerCfg

from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_tasks.contrib.cake_smash.cake_smash_env_cfg import CakeSmashEnvCfg


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_standard_rtx_publishes_cake_without_changing_physics_and_resets(device):
    """Published field is visible, changes under impact, and restores after reset."""
    physics_asset = os.environ.get("ISAACLAB_CAKE_PHYSICS_USD_PATH")
    gaussian_asset = os.environ.get("ISAACLAB_CAKE_GAUSSIAN_USD_PATH")
    if not physics_asset or not gaussian_asset:
        pytest.skip("External Cake physics and Gaussian reference assets were not selected.")
    pytest.importorskip("ovrtx")
    cfg = CakeSmashEnvCfg(physics_asset_path=physics_asset, gaussian_asset_path=gaussian_asset)
    # A nominal one-step episode must not repeatedly restore the cake during
    # interactive inspection: the default task uses manual reset.
    cfg.episode_length_s = 1.0 / 30.0
    cfg.sim.device = device
    cfg.sim.visualizer_cfgs = [
        NewtonRTXVisualizerCfg(
            headless=True,
            window_width=1280,
            window_height=960,
            enable_picking=False,
            show_static=True,
            async_rendering=True,
            distant_light_rotation=(45, -30, 0),
            render_settings=dict(cfg.sim.default_visualizer_cfg.render_settings),
        )
    ]
    env = gym.make("IsaacContrib-Cake-Smash-Direct", cfg=cfg).unwrapped
    try:
        visualizer = env.sim.visualizers[0]
        assert type(visualizer) is NewtonRTXVisualizer
        action = torch.zeros((1, 1), device=device)
        env.reset(seed=42)
        # Capture physical kernels before the first asynchronous RTX frame.
        env.step(action)
        env.reset(seed=42)
        rest = wp.to_torch(env.particle_positions).clone()
        rigid_rest = env.cherry.data.root_state_w.torch.clone()
        for _ in range(3):
            env.sim.render()
            visualizer.render_frame()
        visualizer.finish_frame()
        env.gaussian_stream.verify()
        rest_published = env.gaussian_stream.prepared[1]["positions"][0].numpy().copy()
        pixels = visualizer.capture_rgb_array()
        assert pixels.shape == (960, 1280, 3)
        # Independently detect the authored sponge; hidden/debug particles and
        # rigid meshes alone cannot supply this brown textured face.
        rgb = pixels.astype(np.float32)
        brown = (rgb[..., 0] > 1.12 * rgb[..., 1]) & (rgb[..., 1] > 1.08 * rgb[..., 2]) & (rgb[..., 1] > 15)
        assert np.count_nonzero(brown) > 1000
        # Camera-only redraws must let temporal reconstruction converge. Array
        # uploads of unchanged geometry previously kept this scene shimmering.
        frames = []
        for i in range(50):
            env.sim.render()
            visualizer.render_frame()
            visualizer.finish_frame()
            if i >= 30:
                frames.append(visualizer.capture_rgb_array().astype(np.float32))
        assert np.abs(np.diff(np.stack(frames), axis=0))[:, brown].mean() < 1.0
        torch.testing.assert_close(wp.to_torch(env.particle_positions), rest, rtol=0.0, atol=0.0)
        torch.testing.assert_close(env.cherry.data.root_state_w.torch, rigid_rest, rtol=0.0, atol=0.0)
        for _ in range(30):
            env.step(action)
        before_render = wp.to_torch(env.particle_positions).clone()
        env.sim.render()
        visualizer.render_frame()
        visualizer.finish_frame()
        env.gaussian_stream.verify()
        changed = env.gaussian_stream.prepared[1]["positions"][0].numpy()
        assert np.max(np.abs(changed - rest_published)) > 1e-3
        torch.testing.assert_close(wp.to_torch(env.particle_positions), before_render, rtol=0.0, atol=0.0)
        # Reset while another native render is outstanding exercises the same
        # buffer/history lifecycle as the standard Lab Reset control.
        env.sim.render()
        visualizer.render_frame()
        env.reset(seed=42)
        for _ in range(3):
            env.sim.render()
            visualizer.render_frame()
        visualizer.finish_frame()
        env.gaussian_stream.verify()
        np.testing.assert_allclose(
            env.gaussian_stream.prepared[1]["positions"][0].numpy(), rest_published, rtol=0.0, atol=2e-6
        )
        # Selecting another drop leaves the live state untouched until Reset,
        # then the real rigid solver starts at the selected horizontal position.
        visualizer.set_camera_view((0.40, -0.55, 0.45), (0.0, 0.0, 0.20))
        camera = visualizer._viewer.camera
        camera_pose = (tuple(camera.pos), camera.pitch, camera.yaw, tuple(camera.pivot))
        before_selection = env.cherry.data.root_state_w.torch.clone()
        env.set_drop_offset((-0.055, 0.020))
        torch.testing.assert_close(env.cherry.data.root_state_w.torch, before_selection, rtol=0.0, atol=0.0)
        torch.testing.assert_close(wp.to_torch(env.particle_positions), rest, rtol=0.0, atol=0.0)
        env.reset(seed=42)
        assert (tuple(camera.pos), camera.pitch, camera.yaw, tuple(camera.pivot)) == camera_pose
        torch.testing.assert_close(
            env.cherry.data.root_state_w.torch[0, :2],
            torch.tensor([-0.055, 0.020], device=device),
            rtol=0.0,
            atol=1e-7,
        )
        torch.testing.assert_close(env.cherry.data.root_state_w.torch[:, 2:], rigid_rest[:, 2:], rtol=0.0, atol=0.0)
        torch.testing.assert_close(wp.to_torch(env.particle_positions), rest, rtol=0.0, atol=0.0)
        env.step(action)
        torch.testing.assert_close(
            env.cherry.data.root_state_w.torch[0, :2],
            torch.tensor([-0.055, 0.020], device=device),
            rtol=0.0,
            atol=1e-6,
        )
        # Live XYZ editing after impact must move only the cherry, preserving
        # deformed particles, velocity and native material frames without reset.
        for _ in range(29):
            env.step(action)
        deformed = wp.to_torch(env.particle_positions).clone()
        assert float((deformed - rest).abs().max()) > 1e-3
        particle_velocity = wp.to_torch(NewtonManager.get_state_0().particle_qd).clone()
        material_frames = wp.to_torch(env.gaussian_stream.state.mpm.particle_transform).clone()
        env.sim.render()
        visualizer.render_frame()
        env.set_drop_pose((0.040, -0.060, 0.450))
        expected = torch.tensor([0.040, -0.060, 0.450], device=device)
        torch.testing.assert_close(env.cherry.data.root_state_w.torch[0, :3], expected, rtol=0.0, atol=1e-7)
        assert torch.count_nonzero(env.cherry.data.root_state_w.torch[:, 7:]) == 0
        torch.testing.assert_close(wp.to_torch(env.particle_positions), deformed, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            wp.to_torch(NewtonManager.get_state_0().particle_qd), particle_velocity, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            wp.to_torch(env.gaussian_stream.state.mpm.particle_transform), material_frames, rtol=0.0, atol=0.0
        )
        assert (tuple(camera.pos), camera.pitch, camera.yaw, tuple(camera.pivot)) == camera_pose
        env.sim.render()
        visualizer.render_frame()
        visualizer.finish_frame()
        env.reset(seed=42)
        torch.testing.assert_close(env.cherry.data.root_state_w.torch[0, :3], expected, rtol=0.0, atol=1e-7)
        torch.testing.assert_close(wp.to_torch(env.particle_positions), rest, rtol=0.0, atol=0.0)
        env.step(action)
        assert float(env.cherry.data.root_state_w.torch[0, 2]) < 0.450
        torch.testing.assert_close(env.cherry.data.root_state_w.torch[0, :2], expected[:2], rtol=0.0, atol=1e-6)
        # Close with native GPU work pending; bindings must outlive completion
        # and release before renderer destruction.
        env.sim.render()
        visualizer.render_frame()
    finally:
        env.close()
