# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native contact and full-history reset regression for the external reference cake.

This fixture is intentionally external: generated cake/Gaussian binary assets
are not redistributed with the task source. Select the 936-particle reference
asset with ISAACLAB_CAKE_PHYSICS_USD_PATH.
"""

import os

import gymnasium as gym
import pytest
import torch
import warp as wp

from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_tasks.contrib.cake_smash.cake_smash_env_cfg import CakeSmashEnvCfg


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_native_cake_contact_and_reset_replays_initial_response(device):
    """An ordinary falling rigid body deforms cake and receives reciprocal contact.

    Repeating after crushing must restore material history as well as positions;
    resetting only particle coordinates poisons the next episode's response.
    """
    asset = os.environ.get("ISAACLAB_CAKE_PHYSICS_USD_PATH")
    if not asset:
        pytest.skip("External 936-particle reference cake asset was not selected.")
    cfg = CakeSmashEnvCfg(physics_asset_path=asset)
    cfg.sim.device = device
    env = gym.make("IsaacContrib-Cake-Smash-Direct", cfg=cfg).unwrapped
    try:
        assert sum(layer.particles_per_object for layer in env.layers) == 936
        action = torch.zeros((1, 1), device=device)
        env.reset(seed=42)
        initial = wp.to_torch(env.particle_positions).clone()
        initial_rigid = env.cherry.data.root_state_w.torch.clone()
        for _ in range(30):
            env.step(action)
        response = wp.to_torch(env.particle_positions).clone()
        rigid_response = env.cherry.data.root_state_w.torch.clone()
        assert torch.isfinite(response).all()
        assert float((response - initial).norm(dim=-1).max()) > 0.03
        # The cherry has gained horizontal velocity from asymmetric contact;
        # free fall from the specified zero-velocity pose cannot generate it.
        assert float(rigid_response[0, 7:9].norm()) > 0.05
        assert float(rigid_response[0, 9]) > -3.0
        env.reset(seed=42)
        torch.testing.assert_close(wp.to_torch(env.particle_positions), initial, rtol=0.0, atol=0.0)
        torch.testing.assert_close(env.cherry.data.root_state_w.torch, initial_rigid, rtol=0.0, atol=0.0)
        for _ in range(30):
            env.step(action)
        # Newton does not promise bitwise long-run implicit MPM determinism.
        # The initial one-second response remains close before late debris
        # contacts amplify floating-point differences.
        torch.testing.assert_close(wp.to_torch(env.particle_positions), response, rtol=0.0, atol=2.0e-5)
    finally:
        env.close()
