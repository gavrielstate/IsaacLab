# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-base trained G1 throwing a cherry into native MPM cake."""
import gymnasium as gym
from . import agents

gym.register(
    id="IsaacContrib-G1-Cake-Smash-Direct",
    entry_point=f"{__name__}.g1_cake_env:G1CakeEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.g1_cake_env_cfg:G1CakeEnvCfg",
            "default_agent": "rsl_rl",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg"},
)
