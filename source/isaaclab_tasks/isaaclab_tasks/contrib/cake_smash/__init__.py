# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native Newton MPM cake crushing demonstration."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Cake-Smash-Direct",
    entry_point=f"{__name__}.cake_smash_env:CakeSmashEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.cake_smash_env_cfg:CakeSmashEnvCfg"},
)
