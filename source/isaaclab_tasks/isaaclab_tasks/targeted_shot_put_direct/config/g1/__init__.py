# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Targeted-Shot-Put-Direct",
    entry_point=f"{__name__}.targeted_shot_put_env:TargetedShotPutEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.targeted_shot_put_env_cfg:TargetedShotPutEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Standing-Shot-Put-Direct",
    entry_point=f"{__name__}.standing_shot_put:StandingShotPutEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standing_shot_put:StandingShotPutEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
