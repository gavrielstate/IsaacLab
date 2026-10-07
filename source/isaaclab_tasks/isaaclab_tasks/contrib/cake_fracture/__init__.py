# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental cake fracture using standard Isaac Lab Newton RTX rendering."""
import gymnasium as gym

gym.register(
    id='IsaacContrib-Cake-Fracture-Direct',
    entry_point=f'{__name__}.cake_fracture_env:CakeFractureEnv',
    disable_env_checker=True,
    kwargs={'env_cfg_entry_point': f'{__name__}.cake_fracture_env:CakeFractureEnvCfg'},
)
