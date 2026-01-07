# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import torch
from collections.abc import Sequence

import warp as wp

logger = logging.getLogger(__name__)


def _get_default_device() -> str:
    """Get the default device based on CUDA availability.
    
    Returns:
        "cuda:0" if CUDA is available, otherwise "cpu"
    """
    return "cuda:0" if torch.cuda.is_available() else "cpu"

##
# Frontend conversions - Torch to Warp.
##


# TODO: Perf is atrocious. Need to improve.
# Option 1: Pre-allocate the complete data buffer and fill it with the value.
# Option 2: Create a torch pointer to the warp array and by pass these methods using torch indexing to update
# the warp array. This would save the memory allocation and the generation of the masks.
def make_complete_data_from_torch_single_index(
    value: torch.Tensor,
    N: int,
    ids: Sequence[int] | torch.Tensor | None = None,
    dtype: type = wp.float32,
    device: str | None = None,
) -> wp.array:
    """Converts any Torch frontend data into warp data with single index support.

    Args:
        value: The value to convert. Shape is (N,).
        N: The number of elements in the value.
        ids: The index ids.
        dtype: The dtype of the value.
        device: The device to use for the conversion. Defaults to "cuda:0" if available, else "cpu".

    Returns:
        A warp array.
    """
    if device is None:
        device = _get_default_device()
    
    if ids is None:
        # No ids are provided, so we are expecting complete data.
        value = wp.from_torch(value, dtype=dtype)
    else:
        # Create a complete data buffer from scratch
        complete = torch.zeros((N, *value.shape[1:]), dtype=torch.float32, device=device)
        complete[ids] = value
        value = wp.from_torch(complete, dtype=dtype)
    return value


def make_complete_data_from_torch_dual_index(
    value: torch.Tensor,
    N: int,
    M: int,
    first_ids: Sequence[int] | torch.Tensor | None = None,
    second_ids: Sequence[int] | torch.Tensor | None = None,
    dtype: type = wp.float32,
    device: str | None = None,
) -> wp.array:
    """Converts any Torch frontend data into warp data with dual index support.

    Args:
        value: The value to convert. Shape is (N, M) or (len(first_ids), len(second_ids)).
        N: The number of elements in the first dimension.
        M: The number of elements in the second dimension.
        first_ids: The first index ids.
        second_ids: The second index ids.
        dtype: The dtype of the value.
        device: The device to use for the conversion. Defaults to "cuda:0" if available, else "cpu".

    Returns:
        A tuple of warp data with its two masks.
    """
    if device is None:
        device = _get_default_device()
    
    if (first_ids is None) and (second_ids is None):
        # No ids are provided, so we are expecting complete data.
        value = wp.from_torch(value, dtype=dtype)
    else:
        # Create a complete data buffer from scratch
        complete = torch.zeros((N, M, *value.shape[2:]), dtype=torch.float32, device=device)
        # Fill the complete data buffer with the value.
        if first_ids is None:
            first_ids = slice(None)
        if second_ids is None:
            second_ids = slice(None)
        if first_ids != slice(None) and second_ids != slice(None):
            if isinstance(first_ids, list):
                first_ids = torch.tensor(first_ids, dtype=torch.int32, device=device)
            first_ids = first_ids[:, None]
        complete[first_ids, second_ids] = value
        value = wp.from_torch(complete, dtype=dtype)
    return value


def make_masks_from_torch_ids(
    N: int,
    first_ids: Sequence[int] | torch.Tensor | None = None,
    first_mask: wp.array | torch.Tensor | None = None,
    device: str | None = None,
) -> wp.array | None:
    """Converts any Torch frontend data into warp data with dual index support.

    Args:
        value: The value to convert. Shape is (N, M) or (len(first_ids), len(second_ids)).
        first_ids: The first index ids.
        second_ids: The second index ids.
        first_mask: The first index mask.
        second_mask: The second index mask.
        dtype: The dtype of the value.
        device: The device to use for the conversion. Defaults to "cuda:0" if available, else "cpu".

    Returns:
        A tuple of warp data with its two masks.
    """
    if device is None:
        device = _get_default_device()
    
    if (first_ids is not None) and (first_mask is None):
        # Create a mask from scratch
        first_mask = torch.zeros(N, dtype=torch.bool, device=device)
        first_mask[first_ids] = True
        first_mask = wp.from_torch(first_mask, dtype=wp.bool)
    elif isinstance(first_mask, torch.Tensor):
        first_mask = wp.from_torch(first_mask, dtype=wp.bool)
    return first_mask
