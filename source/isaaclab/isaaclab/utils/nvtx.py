# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NVTX utilities for performance profiling with Nsight Systems."""

from __future__ import annotations

from contextlib import contextmanager

try:
    import torch.cuda.nvtx as nvtx

    _NVTX_AVAILABLE = True
except ImportError:
    _NVTX_AVAILABLE = False


class NVTXMarker:
    """Helper class for NVTX instrumentation in Isaac Lab.

    This class provides utilities for adding NVTX markers to code for profiling
    with Nsight Systems. Markers appear as ranges in the Nsight Systems timeline.

    Examples:
        Basic usage::

            from isaaclab.utils.nvtx import NVTXMarker

            with NVTXMarker.range("my_operation"):
                # your code here
                pass

        Instantaneous mark::

            NVTXMarker.mark("checkpoint_saved")
    """

    @staticmethod
    @contextmanager
    def range(name: str):
        """Create an NVTX range (timed region).

        Args:
            name: Name of the range to display in Nsight Systems.

        Yields:
            None: Context manager yields nothing.

        Examples:
            Basic usage::

                with NVTXMarker.range("my_operation"):
                    # code to profile
                    pass
        """
        if not _NVTX_AVAILABLE:
            yield
            return

        nvtx.range_push(name)

        try:
            yield
        finally:
            nvtx.range_pop()

    @staticmethod
    def mark(name: str):
        """Create an NVTX mark (instantaneous event).

        Marks are useful for marking specific events in time (e.g., checkpoints,
        state changes) rather than timing regions.

        Args:
            name: Name of the mark to display in Nsight Systems.

        Examples:
            Mark a checkpoint::

                NVTXMarker.mark("checkpoint_saved")

            Mark a state change::

                NVTXMarker.mark("environment_reset")
        """
        if not _NVTX_AVAILABLE:
            return
        nvtx.mark(name)

    @staticmethod
    def is_available() -> bool:
        """Check if NVTX is available.

        Returns:
            True if NVTX is available, False otherwise.
        """
        return _NVTX_AVAILABLE
