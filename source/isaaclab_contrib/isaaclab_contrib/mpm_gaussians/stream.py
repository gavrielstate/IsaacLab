# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Persistent native OVRTX Gaussian attribute publication, independent of physics."""

from collections.abc import Mapping, Sequence

import numpy as np


class GaussianArrayStream:
    """Own persistent OVRTX bindings for an authored Gaussian field.

    The renderer must already contain ``prim_paths``. Each publication provides
    one array per prim, in that order. Arrays may implement DLPack, including
    Warp CUDA arrays; this class does not copy them through NumPy. Callers own
    synchronization and must retain inputs until their render completes.

    The bindings establish the animated geometry dataflow used by Nicolas'
    berry task. By-name writes alone do not provide the same renderer contract.
    """

    def __init__(self, renderer, prim_paths: Sequence[str], attributes: Mapping[str, int]):
        from ovrtx import BindingFlag  # noqa: PLC0415

        self._bindings = {}
        if not prim_paths:
            raise ValueError("At least one Gaussian prim path is required.")
        self.prim_paths = tuple(prim_paths)
        try:
            for name, lanes in attributes.items():
                self._bindings[name] = renderer.bind_array_attribute(
                    list(self.prim_paths), name, dtype="float32", shape=(lanes,), flags=BindingFlag.OPTIMIZE
                )
        except Exception:
            self.close()
            raise

    def write(self, values: Mapping[str, Sequence[object]]) -> None:
        """Publish attribute arrays with one array per authored prim."""
        from ovrtx import DataAccess  # noqa: PLC0415

        if values.keys() != self._bindings.keys():
            raise ValueError("Publication must contain exactly the bound attributes.")
        for arrays in values.values():
            if len(arrays) != len(self.prim_paths):
                raise ValueError("Publication requires one array per Gaussian prim.")
        for name, arrays in values.items():
            # GPU buffers require referenced input access. The write call still
            # waits for publication; callers retain arrays through rendering.
            self._bindings[name].write(list(arrays), data_access=DataAccess.ASYNC)

    def verify(self, renderer, expected: Mapping[str, Sequence[np.ndarray]]) -> None:
        """Check published arrays; callers must also validate visible images."""
        for name, arrays in expected.items():
            restored = renderer.read_array_attribute(attribute_name=name, prim_paths=list(self.prim_paths))
            for path, value in zip(self.prim_paths, arrays, strict=True):
                actual = np.from_dlpack(restored[path]).reshape(value.shape)
                if not np.isfinite(value).all() or not np.isfinite(actual).all():
                    raise AssertionError(f"Non-finite Gaussian attribute {name} on {path}")
                np.testing.assert_allclose(actual, value, atol=1e-7, equal_nan=False)

    def close(self) -> None:
        """Unbind before destroying the renderer. Repeated closes are safe."""
        bindings, self._bindings = self._bindings, {}
        for binding in bindings.values():
            binding.unbind()
