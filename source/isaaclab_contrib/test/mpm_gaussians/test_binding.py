# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent affine/covariance and material-region contracts for visual skinning."""

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from isaaclab_contrib.mpm_gaussians import Binding


def test_affine_covariance_and_material_regions():
    """Transport an ellipsoid under shear without pulling from another layer.

    A nearer particle in the other region moves far away. The expected result
    comes from the analytic affine transform and covariance, independently of
    the binding's neighborhood weights and singular-vector ordering. This
    catches region leakage, XYZW/WXYZ confusion and incorrect covariance
    rotation while also checking the one-way physics boundary.
    """
    wp.init()
    rest = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.1, 0.1, 0.1]], np.float32)
    source = np.array([[0.1, 0.1, 0.1]], np.float32)
    scale = np.array([[0.03, 0.01, 0.02]], np.float32)
    orientation = Rotation.from_euler("xyz", [0.2, -0.3, 0.5])
    affine = np.array([[1.2, 0.3, 0], [0, 0.7, 0.2], [0.1, 0, 1.1]], np.float32)
    translation = np.array([0.3, -0.1, 0.2], np.float32)
    current = rest @ affine.T + translation
    current[-1] = [10, 20, 30]
    with wp.ScopedDevice("cpu"):
        binding = Binding(
            {"xyz": source, "scales": scale, "rotations": orientation.as_quat()[None], "regions": np.array([0])},
            rest,
            np.array([0, 0, 0, 0, 1], np.int32),
        )
        positions = wp.array(current, dtype=wp.vec3)
        frames = wp.array(np.repeat(affine[None], len(rest), axis=0), dtype=wp.mat33)
        xyz, scales, rotations = binding.deform_gpu(positions, frames)
        np.testing.assert_array_equal(positions.numpy(), current)
    np.testing.assert_allclose(xyz, source @ affine.T + translation, atol=2e-6)
    before = orientation.as_matrix() @ np.diag(scale[0] ** 2) @ orientation.as_matrix().T
    after_rotation = Rotation.from_quat(rotations[0]).as_matrix()
    after = after_rotation @ np.diag(scales[0] ** 2) @ after_rotation.T
    np.testing.assert_allclose(after, affine @ before @ affine.T, rtol=2e-5, atol=1e-9)
