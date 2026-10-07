# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fragment-aware Gaussian attachments; no interpolation across broken pieces."""

import numpy as np
import warp as wp

from isaaclab_contrib.mpm_gaussians import Binding

from ..cake_smash.gaussian_stream import CakeGaussianStream


@wp.kernel
def transport_fragments(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    offsets: wp.array2d[wp.vec3],
    base: wp.array[wp.mat33],
    bond_ids: wp.array2d[int],
    damage: wp.array[float],
    labels: wp.array[int],
    positions: wp.array[wp.vec3],
    frames: wp.array[wp.mat33],
    xyz: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    rotations: wp.array[wp.quat],
):
    i = wp.tid()
    anchor = ids[i, 0]
    x = wp.vec3(0.0)
    f = wp.mat33(0.0)
    total = float(0.0)
    for k in range(4):
        j = ids[i, k]
        rest_distance = wp.length(offsets[i, 0] - offsets[i, k])
        current_distance = wp.length(positions[j] - positions[anchor])
        intact = True
        bond = bond_ids[i, k]
        if bond >= 0:
            intact = damage[bond] < 0.9999
        if intact and labels[j] == labels[anchor] and current_distance <= 1.5 * rest_distance + 0.001:
            weight = weights[i, k]
            x += weight * (positions[j] + frames[j] @ offsets[i, k])
            f += weight * frames[j]
            total += weight
    xyz[i] = x / total
    u, sigma, v = wp.svd3((f / total) @ base[i])
    if wp.determinant(u) < 0.0:
        for row in range(3):
            u[row, 2] = -u[row, 2]
    scales[i] = wp.vec3(
        wp.max(wp.abs(sigma[0]), 1.0e-9), wp.max(wp.abs(sigma[1]), 1.0e-9), wp.max(wp.abs(sigma[2]), 1.0e-9)
    )
    rotations[i] = wp.quat_from_matrix(u)


class FragmentBinding(Binding):
    def __init__(self, asset, rest, regions, solver):
        super().__init__(asset, rest, regions)
        self.labels = solver.tissue_field
        self.damage = solver.bond_damage
        lookup = {tuple(pair): i for i, pair in enumerate(solver.pairs_host)}
        links = np.array(
            [[lookup.get(tuple(sorted((int(row[0]), int(j)))), -1) for j in row] for row in self.ids], np.int32
        )
        self.bond_ids = wp.array(links, dtype=int, device=solver.model.device)

    def deform_gpu(self, positions, frames, host=True):
        if self.gpu is None:
            super().deform_gpu(positions, frames, host=False)
        wp.launch(
            transport_fragments,
            dim=len(self.xyz),
            inputs=[*self.gpu[:4], self.bond_ids, self.damage, self.labels, positions, frames, *self.gpu[4:]],
            device=positions.device,
        )
        return tuple(a.numpy() for a in self.gpu[4:]) if host else tuple(self.gpu[4:])


class FractureGaussianStream(CakeGaussianStream):
    @property
    def deformation_frames(self):
        return self.solver.skin_frames

    def create_binding(self, rest, physical_regions):
        return FragmentBinding(self.asset, rest, physical_regions, self.solver)

    def advance_frames(self):
        # Explicit solver integrates total display deformation at its substep rate.
        pass
