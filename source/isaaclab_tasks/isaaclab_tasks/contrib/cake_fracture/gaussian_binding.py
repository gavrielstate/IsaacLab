# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fracture-aware Gaussian attachments with optional spatial skinning derivatives."""

import numpy as np
import warp as wp

from isaaclab_contrib.mpm_gaussians import Binding

from ..cake_smash.gaussian_stream import CakeGaussianStream


@wp.kernel
def paste_display_frames(
    native: wp.array[wp.mat33],
    solid: wp.array[wp.mat33],
    paste: wp.array[int],
    volume: wp.array[float],
    rest_volume: wp.array[float],
    strain_limit: float,
    output: wp.array[wp.mat33],
):
    """Retain bounded flow distortion and physical volume for yielding paste."""
    i = wp.tid()
    f = solid[i]
    if paste[i] >= 0:
        u, s, v = wp.svd3(native[i])
        logs = wp.vec3(wp.log(wp.max(s[0], 1.0e-6)), wp.log(wp.max(s[1], 1.0e-6)), wp.log(wp.max(s[2], 1.0e-6)))
        mean = (logs[0] + logs[1] + logs[2]) / 3.0
        deviator = logs - wp.vec3(mean)
        scale = wp.min(1.0, strain_limit / wp.max(wp.length(deviator), 1.0e-6))
        logs = wp.vec3(wp.log(volume[i] / rest_volume[i]) / 3.0) + scale * deviator
        f = u @ wp.diag(wp.vec3(wp.exp(logs[0]), wp.exp(logs[1]), wp.exp(logs[2]))) @ wp.transpose(v)
    output[i] = f


@wp.kernel
def transport_fragments(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    offsets: wp.array2d[wp.vec3],
    base: wp.array[wp.mat33],
    bond_ids: wp.array2d[int],
    damage: wp.array[float],
    labels: wp.array[int],
    paste_fields: wp.array[int],
    skinning_jacobian: int,
    paste_flow: int,
    max_stretch: float,
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
        same_piece = intact and labels[j] == labels[anchor]
        if (skinning_jacobian != 0 or paste_flow != 0) and paste_fields[anchor] >= 0:
            same_piece = paste_fields[j] == paste_fields[anchor]
        if same_piece and current_distance <= 1.5 * rest_distance + 0.001:
            weight = weights[i, k]
            x += weight * (positions[j] + frames[j] @ offsets[i, k])
            f += weight * frames[j]
            total += weight
    xyz[i] = x / total
    f = f / total
    if skinning_jacobian != 0:
        # Different particle translations change the spatial skinning derivative,
        # even if their individual deformation frames remain identity. Include the
        # derivative of the renormalized inverse-distance weights. The current
        # support mask is held fixed; its hard boundary is not differentiable.
        correction = wp.mat33(0.0)
        for k in range(4):
            j = ids[i, k]
            rest_distance = wp.length(offsets[i, 0] - offsets[i, k])
            current_distance = wp.length(positions[j] - positions[anchor])
            intact = True
            bond = bond_ids[i, k]
            if bond >= 0:
                intact = damage[bond] < 0.9999
            same_piece = intact and labels[j] == labels[anchor]
            if paste_fields[anchor] >= 0:
                same_piece = paste_fields[j] == paste_fields[anchor]
            if same_piece and current_distance <= 1.5 * rest_distance + 0.001:
                d2 = wp.dot(offsets[i, k], offsets[i, k])
                if d2 > 0.0002 * 0.0002:
                    grad_log_weight = -2.0 * offsets[i, k] / d2
                    mapped = positions[j] + frames[j] @ offsets[i, k]
                    correction += (weights[i, k] / total) * wp.outer(mapped - xyz[i], grad_log_weight)
        u, stretch, v = wp.svd3(f + correction)
        for axis in range(3):
            stretch[axis] = wp.clamp(stretch[axis], 0.08, max_stretch)
        f = u @ wp.diag(stretch) @ wp.transpose(v)
    u, sigma, v = wp.svd3(f @ base[i])
    if wp.determinant(u) < 0.0:
        for row in range(3):
            u[row, 2] = -u[row, 2]
    scales[i] = wp.vec3(
        wp.max(wp.abs(sigma[0]), 1.0e-9), wp.max(wp.abs(sigma[1]), 1.0e-9), wp.max(wp.abs(sigma[2]), 1.0e-9)
    )
    rotations[i] = wp.quat_from_matrix(u)


class FragmentBinding(Binding):
    def __init__(self, asset, rest, regions, solver, skinning_jacobian=False, max_stretch=3.0, paste_flow=False):
        super().__init__(asset, rest, regions)
        self.skinning_jacobian = int(skinning_jacobian)
        self.max_stretch = max_stretch
        self.paste_flow = int(paste_flow)
        self.paste_fields = solver.paste_fields
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
            inputs=[
                *self.gpu[:4],
                self.bond_ids,
                self.damage,
                self.labels,
                self.paste_fields,
                self.skinning_jacobian,
                self.paste_flow,
                self.max_stretch,
                positions,
                frames,
                *self.gpu[4:],
            ],
            device=positions.device,
        )
        return tuple(a.numpy() for a in self.gpu[4:]) if host else tuple(self.gpu[4:])


class FractureGaussianStream(CakeGaussianStream):
    @property
    def deformation_frames(self):
        if self.env.cfg.gaussian_paste_flow:
            wp.launch(
                paste_display_frames,
                self.solver.model.particle_count,
                inputs=[
                    self.solver.frames,
                    self.solver.skin_frames,
                    self.solver.paste_fields,
                    self.solver.volume,
                    self.solver.initial_volume,
                    self.env.cfg.gaussian_paste_strain_limit,
                    self.paste_frames,
                ],
                device=self.solver.model.device,
            )
            return self.paste_frames
        return self.solver.skin_frames

    def create_binding(self, rest, physical_regions):
        if self.env.cfg.gaussian_paste_flow:
            self.paste_frames = wp.empty_like(self.solver.skin_frames)
        return FragmentBinding(
            self.asset,
            rest,
            physical_regions,
            self.solver,
            skinning_jacobian=self.env.cfg.gaussian_skinning_jacobian,
            max_stretch=self.env.cfg.gaussian_max_stretch,
            paste_flow=self.env.cfg.gaussian_paste_flow,
        )

    def advance_frames(self):
        # Explicit solver integrates total display deformation at its substep rate.
        pass
