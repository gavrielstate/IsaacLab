# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Conservative sparse candidates for the explicit multi-field MPM contact solve."""

import warp as wp

from .explicit_mpm import node_position, static_boundaries


@wp.struct
class _FieldContactCache:
    rows: wp.array[int]
    offsets: wp.array[int]
    energy: wp.array[float]
    degree: wp.array[int]
    neighbors: wp.array2d[int]
    geometry: wp.array2d[wp.vec4]
    mixing: wp.array2d[wp.vec2]


@wp.kernel
def _prepare_field_contacts(
    active: wp.array[int],
    count: wp.array[int],
    origin: wp.vec3,
    resolution: wp.vec3i,
    spacing: float,
    field_count: wp.array[int],
    field_ids: wp.array2d[int],
    dt: float,
    gravity: wp.vec3,
    damping: float,
    field_friction: float,
    contact_broadphase: bool,
    ground: float,
    ground_friction: float,
    vessel_center: wp.array[wp.vec3],
    vessel_size: wp.array[wp.vec3],
    vessel_friction: float,
    corner: float,
    wall: float,
    node_mass: wp.array[float],
    node_momentum: wp.array[wp.vec3],
    mass_gradient: wp.array[wp.vec3],
    mass_moment: wp.array[wp.vec3],
    lower: wp.array2d[float],
    upper: wp.array2d[float],
    node_velocity: wp.array[wp.vec3],
    cache: _FieldContactCache,
):
    tid = wp.tid()
    cache.rows[tid] = 0
    if tid >= wp.min(count[0], active.shape[0]):
        return
    n = active[tid]
    fields = field_count[n]
    cache.rows[tid] = fields
    # Ascending IDs preserve the sequential contact projection order. Shell
    # insertion avoids a full quadratic swap loop on fragmented nodes.
    stride = fields // 2
    while stride > 0:
        for a in range(stride, fields):
            saved = field_ids[n, a]
            b = a
            while b >= stride:
                if field_ids[n, b - stride] <= saved:
                    break
                field_ids[n, b] = field_ids[n, b - stride]
                b -= stride
            field_ids[n, b] = saved
        stride //= 2
    for f in range(fields):
        i = n * field_ids.shape[1] + field_ids[n, f]
        node_velocity[i] = wp.vec3(0.0)
        if node_mass[i] > 1.0e-14:
            node_velocity[i] = (node_momentum[i] / node_mass[i] + dt * gravity) * wp.exp(-damping * dt)
            if mass_gradient.shape[0] > 1:
                # These scratch accumulators are cleared before the next transfer.
                # Cache normalized geometry once instead of dividing it for every pair.
                mass_gradient[i] = mass_gradient[i] / node_mass[i]
                mass_moment[i] = mass_moment[i] / node_mass[i]
    mass = float(0.0)
    momentum = wp.vec3(0.0)
    for f in range(fields):
        i = n * field_ids.shape[1] + field_ids[n, f]
        mass += node_mass[i]
        momentum += node_mass[i] * node_velocity[i]
    mean = momentum / wp.max(mass, 1.0e-20)
    relative_energy = float(0.0)
    for f in range(fields):
        i = n * field_ids.shape[1] + field_ids[n, f]
        relative_energy += 0.5 * node_mass[i] * wp.length_sq(node_velocity[i] - mean)
    cache.energy[tid] = relative_energy


@wp.kernel
def _build_field_contacts(
    active: wp.array[int],
    count: wp.array[int],
    origin: wp.vec3,
    resolution: wp.vec3i,
    spacing: float,
    field_count: wp.array[int],
    field_ids: wp.array2d[int],
    dt: float,
    gravity: wp.vec3,
    damping: float,
    field_friction: float,
    contact_broadphase: bool,
    ground: float,
    ground_friction: float,
    vessel_center: wp.array[wp.vec3],
    vessel_size: wp.array[wp.vec3],
    vessel_friction: float,
    corner: float,
    wall: float,
    node_mass: wp.array[float],
    node_momentum: wp.array[wp.vec3],
    mass_gradient: wp.array[wp.vec3],
    mass_moment: wp.array[wp.vec3],
    lower: wp.array2d[float],
    upper: wp.array2d[float],
    node_velocity: wp.array[wp.vec3],
    cache: _FieldContactCache,
):
    row = wp.tid()
    size = wp.min(count[0], active.shape[0])
    if size == 0:
        return
    if row >= cache.offsets[size - 1] + cache.rows[size - 1]:
        return
    lo = int(0)
    hi = size
    while lo < hi:
        mid = (lo + hi) // 2
        if row >= cache.offsets[mid] + cache.rows[mid]:
            lo = mid + 1
        else:
            hi = mid
    slot = lo
    n = active[slot]
    fields = field_count[n]
    f = row - cache.offsets[slot]
    cache.degree[row] = 0
    i = n * field_ids.shape[1] + field_ids[n, f]
    if node_mass[i] <= 1.0e-14 or fields < 2:
        return
    for g in range(f + 1, fields):
        k = n * field_ids.shape[1] + field_ids[n, g]
        if node_mass[k] <= 1.0e-14:
            continue
        normal = mass_gradient[i] - mass_gradient[k]
        delta = mass_moment[k] - mass_moment[i]
        if wp.length_sq(normal) < 1.0e-10:
            normal = delta
        if wp.dot(normal, delta) < 0.0:
            normal = -normal
        if wp.length_sq(normal) == 0.0:
            continue
        normal = wp.normalize(normal)
        gap = float(0.0)
        for axis in range(3):
            near_i = wp.where(normal[axis] >= 0.0, upper[i, axis], lower[i, axis])
            near_k = wp.where(normal[axis] >= 0.0, lower[k, axis], upper[k, axis])
            gap += normal[axis] * (near_k - near_i)
        # Relative kinetic energy cannot grow during dissipative pair projection.
        # Round conservatively upward before removing any pair from all sweeps.
        speed_bound = (
            1.001 * wp.sqrt(wp.max(0.0, 2.0 * cache.energy[slot] * (1.0 / node_mass[i] + 1.0 / node_mass[k]))) + 1.0e-6
        )
        if gap <= dt * speed_bound + 0.00005001:
            c = cache.degree[row]
            cache.neighbors[row, c] = g
            cache.geometry[row, c] = wp.vec4(normal[0], normal[1], normal[2], gap)
            reduced = node_mass[i] * node_mass[k] / (node_mass[i] + node_mass[k])
            cache.mixing[row, c] = wp.vec2(reduced / node_mass[i], reduced / node_mass[k])
            cache.degree[row] = c + 1


@wp.kernel
def _resolve_field_contacts(
    active: wp.array[int],
    count: wp.array[int],
    origin: wp.vec3,
    resolution: wp.vec3i,
    spacing: float,
    field_count: wp.array[int],
    field_ids: wp.array2d[int],
    dt: float,
    gravity: wp.vec3,
    damping: float,
    field_friction: float,
    contact_broadphase: bool,
    ground: float,
    ground_friction: float,
    vessel_center: wp.array[wp.vec3],
    vessel_size: wp.array[wp.vec3],
    vessel_friction: float,
    corner: float,
    wall: float,
    node_mass: wp.array[float],
    node_momentum: wp.array[wp.vec3],
    mass_gradient: wp.array[wp.vec3],
    mass_moment: wp.array[wp.vec3],
    lower: wp.array2d[float],
    upper: wp.array2d[float],
    node_velocity: wp.array[wp.vec3],
    cache: _FieldContactCache,
):
    tid = wp.tid()
    if tid >= wp.min(count[0], active.shape[0]):
        return
    n = active[tid]
    fields = field_count[n]
    x = node_position(n, origin, resolution, spacing)
    # One thread owns all fields of a node. Each pairwise projection removes approaching relative motion and conserves
    # momentum; separating fields move freely.
    for sweep in range(3):
        for f in range(fields):
            i = n * field_ids.shape[1] + field_ids[n, f]
            mass_i = node_mass[i]
            if mass_i <= 1.0e-14 or fields < 2:
                continue
            velocity_i = node_velocity[i]
            row = cache.offsets[tid] + f
            for candidate in range(cache.degree[row]):
                g = cache.neighbors[row, candidate]
                k = n * field_ids.shape[1] + field_ids[n, g]
                mass_k = node_mass[k]
                if mass_k <= 1.0e-14:
                    continue
                relative = node_velocity[k] - velocity_i
                if contact_broadphase:
                    separated = False
                    for axis in range(3):
                        gap_axis = wp.max(lower[k, axis] - upper[i, axis], lower[i, axis] - upper[k, axis])
                        if gap_axis > wp.abs(relative[axis]) * dt + 0.00005:
                            separated = True
                    if separated:
                        continue
                data = cache.geometry[row, candidate]
                normal = wp.vec3(data[0], data[1], data[2])
                gap = data[3]
                closing = wp.dot(relative, normal)
                if closing < 0.0 and gap <= -closing * dt + 0.00005:
                    tangent = relative - closing * normal
                    tangent *= wp.min(1.0, field_friction * (-closing) / wp.max(wp.length(tangent), 1.0e-12))
                    delta = closing * normal + tangent
                    weights = cache.mixing[row, candidate]
                    velocity_i += weights[0] * delta
                    node_velocity[k] -= weights[1] * delta
            node_velocity[i] = velocity_i
    for f in range(fields):
        i = n * field_ids.shape[1] + field_ids[n, f]
        # Thin vessel bases can lie between nodes, so nodes within half a cell are supported too; particle projection
        # is exact.
        node_velocity[i] = static_boundaries(
            x,
            node_velocity[i],
            ground,
            ground_friction,
            vessel_center,
            vessel_size,
            vessel_friction,
            corner,
            wall,
            0.5 * spacing,
        )


class SparseFieldContacts:
    """Own GPU candidates for conservative pruning without changing projection order.

    Each particle touches at most 27 nodes, so there are at most 27 times the
    particle count field rows. Each row has at most one candidate per field.
    Cache capacities follow those bounds; there is no lossy candidate truncation.
    """

    def __init__(self, particle_count, field_capacity, active_capacity, device, block_dim=1):
        columns = min(particle_count, field_capacity)
        rows = min(27 * particle_count, active_capacity * columns)
        self.block_dim = block_dim
        self.cache = _FieldContactCache()
        self.cache.rows = wp.zeros(active_capacity, dtype=int, device=device)
        self.cache.offsets = wp.zeros_like(self.cache.rows)
        self.cache.energy = wp.zeros(active_capacity, dtype=float, device=device)
        self.cache.degree = wp.empty(rows, dtype=int, device=device)
        self.cache.neighbors = wp.empty((rows, columns), dtype=int, device=device)
        self.cache.geometry = wp.empty((rows, columns), dtype=wp.vec4, device=device)
        self.cache.mixing = wp.empty((rows, columns), dtype=wp.vec2, device=device)
        # Warm scan scratch allocation before simulation CUDA graph capture.
        wp.utils.array_scan(self.cache.rows, self.cache.offsets, inclusive=False)

    def update(self, arguments):
        """Consume the same ordered GPU arguments as the sequential grid kernel."""
        inputs = [*arguments, self.cache]
        wp.launch(
            _prepare_field_contacts,
            len(self.cache.rows),
            inputs=inputs,
            block_dim=self.block_dim,
            device=self.cache.rows.device,
        )
        wp.utils.array_scan(self.cache.rows, self.cache.offsets, inclusive=False)
        wp.launch(_build_field_contacts, len(self.cache.degree), inputs=inputs, device=self.cache.rows.device)
        wp.launch(
            _resolve_field_contacts,
            len(self.cache.rows),
            inputs=inputs,
            block_dim=self.block_dim,
            device=self.cache.rows.device,
        )
