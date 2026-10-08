# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fracture graph, field allocation, and reacting rigid-sphere contact.

This experimental solver separates connected components of an irreversible
rest-neighbor bond graph. Field assignment can run on the GPU at each coupling
boundary. No crack plane or time-dependent break schedule is prescribed.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp
from isaaclab_newton.physics import MPMSolverCfg, NewtonMPMManager
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from isaaclab.utils import configclass

from .explicit_mpm import SolverExplicitMultiFieldMPM, deformation_increment, node_index


def rest_bond_pairs(rest: np.ndarray, neighbor_count: int, graph: str) -> np.ndarray:
    """Build undirected local bonds; directional sampling avoids layer bias.

    Directional mode selects the two closest candidates in each signed dominant
    Cartesian direction. The fracture law and total incident area are unchanged.
    Axes belong to the authored rest frame, so subsequent rigid motion cannot
    change the graph. This is a sampling experiment, not a calibrated quadrature.
    """
    if graph not in ("nearest", "directional"):
        raise ValueError(f"Unknown rest graph: {graph}")
    candidates = neighbor_count if graph == "nearest" else max(32, neighbor_count)
    candidates = min(candidates, len(rest) - 1)
    _, neighbors = cKDTree(rest).query(rest, k=candidates + 1)
    edges = []
    for i, row in enumerate(neighbors[:, 1:]):
        if graph == "directional":
            delta = rest[row] - rest[i]
            axes = np.argmax(np.abs(delta), axis=1)
            sectors = 2 * axes + (delta[np.arange(len(row)), axes] > 0)
            row = np.concatenate([row[sectors == sector][:2] for sector in range(6)])
        edges.extend((i, int(j)) for j in row)
    return np.unique(np.sort(np.asarray(edges, dtype=np.int32), axis=1), axis=0)


@wp.kernel
def initialize_local_components(
    positions: wp.array[wp.vec3],
    origin: wp.vec3,
    spacing: float,
    resolution: wp.vec3i,
    fields: int,
    parent: wp.array[int],
):
    tid = wp.tid()
    p, slot = tid // 27, tid % 27
    q = (positions[p] - origin) / spacing
    base = wp.vec3i(int(wp.floor(q[0] - 0.5)), int(wp.floor(q[1] - 0.5)), int(wp.floor(q[2] - 0.5)))
    node = base + wp.vec3i(slot // 9, (slot % 9) // 3, slot % 3)
    if node[0] >= 0 and node[1] >= 0 and node[2] >= 0:
        if node[0] < resolution[0] and node[1] < resolution[1] and node[2] < resolution[2]:
            parent[node_index(node, resolution) * fields + p] = p


@wp.func
def local_component_root(parent: wp.array[int], start: int, particle: int):
    root = particle
    while parent[start + root] != root:
        root = parent[start + root]
    return root


@wp.kernel
def join_local_components(
    positions: wp.array[wp.vec3],
    pairs: wp.array[wp.vec2i],
    damage: wp.array[float],
    threshold: float,
    origin: wp.vec3,
    spacing: float,
    resolution: wp.vec3i,
    fields: int,
    parent: wp.array[int],
    cracked: wp.array[int],
):
    tid = wp.tid()
    b, slot = tid // 27, tid % 27
    i, j = pairs[b][0], pairs[b][1]
    qi, qj = (positions[i] - origin) / spacing, (positions[j] - origin) / spacing
    base_i = wp.vec3i(int(wp.floor(qi[0] - 0.5)), int(wp.floor(qi[1] - 0.5)), int(wp.floor(qi[2] - 0.5)))
    base_j = wp.vec3i(int(wp.floor(qj[0] - 0.5)), int(wp.floor(qj[1] - 0.5)), int(wp.floor(qj[2] - 0.5)))
    node = base_i + wp.vec3i(slot // 9, (slot % 9) // 3, slot % 3)
    if node[0] < 0 or node[1] < 0 or node[2] < 0:
        return
    if node[0] >= resolution[0] or node[1] >= resolution[1] or node[2] >= resolution[2]:
        return
    delta = node - base_j
    if delta[0] < 0 or delta[1] < 0 or delta[2] < 0 or delta[0] > 2 or delta[1] > 2 or delta[2] > 2:
        return
    node_id = node_index(node, resolution)
    if damage[b] >= threshold:
        wp.atomic_max(cracked, node_id, 1)
        return
    start = node_id * fields
    # Connectivity is restricted to particles interpolating to this node.
    # An intact path outside its stencil cannot weld a local crack shut.
    while True:
        a, c = local_component_root(parent, start, i), local_component_root(parent, start, j)
        if a == c:
            break
        lower, upper = wp.min(a, c), wp.max(a, c)
        if wp.atomic_cas(parent, start + upper, upper, lower) == upper:
            break


@wp.kernel
def initialize_components(parent: wp.array[int], sizes: wp.array[int]):
    i = wp.tid()
    parent[i] = i
    sizes[i] = 0


@wp.func
def component_root(parent: wp.array[int], i: int):
    root = i
    while parent[root] != root:
        root = parent[root]
    return root


@wp.kernel
def join_components(pairs: wp.array[wp.vec2i], damage: wp.array[float], threshold: float, parent: wp.array[int]):
    b = wp.tid()
    if damage[b] < threshold:
        i, j = pairs[b][0], pairs[b][1]
        while True:
            a, c = component_root(parent, i), component_root(parent, j)
            if a == c:
                break
            lower, upper = wp.min(a, c), wp.max(a, c)
            # Only replace a root. Descending parent indices prevent cycles,
            # and retrying a lost race avoids dropping a surviving bond.
            previous = wp.atomic_cas(parent, upper, upper, lower)
            if previous == upper:
                break


@wp.kernel
def label_components(parent: wp.array[int], fields: wp.array[int], sizes: wp.array[int]):
    i = wp.tid()
    root = component_root(parent, i)
    fields[i] = root
    wp.atomic_add(sizes, root, 1)


@wp.kernel
def component_sizes(fields: wp.array[int], sizes: wp.array[int], particle_sizes: wp.array[int]):
    i = wp.tid()
    particle_sizes[i] = sizes[fields[i]]


@wp.kernel
def bonds_step(
    pairs: wp.array[wp.vec2i],
    rest: wp.array[wp.vec3],
    area: wp.array[float],
    strength: float,
    peak: float,
    final: float,
    dt: float,
    grid: float,
    crush_start: float,
    crush_final: float,
    young: wp.array[float],
    compaction: wp.array[float],
    q: wp.array[wp.vec3],
    mass: wp.array[float],
    frames: wp.array[wp.mat33],
    history: wp.array[float],
    damage: wp.array[float],
    v: wp.array[wp.vec3],
    c: wp.array[wp.mat33],
):
    b = wp.tid()
    i, j = pairs[b][0], pairs[b][1]
    u, s, w = wp.svd3(0.5 * (frames[i] + frames[j]))
    rotation = u @ wp.transpose(w)
    normal = rotation @ wp.normalize(rest[b])
    delta = q[j] - q[i] - rotation @ rest[b]
    opening = wp.max(wp.dot(delta, normal), 0.0)
    tangent = delta - wp.dot(delta, normal) * normal
    separation = wp.sqrt(opening * opening + wp.length_sq(tangent))
    history[b] = wp.max(history[b], separation)
    maximum = history[b]
    d = float(0.0)
    if maximum >= final:
        d = 1.0
    elif maximum > peak:
        d = final * (maximum - peak) / (maximum * (final - peak))
    # Permanent pore collapse can destroy sponge cohesion even while the
    # bond is compressed. Cream/frosting retain their original traction law.
    if crush_final > crush_start and crush_final > 0.0:
        crushed = float(0.0)
        if young[i] > 100000.0:
            crushed = wp.max(crushed, compaction[i])
        if young[j] > 100000.0:
            crushed = wp.max(crushed, compaction[j])
        d = wp.max(d, wp.clamp((crushed - crush_start) / (crush_final - crush_start), 0.0, 1.0))
    damage[b] = wp.max(damage[b], d)
    d = damage[b]
    force = (1.0 - d) * strength / peak * area[b] * (opening * normal + tangent)
    wp.atomic_add(v, i, dt / mass[i] * force)
    wp.atomic_add(v, j, -dt / mass[j] * force)
    angular = -0.5 * dt * wp.cross(q[i] - q[j], force)
    moment = 0.25 * grid * grid
    wp.atomic_add(c, i, wp.skew(angular / (2.0 * mass[i] * moment)))
    wp.atomic_add(c, j, wp.skew(angular / (2.0 * mass[j] * moment)))


@wp.kernel
def sphere_initialize(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body: int,
    center: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    angular: wp.array[wp.vec3],
):
    center[0] = wp.transform_get_translation(body_q[body])
    velocity[0] = wp.spatial_top(body_qd[body])
    angular[0] = wp.spatial_bottom(body_qd[body])


@wp.kernel
def sphere_contact(
    q: wp.array[wp.vec3],
    mass: wp.array[float],
    spacing: wp.array[float],
    center: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    angular: wp.array[wp.vec3],
    radius: float,
    sphere_mass: float,
    v: wp.array[wp.vec3],
    impulse: wp.array[wp.vec3],
    reaction: wp.array[wp.vec3],
    moment: wp.array[wp.vec3],
    work: wp.array[float],
):
    """Propose dissipative velocity impulses; overlap never creates a target speed."""
    i = wp.tid()
    impulse[i] = wp.vec3(0.0)
    offset = q[i] - center[0]
    distance = wp.length(offset)
    gap = distance - radius - 0.5 * spacing[i]
    if gap < 0.0 and distance > 1.0e-8:
        n = offset / distance
        relative = v[i] - velocity[0] - wp.cross(angular[0], offset)
        normal_speed = wp.dot(relative, n)
        normal_impulse = wp.max(-normal_speed, 0.0) / (1.0 / mass[i] + 1.0 / sphere_mass)
        tangent = relative - normal_speed * n
        friction = wp.min(0.3 * normal_impulse, mass[i] * wp.length(tangent))
        change = normal_impulse * n - friction * tangent / wp.max(wp.length(tangent), 1.0e-9)
        impulse[i] = change
        wp.atomic_add(reaction, 0, -change)
        wp.atomic_add(moment, 0, wp.cross(offset, -change))
        wp.atomic_add(work, 0, wp.dot(relative, change))
        wp.atomic_add(work, 1, 0.5 * wp.length_sq(change) / mass[i])


@wp.kernel
def sphere_compliant_contact(
    q: wp.array[wp.vec3],
    mass: wp.array[float],
    spacing: wp.array[float],
    center: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    angular: wp.array[wp.vec3],
    radius: float,
    sphere_mass: float,
    dt: float,
    stiffness: float,
    damping_ratio: float,
    v: wp.array[wp.vec3],
    impulse: wp.array[wp.vec3],
    reaction: wp.array[wp.vec3],
    moment: wp.array[wp.vec3],
    work: wp.array[float],
):
    """Implicit Kelvin contact: finite spring energy, damping, and Coulomb friction."""
    i = wp.tid()
    impulse[i] = wp.vec3(0.0)
    offset = q[i] - center[0]
    distance = wp.length(offset)
    if distance > 1.0e-8:
        n = offset / distance
        depth = wp.max(radius + 0.5 * spacing[i] - distance, 0.0)
        relative = v[i] - velocity[0] - wp.cross(angular[0], offset)
        normal_speed = wp.dot(relative, n)
        if depth > 0.0:
            k = stiffness * spacing[i] * spacing[i]
            inverse_mass = 1.0 / mass[i] + 1.0 / sphere_mass
            damping = 2.0 * damping_ratio * wp.sqrt(k / inverse_mass)
            coefficient = damping + dt * k
            normal_impulse = wp.max(dt * (k * depth - coefficient * normal_speed), 0.0)
            normal_impulse /= 1.0 + dt * coefficient * inverse_mass
            tangent = relative - normal_speed * n
            friction = wp.min(0.3 * normal_impulse, mass[i] * wp.length(tangent))
            change = normal_impulse * n - friction * tangent / wp.max(wp.length(tangent), 1.0e-9)
            impulse[i] = change
            wp.atomic_add(reaction, 0, -change)
            wp.atomic_add(moment, 0, wp.cross(offset, -change))
            wp.atomic_add(work, 0, wp.dot(relative, change))
            wp.atomic_add(work, 1, 0.5 * wp.length_sq(change) / mass[i])


@wp.kernel
def compliant_contact_energy(
    q: wp.array[wp.vec3],
    mass: wp.array[float],
    spacing: wp.array[float],
    center: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    radius: float,
    sphere_mass: float,
    dt: float,
    stiffness: float,
    v: wp.array[wp.vec3],
    impulse: wp.array[wp.vec3],
    reaction: wp.array[wp.vec3],
    energy: wp.array[float],
):
    """Quadratic bound on spring-energy change under a common impulse scale."""
    i = wp.tid()
    offset = q[i] - center[0]
    distance = wp.length(offset)
    depth = radius + 0.5 * spacing[i] - distance
    if depth > 0.0 and distance > 1.0e-8:
        n = offset / distance
        k = stiffness * spacing[i] * spacing[i]
        predicted = depth - dt * wp.dot(v[i] - velocity[0], n)
        change = dt * wp.dot(impulse[i] / mass[i] - reaction[0] / sphere_mass, n)
        # The spring is unilateral. Bound its positive compression over
        # scales in [0, 1], without counting fictitious tensile spring energy
        # when an opening contact leaves the surface during this substep.
        if predicted <= 0.0:
            predicted = 0.0
            change = wp.min(change, 0.0)
        else:
            change = wp.min(change, predicted)
        wp.atomic_add(energy, 0, 0.5 * k * (predicted * predicted - depth * depth))
        wp.atomic_add(energy, 1, -k * predicted * change)
        wp.atomic_add(energy, 2, 0.5 * k * change * change)


@wp.kernel
def compliant_contact_scale(
    reaction: wp.array[wp.vec3],
    moment: wp.array[wp.vec3],
    work: wp.array[float],
    spring: wp.array[float],
    sphere_mass: float,
    radius: float,
    scale: wp.array[float],
    failures: wp.array[int],
):
    quadratic = work[1] + 0.5 * wp.length_sq(reaction[0]) / sphere_mass
    quadratic += 0.5 * wp.length_sq(moment[0]) / (0.4 * sphere_mass * radius * radius) + spring[2]
    linear = work[0] + spring[1]
    alpha = float(1.0)
    if spring[0] + linear + quadratic > 1.0e-8:
        alpha = wp.clamp(-linear / wp.max(2.0 * quadratic, 1.0e-30), 0.0, 1.0)
        if spring[0] + alpha * linear + alpha * alpha * quadratic > 1.0e-8:
            wp.atomic_add(failures, 0, 1)
    scale[0] = alpha
    reaction[0] *= alpha
    moment[0] *= alpha


@wp.kernel
def sphere_contact_scale(
    reaction: wp.array[wp.vec3],
    moment: wp.array[wp.vec3],
    work: wp.array[float],
    sphere_mass: float,
    radius: float,
    restitution: float,
    scale: wp.array[float],
):
    """Account for the shared rigid body's simultaneous linear/angular response.

    Contact kinetic work is a*A + a*a*B. Restitution in [0, 1] scales
    the dissipative minimum by at most two, keeping the work nonpositive.
    """
    quadratic = work[1] + 0.5 * wp.length_sq(reaction[0]) / sphere_mass
    quadratic += 0.5 * wp.length_sq(moment[0]) / (0.4 * sphere_mass * radius * radius)
    alpha = float(0.0)
    if quadratic > 0.0:
        alpha = (1.0 + restitution) * wp.clamp(-work[0] / (2.0 * quadratic), 0.0, 1.0)
    scale[0] = alpha
    reaction[0] *= alpha
    moment[0] *= alpha


@wp.kernel
def apply_sphere_contact(
    impulse: wp.array[wp.vec3], mass: wp.array[float], scale: wp.array[float], velocity: wp.array[wp.vec3]
):
    i = wp.tid()
    velocity[i] += scale[0] * impulse[i] / mass[i]


@wp.kernel
def sphere_advance(
    center: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    angular: wp.array[wp.vec3],
    local_impulse: wp.array[wp.vec3],
    local_moment: wp.array[wp.vec3],
    total: wp.array[wp.vec3],
    torque: wp.array[wp.vec3],
    mass: float,
    radius: float,
    dt: float,
):
    velocity[0] += local_impulse[0] / mass
    angular[0] += local_moment[0] / (0.4 * mass * radius * radius)
    center[0] += dt * velocity[0]
    total[0] += local_impulse[0]
    torque[0] += local_moment[0]


@wp.kernel
def sphere_wrench(
    impulse: wp.array[wp.vec3],
    moment: wp.array[wp.vec3],
    body: int,
    mapping: wp.array[int],
    dt: float,
    force: wp.array[wp.spatial_vector],
):
    target = mapping[body]
    if target >= 0:
        force[target] = wp.spatial_vector(impulse[0] / dt, moment[0] / dt)


@wp.kernel
def display_frames(c: wp.array[wp.mat33], dt: float, frame: wp.array[wp.mat33]):
    i = wp.tid()
    f = deformation_increment(c[i], dt) @ frame[i]
    u, s, v = wp.svd3(f)
    for axis in range(3):
        s[axis] = wp.clamp(s[axis], 0.35, 1.8)
    frame[i] = u @ wp.diag(s) @ wp.transpose(v)


@wp.kernel
def plastic_compaction(
    cap: float,
    hardening: float,
    young: wp.array[float],
    poisson: wp.array[float],
    initial_volume: wp.array[float],
    plastic_log_volume: wp.array[float],
    elastic: wp.array[wp.mat33],
    volume: wp.array[float],
    spacing: wp.array[float],
):
    """Pressure cap absorbs pore collapse; hardening limits continued densification."""
    i = wp.tid()
    f = elastic[i]
    log_volume = wp.log(wp.max(wp.determinant(f), 1.0e-8))
    bulk_modulus = young[i] / (3.0 * (1.0 - 2.0 * poisson[i]))
    limit = -(cap + hardening * plastic_log_volume[i]) / bulk_modulus
    if log_volume < limit:
        collapsed = wp.min(limit - log_volume, wp.max(0.7 - plastic_log_volume[i], 0.0))
        plastic_log_volume[i] += collapsed
        elastic[i] = wp.exp(collapsed / 3.0) * f
    volume[i] = initial_volume[i] * wp.exp(-plastic_log_volume[i])
    spacing[i] = wp.pow(volume[i], 1.0 / 3.0)


@wp.kernel
def grain_skin_frames(
    frames: wp.array[wp.mat33],
    sizes: wp.array[int],
    compaction: wp.array[float],
    threshold: int,
    skin_frames: wp.array[wp.mat33],
):
    """Unresolved small grains retain their rest shape, rotation, and pore volume.

    A handful of particles cannot resolve a continuum deformation reliably.
    Their Gaussian support therefore uses the measured rotation and physical
    plastic volume, rather than an accumulating APIC extrapolation.
    """
    i = wp.tid()
    f = frames[i]
    if sizes[i] <= threshold:
        u, s, v = wp.svd3(f)
        f = wp.exp(-compaction[i] / 3.0) * u @ wp.transpose(v)
    skin_frames[i] = f


@wp.kernel
def relax_grains(sizes: wp.array[int], threshold: int, elastic: wp.array[wp.mat33], c: wp.array[wp.mat33]):
    i = wp.tid()
    if sizes[i] <= threshold:
        u, s, v = wp.svd3(elastic[i])
        elastic[i] = u @ wp.transpose(v)
        c[i] = 0.5 * (c[i] - wp.transpose(c[i]))


class SolverCakeFracture(SolverExplicitMultiFieldMPM):
    @dataclass
    class Config(SolverExplicitMultiFieldMPM.Config):
        bond_strength: float = 600.0
        bond_peak: float = 0.002
        bond_final: float = 0.006
        cherry_radius: float = 0.077
        cherry_mass: float = 2.0
        neighbor_count: int = 8
        bond_graph: str = "nearest"
        crush_start: float = 0.0
        crush_final: float = 0.0
        grain_threshold: int = 6
        compression_pressure: float = 1500.0
        compression_hardening: float = 10000.0
        coupling_fracture: bool = False
        """Assign fracture fields on the GPU every coupling step, including CUDA graph replay."""
        field_separation_damage: float = 0.9999
        """Separate fields during cohesive softening; the remaining bond traction still acts across fields."""
        contact_restitution: float = 0.0
        """Fractional rebound in the aggregate particle/sphere contact update, bounded to [0, 1]."""
        contact_stiffness: float = 0.0
        """Contact spring stiffness per represented area [N/m^3]; zero uses velocity contact."""
        contact_damping_ratio: float = 0.75
        node_local_fracture: bool = False
        """Partition each grid node using its local surviving-bond graph, refreshed every MPM substep."""

    def __init__(self, model, config):
        if config.fields == 0:
            config.fields = model.particle_count
        if (config.coupling_fracture or config.node_local_fracture) and config.fields < model.particle_count:
            raise ValueError("GPU fracture assignment requires one field slot per particle")
        if not 0.0 < config.field_separation_damage <= 1.0:
            raise ValueError("Field separation damage must be in (0, 1]")
        if not 0.0 <= config.contact_restitution <= 1.0:
            raise ValueError("Contact restitution must be in [0, 1]")
        if config.contact_stiffness < 0.0 or config.contact_damping_ratio < 0.0:
            raise ValueError("Require nonnegative contact stiffness and damping")
        if config.contact_stiffness > 0.0 and config.contact_restitution != 0.0:
            raise ValueError("Compliant contact uses spring damping instead of contact restitution")
        if not 0.0 < config.bond_peak < config.bond_final or config.bond_strength <= 0.0:
            raise ValueError("Require positive cohesive strength and 0 < peak < final separation")
        if config.neighbor_count < 1 or config.grain_threshold < 0:
            raise ValueError("Require positive neighbor count and nonnegative grain threshold")
        if (config.crush_start != 0.0 or config.crush_final != 0.0) and not (
            0.0 <= config.crush_start < config.crush_final
        ):
            raise ValueError("Require 0 <= crush start < final, or both zero to disable crush damage")
        super().__init__(model, config)
        bodies = [i for i, name in enumerate(model.body_label) if name.endswith("/Cherry")]
        if len(bodies) != 1:
            raise ValueError(f"Expected exactly one cherry proxy, got {model.body_label}")
        self.cherry_body = bodies[0]
        rest = model.particle_q.numpy()
        pairs = rest_bond_pairs(rest, config.neighbor_count, config.bond_graph)
        # Each pair receives its share of a particle's represented surface area.
        degree = np.bincount(pairs.ravel(), minlength=len(rest))
        spacing = self.spacing.numpy()
        areas = 0.5 * (
            spacing[pairs[:, 0]] ** 2 / degree[pairs[:, 0]] + spacing[pairs[:, 1]] ** 2 / degree[pairs[:, 1]]
        )
        self.pairs_host = pairs
        self.bonds = wp.array(pairs, dtype=wp.vec2i, device=model.device)
        if config.node_local_fracture:
            self.local_parent = wp.empty(
                int(np.prod(config.grid_resolution)) * config.fields, dtype=int, device=model.device
            )
            self.local_cracked = wp.zeros(int(np.prod(config.grid_resolution)), dtype=int, device=model.device)
        self.rest_delta = wp.array(rest[pairs[:, 1]] - rest[pairs[:, 0]], dtype=wp.vec3, device=model.device)
        self.bond_area = wp.array(areas, dtype=float, device=model.device)
        self.bond_history = wp.zeros(len(pairs), dtype=float, device=model.device)
        self.bond_damage = wp.zeros(len(pairs), dtype=float, device=model.device)
        self.frames = wp.array(
            np.broadcast_to(np.eye(3, dtype=np.float32), (len(rest), 3, 3)).copy(), dtype=wp.mat33, device=model.device
        )
        self.skin_frames = wp.clone(self.frames)
        self.contact_impulse = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self.contact_work = wp.zeros(2, dtype=float, device=model.device)
        self.contact_scale = wp.zeros(1, dtype=float, device=model.device)
        self.contact_spring_work = wp.zeros(3, dtype=float, device=model.device)
        self.contact_energy_failures = wp.zeros(1, dtype=int, device=model.device)
        for name in ("center", "velocity", "angular", "local_impulse", "local_moment", "impulse", "moment"):
            setattr(self, name, wp.zeros(1, dtype=wp.vec3, device=model.device))
        # Match the authored shear yield approximately; explicit and implicit laws differ.
        self.yield_strain.assign(
            np.maximum(model.mpm.yield_stress.numpy() / model.mpm.young_modulus.numpy(), 0.0001).astype(np.float32)
        )
        self.initial_volume = wp.clone(self.volume)
        self.plastic_log_volume = wp.zeros(model.particle_count, dtype=float, device=model.device)
        self.component_parent = wp.zeros(model.particle_count, dtype=int, device=model.device)
        self.component_counts = wp.zeros(model.particle_count, dtype=int, device=model.device)
        self.fragment_count = 1
        self.field_count = 1
        self.update_fields()

    def update_fields(self):
        if self.config.coupling_fracture:
            # GPU stepping owns the labels; the task only reads diagnostics.
            self.fragment_count = len(np.unique(self.tissue_field.numpy()))
            self.field_count = self.fragment_count
            return
        alive = self.bond_damage.numpy() < self.config.field_separation_damage
        pairs = self.pairs_host[alive]
        count, labels = connected_components(
            coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(self.model.particle_count,) * 2),
            directed=False,
        )
        if count > self.config.fields:
            raise RuntimeError(f"{count} fragments exceed {self.config.fields} field slots; increase capacity.")
        self.tissue_field.assign(labels.astype(np.int32))
        self.fragment_sizes.assign(np.bincount(labels)[labels].astype(np.int32))
        wp.launch(
            relax_grains,
            dim=self.model.particle_count,
            inputs=[self.fragment_sizes, self.config.grain_threshold, self.elastic, self.c],
            device=self.model.device,
        )
        self.fragment_count = count
        self.field_count = count

    def _update_fields_gpu(self):
        device = self.model.device
        count = self.model.particle_count
        wp.launch(initialize_components, count, inputs=[self.component_parent, self.component_counts], device=device)
        wp.launch(
            join_components,
            len(self.bonds),
            inputs=[self.bonds, self.bond_damage, self.config.field_separation_damage, self.component_parent],
            device=device,
        )
        wp.launch(
            label_components,
            count,
            inputs=[self.component_parent, self.tissue_field, self.component_counts],
            device=device,
        )
        wp.launch(
            component_sizes,
            count,
            inputs=[self.tissue_field, self.component_counts, self.fragment_sizes],
            device=device,
        )
        wp.launch(
            relax_grains,
            count,
            inputs=[self.fragment_sizes, self.config.grain_threshold, self.elastic, self.c],
            device=device,
        )

    def _prepare_grid_partition(self, positions):
        if self.config.node_local_fracture:
            cfg = self.config
            self.local_cracked.zero_()
            geometry = [wp.vec3(*cfg.grid_origin), cfg.grid_spacing, wp.vec3i(*cfg.grid_resolution), cfg.fields]
            wp.launch(
                initialize_local_components,
                self.model.particle_count * 27,
                inputs=[positions, *geometry, self.local_parent],
                device=self.model.device,
            )
            wp.launch(
                join_local_components,
                len(self.bonds) * 27,
                inputs=[
                    positions,
                    self.bonds,
                    self.bond_damage,
                    cfg.field_separation_damage,
                    *geometry,
                    self.local_parent,
                    self.local_cracked,
                ],
                device=self.model.device,
            )

    def check(self):
        super().check()
        if self.config.contact_stiffness > 0.0 and self.contact_energy_failures.numpy()[0]:
            raise RuntimeError(
                "Compliant contact exceeded its active spring energy bound; reduce stiffness or timestep"
            )

    def reset(self, state, world_mask=None, flags=None):
        if not self._resets_particle_history(world_mask, flags):
            return
        super().reset(state, world_mask, flags)
        self.bond_history.zero_()
        self.contact_energy_failures.zero_()
        self.bond_damage.zero_()
        self.frames.assign(np.broadcast_to(np.eye(3, dtype=np.float32), (self.model.particle_count, 3, 3)).copy())
        wp.copy(self.skin_frames, self.frames)
        self.plastic_log_volume.zero_()
        wp.copy(self.volume, self.initial_volume)
        self.spacing.assign(np.cbrt(self.initial_volume.numpy()).astype(np.float32))
        if self.config.coupling_fracture:
            self._update_fields_gpu()
        self.update_fields()

    def step(self, state_in, state_out, control, contacts, dt):
        if state_in is not state_out:
            raise ValueError("Cake fracture currently requires in-place stepping")
        cfg = self.config
        if cfg.coupling_fracture:
            self._update_fields_gpu()
        count = round(dt * cfg.substep_rate)
        h = dt / count
        self.impulse.zero_()
        self.moment.zero_()
        wp.launch(
            sphere_initialize,
            dim=1,
            inputs=[state_in.body_q, state_in.body_qd, self.cherry_body, self.center, self.velocity, self.angular],
            device=self.model.device,
        )
        for _ in range(count):
            wp.launch(
                bonds_step,
                dim=len(self.bonds),
                inputs=[
                    self.bonds,
                    self.rest_delta,
                    self.bond_area,
                    cfg.bond_strength,
                    cfg.bond_peak,
                    cfg.bond_final,
                    h,
                    cfg.grid_spacing,
                    cfg.crush_start,
                    cfg.crush_final,
                    self.model.mpm.young_modulus,
                    self.plastic_log_volume,
                    state_in.particle_q,
                    self.model.particle_mass,
                    self.elastic,
                    self.bond_history,
                    self.bond_damage,
                    state_in.particle_qd,
                    self.c,
                ],
                device=self.model.device,
            )
            self.local_impulse.zero_()
            self.local_moment.zero_()
            self.contact_work.zero_()
            inputs = [
                state_in.particle_q,
                self.model.particle_mass,
                self.spacing,
                self.center,
                self.velocity,
                self.angular,
                cfg.cherry_radius,
                cfg.cherry_mass,
            ]
            if cfg.contact_stiffness > 0.0:
                inputs.extend([h, cfg.contact_stiffness, cfg.contact_damping_ratio])
            inputs.extend(
                [
                    state_in.particle_qd,
                    self.contact_impulse,
                    self.local_impulse,
                    self.local_moment,
                    self.contact_work,
                ]
            )
            wp.launch(
                sphere_compliant_contact if cfg.contact_stiffness > 0.0 else sphere_contact,
                self.model.particle_count,
                inputs=inputs,
                device=self.model.device,
            )
            if cfg.contact_stiffness > 0.0:
                self.contact_spring_work.zero_()
                wp.launch(
                    compliant_contact_energy,
                    self.model.particle_count,
                    inputs=[
                        state_in.particle_q,
                        self.model.particle_mass,
                        self.spacing,
                        self.center,
                        self.velocity,
                        cfg.cherry_radius,
                        cfg.cherry_mass,
                        h,
                        cfg.contact_stiffness,
                        state_in.particle_qd,
                        self.contact_impulse,
                        self.local_impulse,
                        self.contact_spring_work,
                    ],
                    device=self.model.device,
                )
                wp.launch(
                    compliant_contact_scale,
                    1,
                    inputs=[
                        self.local_impulse,
                        self.local_moment,
                        self.contact_work,
                        self.contact_spring_work,
                        cfg.cherry_mass,
                        cfg.cherry_radius,
                    ],
                    outputs=[self.contact_scale, self.contact_energy_failures],
                    device=self.model.device,
                )
            else:
                wp.launch(
                    sphere_contact_scale,
                    1,
                    inputs=[
                        self.local_impulse,
                        self.local_moment,
                        self.contact_work,
                        cfg.cherry_mass,
                        cfg.cherry_radius,
                        cfg.contact_restitution,
                    ],
                    outputs=[self.contact_scale],
                    device=self.model.device,
                )
            wp.launch(
                apply_sphere_contact,
                dim=self.model.particle_count,
                inputs=[self.contact_impulse, self.model.particle_mass, self.contact_scale],
                outputs=[state_in.particle_qd],
                device=self.model.device,
            )
            super().step(state_in, state_out, control, contacts, h)
            if cfg.compression_pressure > 0.0:
                wp.launch(
                    plastic_compaction,
                    dim=self.model.particle_count,
                    inputs=[
                        cfg.compression_pressure,
                        cfg.compression_hardening,
                        self.model.mpm.young_modulus,
                        self.model.mpm.poisson_ratio,
                        self.initial_volume,
                        self.plastic_log_volume,
                        self.elastic,
                        self.volume,
                        self.spacing,
                    ],
                    device=self.model.device,
                )
            wp.launch(
                sphere_advance,
                dim=1,
                inputs=[
                    self.center,
                    self.velocity,
                    self.angular,
                    self.local_impulse,
                    self.local_moment,
                    self.impulse,
                    self.moment,
                    cfg.cherry_mass,
                    cfg.cherry_radius,
                    h,
                ],
                device=self.model.device,
            )
            wp.launch(
                display_frames, dim=self.model.particle_count, inputs=[self.c, h, self.frames], device=self.model.device
            )
        wp.launch(
            grain_skin_frames,
            dim=self.model.particle_count,
            inputs=[self.frames, self.fragment_sizes, self.plastic_log_volume, cfg.grain_threshold, self.skin_frames],
            device=self.model.device,
        )

    def coupling_harvest_proxy_wrenches(
        self, body_local_to_proxy_global, out_body_f, *, body_qd_before, state, state_out, contacts, dt
    ):
        out_body_f.zero_()
        wp.launch(
            sphere_wrench,
            dim=1,
            inputs=[self.impulse, self.moment, self.cherry_body, body_local_to_proxy_global, dt, out_body_f],
            device=self.model.device,
        )


class CakeFractureManager(NewtonMPMManager):
    @classmethod
    def _create_solver(cls, model, solver_cfg):
        return SolverCakeFracture(model, solver_cfg.solver_config)


@configclass
class CakeFractureSolverCfg(MPMSolverCfg):
    class_type: type = CakeFractureManager
    solver_config: SolverCakeFracture.Config | None = None
