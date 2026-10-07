# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fracture graph, field allocation, and reacting rigid-sphere contact.

This experimental solver separates connected components of an irreversible
rest-neighbor bond graph. Field labels change only at the uncaptured task-step
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

from .explicit_mpm import SolverExplicitMultiFieldMPM, deformation_increment


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
    damage[b] = d
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
    dt: float,
    v: wp.array[wp.vec3],
    reaction: wp.array[wp.vec3],
    moment: wp.array[wp.vec3],
):
    i = wp.tid()
    offset = q[i] - center[0]
    distance = wp.length(offset)
    gap = distance - radius - 0.5 * spacing[i]
    if gap < 0.0 and distance > 1.0e-8:
        n = offset / distance
        relative = v[i] - velocity[0] - wp.cross(angular[0], offset)
        normal_speed = wp.dot(relative, n)
        impulse = wp.max(-normal_speed - 0.08 * gap / dt, 0.0) / (1.0 / mass[i] + 1.0 / sphere_mass)
        tangent = relative - normal_speed * n
        friction = wp.min(0.3 * impulse, mass[i] * wp.length(tangent))
        change = impulse * n - friction * tangent / wp.max(wp.length(tangent), 1.0e-9)
        v[i] += change / mass[i]
        wp.atomic_add(reaction, 0, -change)
        wp.atomic_add(moment, 0, wp.cross(offset, -change))


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
    frames: wp.array[wp.mat33], sizes: wp.array[int], compaction: wp.array[float], skin_frames: wp.array[wp.mat33]
):
    """Unresolved small grains retain their rest shape, rotation, and pore volume.

    A handful of particles cannot resolve a continuum deformation reliably.
    Their Gaussian support therefore uses the measured rotation and physical
    plastic volume, rather than an accumulating APIC extrapolation.
    """
    i = wp.tid()
    f = frames[i]
    if sizes[i] <= 6:
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
        grain_threshold: int = 6
        compression_pressure: float = 1500.0
        compression_hardening: float = 10000.0

    def __init__(self, model, config):
        if config.fields == 0:
            config.fields = model.particle_count
        if not 0.0 < config.bond_peak < config.bond_final or config.bond_strength <= 0.0:
            raise ValueError("Require positive cohesive strength and 0 < peak < final separation")
        super().__init__(model, config)
        bodies = [i for i, name in enumerate(model.body_label) if name.endswith("/Cherry")]
        if len(bodies) != 1:
            raise ValueError(f"Expected exactly one cherry proxy, got {model.body_label}")
        self.cherry_body = bodies[0]
        rest = model.particle_q.numpy()
        distances, neighbors = cKDTree(rest).query(rest, k=config.neighbor_count + 1)
        pairs = np.unique(
            np.sort(np.array([(i, j) for i, row in enumerate(neighbors[:, 1:]) for j in row], np.int32), axis=1), axis=0
        )
        # Each pair receives its share of a particle's represented surface area.
        degree = np.bincount(pairs.ravel(), minlength=len(rest))
        spacing = self.spacing.numpy()
        areas = 0.5 * (
            spacing[pairs[:, 0]] ** 2 / degree[pairs[:, 0]] + spacing[pairs[:, 1]] ** 2 / degree[pairs[:, 1]]
        )
        self.pairs_host = pairs
        self.bonds = wp.array(pairs, dtype=wp.vec2i, device=model.device)
        self.rest_delta = wp.array(rest[pairs[:, 1]] - rest[pairs[:, 0]], dtype=wp.vec3, device=model.device)
        self.bond_area = wp.array(areas, dtype=float, device=model.device)
        self.bond_history = wp.zeros(len(pairs), dtype=float, device=model.device)
        self.bond_damage = wp.zeros(len(pairs), dtype=float, device=model.device)
        self.frames = wp.array(
            np.broadcast_to(np.eye(3, dtype=np.float32), (len(rest), 3, 3)).copy(), dtype=wp.mat33, device=model.device
        )
        self.skin_frames = wp.clone(self.frames)
        for name in ("center", "velocity", "angular", "local_impulse", "local_moment", "impulse", "moment"):
            setattr(self, name, wp.zeros(1, dtype=wp.vec3, device=model.device))
        # Match the authored shear yield approximately; explicit and implicit laws differ.
        self.yield_strain.assign(
            np.maximum(model.mpm.yield_stress.numpy() / model.mpm.young_modulus.numpy(), 0.0001).astype(np.float32)
        )
        self.initial_volume = wp.clone(self.volume)
        self.plastic_log_volume = wp.zeros(model.particle_count, dtype=float, device=model.device)
        self.fragment_count = 1
        self.field_count = 1
        self.update_fields()

    def update_fields(self):
        alive = self.bond_damage.numpy() < 0.9999
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

    def reset(self, state, world_mask=None, flags=None):
        super().reset(state, world_mask, flags)
        self.bond_history.zero_()
        self.bond_damage.zero_()
        self.frames.assign(np.broadcast_to(np.eye(3, dtype=np.float32), (self.model.particle_count, 3, 3)).copy())
        wp.copy(self.skin_frames, self.frames)
        self.plastic_log_volume.zero_()
        wp.copy(self.volume, self.initial_volume)
        self.spacing.assign(np.cbrt(self.initial_volume.numpy()).astype(np.float32))
        self.update_fields()

    def step(self, state_in, state_out, control, contacts, dt):
        if state_in is not state_out:
            raise ValueError("Cake fracture currently requires in-place stepping")
        cfg = self.config
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
            wp.launch(
                sphere_contact,
                dim=self.model.particle_count,
                inputs=[
                    state_in.particle_q,
                    self.model.particle_mass,
                    self.spacing,
                    self.center,
                    self.velocity,
                    self.angular,
                    cfg.cherry_radius,
                    cfg.cherry_mass,
                    h,
                    state_in.particle_qd,
                    self.local_impulse,
                    self.local_moment,
                ],
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
            inputs=[self.frames, self.fragment_sizes, self.plastic_log_volume, self.skin_frames],
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
