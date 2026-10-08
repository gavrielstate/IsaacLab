Add an experimental ``IsaacContrib-Cake-Fracture-Direct`` task with irreversible
cohesive bonds, fracture-driven Newton MPM material fields, pressure-cap pore
collapse, and fragment-aware Gaussian skinning in the standard ``newton_rtx``
visualizer. Preserve the existing cherry controls and manual episode reset.

Expose optional directional rest-neighbor bonds, irreversible sponge-compaction
damage and field-contact AABB rejection for material sampling experiments.

Accelerate experimental cake fracture contact with node-major grid-field
storage, cached per-field properties and small-grid CUDA scheduling. Preserve
material settings, contact projection order and the 936-particle input.

Prevent cherry-contact energy injection by removing velocity penetration bias
and bounding simultaneous contact impulses using their total linear/angular
kinetic work. Preserve equal/opposite momentum transfer.

Preserve cake damage, plastic compaction and deformation history when moving
only the cherry; clear material history only on a particle/episode reset.

Expose optional GPU connected-component field assignment at the coupling rate
and separation during cohesive softening, retaining remaining bond traction.
Expose energy-bounded partial rebound. Add finite-stiffness Kelvin sphere contact
with a unilateral active-spring energy audit to resist persistent overlap and
preserve the outward response to impact. Keep existing defaults available.
