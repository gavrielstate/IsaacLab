Added
^^^^^

* Added opt-in soft-grain transport to the experimental Cake fracture task.
  Small sponge fragments may retain symmetric affine strain while their
  Gaussian frames preserve plastic volume and bound deviatoric distortion.
  The default retains the existing rigid-grain fallback.

Fixed
^^^^^

* Replaced hard-coded hybrid cream/frosting colours with colours derived from
  the source Gaussian asset and linear RGB native RTX materials. Explicitly
  reapplied retained-mesh material bindings after renderer initialization to
  avoid the pale fallback material. Authored valid initial surface geometry
  before publishing dynamic topology updates.
