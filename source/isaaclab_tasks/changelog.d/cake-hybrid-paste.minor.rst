Added
^^^^^

* Added opt-in hybrid rendering to the experimental Cake fracture task: sponge
  Gaussians share the standard ``newton_rtx`` visualizer with Newton
  ``ParticleSurface`` meshes for cream and frosting. Retained mesh bindings
  publish changing topology through Lab's scene-stream interface.
* Added fracture-triggered cream/frosting paste fields with an exponential
  relaxation of deviatoric strain above yield. Activated particles keep their
  represented volume and affine deformation; intact particles retain the solid
  fracture response. Episode resets restore the solid state, while cherry-only
  resets preserve paste activation and damage. Both features default to disabled.
