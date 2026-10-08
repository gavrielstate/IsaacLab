Added
^^^^^

* Added optional ``gaussian_skinning_jacobian`` covariance transport for the Cake
  fracture task, including spatial derivatives of normalized skinning weights.
  Gaussians can expand and rotate as supporting particles separate, bounded by
  ``gaussian_max_stretch``. Activated paste uses nearby supports in its shared
  physical field. Pure Gaussian rendering remains in the standard Lab OVRTX
  visualizer; mass, material mechanics, contacts and radiance are unchanged.
