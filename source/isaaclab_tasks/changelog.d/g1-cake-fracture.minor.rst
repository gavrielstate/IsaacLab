Added
^^^^^

* Added ``IsaacContrib-G1-Cake-Fracture-Direct`` to run the fixed-base shot-put
  policy with cohesive cake fracture, viscous paste and pure OVRTX Gaussian
  rendering while preserving trained robot mechanics and manual reset.
* Added configurable rigid impactor body selection for the cake fracture
  solver and separate appearance configuration for Gaussian scene streams.
* Framed the robot and cake together with studio lighting in the fracture
  task's initial interactive view.

Fixed
^^^^^

* Preserved owned shot-put action and landing buffers during inference-mode
  playback so manual reset also works outside the inference context.
