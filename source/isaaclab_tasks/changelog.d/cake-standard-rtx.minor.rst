Changed
^^^^^^^

* Integrated Cake Gaussian rendering with the standard Isaac Lab ``newton_rtx`` visualizer and removed the task-specific viewer and window fallback.
* Added a Gaussian asset configuration and GPU scene stream with synchronized reset and shutdown.
* Removed Cake benchmark and recording utilities from the runtime package; interactive execution uses the ordinary Lab task CLI.
* Added cherry XYZ position sliders and drop presets to the standard viewer sidebar, with immediate stationary placement, pause/resume and persistence across manual reset.
* Set the Cake task's default RTX view to 1280x960 with explicit DLSS Quality and frame generation disabled for Gaussian rendering.
* Defaulted the passive Cake example to manual episode reset, with timed resets available through task configuration.

* Enabled Newton's native right-click spring-force picking for the rigid cherry.

Fixed
^^^^^

* Kept unchanged Cake Gaussian geometry published during camera-only redraws
  so RTX temporal reconstruction converged while simulation was paused.

Added
^^^^^

* Added a positive ``material_strength_scale`` task setting for tuning authored
  Cake elastic stiffness and yield limits without changing its mass or geometry.
