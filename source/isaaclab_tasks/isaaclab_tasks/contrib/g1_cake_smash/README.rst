G1 cherry throw into native MPM cake
===================================

``IsaacContrib-G1-Cake-Smash-Direct`` plays the fixed-base targeted-shot-put
policy with one physically simulated cherry from hand contact through impact.
The robot root is fixed; this is separate from the standing policy. The task
uses standard Lab ``newton_rtx`` and the Cake Gaussian scene stream.

The collider preserves the trained 35 mm radius. The procedural cherry mesh
and stem are scaled to it; the stem is visual decoration. The default mass is
2 kg, and ``env.cherry_mass`` accepts the trained range of 1.5--2.5 kg.
The recorded policy mechanics are restored from ``params/env.yaml``, including
robot assets, initial joints, actuator gains, timing and rigid solver settings.
The RSL runner loads the checkpoint's observation normalizer.

This is a playback demonstration. It preserves the 53 state observations and
12 actions of the fixed-base policy. The robot holds its final action after
the recorded three-second policy horizon; cake dynamics continue. Use the
standard viewer pause/resume and Reset controls to inspect and repeat a throw.
No automatic episode reset occurs.

Targets remain ground-plane XY. The default target is (1.2, 0) m, with cake
placement fixed before flight at (1.08, -0.025, 0) m. That placement was
calibrated for the supplied 2 kg fixed-base checkpoint. Other masses and targets
can change the trajectory; no cake relocation or projectile teleport happens
at release.

Launch
------

With the task and Newton packages installed from this checkout:

.. code-block:: bash

   export ISAACLAB_CAKE_PHYSICS_USD_PATH=/path/to/isaaclab_cake_all_firm_stand64_visuals.usdc
   export ISAACLAB_CAKE_GAUSSIAN_USD_PATH=/path/to/cake_live_ovrtx_lod_surface2_interior4.usdc
   export ISAACLAB_SHOT_POLICY_PARAMS_PATH=/path/to/shot_policy/params
   uv run --no-sync isaaclab play --task IsaacContrib-G1-Cake-Smash-Direct \
       --visualizer newton_rtx --num_envs 1 --checkpoint /path/to/shot_policy/model_2999.pt

On tor-datastudio-2, the existing ``work/host/run-isaaclab`` helper additionally
preserves the validated Triton/LLVM initialization order. Exact host commands
and recordings are kept under ``work/cake``, outside candidate repository code.

The fixed-base task implementation and agent configuration were selected from
``experiment/targeted-shot-put`` at ``853400c26a1``; its unrelated baseline and
dependency changes are not included. Asset data and policy weights are external.
