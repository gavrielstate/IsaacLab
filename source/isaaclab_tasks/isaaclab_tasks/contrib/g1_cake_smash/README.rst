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

Fracture cake variant
---------------------

``IsaacContrib-G1-Cake-Fracture-Direct`` keeps the same fixed-base policy and
trained robot mechanics, with the newer explicit multi-field cake solver.
The default is a 1.5 kg cherry and a fixed cake offset (1.13, 0.01, 0) m,
calibrated for that mass and the supplied checkpoint's (1.2, 0) m ground target.
The original 35 mm collision radius is retained. Fracture, compacting sponge,
yielding viscous paste and full Gaussian rendering are enabled; no launch
impulse or release teleport is prescribed. Native MPM remains available in
``IsaacContrib-G1-Cake-Smash-Direct``.

.. code-block:: bash

   export ISAACLAB_CAKE_PHYSICS_USD_PATH=/path/to/sponge_30k.usdc
   export ISAACLAB_CAKE_GAUSSIAN_USD_PATH=/path/to/cake_live_ovrtx.usdc
   export ISAACLAB_SHOT_POLICY_PARAMS_PATH=/path/to/shot_policy/params
   uv run --no-sync isaaclab play --task IsaacContrib-G1-Cake-Fracture-Direct \
       --visualizer newton_rtx --num_envs 1 --checkpoint /path/to/shot_policy/model_2999.pt

The physics asset is the validated 936-particle soft-sponge variant, with its
relative dependencies preserved. Cake material and skinning overrides use
the nested ``env.cake`` configuration; for example, ``env.cake.paste_viscosity``
and ``env.cake.gaussian_paste_flow``. The solver grid and analytic pedestal
are translated with the cake before construction. MPM substeps remain at
1200 Hz, while rigid coupling follows the recorded 240 Hz robot timestep.
Control remains at 60 Hz. Reset clears fracture, compaction, paste and
coupled-solver history before returning the ball to the trained palm pose.

This task accepts the fixed-base 53-observation / 12-action contract only.
A free-standing checkpoint with different observations/actions requires its
own matching robot task integration; it must not be loaded into this task.
