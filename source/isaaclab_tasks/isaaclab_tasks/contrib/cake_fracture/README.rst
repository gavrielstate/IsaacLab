Cake fracture with multi-field MPM
=================================

``IsaacContrib-Cake-Fracture-Direct`` runs the experimental cake fracture
solver with standard Isaac Lab ``newton_rtx`` / OVRTX Gaussian rendering.
Its explicit multi-field MPM core derives from Nicolas' Raspberry solver at
commit ``2496558148d586114bc908d9d3fe63b6a02a419b``. It adds irreversible
cohesive bonds, fracture-driven velocity fields, fragment contact, plastic
sponge compaction, yielding viscous paste and compliant cherry contact.
It is an experimental solver integrated with Newton's coupling interfaces;
it is not Newton's built-in implicit MPM solver.

Run interactively
-----------------

Install Isaac Lab's Newton and OVRTX dependencies and obtain the external
cake asset bundle. The validated low-resolution example uses the
936-particle soft-sponge physics USD, ``sponge_30k.usdc``, and the full
Gaussian asset, ``cake_live_ovrtx.usdc``. Keep the physics USD's relative
dependencies alongside it. From the Isaac Lab repository root:

.. code-block:: bash

   export ISAACLAB_CAKE_PHYSICS_USD_PATH=/path/to/sponge_30k.usdc
   export ISAACLAB_CAKE_GAUSSIAN_USD_PATH=/path/to/cake_live_ovrtx.usdc
   uv run --no-sync isaaclab zero_agent \
       --task IsaacContrib-Cake-Fracture-Direct --visualizer newton_rtx --num_envs 1 \
       env.gaussian_paste_flow=True

The default cherry mass is 1.5 kg, with center position (0.000, 0.020, 0.500) m.
Use the standard Pause/Resume and Reset episode controls and Cake drop XYZ
sliders. Repositioning the cherry preserves the crushed cake; resetting the
episode restores it. Episodes do not reset automatically by default.
Append ``env.drop_offset=[0.011,0.011]`` to reproduce the demonstrated impact
position. The explicit solver runs at 1200 Hz with 120 Hz rigid coupling.
The simulation and physical material settings are qualitative cake tuning,
rather than measured material calibration.

``env.gaussian_paste_flow=True`` enables volume-preserving Gaussian shape
transport for physically yielding cream and frosting. It changes appearance,
not physics. Original Gaussians remain the rendered cake geometry; the
optional hybrid surface renderer is disabled by default. The optional
Gaussian skinning Jacobian is also disabled by default. Rendering defaults
remain interactive RTPT with DLSS Quality and one requested sample per pixel.
Offline videos using 64-sample PathTracing and spatial OptiX denoising use
separate capture settings and do not alter interactive defaults.

Use Newton's built-in MPM
------------------------

Select the original task to use Newton's native implicit MPM solver:

.. code-block:: bash

   uv run --no-sync isaaclab zero_agent \
       --task IsaacContrib-Cake-Smash-Direct --visualizer newton_rtx --num_envs 1

Both tasks accept the same asset-path exports and use the standard Lab
visualizer. Solver selection is through the task ID, not a boolean within
the fracture task. The native task retains its own defaults, including a
2 kg cherry, and does not include the experimental cohesive fracture or
paste-flow appearance option. Its behavior is consequently different.
See ``../cake_smash/README.rst`` for native task settings and validation.

Implementation
--------------

``explicit_mpm.py`` contains the explicit particle/grid transfers and material
integration. ``solver.py`` adds cohesive fracture, field assignment and
cherry coupling. ``field_contacts.py`` accelerates grid contact with conservative
sparse candidate generation. ``gaussian_binding.py`` handles fracture-aware
Gaussian skinning without advancing physics. Assets, benchmark scripts,
captures and development reports are external to this runtime package.
