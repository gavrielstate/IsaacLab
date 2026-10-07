Native cake crushing with Newton MPM
====================================

``IsaacContrib-Cake-Smash-Direct`` is a single-workcell Isaac Lab task. A solid
2 kg cherry starts at X=0.000 m, Y=0.020 m, Z=0.500 m and falls onto a
layered cake. Newton's native implicit MPM,
MuJoCo Warp and coupled proxy solvers produce the motion. The standard Lab
``newton_rtx`` visualizer renders a deforming Gaussian field, stand and cherry.
There is no task-specific viewer or simulation loop.

Run interactively
-----------------

Install Isaac Lab's Newton and OVRTX dependencies and obtain the external
physics and Gaussian assets. From the Isaac Lab repository root:

.. code-block:: bash

   export ISAACLAB_CAKE_PHYSICS_USD_PATH=/path/to/isaaclab_cake_all_firm_stand64_visuals.usdc
   export ISAACLAB_CAKE_GAUSSIAN_USD_PATH=/path/to/cake_live_ovrtx_lod_surface2_interior4.usdc
   uv run --no-sync isaaclab zero_agent \
       --task IsaacContrib-Cake-Smash-Direct --visualizer newton_rtx --num_envs 1

Use **Pause/Resume simulation** and **Reset episode** in the standard sidebar.
Right-click and hold on the cherry, then drag to apply Newton's native spring
force while simulation runs. Release the mouse button to stop pulling. Picking
uses rigid collision geometry; it does not select the deformable Gaussian cake.
While paused, moving the spring target affects the cherry on Resume or a
single physics step, rather than immediately placing it.
The **Cake drop** panel contains cherry X/Y and center Z sliders and Center,
Left, Right, Rim and Cut face presets. Z is the cherry center height in meters,
not a gap above the cake. Changing a slider or preset immediately places an
upright, stationary cherry and pauses windowed simulation. Resume releases it
from that position. The cake and camera stay unchanged; Reset episode restores
the cake and uses the selected XYZ position. Episodes reset manually by default. Set
``env.reset_on_timeout=True`` to enable ordinary timed Gym resets.

The task registers ``_render_drop_controls`` with the standard visualizer's
``register_ui_callback(..., position="panel")``. The callback receives ImGui
and draws the controls each UI frame. ``set_drop_pose`` writes the rigid pose
and velocity through Lab's asset API and updates kinematics without advancing
physics. Renderer history clearing is deferred until the next frame boundary.

The default render is 1280x960 with DLSS Quality, one requested RTPT sample
per pixel and frame generation disabled. Append ``env.render_samples=8`` to
the launch command to request eight samples per pixel; use a positive integer.
Drag the window border or maximize it to enlarge the view. Newton keeps the
render resolution fixed during window resizing, so enlargement scales the
image. Override ``env.sim.default_visualizer_cfg.window_width`` and
``env.sim.default_visualizer_cfg.window_height`` in the Lab command to choose
another render size. Mouse controls orbit, pan and zoom the camera.
No recording or output directory is required.

Assets and physical configuration
---------------------------------

Large generated assets are external to the source package. The reference cake
and pedestal geometry were generated procedurally, and the Gaussian appearance
was fitted to renders of that geometry. Share the inputs as a versioned asset
bundle alongside setup instructions.
The reference physics USD requires ``stand_mesh.usdc`` and the referenced
``textures/color_3F4738.exr`` beside it and contains
``/World/Cake`` particle layers, ``/World/Cherry`` with a sphere collider and
``/World/Stand`` with a solid mesh collider. The Gaussian USD contains
``/World/Cake`` as ``ParticleField3DGaussianSplat``, radiance materials under
``/World/Looks`` and ``cake:component_ids`` matching the physical layers.

The task imports authored positions, masses, widths and constitutive materials.
A temporary flattened USD adapts paths to the existing MPMObject geometry
convention. Reset restores native material and coupled-solver history.
Gaussian skinning consumes particle positions and native material frames;
it does not change physical motion. Pending rendering finishes before reset
or buffer reuse, and renderer bindings release before renderer destruction.
Camera-only redraws retain the published field until physics advances or an
episode reset invalidates it. This lets RTX temporal reconstruction converge
instead of repeatedly invalidating it with unchanged Gaussian array uploads.
Motion-related shimmer remains under investigation; increasing the requested
RTPT samples alone did not reduce it in the validated runtime.

Physics runs at 120 Hz with four physics ticks per Gym step and 30 Hz rendering.
The default reference uses 936 particles, a 26 mm grid,
60 solver iterations, S2 collider basis and a 1,024-cell sparse grid budget. Cherry mass, gap and horizontal
position are task configuration fields. Material values are qualitative cake
calibration. The reserved scalar action has no effect and reward is zero.
Multi-workcell execution is rejected until isolation and per-world reset are
verified. Robot decimation and multiple MPM entry substeps require additional
Gaussian callback scheduling support.

Validation
----------

With the external reference assets selected, run the standard RTX publication
and reset test:

.. code-block:: bash

   uv run --no-sync python -m pytest \
       source/isaaclab_tasks/test/contrib/test_cake_smash_newton_rtx.py -q

The RTX test checks visible cake publication, deformation without rendering
changing physics, manual reset behavior, cherry repositioning and GPU resource
lifetime. Benchmark campaigns and video composition tools are development
working material outside this runtime package.

The historical native contact/replay fixture separately requires the original
936-particle ``isaaclab_cake_all_firm_stand64_visuals.usdc`` selected through
``ISAACLAB_CAKE_PHYSICS_USD_PATH``. It retains that asset's original 26 mm grid
to compare the same authored response. Run it with
``python -m pytest source/isaaclab_tasks/test/contrib/test_cake_smash_native.py -q``.

Material softness
-----------------

Append ``env.material_strength_scale=0.8`` to the ordinary Lab launch command
to soften the authored cake material. The default ``1.0`` preserves the USD
material. Positive lower values scale Young's modulus, yield pressure, yield
stress and elastic damping together; mass, density, geometry, friction and
viscosity stay authored. This setting applies at startup, so restart the task
after changing it. Keep the original 936-particle asset paired with the default 26 mm grid.
The experimental 26,832-particle asset requires an 8 mm grid and a 32,768-cell
sparse budget, configured through ``env.sim.physics.solver_cfg.entries.1.solver_cfg.voxel_size``
and ``env.sim.physics.solver_cfg.entries.1.solver_cfg.max_active_cell_count``.
