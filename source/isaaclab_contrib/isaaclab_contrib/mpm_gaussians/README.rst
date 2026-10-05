MPM objects with Gaussian appearance
====================================

This module shares Nicolas' berry-task skinning code and persistent OVRTX
array publication between tasks. It is a rendering consumer: it does not
advance particles, define a constitutive model, or implement fracture.

``binding.py`` and ``local_frame.py`` preserve the supplied numerical
implementations. ``make_binding`` selects affine, MLS, or fracture-aware MLS
visual bindings. A visual fracture binding is not a discontinuous MPM solver.
The original berry import paths remain compatibility shims.

Input contracts
----------------

Rest and current particle positions and Gaussian positions are in meters in
the same coordinate system. Gaussian rotations use XYZW quaternions. Binding
supports are restricted by the supplied physical and visual region IDs.
Particle ordering must remain stable for the binding lifetime. Native Newton
display frames must be updated at each actual MPM substep, with the matching
timestep; the rendering module does not schedule physics.

``GaussianLocalFrame`` computes a GPU bounds reduction, reads six scalars to
the CPU, and returns normalized geometry plus a compensating transform. It
does not copy the complete Gaussian arrays to CPU.

Persistent publication
-----------------------

Author native ``ParticleField3DGaussianSplat`` geometry and materials before
creating the renderer. Declare animated geometry attributes as time-varying
using the task's USD authoring code. After renderer creation, construct a
``GaussianArrayStream`` with explicit prim paths and attribute lane counts.
Publish one array per prim for every bound attribute. Warp CUDA arrays can be
passed directly through DLPack using OVRTX asynchronous data access.
Publication completion is awaited; this is distinct from render completion.
Callers retain the arrays until rendering
completes and own CUDA/renderer synchronization. Close the stream before
destroying the renderer.

Live Gaussian geometry requires OVRTX 0.6 and OVStage 0.3. Call
``require_live_gaussian_renderer`` before initialization. The older renderer
can return correctly updated arrays while rendering stale acceleration data;
array readback alone is not a visibility test. ``verify`` checks array equality
and finiteness only, outside the timed loop.
An integration must also inspect rendered frame continuity and motion.

``apply_sampling_settings`` optionally selects native DLAA or DLSS Quality
through the public RenderProduct token attributes. Default preserves inherited
anti-aliasing. Quality reduces internal rendering resolution and can soften
fine geometry; consumers own its visible-frame validation and performance
measurements. It does not change Gaussian opacity or physics.

Consumers and limits
--------------------

``franka_pick_berries`` retains its supplied explicit tissue physics and uses
this module for binding, local coordinates, settings and array publication.
Its asset-specific SH orientation and damage tint stay in the berry task.
``cake_smash`` consumes existing Newton implicit MPM particle positions and
native display frames. Its degree-zero exterior and interior Gaussians share
the same binding; the stand and cherry remain rigid meshes.

USD asset loading, camera setup, renderer lifecycle, sensors and task logic
remain with consumers. This package imports no task modules. Multiple worlds,
robot control with folded solver decimation, sensor integration and each new consumer's visible-frame correctness need
separate integration validation before shipping. The supplied berry squash
and native cake drop have been rendered with OVRTX 0.6 / OVStage 0.3.
