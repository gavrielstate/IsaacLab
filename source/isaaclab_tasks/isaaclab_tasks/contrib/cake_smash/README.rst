Native layered cake crushing
============================

``IsaacContrib-Cake-Smash-Direct`` is a single-workcell Isaac Lab Gym demonstration.
An initially stationary solid 2 kg cherry falls 30 mm onto an off-center region
of the cake. The stand stays a solid mesh. Nine separately authored cake layers
are native ``MPMObject`` assets with heterogeneous masses and constitutive
materials, coupled to ordinary rigid dynamics through
``CouplerProxyCfg`` with physical proxy mass scale 1.0.

The task uses unmodified Newton ``SolverImplicitMPM``, ``SolverMuJoCo`` on the
GPU and ``SolverCoupledProxy``. It contains no particle motion, fracture,
damage or seam-release kernels. Dropping is an initial condition; the reserved
scalar action has no effect and the reward is zero. This is a reproducible
physics demonstration, not a trained robot policy. Multi-workcell execution is
rejected until isolation and per-world reset are verified.

External assets
---------------

Set ``ISAACLAB_CAKE_PHYSICS_USD_PATH`` or pass ``physics_asset_path`` explicitly.
The current reference asset is
``tradeoff_p936_h120_gap03_sparse_i60_c1024.usda``. It requires the referenced
``stand_mesh.usdc`` alongside it. Large generated binary assets are not included
in Isaac Lab source; public redistribution and hosting must be agreed before
shipping. The source schema contract is ``/World/Cake`` with separate
``UsdGeom.Points`` components, ``/World/Cherry`` with a sphere collider and
``/World/Stand`` with a solid mesh collider.

At startup a temporary flattened USD file adapts component prim paths to the
existing MPMObject ``geometry/points`` convention. Particle position, mass,
width, per-layer material binding and initial plastic volume strain are copied
from USD. Cherry mass, gap and horizontal offset are ordinary configurable
physical initial conditions. Reset restores positions, velocities and native
MPM/rigid solver history; it does not retain prior cake deformation.

Run camera-free physics
-----------------------

From the Isaac Lab repository root:

.. code-block:: bash

   uv run --no-sync python -m isaaclab_tasks.contrib.cake_smash.demo \
       --physics-asset /path/to/tradeoff_p936_h120_gap03_sparse_i60_c1024.usda \
       --device cuda:0 --seconds 5 --trials 3 --output /tmp/cake_physics.json

Use ``--gym-step`` to include Gym observation, reward, termination and task
bookkeeping. Use ``--gravity-only`` to move the cherry outside the workcell and
measure the cake's response under its own weight. ``--physics-hz``,
``--iterations``, ``--capacity``, ``--mass``, ``--gap`` and ``--offset`` select
native configurations and initial conditions. ``--physics-hz`` must be a positive multiple of 30 Hz.

Physics timing excludes startup, compilation, CUDA graph capture, reset,
rendering and diagnostics. Particle state remains on the GPU during the timed
loop, with one explicit synchronization at the end. The simulation context also
reads Newton's latest and accumulated sparse-grid safety status arrays
(one uint32 each) after every physics tick, synchronizing the device.
Use ``--graph-replay`` for pure native graph timing with that safety status
checked only after the loop; this mode has no CPU data readback per tick. The report checks native solver
types, layer particle counts, cake mass, finite particles, exact state reset
and repeated response. Repeated long episodes are not claimed bitwise
deterministic: late fragment contacts can amplify tiny floating-point
differences. Separate the physics-only rate from end-to-end rendering
and capture performance.

The current passive environment has no actuated articulation, so
``NewtonManager.handles_decimation()`` returns False. One simulation-context
step advances **one** 120 Hz physics tick; one Gym step advances four ticks.
Benchmarking 150 simulation-context calls would simulate only 1.25 s, not 5 s.
The benchmark uses 600 calls for five seconds. A post-step visual material-frame
callback also runs at every physics tick in this configuration. Adding an
actuated robot or changing MPM entry substeps requires revisiting that cadence.

The 1024-cell sparse budget is validated only for the reference episode and
must be increased for motions with greater spread. Material values are
qualitative cake calibration, not food measurements. Gravity-only checks
remain important: finite particles and a fast benchmark do not establish that
a cake stands still or looks realistic.

Focused physical/reset regression
---------------------------------

.. code-block:: bash

   ISAACLAB_CAKE_PHYSICS_USD_PATH=/path/to/reference.usda \
       uv run --no-sync python -m pytest source/isaaclab_tasks/test/contrib/test_cake_smash_native.py -q

The test protects ordinary reciprocal collision and replay after a crushing episode. It is skipped when the external reference asset is absent.
Gaussian skinning/rendering is an independent consumer of these native
positions and material frames; it must not alter the physics state.

Gaussian preparation and live rendering
---------------------------------------

The external Gaussian USD must contain ``/World/Cake`` as a native
``ParticleField3DGaussianSplat`` with ``cake:component_ids`` matching the
nine physical component names. ``/World/Looks`` contains authored radiance
materials. Decorative Gaussians with negative component IDs bind to their
nearest physical layer; named layer supports stay within that component.
The current degree-zero field includes both exterior and interior Gaussians.

Validate skinning, native display frames and reset without a renderer:

.. code-block:: bash

   uv run --no-sync python -m isaaclab_tasks.contrib.cake_smash.validate_skinning \
       --physics-asset /path/to/reference.usda \
       --gaussian-asset /path/to/cake_live_ovrtx.usdc \
       --output /tmp/cake_skinning.json

This preparation check is independent of rendered visibility. Live cake
Gaussian deformation and solid stand/cherry rendering have also been verified
with OVRTX 0.6 and OVStage 0.3. Install an authorized wheel pair with Nicolas'
berry setup script and preserve those builds with ``uv run --no-sync``:

.. code-block:: bash

   bash source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/setup/setup.sh \
       --renderer-wheels /path/to/wheelhouse
   export UV_PROJECT_ENVIRONMENT="$PWD/.venv-tasks"

Run the live native viewer:

.. code-block:: bash

   uv run --no-sync python -m isaaclab_tasks.contrib.cake_smash.render_demo \
       --physics-asset /path/to/reference.usda \
       --gaussian-asset /path/to/cake_live_ovrtx.usdc \
       --seconds 5 --width 960 --height 720 --video --verify-render --output /tmp/cake_render

Open the native RTX viewer from a graphics desktop:

.. code-block:: bash

   uv run --no-sync python -m isaaclab_tasks.contrib.cake_smash.render_demo \
       --physics-asset /path/to/reference.usda \
       --gaussian-asset /path/to/cake_live_ovrtx.usdc \
       --window --paused --lighting front --offset 0.09 0.02

Space toggles pause, period steps one display frame, and the native Reset
button restarts physics and clears renderer history. H hides/shows the native
panels, and native mouse controls orbit or zoom the camera. The final state
remains available after ``--seconds`` until reset or window closure. Window
mode needs no output directory. Use the headless command separately for video
and array verification. Direct CUDA/OpenGL presentation uses an NVIDIA graphics
context on the simulation GPU. If texture registration is unavailable (for
example, a software Xvfb/llvmpipe display), the task automatically presents the
finished RTX color image through CPU upload in a pyglet window, retaining
Newton's native GUI/camera controls. Physics, skinning and RTX rendering remain
on GPU. This presentation fallback copies only the finished RGBA image and adds
display cost. Window performance is separate from headless measurements.

``--physics-hz`` selects 30, 60, 90 or 120 Hz, and ``--render-fps`` selects
15 or 30 frames/s. The default is 120 Hz physics with 30 frames/s rendering;
changing the display rate does not lower physics frequency. ``--async-render``
uses OVRTX's existing asynchronous renderer/data-access path to overlap rendering
with subsequent physics. Reported loop time includes the final renderer
completion wait. ``--width``, ``--height`` and ``--samples`` control rendering
cost; ``--iterations`` and ``--capacity`` control the native solver. Recheck
gravity support and impact behavior when changing physics frequency or solve
accuracy. Skinning uses Nicolas' existing affine4 binding and native MPM frames.
``--antialiasing quality`` selects native DLSS Quality upscaling; ``default``
preserves inherited settings and ``dlaa`` selects full-resolution DLAA. Quality
can soften small crumbs and curls. ``--collider-basis Q1`` is an experimental
native alternative; the validated reference retains S2.
The camera includes the full cherry stem. ``--lighting front`` selects the
native dome/distant light rig aimed at the cut face; ``studio`` remains the
default. These choices change appearance only.

The full field contains 620,171 Gaussians. An external 2 mm surface / 4 mm interior
candidate retains 267,055 Gaussians, including 75,944 interior Gaussians and
all decorative curls. Paired with a profile-preserving 64-segment solid stand
collider, 120 Hz physics, 60 native iterations, 640x480 output, 15 fps and DLSS
Quality with the current wide camera and front light, three warmed single-L40
trials take 4.953, 4.913 and 4.893 s for five simulated seconds (median 1.018x).
This has very little realtime margin. At 30 fps, one DLSS
Quality trial takes 5.896 s (0.848x). The original full-detail 960x720 reference
takes approximately 0.40x on the unconstrained two-L40 host. These are
physics/skinning/render rates excluding CPU video capture and encoding;
they are different presets, not isolated Gaussian-count comparisons.

Restrict ``CUDA_VISIBLE_DEVICES=0`` for a single-GPU measurement: the native
OVRTX default can use every visible GPU. Include final renderer completion
in timing. Video capture waits for each current frame and serializes overlap,
so a no-capture benchmark is not an encoded-video throughput claim. Assets
and physics tunings remain external; physics/solver defaults are unchanged.

``render_demo`` uses Newton's existing ``ViewerRTX`` and shared persistent
OVRTX array bindings. Particle and Gaussian arrays stay on GPU; local-frame
bounds read six scalars per rendered frame. Rigid transform submission uses
the native viewer path. Video capture copies rendered pixels to CPU and is
timed separately. The first render and initialization are excluded from the
reported warmed loop rate. No Blender renderer or trajectory playback is used.
``--verify-render`` reads back the final native Gaussian arrays outside loop
timing. It does not substitute for inspecting visible frames.

Construct the stream before the first physics step so its native display-frame
callback is included in graph capture. The adapter currently rejects folded
robot decimation and multiple MPM substeps. Degree-zero cake radiance needs
no berry SH damage shader. Check persistent array readback and visible motion
when changing renderer builds; the 0.5 version guard must not be bypassed.
Isaac Lab sensor rendering requires separate validation from the standalone
native viewer.
