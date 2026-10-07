# G1 targeted shot-put experiment

This experiment trains a Unitree G1 to release a physically simulated ball toward
sampled ground targets. The ball starts at the right articulated hand. It is not
attached to the robot; release and flight result from contact and joint control.
Robot self-collision is enabled. The default physics backend is Newton/MJWarp.
PhysX collision settings are configured, but PhysX training has not been validated.

Two Gym tasks are registered:

| Task | Base | Controls | Episode | Default objective |
| --- | --- | --- | --- | --- |
| `Isaac-Targeted-Shot-Put-Direct` | Fixed | 12 right arm/hand joints | 3 s | First-impact target accuracy |
| `Isaac-Standing-Shot-Put-Direct` | Free | 30 arm/hand/leg/torso joints | 6 s | Hold the ball while remaining upright |

Fixed-base targets span 0.5–2 m in the front ±60° sector. Standing targets span
0.5–0.75 m in the front ±30° sector and remain fixed if the robot moves.
The standing task defaults to `env.hold_only=true`. Setting it to false enables
an experimental throwing/recovery objective, requiring two continuous seconds
of settled balance after first impact. This stage has not passed the promotion gate.

## Training

Run from the repository root with the normal Isaac Lab frontend. This branch pins
PyTorch 2.11 / torchvision 0.26 to CUDA 12.8, the stack used for the reported runs.
The regenerated `uv.lock` records that compatibility stack.

```bash
uv run --frozen isaaclab train_multigpu --num_gpus 2 \
  --task Isaac-Targeted-Shot-Put-Direct --num_envs 4096 --max_iterations 3000 \
  agent.experiment_name=g1_front_accuracy_self_collision \
  agent.actor.distribution_cfg.init_std=0.35 agent.algorithm.entropy_coef=0.001

uv run --frozen isaaclab train_multigpu --num_gpus 2 \
  --task Isaac-Standing-Shot-Put-Direct --num_envs 4096 --max_iterations 3000 \
  agent.experiment_name=g1_front_standing_self_collision \
  agent.actor.distribution_cfg.init_std=0.25 agent.algorithm.entropy_coef=0.001
```

Each command uses 4096 environments per GPU. Single-GPU runs use `isaaclab train`
with the same task and agent overrides. On hosts with restricted GPU peer access,
the validated runs used `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`.

The fixed-base ball mass starts at 1.5 kg. Its uniformly sampled upper bound
ramps to 2.5 kg over 16,000 policy steps (1000 iterations at 16 rollout steps).
Standing holding uses a fixed 1.5 kg ball. The mass curriculum counter is not
restored from checkpoints; use `env.mass_curriculum_initial_steps=<offset>`
when resuming, or `16000` to evaluate the full range immediately.

## Evaluation snapshot

The following results are from the October 7, 2026 runs, checkpoint iteration
2999, using deterministic inference with seed 9731 and one completed episode
per environment across 2048 environments. Self-collision was enabled throughout.

| Stage | Result |
| --- | --- |
| Fixed-base throwing, 1.5–2.5 kg | 90.87% within 5 cm; 99.37% within 15 cm; mean miss 2.78 cm; p90 miss 4.84 cm |
| Standing holding, 1.5 kg | 94.87% held for the full six seconds and ended balanced; 0% falls; mean duration 5.78 s |

Hit rates include only valid throws: upward release after at least six policy
steps, horizontal release speed ≥0.4 m/s, and flight ≥0.12 s. Miss is measured
at interpolated first ground impact; subsequent rolling or bouncing cannot
improve the score. Standing holding terminates on release or a fall. Balance
requires bounded tilt, linear velocity, and angular velocity.

Training reset-batch metrics are diagnostics, not substitutes for completed-episode
evaluation. Checkpoints, videos, and generated run artifacts are not included
in this source branch. The snapshot documents existing evaluated policies;
training commands may produce different results with different random seeds.

## Limits and progression gates

The robot and ball use a fixed initial pose, without disturbances or learned
pickup. Arm limits (25 Nm), finger limits (5 Nm), and standing leg limits (60 Nm)
are provisional simulation settings and are not calibrated hardware claims.
The 3.5 cm ball radius and 1.5–2.5 kg masses are task assumptions.

Standing holding remains below the 99% success gate. Standing throwing,
walking, disturbed balance, hardware transfer, and competition shot-put rules
remain unvalidated. Promotion must be based on completed-episode evaluation
and inspection of whole-body contact/recovery videos.
