# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run or benchmark the native cake task without rendering.

Example:
    uv run python -m isaaclab_tasks.contrib.cake_smash.demo --physics-asset /path/to/cake.usda

Physics timing uses the environment's simulation context and its captured
native solver graph. Gym timing additionally includes observations, rewards,
termination checks and environment bookkeeping. Neither loop reads particle
positions back to the CPU or renders images.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton.solvers import SolverImplicitMPM, SolverMuJoCo
from newton.solvers.experimental.coupled import SolverCoupledProxy

from .cake_smash_env_cfg import CakeSmashEnvCfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-asset", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--physics-hz", type=int, default=120)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--mass", type=float, default=2.0)
    parser.add_argument("--gap", type=float, default=0.03)
    parser.add_argument("--offset", type=float, nargs=2, default=(0.055, 0.020))
    parser.add_argument("--gravity-only", action="store_true")
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--gym-step", action="store_true", help="Time Gym step, including task bookkeeping.")
    timing.add_argument("--graph-replay", action="store_true", help="Replay the captured native CUDA graph directly.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.physics_hz <= 0 or args.physics_hz % 30 != 0:
        raise ValueError("physics-hz must be a positive multiple of 30.")
    cfg = CakeSmashEnvCfg(
        physics_asset_path=args.physics_asset,
        cherry_mass=args.mass,
        drop_gap=args.gap,
        drop_offset=tuple(args.offset),
        gravity_only=args.gravity_only,
        episode_length_s=args.seconds + 1.0,
        decimation=args.physics_hz // 30,
    )
    cfg.sim.device = args.device
    cfg.sim.dt = 1.0 / args.physics_hz
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.physics.solver_cfg.entries[1].solver_cfg.max_iterations = args.iterations
    cfg.sim.physics.solver_cfg.entries[1].solver_cfg.max_active_cell_count = args.capacity
    interval = cfg.sim.dt * cfg.decimation if args.gym_step else cfg.sim.dt
    steps = round(args.seconds / interval)
    if steps <= 0 or args.trials <= 0:
        raise ValueError("seconds and trials must be positive.")
    env = gym.make("IsaacContrib-Cake-Smash-Direct", cfg=cfg).unwrapped
    try:
        model = NewtonManager.get_model()
        solver = NewtonManager._solver
        assert isinstance(solver, SolverCoupledProxy)
        assert isinstance(solver.solver("cake"), SolverImplicitMPM)
        assert isinstance(solver.solver("rigid"), SolverMuJoCo)
        action = torch.zeros((1, 1), device=args.device)
        env.reset(seed=42)
        initial_particles = wp.to_torch(env.particle_positions).clone()
        initial_cherry = env.cherry.data.root_state_w.torch.clone()
        # Compile and capture once. Reset starts every trial from the same asset.
        env.step(action)
        wp.synchronize_device(args.device)
        if args.graph_replay and env.sim.physics_manager.handles_decimation():
            raise RuntimeError("Direct graph timing requires one physics tick per captured graph.")
        timings = []
        finals = []
        final_cherries = []
        final_velocities = []
        for trial in range(args.trials):
            env.reset(seed=42)
            wp.synchronize_device(args.device)
            start = time.perf_counter()
            for _ in range(steps):
                if args.gym_step:
                    env.step(action)
                elif args.graph_replay:
                    wp.capture_launch(NewtonManager._graph)
                else:
                    env.sim.step(render=False)
            wp.synchronize_device(args.device)
            timings.append(time.perf_counter() - start)
            # Diagnostics and particle readback happen after the timing boundary.
            solver.solver("cake").check_sparse_grid_rebuild_status()
            env.scene.update(env.step_dt)
            finals.append(wp.to_torch(env.particle_positions).clone())
            final_velocities.append(wp.to_torch(NewtonManager.get_state_0().particle_qd).clone())
            final_cherries.append(env.cherry.data.root_state_w.torch.clone())
            print(f"Trial {trial + 1}: {timings[-1]:.6f} s", flush=True)
        finite = all(bool(torch.isfinite(state).all()) for state in finals)
        if not finite:
            raise RuntimeError("Native simulation produced non-finite particle positions.")
        displacement = (finals[0] - initial_particles).norm(dim=-1)
        replay_error = max(float((state - finals[0]).abs().max()) for state in finals)
        env.reset(seed=42)
        reset_error = float((wp.to_torch(env.particle_positions) - initial_particles).abs().max())
        cherry_reset_error = float((env.cherry.data.root_state_w.torch - initial_cherry).abs().max())
        if reset_error > 1.0e-6 or cherry_reset_error > 1.0e-6:
            raise RuntimeError("Reset did not restore the authored particle and rigid state.")
        layer_metrics = {}
        for name in env.component_names:
            start, count = next(
                interval
                for path, interval in NewtonManager.backend.particle_ranges.items()
                if f"/CakeLayers/{name}/" in path
            )
            difference = finals[0][start : start + count] - initial_particles[start : start + count]
            layer_metrics[name] = {
                "median_vertical_displacement_m": float(difference[:, 2].median()),
                "max_downward_displacement_m": float((-difference[:, 2]).clamp_min(0).max()),
                "max_displacement_m": float(difference.norm(dim=-1).max()),
            }
        particle_mass = wp.to_torch(model.particle_mass)
        particle_kinetic_energy = float(0.5 * (particle_mass * final_velocities[0].square().sum(-1)).sum())
        gravity = torch.tensor(cfg.sim.gravity, device=args.device)
        potential_change = float(-(particle_mass * ((finals[0] - initial_particles) * gravity).sum(-1)).sum())
        median = float(np.median(timings))
        simulated = steps * interval
        report = {
            "task": "IsaacContrib-Cake-Smash-Direct",
            "physics_asset": str(Path(args.physics_asset).resolve()),
            "newton_version": importlib.metadata.version("newton"),
            "warp_version": wp.__version__,
            "physics": "unmodified SolverCoupledProxy + SolverImplicitMPM + SolverMuJoCo (GPU)",
            "solver_entry_names": list(solver.entry_names()),
            "particle_count": model.particle_count,
            "layer_particle_counts": dict(
                zip(env.component_names, [x.particles_per_object for x in env.layers], strict=True)
            ),
            "cake_mass_kg": float(wp.to_torch(model.particle_mass).sum()),
            "cherry_mass_kg": args.mass,
            "proxy_mass_scale": cfg.sim.physics.solver_cfg.proxies[0].mass_scale,
            "device": args.device,
            "physics_hz": args.physics_hz,
            "rigid_substeps": 2,
            "iterations": args.iterations,
            "sparse_capacity": args.capacity,
            "gravity_only": args.gravity_only,
            "timing_mode": (
                "Gym step"
                if args.gym_step
                else "native CUDA graph replay"
                if args.graph_replay
                else "simulation context physics step"
            ),
            "physics_steps": round(simulated * args.physics_hz),
            "native_status_cpu_readback_per_step": not args.graph_replay,
            "rendering": False,
            "runtime_particle_cpu_readback": False,
            "startup_capture_reset_and_diagnostics_excluded": True,
            "simulated_seconds": simulated,
            "trial_wall_seconds": timings,
            "median_wall_seconds": median,
            "realtime_factor": simulated / median,
            "finite_particles": finite,
            "layer_displacements": layer_metrics,
            "final_particle_kinetic_energy_j": particle_kinetic_energy,
            "particle_gravity_potential_energy_change_j": potential_change,
            "final_cherry_translational_kinetic_energy_j": float(
                0.5 * args.mass * final_cherries[0][:, 7:10].square().sum()
            ),
            "maximum_particle_displacement_m": float(displacement.max()),
            "median_particle_displacement_m": float(displacement.median()),
            "same_seed_replay_max_position_difference_m": replay_error,
            "reset_particle_position_error_m": reset_error,
            "reset_cherry_state_error": cherry_reset_error,
            "initial_cherry_state": initial_cherry.cpu().tolist(),
            "final_cherry_state": final_cherries[0].cpu().tolist(),
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
