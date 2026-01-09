#!/usr/bin/env python
"""
Parallel version of evaluate_initial_states.py

Evaluates policy success rate across random initial object placements using
multiple processes to speed up evaluation.

Usage:
    python evaluate_initial_states_parallel.py \
        --checkpoint /path/to/model.pth \
        --n_samples 100 \
        --n_trials 5 \
        --num_workers 8
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import multiprocessing as mp
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils


class StateSampler:
    """
    Simple random sampler for object placements with collision checking.
    """

    def __init__(self, env, fixed_robot_qpos=None):
        """
        Args:
            env: robomimic environment wrapper
            fixed_robot_qpos: Fixed robot joint angles. If None, uses current state.
        """
        self.env = env
        self.base_env = env.base_env if hasattr(env, 'base_env') else env

        # Extract constraints from robosuite
        if hasattr(self.base_env, 'placement_initializer'):
            sampler = self.base_env.placement_initializer
            self.x_range = sampler.x_range
            self.y_range = sampler.y_range
            self.z = sampler.reference_pos[2] + sampler.z_offset
        else:
            # Fallback defaults
            self.x_range = [-0.1, 0.1]
            self.y_range = [-0.1, 0.1]
            self.z = 0.82

        # Query MuJoCo model structure to find indices
        self._find_state_indices()

        # Get or set fixed robot configuration
        if fixed_robot_qpos is not None:
            self.robot_qpos = fixed_robot_qpos
        else:
            obs = self.env.reset()
            state = self.env.get_state()
            self.robot_qpos = state['states'][self.robot_qpos_start:self.gripper_qpos_start].copy()

        self.gripper_qpos = np.array([0.020833, -0.020833])

    def _find_state_indices(self):
        """
        Query MuJoCo model to find the correct indices for robot, gripper, and object.
        This ensures we're robust to different robot models and configurations.
        """
        sim = self.base_env.sim

        # The flattened state vector from robosuite has structure:
        # [time_offset(1), qpos(nq), qvel(nv)]
        # But get_state().flatten() gives: [qpos(nq), qvel(nv), time(1)]
        # Let's verify by checking the actual structure

        # Find object joint (typically a free joint for the cube)
        obj_joint_idx = None
        obj_joint_qpos_addr = None

        for i in range(sim.model.njnt):
            jnt_name = sim.model.joint_id2name(i)
            if 'cube' in jnt_name.lower() or 'object' in jnt_name.lower():
                obj_joint_idx = i
                obj_joint_qpos_addr = sim.model.jnt_qposadr[i]
                break

        if obj_joint_qpos_addr is None:
            # Fallback: assume standard structure
            print("Warning: Could not find object joint, using default indices")
            self.robot_qpos_start = 0
            self.robot_qpos_end = 7
            self.gripper_qpos_start = 7
            self.gripper_qpos_end = 9
            self.obj_pos_start = 9
            self.obj_pos_end = 12
            self.obj_quat_start = 12
            self.obj_quat_end = 16
            self.qvel_start = 16
        else:
            # Object is at obj_joint_qpos_addr (position = 3 DOF, quat = 4 DOF for free joint)
            self.obj_pos_start = obj_joint_qpos_addr
            self.obj_pos_end = obj_joint_qpos_addr + 3
            self.obj_quat_start = obj_joint_qpos_addr + 3
            self.obj_quat_end = obj_joint_qpos_addr + 7

            # Find gripper joints (look for gripper in name)
            gripper_start = None
            for i in range(sim.model.njnt):
                jnt_name = sim.model.joint_id2name(i)
                if 'gripper' in jnt_name.lower():
                    gripper_start = sim.model.jnt_qposadr[i]
                    break

            if gripper_start is not None:
                self.gripper_qpos_start = gripper_start
                self.gripper_qpos_end = self.obj_pos_start  # Gripper ends where object begins
                self.robot_qpos_start = 0
                self.robot_qpos_end = gripper_start
            else:
                # Fallback
                self.robot_qpos_start = 0
                self.robot_qpos_end = obj_joint_qpos_addr - 2  # Assume 2 gripper joints
                self.gripper_qpos_start = obj_joint_qpos_addr - 2
                self.gripper_qpos_end = obj_joint_qpos_addr

            # qvel starts after qpos
            self.qvel_start = sim.model.nq

    def sample_random(self, n_samples=1, check_collision=True, max_attempts=100):
        """
        Sample random valid states.

        Args:
            n_samples: Number of valid samples to generate
            check_collision: If True, validate with physics simulation
            max_attempts: Max attempts per sample if collision checking enabled

        Returns:
            Array of shape (n_samples, 3) with [x, y, theta] states
        """
        valid_samples = []

        for _ in range(n_samples):
            attempts = 0
            while attempts < max_attempts:
                # Sample uniformly
                x = np.random.uniform(*self.x_range)
                y = np.random.uniform(*self.y_range)
                theta = np.random.uniform(0, 2*np.pi)

                # Check collision if requested
                if check_collision:
                    if self._check_collision(x, y, theta):
                        valid_samples.append([x, y, theta])
                        break
                    attempts += 1
                else:
                    valid_samples.append([x, y, theta])
                    break

            if attempts >= max_attempts:
                print(f"Warning: Failed to find valid sample after {max_attempts} attempts")

        return np.array(valid_samples)

    def _check_collision(self, x, y, theta):
        """Check if state is collision-free."""
        # Create state
        state_dict = self.create_state_dict(x, y, theta)

        # Apply to simulation
        self.base_env.sim.set_state_from_flattened(state_dict['states'])
        self.base_env.sim.forward()

        # Check contacts
        n_contacts = self.base_env.sim.data.ncon

        if n_contacts == 0:
            return True

        # Check if any contact is problematic (robot-object collision, not gripper)
        for i in range(n_contacts):
            contact = self.base_env.sim.data.contact[i]
            geom1_name = self.base_env.sim.model.geom_id2name(contact.geom1)
            geom2_name = self.base_env.sim.model.geom_id2name(contact.geom2)

            is_robot = 'robot0' in geom1_name or 'robot0' in geom2_name
            is_gripper = 'gripper' in geom1_name or 'gripper' in geom2_name
            is_object = any(key in geom1_name or key in geom2_name
                          for key in ['cube', 'object', 'Box'])

            # Robot (not gripper) colliding with object is invalid
            if is_robot and is_object and not is_gripper:
                return False

        return True

    def create_state_dict(self, x, y, theta):
        """
        Create state dictionary for env.reset_to().

        Args:
            x, y: Object position
            theta: Object rotation (radians)

        Returns:
            Dict with 'states' key containing full MuJoCo state
        """
        # Get template
        obs = self.env.reset()
        template_state = self.env.get_state()
        state_vector = template_state['states'].copy()

        # Set robot joints
        state_vector[self.robot_qpos_start:self.robot_qpos_end] = self.robot_qpos

        # Set gripper joints
        state_vector[self.gripper_qpos_start:self.gripper_qpos_end] = self.gripper_qpos

        # Set object position
        state_vector[self.obj_pos_start] = x
        state_vector[self.obj_pos_start + 1] = y
        state_vector[self.obj_pos_start + 2] = self.z

        # Set object orientation (quaternion from theta, MuJoCo uses [w,x,y,z])
        state_vector[self.obj_quat_start] = np.cos(theta / 2)      # w
        state_vector[self.obj_quat_start + 1] = 0.0                # x
        state_vector[self.obj_quat_start + 2] = 0.0                # y
        state_vector[self.obj_quat_start + 3] = np.sin(theta / 2)  # z

        # Zero velocities
        state_vector[self.qvel_start:-1] = 0.0
        state_vector[-1] = 0.0  # time (last element)

        return {"states": state_vector}


def evaluate_single_state_worker(args):
    """
    Worker function to evaluate a single state.
    This runs in a separate process.

    Args:
        args: tuple of (state_idx, state, checkpoint_path, horizon, n_trials, device_id)

    Returns:
        tuple of (state_idx, state, success_rate)
    """
    state_idx, state, checkpoint_path, horizon, n_trials, device_id = args

    # Suppress stdout/stderr in worker processes
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    # Set device for this worker
    if device_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

    # Load policy and environment in this worker
    device = TorchUtils.get_torch_device(try_to_use_cuda=(device_id is not None))

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=checkpoint_path,
        device=device,
        verbose=False
    )

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        render=False,
        render_offscreen=False,
        verbose=False
    )

    # Create state dict from state
    sampler = StateSampler(env)
    x, y, theta = state
    state_dict = sampler.create_state_dict(x, y, theta)

    # Evaluate
    successes = []
    for _ in range(n_trials):
        obs = env.reset_to(state_dict)
        policy.start_episode()
        success = False

        for t in range(horizon):
            action = policy(ob=obs)
            obs, reward, done, info = env.step(action)

            if env.is_success()["task"]:
                success = True
                break

            if done:
                break

        successes.append(success)

    success_rate = np.mean(successes)

    return (state_idx, state, success_rate)


def create_heatmap(states, success_rates, x_range, y_range, output_path):
    """Create heatmap visualization of success rates."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Scatter plot
    ax = axes[0]
    scatter = ax.scatter(states[:, 0], states[:, 1], c=success_rates,
                        s=100, cmap='RdYlGn', vmin=0, vmax=1,
                        edgecolors='black', linewidth=0.5, alpha=0.8)

    ax.set_xlabel('X position (m)', fontsize=12)
    ax.set_ylabel('Y position (m)', fontsize=12)
    ax.set_title('Success Rate vs. Object Position', fontsize=14, fontweight='bold')
    ax.set_xlim(x_range[0] - 0.01, x_range[1] + 0.01)
    ax.set_ylim(y_range[0] - 0.01, y_range[1] + 0.01)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Success Rate', fontsize=11)

    # Draw boundary
    rect = patches.Rectangle(
        (x_range[0], y_range[0]),
        x_range[1] - x_range[0],
        y_range[1] - y_range[0],
        linewidth=2, edgecolor='blue', facecolor='none',
        linestyle='--', label='Valid region'
    )
    ax.add_patch(rect)
    ax.legend()

    # Plot 2: Histogram
    ax = axes[1]
    ax.hist(success_rates, bins=20, edgecolor='black', color='skyblue', alpha=0.7)
    ax.set_xlabel('Success Rate', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Success Rates', fontsize=14, fontweight='bold')
    ax.axvline(success_rates.mean(), color='red', linestyle='--',
              linewidth=2, label=f'Mean: {success_rates.mean():.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    plt.close()


def run_evaluation(args):
    """Main evaluation loop with parallelization."""

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Parallel Policy Evaluation Across Initial Object Placements")
    print("=" * 70)

    # Load environment once to get constraints
    print("\n[1/5] Loading environment to extract constraints...")
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    _, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.checkpoint,
        device=device,
        verbose=False
    )

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        render=False,
        render_offscreen=False,
        verbose=False
    )

    print(f"  ✓ Environment: {env.name}")

    # Get horizon
    if args.horizon is None:
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        horizon = config.experiment.rollout.horizon
    else:
        horizon = args.horizon
    print(f"  ✓ Rollout horizon: {horizon}")

    # Initialize sampler
    print("\n[2/5] Initializing state sampler...")
    sampler = StateSampler(env)
    print(f"  ✓ Valid region: x ∈ {sampler.x_range}, y ∈ {sampler.y_range}")
    print(f"  ✓ Object height: z = {sampler.z:.3f}")
    print(f"  ✓ Fixed robot config: {sampler.robot_qpos[:3]} ...")

    # Sample states
    print(f"\n[3/5] Sampling {args.n_samples} initial states...")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    t_start = time.time()
    states = sampler.sample_random(
        n_samples=args.n_samples,
        check_collision=not args.skip_collision_check,
        max_attempts=args.max_collision_attempts
    )
    t_elapsed = time.time() - t_start

    print(f"  ✓ Sampled {len(states)} valid states in {t_elapsed:.1f}s")
    if len(states) < args.n_samples:
        print(f"  ⚠ Warning: Only found {len(states)}/{args.n_samples} valid states")

    # Prepare worker arguments
    print(f"\n[4/5] Evaluating policy with {args.num_workers} workers...")
    print(f"  ({args.n_trials} trials per state, {len(states)} states total)")

    # Determine device assignment
    if args.gpu_ids is not None:
        gpu_ids = args.gpu_ids
        device_assignment = [gpu_ids[i % len(gpu_ids)] for i in range(args.num_workers)]
    else:
        device_assignment = [None] * args.num_workers

    # Create arguments for each state
    worker_args = [
        (i, state, args.checkpoint, horizon, args.n_trials, device_assignment[i % args.num_workers])
        for i, state in enumerate(states)
    ]

    # Run parallel evaluation
    t_start = time.time()

    if args.num_workers == 1:
        # Sequential for debugging
        results = [evaluate_single_state_worker(arg) for arg in worker_args]
    else:
        # Parallel
        with mp.Pool(processes=args.num_workers) as pool:
            results = []
            for i, result in enumerate(pool.imap_unordered(evaluate_single_state_worker, worker_args)):
                results.append(result)
                if (i + 1) % max(1, len(states) // 10) == 0:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed
                    eta = (len(states) - i - 1) / rate
                    print(f"  [{i+1:3d}/{len(states)}] {rate:.1f} states/s | ETA: {eta:.0f}s")

    t_elapsed = time.time() - t_start

    # Sort results by index
    results.sort(key=lambda x: x[0])

    # Extract data
    evaluated_states = np.array([r[1] for r in results])
    success_rates = np.array([r[2] for r in results])

    print(f"  ✓ Completed in {t_elapsed:.1f}s ({len(states) / t_elapsed:.2f} states/s)")
    print(f"  ✓ Total rollouts: {len(states) * args.n_trials}")

    # Create visualizations
    print("\n[5/5] Creating visualizations...")
    create_heatmap(
        evaluated_states, success_rates,
        sampler.x_range, sampler.y_range,
        output_dir / 'success_heatmap.png'
    )

    # Save raw data
    np.savez(
        output_dir / 'evaluation_results.npz',
        states=evaluated_states,
        success_rates=success_rates,
        x_range=sampler.x_range,
        y_range=sampler.y_range,
        z=sampler.z,
    )
    print(f"  ✓ Saved data to {output_dir / 'evaluation_results.npz'}")

    # Save config
    config = vars(args)
    config['x_range'] = sampler.x_range
    config['y_range'] = sampler.y_range
    config['z'] = sampler.z
    config['evaluation_time_seconds'] = t_elapsed
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Saved config to {output_dir / 'config.json'}")

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total states evaluated:  {len(evaluated_states)}")
    print(f"Total rollouts:          {len(evaluated_states) * args.n_trials}")
    print(f"Evaluation time:         {t_elapsed:.1f}s ({len(evaluated_states) / t_elapsed:.2f} states/s)")
    print(f"Workers used:            {args.num_workers}")
    print(f"\nSuccess Rate Statistics:")
    print(f"  Mean:   {success_rates.mean():.3f}")
    print(f"  Std:    {success_rates.std():.3f}")
    print(f"  Min:    {success_rates.min():.3f}")
    print(f"  Max:    {success_rates.max():.3f}")
    print(f"  Median: {np.median(success_rates):.3f}")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    # Required for multiprocessing on some systems
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(
        description="Parallel evaluation of policy across random initial object placements"
    )

    # Required
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained policy checkpoint (.pth)"
    )

    # Sampling
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of random states to sample and evaluate"
    )
    parser.add_argument(
        "--skip_collision_check",
        action="store_true",
        help="Skip collision checking (faster but may sample invalid states)"
    )
    parser.add_argument(
        "--max_collision_attempts",
        type=int,
        default=100,
        help="Max attempts to find collision-free state"
    )

    # Evaluation
    parser.add_argument(
        "--n_trials",
        type=int,
        default=5,
        help="Number of trials per state to estimate success rate"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Rollout horizon (default: from checkpoint config)"
    )

    # Parallelization
    parser.add_argument(
        "--num_workers",
        type=int,
        default=mp.cpu_count(),
        help=f"Number of parallel workers (default: {mp.cpu_count()})"
    )
    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs='+',
        default=None,
        help="GPU IDs to use (e.g., --gpu_ids 0 1 2). If not specified, uses CPU."
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()
    run_evaluation(args)
