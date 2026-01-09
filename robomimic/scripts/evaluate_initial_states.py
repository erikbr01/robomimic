#!/usr/bin/env python
"""
Evaluate policy success rate across random initial object placements.

This script:
1. Loads a trained policy from a checkpoint
2. Randomly samples initial object positions (x, y, theta) from valid region
3. Validates each state using robosuite's collision detection
4. Evaluates policy at each valid state
5. Outputs a heatmap of success rate vs. initial object pose

Usage:
    python evaluate_initial_states.py --checkpoint /path/to/model.pth --n_samples 100 --n_trials 5
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import h5py
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

    Extracts valid region from robosuite's placement_initializer and validates
    sampled states using physics simulation.
    """

    def __init__(self, env, fixed_robot_qpos=None):
        """
        Args:
            env: robomimic environment wrapper
            fixed_robot_qpos: Fixed robot joint angles (7,). If None, uses current state.
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

        # Get or set fixed robot configuration
        if fixed_robot_qpos is not None:
            self.robot_qpos = fixed_robot_qpos
        else:
            obs = self.env.reset()
            state = self.env.get_state()
            self.robot_qpos = state['states'][:8].copy()  # Panda has 8 joints

        self.gripper_qpos = np.array([0.020833, -0.020833])

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

        # Set robot (Panda has 8 joints: 7 arm + 1 wrist)
        state_vector[0:8] = self.robot_qpos
        state_vector[8:10] = self.gripper_qpos

        # Set object position
        state_vector[10] = x
        state_vector[11] = y
        state_vector[12] = self.z

        # Set object orientation (quaternion from theta, MuJoCo uses [w,x,y,z])
        state_vector[13] = np.cos(theta / 2)  # w
        state_vector[14] = 0.0                 # x
        state_vector[15] = 0.0                 # y
        state_vector[16] = np.sin(theta / 2)  # z

        # Zero velocities (qvel starts at index 17)
        state_vector[17:32] = 0.0
        state_vector[31] = 0.0  # time

        return {"states": state_vector}


def evaluate_policy_at_state(policy, env, state_dict, horizon, n_trials=1):
    """
    Evaluate policy at a specific initial state.

    Args:
        policy: Loaded policy
        env: Environment
        state_dict: Initial state for reset_to()
        horizon: Rollout horizon
        n_trials: Number of trials to average

    Returns:
        success_rate: Float in [0, 1]
    """
    successes = []

    for _ in range(n_trials):
        # Reset to state
        obs = env.reset_to(state_dict)

        # Run rollout
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

    return np.mean(successes)


def create_heatmap(states, success_rates, x_range, y_range, output_path):
    """
    Create heatmap visualization of success rates.

    Args:
        states: Array of shape (n, 3) with [x, y, theta]
        success_rates: Array of shape (n,) with success rates
        x_range, y_range: Valid region bounds
        output_path: Where to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Scatter plot (average over theta)
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

    # Add colorbar
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

    # Plot 2: Histogram of success rates
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
    """Main evaluation loop."""

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Evaluating Policy Across Initial Object Placements")
    print("=" * 70)

    # Load policy and environment
    print("\n[1/5] Loading policy and environment...")
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.checkpoint,
        device=device,
        verbose=False
    )
    print(f"  ✓ Policy loaded from {args.checkpoint}")

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

    # Evaluate policy at each state
    print(f"\n[4/5] Evaluating policy ({args.n_trials} trials per state)...")
    success_rates = []

    for i, state in enumerate(states):
        x, y, theta = state

        # Create state dict
        state_dict = sampler.create_state_dict(x, y, theta)

        # Evaluate
        t_start = time.time()
        success_rate = evaluate_policy_at_state(
            policy, env, state_dict, horizon, n_trials=args.n_trials
        )
        t_elapsed = time.time() - t_start

        success_rates.append(success_rate)

        # Print progress
        if (i + 1) % 10 == 0 or (i + 1) == len(states):
            avg_success = np.mean(success_rates)
            print(f"  [{i+1:3d}/{len(states)}] "
                  f"x={x:6.3f}, y={y:6.3f}, θ={theta:5.2f} | "
                  f"success={success_rate:.2f} | "
                  f"avg={avg_success:.3f} | "
                  f"{t_elapsed:.1f}s")

    success_rates = np.array(success_rates)

    # Create visualizations
    print("\n[5/5] Creating visualizations...")
    create_heatmap(
        states, success_rates,
        sampler.x_range, sampler.y_range,
        output_dir / 'success_heatmap.png'
    )

    # Save raw data
    np.savez(
        output_dir / 'evaluation_results.npz',
        states=states,
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
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Saved config to {output_dir / 'config.json'}")

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total states evaluated:  {len(states)}")
    print(f"Total rollouts:          {len(states) * args.n_trials}")
    print(f"\nSuccess Rate Statistics:")
    print(f"  Mean:   {success_rates.mean():.3f}")
    print(f"  Std:    {success_rates.std():.3f}")
    print(f"  Min:    {success_rates.min():.3f}")
    print(f"  Max:    {success_rates.max():.3f}")
    print(f"  Median: {np.median(success_rates):.3f}")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate policy across random initial object placements"
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
