#!/usr/bin/env python
"""
Evaluate variance in performance estimation vs number of rollouts.

This script evaluates how the variance in success rate estimates decreases as the
number of rollouts increases. For each evaluation count M (from 1 to max_rollouts),
it runs N evaluations with different seeds and computes the mean and variance
of the estimated success rates.

Usage:
    python evaluate_variance_vs_rollouts.py \
        --checkpoint /path/to/model.pth \
        --max_rollouts 100 \
        --n_seeds 10 \
        --num_workers 8

Output:
    - A plot showing mean success rate and variance bands vs number of rollouts
    - Raw data saved as .npz file
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils


def qvel_width(joint_type: mujoco.mjtJoint) -> int:
    """Get the dimensionality of the joint in qvel."""
    return {0: 6, 1: 3, 2: 1, 3: 1}[joint_type]


def qpos_width(joint_type: mujoco.mjtJoint) -> int:
    """Get the dimensionality of the joint in qpos."""
    return {0: 7, 1: 4, 2: 1, 3: 1}[joint_type]


class StateSampler:
    """Simple random sampler for object placements with collision checking."""

    def __init__(self, env, fixed_robot_qpos=None, fixed_gripper_qpos=None):
        self.env = env
        self.base_env = env.base_env if hasattr(env, "base_env") else env

        if hasattr(self.base_env, "placement_initializer"):
            sampler = self.base_env.placement_initializer
            self.x_range = sampler.x_range
            self.y_range = sampler.y_range

            self.x_range[0] = self.x_range[0]
            self.x_range[1] = self.x_range[1]
            self.z = sampler.reference_pos[2] + sampler.z_offset
        else:
            self.x_range = [-0.1, 0.1]
            self.y_range = [-0.1, 0.1]
            self.z = 0.82

        self._find_state_indices()

        obs = self.env.reset()
        self.template_state = self.env.get_state()

        # Store fixed robot and gripper qpos - these will be the same across all seeds
        if fixed_robot_qpos is not None:
            self.robot_qpos = fixed_robot_qpos
        else:
            self.robot_qpos = self.template_state["states"][
                self.robot_qpos_start : self.robot_qpos_end
            ].copy()

        if fixed_gripper_qpos is not None:
            self.gripper_qpos = fixed_gripper_qpos
        else:
            self.gripper_qpos = self.template_state["states"][
                self.gripper_qpos_start : self.gripper_qpos_end
            ].copy()

    def _find_state_indices(self):
        sim = self.base_env.sim
        self.joint_map = {}

        for i in range(sim.model.njnt):
            jnt_name = sim.model.joint_id2name(i)
            jnt_id = sim.model.joint(jnt_name).id
            jnt_type = sim.model.jnt_type[jnt_id]

            qpos_addr = sim.model.jnt_qposadr[i]
            qpos_size = qpos_width(jnt_type)
            qvel_size = qvel_width(jnt_type)
            qvel_addr = sim.model.jnt_dofadr[i]

            self.joint_map[jnt_name] = {
                "type": jnt_type,
                "qpos_start": qpos_addr,
                "qpos_end": qpos_addr + qpos_size,
                "qpos_size": qpos_size,
                "qvel_start": qvel_addr,
                "qvel_end": qvel_addr + qvel_size,
                "qvel_size": qvel_size,
            }

        robot_joints = []
        gripper_joints = []
        object_joint = None

        for jnt_name, jnt_info in self.joint_map.items():
            if "cube" in jnt_name.lower() or "object" in jnt_name.lower():
                object_joint = jnt_name
            elif "gripper" in jnt_name.lower():
                gripper_joints.append(jnt_name)
            else:
                robot_joints.append(jnt_name)

        if object_joint is None:
            raise ValueError("Could not find object joint in MuJoCo model")

        if robot_joints:
            robot_joints.sort(key=lambda x: self.joint_map[x]["qpos_start"])
            self.robot_qpos_start = self.joint_map[robot_joints[0]]["qpos_start"]
            self.robot_qpos_end = self.joint_map[robot_joints[-1]]["qpos_end"]
        else:
            self.robot_qpos_start = 0
            self.robot_qpos_end = 0

        if gripper_joints:
            gripper_joints.sort(key=lambda x: self.joint_map[x]["qpos_start"])
            self.gripper_qpos_start = self.joint_map[gripper_joints[0]]["qpos_start"]
            self.gripper_qpos_end = self.joint_map[gripper_joints[-1]]["qpos_end"]
        else:
            self.gripper_qpos_start = self.robot_qpos_end
            self.gripper_qpos_end = self.robot_qpos_end

        obj_info = self.joint_map[object_joint]
        obj_qpos_start = obj_info["qpos_start"]

        if obj_info["type"] == 0:  # Free joint
            self.obj_pos_start = obj_qpos_start + 1
            self.obj_pos_end = obj_qpos_start + 4
            self.obj_quat_start = obj_qpos_start + 4
            self.obj_quat_end = obj_qpos_start + 8
        else:
            self.obj_pos_start = obj_qpos_start + 1
            self.obj_pos_end = obj_qpos_start + 1 + min(3, obj_info["qpos_size"])
            self.obj_quat_start = self.obj_pos_end
            self.obj_quat_end = obj_info["qpos_end"] + 1

        self.qvel_start = sim.model.nq + 1

    def sample_random(self, n_samples=1, check_collision=True, max_attempts=100):
        valid_samples = []

        while len(valid_samples) < n_samples:
            attempts = 0
            while attempts < max_attempts:
                attempts += 1
                x = np.random.uniform(*self.x_range)
                y = np.random.uniform(*self.y_range)
                theta = np.random.uniform(0, 2 * np.pi)

                if check_collision:
                    if not self._check_collision(x, y, theta):
                        continue

                valid_samples.append([x, y, theta])
                break

            if attempts >= max_attempts:
                print(
                    f"Warning: Failed to find valid sample after {max_attempts} attempts"
                )

        return np.array(valid_samples)

    def _check_collision(self, x, y, theta):
        state_dict = self.create_state_dict(x, y, theta)
        self.base_env.sim.set_state_from_flattened(state_dict["states"])
        self.base_env.sim.forward()

        n_contacts = self.base_env.sim.data.ncon
        if n_contacts == 0:
            return True

        for i in range(n_contacts):
            contact = self.base_env.sim.data.contact[i]
            geom1_name = self.base_env.sim.model.geom_id2name(contact.geom1)
            geom2_name = self.base_env.sim.model.geom_id2name(contact.geom2)

            is_robot = "robot0" in geom1_name or "robot0" in geom2_name
            is_gripper = "gripper" in geom1_name or "gripper" in geom2_name
            is_object = any(
                key in geom1_name or key in geom2_name
                for key in ["cube", "object", "Box"]
            )

            if is_robot and is_object and not is_gripper:
                return False

        return True

    def create_state_dict(self, x, y, theta, robot_qpos=None, gripper_qpos=None):
        state_vector = self.template_state["states"].copy()

        if robot_qpos is None:
            robot_qpos = self.robot_qpos
        state_vector[self.robot_qpos_start : self.robot_qpos_end] = robot_qpos

        if gripper_qpos is None:
            gripper_qpos = self.gripper_qpos
        state_vector[self.gripper_qpos_start : self.gripper_qpos_end] = gripper_qpos

        state_vector[self.obj_pos_start] = x
        state_vector[self.obj_pos_start + 1] = y
        state_vector[self.obj_pos_start + 2] = self.z

        state_vector[self.obj_quat_start] = np.cos(theta / 2)
        state_vector[self.obj_quat_start + 1] = 0.0
        state_vector[self.obj_quat_start + 2] = 0.0
        state_vector[self.obj_quat_start + 3] = np.sin(theta / 2)

        state_vector[self.qvel_start : -1] = 0.0
        state_vector[-1] = 0.0

        return {"states": state_vector}


def evaluate_seed_worker(args):
    """
    Worker function to evaluate a policy with a specific seed.
    Samples max_rollouts states and evaluates each one.

    Args:
        args: tuple of (seed, checkpoint_path, horizon, max_rollouts, n_trials, device_id,
                       fixed_robot_qpos, fixed_gripper_qpos)

    Returns:
        tuple of (seed, states, success_results) where:
            - states is array of shape (max_rollouts, 3) with (x, y, theta)
            - success_results is array of shape (max_rollouts,) with success values
    """
    (
        seed,
        checkpoint_path,
        horizon,
        max_rollouts,
        n_trials,
        device_id,
        fixed_robot_qpos,
        fixed_gripper_qpos,
    ) = args

    # Suppress stdout/stderr
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    # Set seed for this worker (only affects object placement randomization)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = TorchUtils.get_torch_device(try_to_use_cuda=(device_id is not None))

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=checkpoint_path, device=device, verbose=False
    )

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, render=False, render_offscreen=False, verbose=False
    )

    # Use fixed robot and gripper qpos across all seeds
    sampler = StateSampler(
        env, fixed_robot_qpos=fixed_robot_qpos, fixed_gripper_qpos=fixed_gripper_qpos
    )

    # Sample all states with this seed
    states = sampler.sample_random(n_samples=max_rollouts, check_collision=True)

    # Evaluate each state
    success_results = np.zeros(max_rollouts)

    for i, state in enumerate(states):
        x, y, theta = state
        state_dict = sampler.create_state_dict(x, y, theta)

        # Run n_trials and take majority vote (or just single trial if n_trials=1)
        trial_successes = []
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

            trial_successes.append(success)

        # Use mean success rate across trials
        success_results[i] = np.mean(trial_successes)

    return (seed, states, success_results)


def compute_cumulative_success_rates(success_results):
    """
    Compute cumulative success rate estimates for increasing number of rollouts.

    Args:
        success_results: array of shape (n_seeds, max_rollouts) with success values

    Returns:
        cumulative_rates: array of shape (n_seeds, max_rollouts) where
            cumulative_rates[seed, m] = mean success rate using first m+1 rollouts
    """
    n_seeds, max_rollouts = success_results.shape
    cumulative_rates = np.zeros((n_seeds, max_rollouts))

    for seed_idx in range(n_seeds):
        for m in range(max_rollouts):
            cumulative_rates[seed_idx, m] = np.mean(success_results[seed_idx, : m + 1])

    return cumulative_rates


def create_individual_seed_plots(
    all_states, all_success_results, seeds, x_range, y_range, output_dir
):
    """
    Create individual detailed plots for each seed showing position and theta vs success.

    Args:
        all_states: list of arrays, each of shape (max_rollouts, 3) with (x, y, theta)
        all_success_results: list of arrays, each of shape (max_rollouts,)
        seeds: list of seed values
        x_range, y_range: valid sampling regions
        output_dir: directory to save individual plots
    """
    seed_plots_dir = output_dir / "seed_plots"
    seed_plots_dir.mkdir(exist_ok=True)

    for seed, states, success_results in zip(seeds, all_states, all_success_results):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left plot: X-Y positions colored by success
        ax = axes[0]
        scatter = ax.scatter(
            states[:, 0],
            states[:, 1],
            c=success_results,
            s=100,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        ax.set_xlabel("X position (m)", fontsize=14)
        ax.set_ylabel("Y position (m)", fontsize=14)
        ax.set_title(f"Sampled Positions (X, Y)", fontsize=16, fontweight="bold")
        ax.set_xlim(x_range[0] - 0.02, x_range[1] + 0.02)
        ax.set_ylim(y_range[0] - 0.02, y_range[1] + 0.02)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Success Rate", fontsize=12)

        # Right plot: Theta vs Success
        ax = axes[1]
        scatter = ax.scatter(
            states[:, 2],
            success_results,
            c=success_results,
            s=100,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        ax.set_xlabel("Theta (radians)", fontsize=14)
        ax.set_ylabel("Success Rate", fontsize=14)
        ax.set_title(f"Theta vs Success Rate", fontsize=16, fontweight="bold")
        ax.set_xlim(-0.2, 2 * np.pi + 0.2)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # Add horizontal line at mean success rate
        mean_success = np.mean(success_results)
        ax.axhline(
            mean_success,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_success:.3f}",
        )
        ax.legend(fontsize=12)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Success Rate", fontsize=12)

        # Overall title
        fig.suptitle(
            f"Seed {seed} - Mean Success Rate: {mean_success:.3f} ({len(states)} samples)",
            fontsize=18,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()
        plt.savefig(
            seed_plots_dir / f"seed_{seed}_analysis.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

    print(f"Saved individual seed plots to {seed_plots_dir}")


def create_position_heatmaps(
    all_states, all_success_results, seeds, x_range, y_range, output_path
):
    """
    Create heatmaps showing sampled positions and success rates for each seed.

    Args:
        all_states: list of arrays, each of shape (max_rollouts, 3) with (x, y, theta)
        all_success_results: list of arrays, each of shape (max_rollouts,)
        seeds: list of seed values
        x_range, y_range: valid sampling regions
        output_path: path to save the plot
    """
    n_seeds = len(seeds)

    # Create grid layout - 2 rows per seed (positions + success heatmap)
    n_cols = min(5, n_seeds)
    n_rows = (n_seeds - 1) // n_cols + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_seeds == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (seed, states, success_results) in enumerate(
        zip(seeds, all_states, all_success_results)
    ):
        ax = axes[idx]

        # Scatter plot of positions colored by success
        scatter = ax.scatter(
            states[:, 0],
            states[:, 1],
            c=success_results,
            s=20,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            alpha=0.6,
            edgecolors="none",
        )

        ax.set_xlabel("X position (m)", fontsize=10)
        ax.set_ylabel("Y position (m)", fontsize=10)
        ax.set_title(
            f"Seed {seed}\nMean: {np.mean(success_results):.3f}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlim(x_range[0] - 0.01, x_range[1] + 0.01)
        ax.set_ylim(y_range[0] - 0.01, y_range[1] + 0.01)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Add colorbar
        plt.colorbar(scatter, ax=ax, label="Success")

    # Hide unused subplots
    for idx in range(n_seeds, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved position heatmaps to {output_path}")
    plt.close()


def create_position_distribution_analysis(
    all_states, seeds, x_range, y_range, output_path
):
    """
    Analyze and visualize the distribution of sampled positions across seeds.

    Args:
        all_states: list of arrays, each of shape (max_rollouts, 3)
        seeds: list of seed values
        x_range, y_range: valid sampling regions
        output_path: path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: X position distributions
    ax = axes[0, 0]
    for seed, states in zip(seeds, all_states):
        ax.hist(
            states[:, 0],
            bins=30,
            alpha=0.3,
            label=f"Seed {seed}" if seed in seeds[:5] else None,
        )
    ax.axvline(
        np.mean([np.mean(states[:, 0]) for states in all_states]),
        color="red",
        linestyle="--",
        linewidth=2,
        label="Mean of means",
    )
    ax.set_xlabel("X position (m)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("X Position Distribution Across Seeds", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Y position distributions
    ax = axes[0, 1]
    for seed, states in zip(seeds, all_states):
        ax.hist(
            states[:, 1],
            bins=30,
            alpha=0.3,
            label=f"Seed {seed}" if seed in seeds[:5] else None,
        )
    ax.axvline(
        np.mean([np.mean(states[:, 1]) for states in all_states]),
        color="red",
        linestyle="--",
        linewidth=2,
        label="Mean of means",
    )
    ax.set_xlabel("Y position (m)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Y Position Distribution Across Seeds", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Theta distributions
    ax = axes[1, 0]
    for seed, states in zip(seeds, all_states):
        ax.hist(
            states[:, 2],
            bins=30,
            alpha=0.3,
            label=f"Seed {seed}" if seed in seeds[:5] else None,
        )
    ax.set_xlabel("Theta (radians)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Theta Distribution Across Seeds", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Mean positions per seed
    ax = axes[1, 1]
    mean_x = [np.mean(states[:, 0]) for states in all_states]
    mean_y = [np.mean(states[:, 1]) for states in all_states]

    ax.scatter(
        mean_x,
        mean_y,
        s=100,
        c=range(len(seeds)),
        cmap="tab10",
        edgecolors="black",
        linewidth=2,
    )
    for i, seed in enumerate(seeds):
        ax.annotate(
            f"S{seed}", (mean_x[i], mean_y[i]), fontsize=8, ha="center", va="center"
        )

    ax.axhline(np.mean(mean_y), color="red", linestyle="--", alpha=0.5, label="Mean Y")
    ax.axvline(np.mean(mean_x), color="red", linestyle="--", alpha=0.5, label="Mean X")
    ax.set_xlabel("Mean X position (m)", fontsize=12)
    ax.set_ylabel("Mean Y position (m)", fontsize=12)
    ax.set_title("Mean Sampled Position per Seed", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved position distribution analysis to {output_path}")
    plt.close()


def create_variance_plot(cumulative_rates, output_path):
    """
    Create plot showing mean and variance of success rate estimates vs number of rollouts.

    Args:
        cumulative_rates: array of shape (n_seeds, max_rollouts)
        output_path: path to save the plot
    """
    n_seeds, max_rollouts = cumulative_rates.shape
    rollout_counts = np.arange(1, max_rollouts + 1)

    # Compute statistics across seeds for each rollout count
    mean_rates = np.mean(cumulative_rates, axis=0)
    std_rates = np.std(cumulative_rates, axis=0)
    min_rates = np.min(cumulative_rates, axis=0)
    max_rates = np.max(cumulative_rates, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Mean with std bands
    ax = axes[0, 0]
    ax.plot(rollout_counts, mean_rates, "b-", linewidth=2, label="Mean")
    ax.fill_between(
        rollout_counts,
        mean_rates - std_rates,
        mean_rates + std_rates,
        alpha=0.3,
        color="blue",
        label="1 std",
    )
    ax.fill_between(
        rollout_counts,
        mean_rates - 2 * std_rates,
        mean_rates + 2 * std_rates,
        alpha=0.15,
        color="blue",
        label="2 std",
    )
    ax.set_xlabel("Number of Rollouts", fontsize=12)
    ax.set_ylabel("Estimated Success Rate", fontsize=12)
    ax.set_title(
        "Success Rate Estimate vs Number of Rollouts", fontsize=14, fontweight="bold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max_rollouts)
    ax.set_ylim(0, 1)

    # Plot 2: Variance/Std vs rollouts
    ax = axes[0, 1]
    ax.plot(rollout_counts, std_rates, "r-", linewidth=2, label="Std Dev")
    ax.plot(rollout_counts, std_rates**2, "g--", linewidth=2, label="Variance")
    ax.set_xlabel("Number of Rollouts", fontsize=12)
    ax.set_ylabel("Variance / Std Dev", fontsize=12)
    ax.set_title(
        "Estimation Variance vs Number of Rollouts", fontsize=14, fontweight="bold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max_rollouts)

    # Plot 3: All individual seed trajectories
    ax = axes[1, 0]
    for seed_idx in range(n_seeds):
        ax.plot(
            rollout_counts,
            cumulative_rates[seed_idx],
            alpha=0.5,
            linewidth=1,
            label=f"Seed {seed_idx}" if seed_idx < 5 else None,
        )
    ax.plot(rollout_counts, mean_rates, "k-", linewidth=3, label="Mean")
    ax.set_xlabel("Number of Rollouts", fontsize=12)
    ax.set_ylabel("Estimated Success Rate", fontsize=12)
    ax.set_title("Individual Seed Trajectories", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max_rollouts)
    ax.set_ylim(0, 1)

    # Plot 4: Min/Max range
    ax = axes[1, 1]
    ax.plot(rollout_counts, mean_rates, "b-", linewidth=2, label="Mean")
    ax.fill_between(
        rollout_counts,
        min_rates,
        max_rates,
        alpha=0.3,
        color="orange",
        label="Min-Max Range",
    )
    ax.set_xlabel("Number of Rollouts", fontsize=12)
    ax.set_ylabel("Estimated Success Rate", fontsize=12)
    ax.set_title("Min-Max Range Across Seeds", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max_rollouts)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


def run_evaluation(args):
    """Main evaluation loop."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Variance in Performance Estimation vs Number of Rollouts")
    print("=" * 70)

    # Load environment once to get constraints and horizon
    print("\n[1/4] Loading environment...")
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    _, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.checkpoint, device=device, verbose=False
    )

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, render=False, render_offscreen=False, verbose=False
    )

    print(f"  Environment: {env.name}")

    if args.horizon is None:
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        horizon = config.experiment.rollout.horizon
    else:
        horizon = args.horizon
    print(f"  Rollout horizon: {horizon}")

    sampler = StateSampler(env)
    print(f"  Valid region: x in {sampler.x_range}, y in {sampler.y_range}")

    # Extract fixed robot and gripper qpos to use across all seeds
    fixed_robot_qpos = sampler.robot_qpos.copy()
    fixed_gripper_qpos = sampler.gripper_qpos.copy()
    print(f"  Fixed robot qpos: {fixed_robot_qpos}")
    print(f"  Fixed gripper qpos: {fixed_gripper_qpos}")

    # Prepare worker arguments
    print(f"\n[2/4] Preparing {args.n_seeds} seed evaluations...")
    print(f"  Max rollouts per seed: {args.max_rollouts}")
    print(f"  Trials per state: {args.n_trials}")
    print(f"  Workers: {args.num_workers}")

    base_seed = args.seed
    seeds = [base_seed + i * 1000 for i in range(args.n_seeds)]

    if args.gpu_ids is not None:
        device_assignment = [
            args.gpu_ids[i % len(args.gpu_ids)] for i in range(args.num_workers)
        ]
    else:
        device_assignment = [None] * args.num_workers

    worker_args = [
        (
            seed,
            args.checkpoint,
            horizon,
            args.max_rollouts,
            args.n_trials,
            device_assignment[i % args.num_workers],
            fixed_robot_qpos,
            fixed_gripper_qpos,
        )
        for i, seed in enumerate(seeds)
    ]

    # Run evaluation
    print(f"\n[3/4] Running evaluations...")
    t_start = time.time()

    if args.num_workers == 1:
        results = [evaluate_seed_worker(arg) for arg in worker_args]
    else:
        with Pool(processes=args.num_workers) as pool:
            results = []
            for i, result in enumerate(
                pool.imap_unordered(evaluate_seed_worker, worker_args)
            ):
                results.append(result)
                elapsed = time.time() - t_start
                print(
                    f"  [{i + 1}/{args.n_seeds}] Seed {result[0]} completed ({elapsed:.1f}s)"
                )

    t_elapsed = time.time() - t_start

    # Sort by seed and extract data
    results.sort(key=lambda x: x[0])
    all_states = [r[1] for r in results]
    success_results = np.array([r[2] for r in results])

    print(f"  Completed in {t_elapsed:.1f}s")
    print(f"  Total rollouts: {args.n_seeds * args.max_rollouts}")

    # Compute cumulative success rates
    print(f"\n[4/5] Computing statistics and creating plots...")
    cumulative_rates = compute_cumulative_success_rates(success_results)

    # Create variance plot
    create_variance_plot(cumulative_rates, output_dir / "variance_vs_rollouts.png")

    # Create position visualizations
    print(f"\n[5/5] Creating position visualizations...")

    # Individual detailed plots for each seed
    create_individual_seed_plots(
        all_states, success_results, seeds, sampler.x_range, sampler.y_range, output_dir
    )

    # Summary heatmaps
    create_position_heatmaps(
        all_states,
        success_results,
        seeds,
        sampler.x_range,
        sampler.y_range,
        output_dir / "position_heatmaps.png",
    )

    # Distribution analysis
    create_position_distribution_analysis(
        all_states,
        seeds,
        sampler.x_range,
        sampler.y_range,
        output_dir / "position_distributions.png",
    )

    # Save raw data
    np.savez(
        output_dir / "variance_analysis_results.npz",
        success_results=success_results,
        cumulative_rates=cumulative_rates,
        all_states=np.array(all_states),
        seeds=np.array(seeds),
        rollout_counts=np.arange(1, args.max_rollouts + 1),
    )
    print(f"  Saved data to {output_dir / 'variance_analysis_results.npz'}")

    # Save config
    config = vars(args).copy()
    config["evaluation_time_seconds"] = t_elapsed
    config["seeds_used"] = seeds
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config to {output_dir / 'config.json'}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Seeds evaluated: {args.n_seeds}")
    print(f"Max rollouts per seed: {args.max_rollouts}")
    print(f"Total rollouts: {args.n_seeds * args.max_rollouts}")
    print(f"Evaluation time: {t_elapsed:.1f}s")

    # Statistics at different rollout counts
    for n_rollouts in [1, 5, 10, 25, 50, args.max_rollouts]:
        if n_rollouts <= args.max_rollouts:
            rates_at_n = cumulative_rates[:, n_rollouts - 1]
            print(f"\nAt {n_rollouts} rollouts:")
            print(f"  Mean success rate: {np.mean(rates_at_n):.3f}")
            print(f"  Std dev: {np.std(rates_at_n):.3f}")
            print(f"  Min: {np.min(rates_at_n):.3f}, Max: {np.max(rates_at_n):.3f}")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Evaluate variance in success rate estimation vs number of rollouts"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained policy checkpoint (.pth)",
    )

    parser.add_argument(
        "--max_rollouts",
        type=int,
        default=100,
        help="Maximum number of rollouts per seed (M)",
    )

    parser.add_argument(
        "--n_seeds",
        type=int,
        default=10,
        help="Number of different seeds to evaluate (N)",
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=1,
        help="Number of trials per state (for stochastic policies)",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Rollout horizon (default: from checkpoint config)",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=mp.cpu_count(),
        help=f"Number of parallel workers (default: {mp.cpu_count()})",
    )

    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="+",
        default=None,
        help="GPU IDs to use (e.g., --gpu_ids 0 1 2). If not specified, uses CPU.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="variance_eval_results",
        help="Directory to save results",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (each evaluation uses seed + i*1000)",
    )

    args = parser.parse_args()
    run_evaluation(args)
