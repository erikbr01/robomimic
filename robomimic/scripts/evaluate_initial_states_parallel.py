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
import multiprocessing as mp
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.envs.env_base import EnvBase


def qvel_width(joint_type: mujoco.mjtJoint) -> int:
    """Get the dimensionality of the joint in qvel."""
    return {0: 6, 1: 3, 2: 1, 3: 1}[joint_type]


def qpos_width(joint_type: mujoco.mjtJoint) -> int:
    """Get the dimensionality of the joint in qpos."""
    return {0: 7, 1: 4, 2: 1, 3: 1}[joint_type]


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
        self.base_env = env.base_env if hasattr(env, "base_env") else env

        # Extract constraints from robosuite
        if hasattr(self.base_env, "placement_initializer"):
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

        # Get template state and fixed robot configuration ONCE
        obs = self.env.reset()
        self.template_state = self.env.get_state()

        # Get or set fixed robot configuration
        if fixed_robot_qpos is not None:
            self.robot_qpos = fixed_robot_qpos
        else:
            self.robot_qpos = self.template_state["states"][
                self.robot_qpos_start : self.gripper_qpos_start
            ].copy()

        # Gripper configuration from dataset: first joint ~0.79 (open), second joint 0.020833
        self.gripper_qpos = self.template_state["states"][
            self.gripper_qpos_start : self.gripper_qpos_end
        ].copy()

    def _find_state_indices(self):
        """
        Query MuJoCo model to find the correct indices for robot, gripper, and object.
        This ensures we're robust to different robot models and configurations.

        Creates a mapping of joint names to their qpos and qvel indices.
        """
        sim = self.base_env.sim

        # Build comprehensive joint mapping
        self.joint_map = {}
        qvel_offset = 0

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

        # Identify joint categories
        robot_joints = []
        gripper_joints = []
        object_joint = None

        for jnt_name, jnt_info in self.joint_map.items():
            if "cube" in jnt_name.lower() or "object" in jnt_name.lower():
                object_joint = jnt_name
            elif "gripper" in jnt_name.lower():
                gripper_joints.append(jnt_name)
            else:
                # Assume remaining joints are robot arm joints
                robot_joints.append(jnt_name)

        if object_joint is None:
            raise ValueError("Could not find object joint in MuJoCo model")

        # Set robot joint indices (continuous range from first robot joint)
        if robot_joints:
            robot_joints.sort(key=lambda x: self.joint_map[x]["qpos_start"])
            self.robot_qpos_start = self.joint_map[robot_joints[0]]["qpos_start"]
            self.robot_qpos_end = self.joint_map[robot_joints[-1]]["qpos_end"]
        else:
            self.robot_qpos_start = 0
            self.robot_qpos_end = 0

        # Set gripper joint indices (continuous range from first gripper joint)
        if gripper_joints:
            gripper_joints.sort(key=lambda x: self.joint_map[x]["qpos_start"])
            self.gripper_qpos_start = self.joint_map[gripper_joints[0]]["qpos_start"]
            self.gripper_qpos_end = self.joint_map[gripper_joints[-1]]["qpos_end"]
        else:
            # No gripper joints found, use empty range
            self.gripper_qpos_start = self.robot_qpos_end
            self.gripper_qpos_end = self.robot_qpos_end

        # Set object joint indices
        obj_info = self.joint_map[object_joint]
        obj_qpos_start = obj_info["qpos_start"]

        # For free joints, position is first 3 DOF, quaternion is next 4 DOF
        # The +1 offset accounts for an extra gripper actuator value in the
        # flattened state vector that isn't in raw qpos
        if obj_info["type"] == 0:  # Free joint
            self.obj_pos_start = obj_qpos_start + 1
            self.obj_pos_end = obj_qpos_start + 4
            self.obj_quat_start = obj_qpos_start + 4
            self.obj_quat_end = obj_qpos_start + 8
        else:
            # For other joint types, adjust accordingly
            self.obj_pos_start = obj_qpos_start + 1
            self.obj_pos_end = obj_qpos_start + 1 + min(3, obj_info["qpos_size"])
            self.obj_quat_start = self.obj_pos_end
            self.obj_quat_end = obj_info["qpos_end"] + 1

        # qvel starts after qpos, with +1 offset for time at the beginning
        self.qvel_start = sim.model.nq + 1

    def sample_random(
        self, n_samples=1, check_collision=True, max_attempts=100, fixed_theta=None
    ):
        """
        Sample random valid states.

        Args:
            n_samples: Number of valid samples to generate
            check_collision: If True, validate with physics simulation
            max_attempts: Maximum attempts to find each valid state
            fixed_theta: If provided, use this theta value for all samples (in radians)

        Returns:
            np.array of shape (n_samples, 3) with (x, y, theta) for each state
        """
        valid_samples = []

        while len(valid_samples) < n_samples:
            attempts = 0
            while attempts < max_attempts:
                attempts += 1

                # Sample uniformly
                x = np.random.uniform(*self.x_range)
                y = np.random.uniform(*self.y_range)

                # Use fixed theta if provided, otherwise sample randomly
                if fixed_theta is not None:
                    theta = fixed_theta
                else:
                    theta = np.random.uniform(0, 2 * np.pi)

                # Check collision if requested
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
        """Check if state is collision-free."""
        # Create state
        state_dict = self.create_state_dict(x, y, theta)

        # Apply to simulation
        self.base_env.sim.set_state_from_flattened(state_dict["states"])
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

            is_robot = "robot0" in geom1_name or "robot0" in geom2_name
            is_gripper = "gripper" in geom1_name or "gripper" in geom2_name
            is_object = any(
                key in geom1_name or key in geom2_name
                for key in ["cube", "object", "Box"]
            )

            # Robot (not gripper) colliding with object is invalid
            if is_robot and is_object and not is_gripper:
                return False

        return True

    def create_state_dict(self, x, y, theta, robot_qpos=None, gripper_qpos=None):
        """
        Create state dictionary for env.reset_to().

        Args:
            x, y: Object position
            theta: Object rotation (radians)
            robot_qpos: Optional robot joint angles (if None, uses default)
            gripper_qpos: Optional gripper joint angles (if None, uses default)

        Returns:
            Dict with 'states' key containing full MuJoCo state
        """
        # Use cached template (don't call reset() again!)
        state_vector = self.template_state["states"].copy()

        # Set robot joints
        if robot_qpos is None:
            robot_qpos = self.robot_qpos
        state_vector[self.robot_qpos_start : self.robot_qpos_end] = robot_qpos

        # Set gripper joints
        if gripper_qpos is None:
            gripper_qpos = self.gripper_qpos
        state_vector[self.gripper_qpos_start : self.gripper_qpos_end] = gripper_qpos

        # Set object position
        state_vector[self.obj_pos_start] = x
        state_vector[self.obj_pos_start + 1] = y
        state_vector[self.obj_pos_start + 2] = self.z

        # Set object orientation (quaternion from theta, MuJoCo uses [w,x,y,z])
        state_vector[self.obj_quat_start] = np.cos(theta / 2)  # w
        state_vector[self.obj_quat_start + 1] = 0.0  # x
        state_vector[self.obj_quat_start + 2] = 0.0  # y
        state_vector[self.obj_quat_start + 3] = np.sin(theta / 2)  # z

        # Zero velocities
        state_vector[self.qvel_start : -1] = 0.0
        state_vector[-1] = 0.0  # time (last element)

        return {"states": state_vector}


def add_text_overlay(frame, text_lines, success=None):
    """
    Add text overlay to a video frame.

    Args:
        frame: numpy array (H, W, 3)
        text_lines: list of strings to display
        success: None, True, or False (controls color)

    Returns:
        frame with text overlay
    """
    # Convert to PIL Image
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24
            )
            small_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18
            )
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

    # Determine background color based on success
    if success is True:
        bg_color = (0, 180, 0, 200)  # Green with transparency
        text_color = (255, 255, 255)
    elif success is False:
        bg_color = (180, 0, 0, 200)  # Red with transparency
        text_color = (255, 255, 255)
    else:
        bg_color = (0, 0, 0, 200)  # Black with transparency
        text_color = (255, 255, 255)

    # Create semi-transparent overlay
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # Calculate text box size
    y_offset = 10
    max_width = 0
    total_height = 0

    for line in text_lines:
        bbox = draw.textbbox(
            (0, 0),
            line,
            font=font if "SUCCESS" in line or "FAILURE" in line else small_font,
        )
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        max_width = max(max_width, line_width)
        total_height += line_height + 5

    # Draw background rectangle
    padding = 15
    overlay_draw.rectangle(
        [
            (10, y_offset - 5),
            (10 + max_width + padding * 2, y_offset + total_height + padding),
        ],
        fill=bg_color,
    )

    # Composite the overlay
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw text
    y_position = y_offset + 5
    for line in text_lines:
        current_font = font if ("SUCCESS" in line or "FAILURE" in line) else small_font
        draw.text((10 + padding, y_position), line, fill=text_color, font=current_font)
        bbox = draw.textbbox((0, 0), line, font=current_font)
        y_position += (bbox[3] - bbox[1]) + 5

    return np.array(img)


def render_video_for_state(args):
    """
    Render a video for a single state with text overlay.

    Args:
        args: tuple of (state, success_rate, checkpoint_path, horizon, video_path, camera_names)

    Returns:
        success: whether the rollout succeeded
    """
    state, expected_success_rate, checkpoint_path, horizon, video_path, camera_names = (
        args
    )

    # Suppress output
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    # Load policy and environment
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=checkpoint_path, device=device, verbose=False
    )

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        render=False,
        render_offscreen=True,
        verbose=False,
    )

    # Create state dict
    sampler = StateSampler(env)
    x, y, theta = state
    state_dict = sampler.create_state_dict(x, y, theta)

    # Create video writer
    video_writer = imageio.get_writer(video_path, fps=20)

    try:
        # Reset to state
        obs = env.reset_to(state_dict)
        policy.start_episode()

        success = False
        video_skip = 5
        video_count = 0
        timestep = 0

        for t in range(horizon):
            action = policy(ob=obs)
            obs, reward, done, info = env.step(action)
            timestep += 1

            # Check success
            if env.is_success()["task"]:
                success = True

            # Render frame
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    video_img.append(
                        env.render(
                            mode="rgb_array",
                            height=512,
                            width=512,
                            camera_name=cam_name,
                        )
                    )
                video_img = np.concatenate(video_img, axis=1)

                # Add text overlay
                text_lines = [
                    "SUCCESS" if success else "FAILURE",
                    f"x={x:.3f}, y={y:.3f}, θ={theta:.2f}rad",
                    f"Expected rate: {expected_success_rate:.2f}",
                    f"Step: {timestep}/{horizon}",
                ]
                video_img = add_text_overlay(video_img, text_lines, success=success)

                video_writer.append_data(video_img)
            video_count += 1

            if success or done:
                # Render a few more frames after success/failure to see final state
                for _ in range(10):
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(
                            env.render(
                                mode="rgb_array",
                                height=512,
                                width=512,
                                camera_name=cam_name,
                            )
                        )
                    video_img = np.concatenate(video_img, axis=1)
                    text_lines = [
                        "SUCCESS" if success else "FAILURE",
                        f"x={x:.3f}, y={y:.3f}, θ={theta:.2f}rad",
                        f"Expected rate: {expected_success_rate:.2f}",
                        f"Final step: {timestep}/{horizon}",
                    ]
                    video_img = add_text_overlay(video_img, text_lines, success=success)
                    video_writer.append_data(video_img)
                break

        video_writer.close()
        return success

    except Exception as e:
        video_writer.close()
        return False


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
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    # Set device for this worker
    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    # Load policy and environment in this worker
    device = TorchUtils.get_torch_device(try_to_use_cuda=(device_id is not None))

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=checkpoint_path, device=device, verbose=False
    )

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, render=False, render_offscreen=False, verbose=False
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
    scatter = ax.scatter(
        states[:, 0],
        states[:, 1],
        c=success_rates,
        s=100,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        edgecolors="black",
        linewidth=0.5,
        alpha=0.8,
    )

    ax.set_xlabel("X position (m)", fontsize=12)
    ax.set_ylabel("Y position (m)", fontsize=12)
    ax.set_title("Success Rate vs. Object Position", fontsize=14, fontweight="bold")
    ax.set_xlim(x_range[0] - 0.01, x_range[1] + 0.01)
    ax.set_ylim(y_range[0] - 0.01, y_range[1] + 0.01)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Success Rate", fontsize=11)

    # Draw boundary
    rect = patches.Rectangle(
        (x_range[0], y_range[0]),
        x_range[1] - x_range[0],
        y_range[1] - y_range[0],
        linewidth=2,
        edgecolor="blue",
        facecolor="none",
        linestyle="--",
        label="Valid region",
    )
    ax.add_patch(rect)
    ax.legend()

    # Plot 2: Histogram
    ax = axes[1]
    ax.hist(success_rates, bins=20, edgecolor="black", color="skyblue", alpha=0.7)
    ax.set_xlabel("Success Rate", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Success Rates", fontsize=14, fontweight="bold")
    ax.axvline(
        success_rates.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {success_rates.mean():.3f}",
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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
        ckpt_path=args.checkpoint, device=device, verbose=False
    )

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, render=False, render_offscreen=False, verbose=False
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
    if args.fixed_theta is not None:
        print(
            f"  ✓ Using fixed theta: {args.fixed_theta:.3f} rad ({np.degrees(args.fixed_theta):.1f}°)"
        )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    t_start = time.time()
    states = sampler.sample_random(
        n_samples=args.n_samples,
        check_collision=not args.skip_collision_check,
        max_attempts=args.max_collision_attempts,
        fixed_theta=args.fixed_theta,
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
        (
            i,
            state,
            args.checkpoint,
            horizon,
            args.n_trials,
            device_assignment[i % args.num_workers],
        )
        for i, state in enumerate(states)
    ]

    # Run parallel evaluation
    t_start = time.time()

    if args.num_workers == 1:
        # Sequential for debugging
        results = [evaluate_single_state_worker(arg) for arg in worker_args]
    else:
        # Parallel
        with Pool(processes=args.num_workers) as pool:
            results = []
            for i, result in enumerate(
                pool.imap_unordered(evaluate_single_state_worker, worker_args)
            ):
                results.append(result)
                if (i + 1) % max(1, len(states) // 10) == 0:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed
                    eta = (len(states) - i - 1) / rate
                    print(
                        f"  [{i + 1:3d}/{len(states)}] {rate:.1f} states/s | ETA: {eta:.0f}s"
                    )

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
        evaluated_states,
        success_rates,
        sampler.x_range,
        sampler.y_range,
        output_dir / "success_heatmap.png",
    )

    # Save raw data
    np.savez(
        output_dir / "evaluation_results.npz",
        states=evaluated_states,
        success_rates=success_rates,
        x_range=sampler.x_range,
        y_range=sampler.y_range,
        z=sampler.z,
    )
    print(f"  ✓ Saved data to {output_dir / 'evaluation_results.npz'}")

    # Save config
    config = vars(args)
    config["x_range"] = sampler.x_range
    config["y_range"] = sampler.y_range
    config["z"] = sampler.z
    config["evaluation_time_seconds"] = t_elapsed
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Saved config to {output_dir / 'config.json'}")

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total states evaluated:  {len(evaluated_states)}")
    print(f"Total rollouts:          {len(evaluated_states) * args.n_trials}")
    print(
        f"Evaluation time:         {t_elapsed:.1f}s ({len(evaluated_states) / t_elapsed:.2f} states/s)"
    )
    print(f"Workers used:            {args.num_workers}")
    print(f"\nSuccess Rate Statistics:")
    print(f"  Mean:   {success_rates.mean():.3f}")
    print(f"  Std:    {success_rates.std():.3f}")
    print(f"  Min:    {success_rates.min():.3f}")
    print(f"  Max:    {success_rates.max():.3f}")
    print(f"  Median: {np.median(success_rates):.3f}")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)

    # Render videos if requested
    if args.render_videos:
        print("\n" + "=" * 70)
        print("RENDERING VIDEOS")
        print("=" * 70)

        video_dir = output_dir / "videos"
        video_dir.mkdir(exist_ok=True)

        # Render ALL states if requested, otherwise select subset
        if args.render_all:
            print(f"\nRendering ALL {len(evaluated_states)} states...")
            selected_indices = list(range(len(evaluated_states)))
        else:
            # Select states to render
            selected_indices = []

            # High success states
            high_success_indices = np.argsort(success_rates)[-args.n_videos_success :]
            selected_indices.extend(high_success_indices)

            # Low success states
            low_success_indices = np.argsort(success_rates)[: args.n_videos_failure]
            selected_indices.extend(low_success_indices)

            print(f"\nRendering {len(selected_indices)} videos...")
            print(f"  {args.n_videos_success} high-success states")
            print(f"  {args.n_videos_failure} low-success states")

        print(f"  Cameras: {args.video_cameras}")

        # Prepare video rendering arguments
        camera_names = args.video_cameras.split(",")
        video_args = []
        for idx in selected_indices:
            state = evaluated_states[idx]
            success_rate = success_rates[idx]
            x, y, theta = state
            video_filename = f"state{idx:03d}_rate{success_rate:.2f}_x{x:.3f}_y{y:.3f}_theta{theta:.2f}.mp4"
            video_path = video_dir / video_filename
            video_args.append(
                (
                    state,
                    success_rate,
                    args.checkpoint,
                    horizon,
                    str(video_path),
                    camera_names,
                )
            )

        # Render videos (sequentially to avoid issues)
        t_start = time.time()
        for i, args_tuple in enumerate(video_args):
            state = args_tuple[0]
            expected_rate = args_tuple[1]
            x, y, theta = state
            print(
                f"  [{i + 1}/{len(video_args)}] Rendering state {selected_indices[i]}: x={x:.3f}, y={y:.3f}, θ={theta:.2f}, expected={expected_rate:.2f}"
            )
            success = render_video_for_state(args_tuple)
            result_str = "✓ SUCCESS" if success else "✗ FAILURE"
            print(f"      {result_str}")

        t_elapsed = time.time() - t_start
        print(f"\n  ✓ Rendered {len(video_args)} videos in {t_elapsed:.1f}s")
        print(f"  ✓ Videos saved to {video_dir}")
        print("=" * 70)


if __name__ == "__main__":
    # Required for multiprocessing on some systems
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Parallel evaluation of policy across random initial object placements"
    )

    # Required
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained policy checkpoint (.pth)",
    )

    # Sampling
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of random states to sample and evaluate",
    )
    parser.add_argument(
        "--skip_collision_check",
        action="store_true",
        help="Skip collision checking (faster but may sample invalid states)",
    )
    parser.add_argument(
        "--max_collision_attempts",
        type=int,
        default=100,
        help="Max attempts to find collision-free state",
    )

    # Evaluation
    parser.add_argument(
        "--n_trials",
        type=int,
        default=5,
        help="Number of trials per state to estimate success rate",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Rollout horizon (default: from checkpoint config)",
    )

    # Parallelization
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

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save results",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--fixed_theta",
        type=float,
        default=None,
        help="Fix theta to this value (in radians) instead of sampling randomly",
    )

    # Video rendering options
    parser.add_argument(
        "--render_videos",
        action="store_true",
        help="Render videos of selected rollouts after evaluation",
    )
    parser.add_argument(
        "--render_all",
        action="store_true",
        help="Render videos for ALL evaluated states (overrides n_videos_* if set)",
    )
    parser.add_argument(
        "--n_videos_success",
        type=int,
        default=5,
        help="Number of high-success states to render (ignored if --render_all is set)",
    )
    parser.add_argument(
        "--n_videos_failure",
        type=int,
        default=5,
        help="Number of low-success states to render (ignored if --render_all is set)",
    )
    parser.add_argument(
        "--video_cameras",
        type=str,
        default="frontview",
        help="Camera names for video rendering (comma-separated)",
    )

    args = parser.parse_args()
    run_evaluation(args)
