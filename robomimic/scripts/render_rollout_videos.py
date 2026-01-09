#!/usr/bin/env python
"""
Render videos of policy rollouts at specific initial states.

This script:
1. Loads a trained policy and evaluation results
2. Selects interesting states (high/low success, specific theta ranges)
3. Renders videos of rollouts from those states

Usage:
    python render_rollout_videos.py --checkpoint /path/to/model.pth --results /path/to/evaluation_results.npz --output_dir videos/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import imageio
import numpy as np

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.envs.env_base import EnvBase


class StateSampler:
    """
    Helper to create state dictionaries for reset_to().
    """

    def __init__(self, env):
        """
        Args:
            env: robomimic environment wrapper
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
        """
        sim = self.base_env.sim

        # Find object joint (typically a free joint for the cube)
        obj_joint_idx = None
        obj_joint_qpos_addr = None

        for i in range(sim.model.njnt):
            jnt_name = sim.model.joint_id2name(i)
            if "cube" in jnt_name.lower() or "object" in jnt_name.lower():
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
            # Object is at obj_joint_qpos_addr + 1 (position = 3 DOF, quat = 4 DOF for free joint)
            # The +1 offset accounts for the gripper having 2 joints that occupy qpos[7] and qpos[8]
            # even though jnt_qposadr reports the cube joint starts at 9
            self.obj_pos_start = obj_joint_qpos_addr + 1
            self.obj_pos_end = obj_joint_qpos_addr + 4
            self.obj_quat_start = obj_joint_qpos_addr + 4
            self.obj_quat_end = obj_joint_qpos_addr + 8

            # Find gripper joints (look for gripper in name)
            gripper_start = None
            for i in range(sim.model.njnt):
                jnt_name = sim.model.joint_id2name(i)
                if "gripper" in jnt_name.lower():
                    gripper_start = sim.model.jnt_qposadr[i]
                    break

            if gripper_start is not None:
                self.gripper_qpos_start = gripper_start
                self.gripper_qpos_end = (
                    gripper_start + 3
                )  # 3 gripper joints (including coupled joint)
                self.robot_qpos_start = 0
                self.robot_qpos_end = gripper_start
            else:
                # Fallback
                self.robot_qpos_start = 0
                self.robot_qpos_end = obj_joint_qpos_addr - 3  # Assume 3 gripper joints
                self.gripper_qpos_start = obj_joint_qpos_addr - 3
                self.gripper_qpos_end = obj_joint_qpos_addr

            # qvel starts after qpos (which is 17 for this robot configuration)
            self.qvel_start = 17

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


def rollout_with_video(
    policy, env, state_dict, horizon, video_writer, video_skip=5, camera_names=None
):
    """
    Perform a single rollout and write to video.

    Args:
        policy: Loaded policy
        env: Environment
        state_dict: Initial state for reset_to()
        horizon: Rollout horizon
        video_writer: imageio video writer
        video_skip: Frame skip for video
        camera_names: List of camera names for rendering

    Returns:
        success: Whether the rollout succeeded
        horizon_reached: Number of steps taken
    """
    if camera_names is None:
        camera_names = ["frontview"]

    # Reset to state
    obs = env.reset_to(state_dict)

    video_count = 0
    success = False

    # Render the initial frame BEFORE taking any actions
    video_img = []
    for cam_name in camera_names:
        video_img.append(
            env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name)
        )
    video_img = np.concatenate(video_img, axis=1)
    video_writer.append_data(video_img)
    video_count += 1

    for step_i in range(horizon):
        # Get action from policy
        act = policy(ob=obs)

        # Step environment
        obs, r, done, _ = env.step(act)

        # Check success
        success = env.is_success()["task"]

        # Render to video
        if video_count % video_skip == 0:
            video_img = []
            for cam_name in camera_names:
                video_img.append(
                    env.render(
                        mode="rgb_array", height=512, width=512, camera_name=cam_name
                    )
                )
            video_img = np.concatenate(video_img, axis=1)  # concatenate horizontally
            video_writer.append_data(video_img)
        video_count += 1

        # Break if done or success
        if done or success:
            break

    return success, step_i + 1


def select_states_to_render(
    states, success_rates, n_success=5, n_failure=5, theta_bins=None
):
    """
    Select interesting states to render.

    Args:
        states: Array of (x, y, theta) states
        success_rates: Array of success rates for each state
        n_success: Number of success cases to render
        n_failure: Number of failure cases to render
        theta_bins: Optional list of theta values to focus on

    Returns:
        selected_states: List of (x, y, theta, success_rate, label) tuples
    """
    selected = []

    # Find high success cases
    success_indices = np.argsort(success_rates)[-n_success:]
    for idx in success_indices:
        x, y, theta = states[idx]
        selected.append((x, y, theta, success_rates[idx], "success"))

    # Find low success cases
    failure_indices = np.argsort(success_rates)[:n_failure]
    for idx in failure_indices:
        x, y, theta = states[idx]
        selected.append((x, y, theta, success_rates[idx], "failure"))

    # If theta bins specified, add cases from those bins
    if theta_bins is not None:
        for target_theta in theta_bins:
            # Find closest theta
            theta_diffs = np.abs(states[:, 2] - target_theta)
            closest_idx = np.argmin(theta_diffs)
            x, y, theta = states[closest_idx]
            selected.append(
                (x, y, theta, success_rates[closest_idx], f"theta_{target_theta:.2f}")
            )

    return selected


def render_videos(args):
    """
    Main rendering function.
    """
    print(f"\n{'=' * 60}")
    print("Rendering Rollout Videos")
    print(f"{'=' * 60}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load policy
    print("\n[1/5] Loading policy...")
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.checkpoint, device=device, verbose=False
    )
    print(f"  ✓ Policy loaded from {args.checkpoint}")

    # Load environment with rendering enabled
    print("\n[2/5] Loading environment...")
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        render=False,
        render_offscreen=True,
        verbose=False,
    )
    print(f"  ✓ Environment: {env.name}")

    # Get horizon
    horizon = args.horizon
    if horizon is None:
        try:
            horizon = ckpt_dict["config"].train.rollout.horizon
        except (KeyError, AttributeError):
            # Fallback to default horizon
            horizon = 400
            print(f"  ! Using default horizon: {horizon}")
    print(f"  ✓ Horizon: {horizon}")

    # Load evaluation results
    print("\n[3/5] Loading evaluation results...")
    if args.results:
        results = np.load(args.results)
        states = results["states"]
        success_rates = results["success_rates"]
        print(f"  ✓ Loaded {len(states)} evaluated states")
        print(f"  ✓ Mean success rate: {success_rates.mean():.3f}")
    else:
        # Manual state specification
        if args.manual_states is None:
            print("ERROR: Must provide either --results or --manual_states")
            return
        # Parse manual states: "x,y,theta;x,y,theta;..."
        states = []
        for state_str in args.manual_states.split(";"):
            x, y, theta = map(float, state_str.split(","))
            states.append([x, y, theta])
        states = np.array(states)
        success_rates = np.zeros(len(states))  # Unknown success rates
        print(f"  ✓ Using {len(states)} manually specified states")

    # Select states to render
    print("\n[4/5] Selecting states to render...")
    theta_bins = None
    if args.theta_bins:
        theta_bins = [float(t) for t in args.theta_bins.split(",")]
        print(f"  ✓ Theta bins: {theta_bins}")

    selected_states = select_states_to_render(
        states,
        success_rates,
        n_success=args.n_success,
        n_failure=args.n_failure,
        theta_bins=theta_bins,
    )
    print(f"  ✓ Selected {len(selected_states)} states to render")

    # Create state sampler
    sampler = StateSampler(env)

    # Render videos
    print("\n[5/5] Rendering videos...")
    camera_names = args.camera_names.split(",") if args.camera_names else ["frontview"]
    print(f"  ✓ Cameras: {camera_names}")

    for i, (x, y, theta, expected_success, label) in enumerate(selected_states):
        print(
            f"\n  [{i + 1}/{len(selected_states)}] Rendering {label}: "
            f"x={x:.3f}, y={y:.3f}, θ={theta:.2f}, expected_success={expected_success:.2f}"
        )

        # Create state dict
        state_dict = sampler.create_state_dict(x, y, theta)

        # Create video writer
        video_filename = (
            f"{label}_x{x:.3f}_y{y:.3f}_theta{theta:.2f}_exp{expected_success:.2f}.mp4"
        )
        video_path = output_dir / video_filename
        video_writer = imageio.get_writer(video_path, fps=20)

        try:
            # Run rollout
            success, steps = rollout_with_video(
                policy=policy,
                env=env,
                state_dict=state_dict,
                horizon=horizon,
                video_writer=video_writer,
                video_skip=args.video_skip,
                camera_names=camera_names,
            )

            video_writer.close()

            # Report result
            result_str = "✓ SUCCESS" if success else "✗ FAILURE"
            print(f"    {result_str} in {steps} steps → {video_path}")

        except Exception as e:
            video_writer.close()
            print(f"    ERROR: {e}")
            continue

    print(f"\n{'=' * 60}")
    print(f"Done! Videos saved to {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render videos of policy rollouts")

    # Required
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained policy checkpoint (.pth)",
    )

    # State selection
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Path to evaluation results (.npz) from evaluate_initial_states.py",
    )
    parser.add_argument(
        "--manual_states",
        type=str,
        default=None,
        help="Manual states to render (format: 'x,y,theta;x,y,theta;...')",
    )
    parser.add_argument(
        "--n_success",
        type=int,
        default=5,
        help="Number of high-success cases to render",
    )
    parser.add_argument(
        "--n_failure",
        type=int,
        default=5,
        help="Number of low-success cases to render",
    )
    parser.add_argument(
        "--theta_bins",
        type=str,
        default=None,
        help="Specific theta values to render (comma-separated, e.g., '0.0,1.57,3.14')",
    )

    # Video settings
    parser.add_argument(
        "--camera_names",
        type=str,
        default="frontview",
        help="Camera names for rendering (comma-separated, e.g., 'frontview,birdview')",
    )
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="Frame skip for video (every Nth frame)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Rollout horizon (default: from checkpoint config)",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="rollout_videos",
        help="Directory to save videos",
    )

    args = parser.parse_args()
    render_videos(args)
