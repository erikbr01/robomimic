#!/usr/bin/env python
"""
Debug script to verify cube visibility in rendered frames.

This script:
1. Creates a state with specified object pose
2. Resets environment to that state
3. Renders frames from multiple camera angles
4. Saves images to verify cube is visible

Usage:
    python debug_cube_visibility.py --checkpoint /path/to/model.pth --x 0.0 --y 0.0 --theta 0.0
"""

import argparse
import os
from pathlib import Path

import imageio
import numpy as np

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils


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
        print("\n[DEBUG] Getting template state from environment...")
        obs = self.env.reset()
        self.template_state = self.env.get_state()
        print(
            f"[DEBUG] Template state vector shape: {self.template_state['states'].shape}"
        )

        self.robot_qpos = self.template_state["states"][
            self.robot_qpos_start : self.gripper_qpos_start
        ].copy()
        # Gripper configuration from dataset: first joint ~0.79 (open), second joint 0.020833
        self.gripper_qpos = self.template_state["states"][
            self.gripper_qpos_start : self.gripper_qpos_end
        ].copy()

        print(
            f"[DEBUG] Robot qpos indices: [{self.robot_qpos_start}:{self.gripper_qpos_start}]"
        )
        print(
            f"[DEBUG] Gripper qpos indices: [{self.gripper_qpos_start}:{self.gripper_qpos_end}]"
        )
        print(f"[DEBUG] Object pos indices: [{self.obj_pos_start}:{self.obj_pos_end}]")
        print(
            f"[DEBUG] Object quat indices: [{self.obj_quat_start}:{self.obj_quat_end}]"
        )
        print(f"[DEBUG] qvel start index: {self.qvel_start}")

    def _find_state_indices(self):
        """
        Query MuJoCo model to find the correct indices for robot, gripper, and object.
        """
        sim = self.base_env.sim

        print("\n[DEBUG] Querying MuJoCo model structure...")
        print(f"[DEBUG] Number of joints: {sim.model.njnt}")
        print(f"[DEBUG] Number of qpos: {sim.model.nq}")
        print(f"[DEBUG] Number of qvel: {sim.model.nv}")

        # Print all joint names and addresses
        print("\n[DEBUG] Joint structure:")
        for i in range(sim.model.njnt):
            jnt_name = sim.model.joint_id2name(i)
            jnt_addr = sim.model.jnt_qposadr[i]
            print(f"[DEBUG]   Joint {i}: {jnt_name} @ qpos[{jnt_addr}]")

        # Find object joint (typically a free joint for the cube)
        obj_joint_idx = None
        obj_joint_qpos_addr = None

        for i in range(sim.model.njnt):
            jnt_name = sim.model.joint_id2name(i)
            if "cube" in jnt_name.lower() or "object" in jnt_name.lower():
                obj_joint_idx = i
                obj_joint_qpos_addr = sim.model.jnt_qposadr[i]
                print(
                    f"\n[DEBUG] Found object joint: {jnt_name} at index {i}, qpos address {obj_joint_qpos_addr}"
                )
                break

        if obj_joint_qpos_addr is None:
            # Fallback: assume standard structure
            print("[DEBUG] WARNING: Could not find object joint, using default indices")
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
                    print(
                        f"[DEBUG] Found gripper joint: {jnt_name} at qpos address {gripper_start}"
                    )
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

        print(
            f"\n[DEBUG] Creating state dict for x={x:.3f}, y={y:.3f}, theta={theta:.3f}"
        )
        print(
            f"[DEBUG] Original object position: {state_vector[self.obj_pos_start : self.obj_pos_end]}"
        )
        print(
            f"[DEBUG] Original object quaternion: {state_vector[self.obj_quat_start : self.obj_quat_end]}"
        )

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

        print(
            f"[DEBUG] New object position: {state_vector[self.obj_pos_start : self.obj_pos_end]}"
        )
        print(
            f"[DEBUG] New object quaternion: {state_vector[self.obj_quat_start : self.obj_quat_end]}"
        )

        return {"states": state_vector}


def debug_cube_visibility(args):
    """
    Main debug function.
    """
    print(f"\n{'=' * 60}")
    print("Debug Cube Visibility")
    print(f"{'=' * 60}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load policy
    print("\n[1/4] Loading policy...")
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.checkpoint, device=device, verbose=False
    )
    print(f"  ✓ Policy loaded from {args.checkpoint}")

    # Load environment with rendering enabled
    print("\n[2/4] Loading environment...")
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        render=False,
        render_offscreen=True,
        verbose=False,
    )
    print(f"  ✓ Environment: {env.name}")

    # Create state sampler
    print("\n[3/4] Creating state sampler...")
    sampler = StateSampler(env)

    # Create state dict
    print("\n[4/4] Creating and resetting to target state...")
    state_dict = sampler.create_state_dict(args.x, args.y, args.theta)

    # Reset to state
    print("\n[DEBUG] Calling env.reset_to()...")
    obs = env.reset_to(state_dict)

    # Get actual state after reset
    actual_state = env.get_state()
    actual_state_vector = actual_state["states"]
    print(f"\n[DEBUG] Actual state after reset_to():")
    print(
        f"[DEBUG] Object position: {actual_state_vector[sampler.obj_pos_start : sampler.obj_pos_end]}"
    )
    print(
        f"[DEBUG] Object quaternion: {actual_state_vector[sampler.obj_quat_start : sampler.obj_quat_end]}"
    )

    # Check if object observation is present
    if "object" in obs:
        print(f"\n[DEBUG] Object in observation: {obs['object']}")
    else:
        print("\n[DEBUG] WARNING: 'object' key not found in observation!")

    # Render from multiple camera angles
    camera_names = ["frontview", "birdview", "agentview", "sideview"]
    print(f"\n[DEBUG] Rendering from cameras: {camera_names}")

    for cam_name in camera_names:
        try:
            print(f"\n[DEBUG] Rendering {cam_name}...")
            img = env.render(
                mode="rgb_array", height=512, width=512, camera_name=cam_name
            )

            # Save image
            img_path = output_dir / f"test_{cam_name}.png"
            imageio.imwrite(img_path, img)
            print(f"  ✓ Saved {img_path}")

            # Check if image is not all zeros (would indicate rendering issue)
            if img.sum() == 0:
                print(f"  WARNING: Image is all black!")
            else:
                print(f"  ✓ Image has content (sum={img.sum()})")

        except Exception as e:
            print(f"  ERROR rendering {cam_name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Done! Check images in {output_dir}")
    print(f"{'=' * 60}\n")

    # Print summary
    print("\nSUMMARY:")
    print(f"  Target position: ({args.x:.3f}, {args.y:.3f}, {sampler.z:.3f})")
    print(f"  Target theta: {args.theta:.3f} rad ({np.degrees(args.theta):.1f}°)")
    print(
        f"  Actual position: {actual_state_vector[sampler.obj_pos_start : sampler.obj_pos_end]}"
    )
    print(
        f"  Actual quaternion: {actual_state_vector[sampler.obj_quat_start : sampler.obj_quat_end]}"
    )
    print(f"\n  If the cube is not visible, check:")
    print(f"    1. Object position is on the table")
    print(f"    2. Object is not clipping through table")
    print(f"    3. Camera angle includes the table area")
    print(f"    4. State indices are correct (see debug output above)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug cube visibility in rendered frames"
    )

    # Required
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained policy checkpoint (.pth)",
    )

    # Object pose
    parser.add_argument(
        "--x",
        type=float,
        default=0.0,
        help="Object x position (meters)",
    )
    parser.add_argument(
        "--y",
        type=float,
        default=0.0,
        help="Object y position (meters)",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.0,
        help="Object rotation (radians)",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="debug_cube_visibility",
        help="Directory to save debug images",
    )

    args = parser.parse_args()
    debug_cube_visibility(args)
