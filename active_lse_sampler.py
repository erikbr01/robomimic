"""
Constraint-aware state sampler for Active Level Set Estimation (LSE).

This module provides tools to sample object placements that respect robosuite's
placement constraints (bounds, collision checking, etc.) for use in Active LSE
policy evaluation.

Example usage:
    >>> import robomimic.utils.file_utils as FileUtils
    >>> import robomimic.utils.env_utils as EnvUtils
    >>> from active_lse_sampler import ConstraintAwareSampler
    >>>
    >>> # Load environment
    >>> env_meta = FileUtils.get_env_metadata_from_dataset('dataset.hdf5')
    >>> env = EnvUtils.create_env_from_metadata(env_meta, render=False)
    >>>
    >>> # Create sampler
    >>> sampler = ConstraintAwareSampler(env)
    >>>
    >>> # Sample valid states
    >>> x, y, theta = sampler.sample_uniform()
    >>>
    >>> # Check validity
    >>> if sampler.is_valid_state(x, y, theta, check_physics=True):
    >>>     state = sampler.create_initial_state(x, y, theta)
    >>>     obs = env.reset_to(state)
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union


class ConstraintAwareSampler:
    """
    Samples object placements that respect robosuite's constraints.

    This class extracts constraints from robosuite's placement_initializer
    and provides methods for sampling valid states, checking validity, and
    creating initial states for policy evaluation.
    """

    def __init__(self, env, robot_qpos: Optional[np.ndarray] = None):
        """
        Initialize sampler with environment constraints.

        Args:
            env: robomimic environment (EnvRobosuite wrapper)
            robot_qpos: Fixed robot joint angles (7,). If None, uses current robot state.
                       Recommended to fix this for consistent Active LSE experiments.
        """
        self.env = env
        self.base_env = env.base_env if hasattr(env, 'base_env') else env

        # Extract placement sampler from robosuite
        if hasattr(self.base_env, 'placement_initializer'):
            self.placement_sampler = self.base_env.placement_initializer

            # Extract constraints
            self.x_range = self.placement_sampler.x_range
            self.y_range = self.placement_sampler.y_range
            self.z_offset = self.placement_sampler.z_offset
            self.reference_pos = self.placement_sampler.reference_pos
            self.z = self.reference_pos[2] + self.z_offset
            self.rotation_axis = self.placement_sampler.rotation_axis
        else:
            # Fallback defaults if no placement sampler
            self.placement_sampler = None
            self.x_range = [-0.1, 0.1]
            self.y_range = [-0.1, 0.1]
            self.z = 0.82
            self.rotation_axis = 'z'
            print("Warning: No placement_initializer found, using default constraints")

        # Get or set robot configuration
        if robot_qpos is not None:
            self.robot_qpos = robot_qpos
        else:
            # Get current robot state
            obs = self.env.reset()
            state = self.env.get_state()
            self.robot_qpos = state['states'][:7].copy()

        # Store gripper state (typically open position)
        self.gripper_qpos = np.array([0.020833, -0.020833])

    @property
    def state_dim(self) -> int:
        """Dimension of state space (x, y, theta)."""
        return 3

    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return state space bounds."""
        return {
            'x': tuple(self.x_range),
            'y': tuple(self.y_range),
            'theta': (0, 2*np.pi),
        }

    def is_valid_state(self, x: float, y: float, theta: float = 0.0,
                       check_physics: bool = False) -> bool:
        """
        Check if state (x, y, theta) is valid.

        Args:
            x: Object x position
            y: Object y position
            theta: Object rotation around z-axis (radians)
            check_physics: If True, validate by simulating and checking collisions

        Returns:
            True if state is valid (within bounds and no collisions if checked)
        """
        # Check bounds
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range

        if not (x_min <= x <= x_max):
            return False
        if not (y_min <= y <= y_max):
            return False

        # Optionally check physics
        if check_physics:
            return self._check_physics_validity(x, y, theta)

        return True

    def _check_physics_validity(self, x: float, y: float, theta: float) -> bool:
        """
        Check validity by simulating state and detecting collisions.

        Returns:
            True if no problematic collisions detected
        """
        # Create and apply state
        state = self.create_initial_state(x, y, theta)

        # Set state in simulation
        self.base_env.sim.set_state_from_flattened(state['states'])
        self.base_env.sim.forward()

        # Check for collisions
        n_contacts = self.base_env.sim.data.ncon

        if n_contacts == 0:
            return True

        # Check each contact
        for i in range(n_contacts):
            contact = self.base_env.sim.data.contact[i]
            geom1_name = self.base_env.sim.model.geom_id2name(contact.geom1)
            geom2_name = self.base_env.sim.model.geom_id2name(contact.geom2)

            # Check for problematic collisions
            # Robot-object collision (excluding gripper which is allowed)
            is_robot = 'robot0' in geom1_name or 'robot0' in geom2_name
            is_gripper = 'gripper' in geom1_name or 'gripper' in geom2_name
            is_object = 'cube' in geom1_name or 'cube' in geom2_name or \
                       'object' in geom1_name or 'object' in geom2_name

            if is_robot and is_object and not is_gripper:
                return False  # Invalid collision

        return True

    def sample_uniform(self, n_samples: int = 1) -> Union[Tuple[float, float, float],
                                                           np.ndarray]:
        """
        Sample uniformly from valid state space.

        Args:
            n_samples: Number of samples to generate

        Returns:
            If n_samples=1: (x, y, theta) tuple
            If n_samples>1: array of shape (n_samples, 3)
        """
        x = np.random.uniform(*self.x_range, n_samples)
        y = np.random.uniform(*self.y_range, n_samples)
        theta = np.random.uniform(0, 2*np.pi, n_samples)

        if n_samples == 1:
            return float(x[0]), float(y[0]), float(theta[0])

        return np.stack([x, y, theta], axis=1)

    def sample_grid(self, nx: int = 10, ny: int = 10, ntheta: int = 8) -> np.ndarray:
        """
        Generate regular grid of states in valid space.

        Args:
            nx: Number of points in x direction
            ny: Number of points in y direction
            ntheta: Number of rotation samples

        Returns:
            Array of shape (nx*ny*ntheta, 3) with [x, y, theta] states
        """
        x = np.linspace(*self.x_range, nx)
        y = np.linspace(*self.y_range, ny)
        theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)

        X, Y, Theta = np.meshgrid(x, y, theta, indexing='ij')
        states = np.stack([X.ravel(), Y.ravel(), Theta.ravel()], axis=1)

        return states

    def sample_rejection(self, n_samples: int = 1,
                        max_attempts: int = 100) -> Union[Tuple[float, float, float],
                                                          np.ndarray]:
        """
        Sample with rejection (physics validation).

        Args:
            n_samples: Number of valid samples to generate
            max_attempts: Maximum rejection attempts per sample

        Returns:
            Valid samples that pass physics checks
        """
        valid_samples = []

        while len(valid_samples) < n_samples:
            # Sample candidate
            x, y, theta = self.sample_uniform(n_samples=1)

            # Check validity with physics
            attempts = 0
            while not self.is_valid_state(x, y, theta, check_physics=True):
                x, y, theta = self.sample_uniform(n_samples=1)
                attempts += 1
                if attempts >= max_attempts:
                    raise RuntimeError(f"Failed to sample valid state after {max_attempts} attempts")

            valid_samples.append([x, y, theta])

        valid_samples = np.array(valid_samples)

        if n_samples == 1:
            return tuple(valid_samples[0])

        return valid_samples

    def create_initial_state(self, x: float, y: float, theta: float,
                            robot_qpos: Optional[np.ndarray] = None) -> Dict:
        """
        Create initial state dictionary for env.reset_to().

        Args:
            x: Object x position
            y: Object y position
            theta: Object rotation around z-axis (radians)
            robot_qpos: Robot joint angles (7,). If None, uses self.robot_qpos

        Returns:
            State dict with 'states' key for reset_to()
        """
        # Get a template state
        obs = self.env.reset()
        template_state = self.env.get_state()

        # Copy and modify
        state_vector = template_state['states'].copy()

        # Set robot configuration
        qpos_robot = robot_qpos if robot_qpos is not None else self.robot_qpos
        state_vector[0:7] = qpos_robot

        # Set gripper configuration
        state_vector[7:9] = self.gripper_qpos

        # Set object position
        state_vector[9] = x
        state_vector[10] = y
        state_vector[11] = self.z

        # Set object orientation (quaternion from theta)
        # Rotation around z-axis: q = [cos(θ/2), 0, 0, sin(θ/2)]
        state_vector[12] = np.cos(theta / 2)  # quat_w
        state_vector[13] = 0.0                 # quat_x
        state_vector[14] = 0.0                 # quat_y
        state_vector[15] = np.sin(theta / 2)  # quat_z

        # Zero out velocities for clean initial state
        state_vector[16:31] = 0.0

        # Reset time
        state_vector[31] = 0.0

        return {"states": state_vector}

    def theta_to_quat(self, theta: float) -> np.ndarray:
        """
        Convert z-axis rotation to quaternion.

        Args:
            theta: Rotation angle in radians

        Returns:
            Quaternion [w, x, y, z]
        """
        return np.array([
            np.cos(theta / 2),
            0.0,
            0.0,
            np.sin(theta / 2)
        ])

    def quat_to_theta(self, quat: np.ndarray) -> float:
        """
        Extract z-axis rotation from quaternion.

        Args:
            quat: Quaternion [w, x, y, z]

        Returns:
            Rotation angle in radians [0, 2π)
        """
        w, x, y, z = quat
        theta = 2 * np.arctan2(z, w)
        return theta % (2 * np.pi)

    def get_state_from_obs(self, obs: Dict) -> Tuple[float, float, float]:
        """
        Extract (x, y, theta) from observation.

        Args:
            obs: Observation dictionary from environment

        Returns:
            (x, y, theta) tuple
        """
        if 'object' in obs:
            # robomimic format
            obj = obs['object']
            x, y, z = obj[:3]

            # Extract orientation if available
            if len(obj) >= 7:
                quat = obj[3:7]  # [x, y, z, w] format in obs
                # Convert to [w, x, y, z] for quat_to_theta
                quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
                theta = self.quat_to_theta(quat_wxyz)
            else:
                theta = 0.0
        elif 'cube_pos' in obs:
            # robosuite format
            x, y, z = obs['cube_pos']
            if 'cube_quat' in obs:
                quat = obs['cube_quat']  # [w, x, y, z] format
                theta = self.quat_to_theta(quat)
            else:
                theta = 0.0
        else:
            raise ValueError("Cannot find object position in observation")

        return x, y, theta

    def visualize_bounds(self, ax=None, show_grid: bool = True):
        """
        Visualize valid state space bounds.

        Args:
            ax: matplotlib axis (created if None)
            show_grid: If True, show sample grid points

        Returns:
            matplotlib axis
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("matplotlib not available for visualization")
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # Draw valid region
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        width = x_max - x_min
        height = y_max - y_min

        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor='b', facecolor='lightblue', alpha=0.3,
            label='Valid region'
        )
        ax.add_patch(rect)

        # Show grid if requested
        if show_grid:
            grid = self.sample_grid(nx=5, ny=5, ntheta=1)
            ax.scatter(grid[:, 0], grid[:, 1], c='red', s=20, alpha=0.6,
                      label='Sample points')

        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_title('Object Placement Constraints')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()

        return ax

    def __repr__(self) -> str:
        return (f"ConstraintAwareSampler(\n"
                f"  x_range={self.x_range},\n"
                f"  y_range={self.y_range},\n"
                f"  z={self.z:.3f},\n"
                f"  rotation_axis='{self.rotation_axis}'\n"
                f")")
