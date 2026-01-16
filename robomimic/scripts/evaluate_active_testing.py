#!/usr/bin/env python
"""
Active Testing for Sample-Efficient Policy Evaluation.

This script implements an active testing loop using a Gaussian Process (GP)
surrogate model to predict success rates across the object pose space (ODD).
The acquisition function uses GP variance to find maximally uncertain samples,
enabling more sample-efficient evaluation.

The final success rate is estimated using the ASE (Average Success Estimate)
mechanism: uniformly sampling points from the ODD and averaging the GP's
predicted success rates.

Usage:
    python evaluate_active_testing.py \
        --checkpoint /path/to/model.pth \
        --total_budget 100 \
        --init_samples 10 \
        --n_seeds 5 \
        --num_workers 8

Output:
    - A plot showing mean success rate estimate and variance vs number of rollouts
    - Comparison between active and random sampling
    - Raw data saved as .npz file
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from abc import ABC, abstractmethod
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import norm

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

# =============================================================================
# Gaussian Process Implementation
# =============================================================================


class GaussianProcess:
    """
    Simple Gaussian Process for regression with RBF kernel.

    Predicts success probability as a function of object pose (x, y, theta).
    """

    def __init__(
        self,
        length_scales: np.ndarray = None,
        signal_variance: float = 1.0,
        noise_variance: float = 0.1,
        normalize_inputs: bool = True,
    ):
        """
        Initialize GP with RBF kernel.

        Args:
            length_scales: Per-dimension length scales for RBF kernel.
                           If None, defaults to [0.05, 0.05, 1.0] for (x, y, theta).
            signal_variance: Signal variance (amplitude) of the kernel.
            noise_variance: Observation noise variance.
            normalize_inputs: Whether to normalize inputs to [0, 1].
        """
        self.length_scales = (
            length_scales if length_scales is not None else np.array([0.05, 0.05, 1.0])
        )
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.normalize_inputs = normalize_inputs

        # Training data
        self.X_train = None
        self.y_train = None
        self.K_inv = None

        # Normalization parameters
        self.X_min = None
        self.X_max = None

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize inputs to [0, 1] range."""
        if not self.normalize_inputs or self.X_min is None:
            return X
        return (X - self.X_min) / (self.X_max - self.X_min + 1e-8)

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute RBF (squared exponential) kernel between two sets of points.

        K(x, x') = signal_variance * exp(-0.5 * sum_d ((x_d - x'_d) / l_d)^2)
        """
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

        # Scaled squared distances
        X1_scaled = X1 / self.length_scales
        X2_scaled = X2 / self.length_scales

        # Compute pairwise squared distances
        sq_dist = (
            np.sum(X1_scaled**2, axis=1, keepdims=True)
            + np.sum(X2_scaled**2, axis=1)
            - 2 * X1_scaled @ X2_scaled.T
        )

        return self.signal_variance * np.exp(-0.5 * sq_dist)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the GP to training data.

        Args:
            X: Training inputs of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        # Store normalization parameters
        if self.normalize_inputs:
            self.X_min = X.min(axis=0)
            self.X_max = X.max(axis=0)

        self.X_train = self._normalize(X)
        self.y_train = y

        # Compute kernel matrix with noise
        K = self._rbf_kernel(self.X_train, self.X_train)
        K += self.noise_variance * np.eye(len(X))

        # Compute inverse (with regularization for numerical stability)
        try:
            self.K_inv = np.linalg.inv(K + 1e-6 * np.eye(len(X)))
        except np.linalg.LinAlgError:
            # Use pseudo-inverse as fallback
            self.K_inv = np.linalg.pinv(K)

    def predict(
        self, X: np.ndarray, return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict mean and optionally standard deviation at new points.

        Args:
            X: Test inputs of shape (n_samples, n_features)
            return_std: Whether to return standard deviation

        Returns:
            mean: Predicted means of shape (n_samples,)
            std: Predicted standard deviations (if return_std=True)
        """
        if self.X_train is None:
            raise ValueError("GP has not been fitted yet. Call fit() first.")

        X = np.atleast_2d(X)
        X_norm = self._normalize(X)

        # Compute kernel between test and training points
        K_star = self._rbf_kernel(X_norm, self.X_train)

        # Predictive mean
        mean = K_star @ self.K_inv @ self.y_train

        # Clip to valid probability range
        mean = np.clip(mean, 0, 1)

        if not return_std:
            return mean, None

        # Predictive variance
        K_star_star = self._rbf_kernel(X_norm, X_norm)
        var = np.diag(K_star_star) - np.sum(K_star @ self.K_inv * K_star, axis=1)
        var = np.maximum(var, 1e-8)  # Numerical stability
        std = np.sqrt(var)

        return mean, std

    def update(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        Update the GP with new observations (incremental update).

        Args:
            X_new: New inputs of shape (n_new, n_features)
            y_new: New targets of shape (n_new,)
        """
        X_new = np.atleast_2d(X_new)
        y_new = np.atleast_1d(y_new)

        if self.X_train is None:
            self.fit(X_new, y_new)
        else:
            # Denormalize existing data, concatenate, and refit
            if self.normalize_inputs and self.X_min is not None:
                X_orig = self.X_train * (self.X_max - self.X_min + 1e-8) + self.X_min
            else:
                X_orig = self.X_train

            X_combined = np.vstack([X_orig, X_new])
            y_combined = np.concatenate([self.y_train, y_new])
            self.fit(X_combined, y_combined)


# =============================================================================
# Acquisition Functions
# =============================================================================


class AcquisitionFunction(ABC):
    """Base class for acquisition functions."""

    @abstractmethod
    def __call__(self, X: np.ndarray, gp: GaussianProcess) -> np.ndarray:
        """
        Compute acquisition value at points X.

        Args:
            X: Points to evaluate, shape (n_points, n_features)
            gp: Fitted Gaussian Process

        Returns:
            Acquisition values, shape (n_points,)
        """
        pass


class MaxVarianceAcquisition(AcquisitionFunction):
    """
    Maximum variance (uncertainty) acquisition function.

    Selects points with highest predictive uncertainty.
    """

    def __call__(self, X: np.ndarray, gp: GaussianProcess) -> np.ndarray:
        _, std = gp.predict(X, return_std=True)
        return std


class UCBAcquisition(AcquisitionFunction):
    """
    Upper Confidence Bound acquisition function.

    UCB(x) = mu(x) + beta * sigma(x)

    Can be used to explore high-uncertainty regions that might also have
    high success rates.
    """

    def __init__(self, beta: float = 2.0):
        self.beta = beta

    def __call__(self, X: np.ndarray, gp: GaussianProcess) -> np.ndarray:
        mean, std = gp.predict(X, return_std=True)
        return mean + self.beta * std


class LCBAcquisition(AcquisitionFunction):
    """
    Lower Confidence Bound acquisition function.

    LCB(x) = mu(x) - beta * sigma(x)

    Useful for finding regions of potential failure.
    """

    def __init__(self, beta: float = 2.0):
        self.beta = beta

    def __call__(self, X: np.ndarray, gp: GaussianProcess) -> np.ndarray:
        mean, std = gp.predict(X, return_std=True)
        return -(mean - self.beta * std)  # Negate for maximization


# =============================================================================
# Success Rate Estimators
# =============================================================================


class SuccessRateEstimator(ABC):
    """Base class for success rate estimation mechanisms."""

    @abstractmethod
    def estimate(
        self,
        gp: GaussianProcess,
        observed_X: np.ndarray,
        observed_y: np.ndarray,
        bounds: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Estimate the overall success rate.

        Args:
            gp: Fitted Gaussian Process
            observed_X: Observed inputs, shape (n_obs, n_features)
            observed_y: Observed outcomes, shape (n_obs,)
            bounds: Input bounds, shape (n_features, 2)

        Returns:
            mean_estimate: Estimated mean success rate
            std_estimate: Estimated uncertainty in the estimate
        """
        pass


class ASEEstimator(SuccessRateEstimator):
    """
    Average Success Estimate (ASE) mechanism.

    Estimates success rate by uniformly sampling points from the ODD
    and averaging the GP's predicted success rates.
    """

    def __init__(self, n_samples: int = 1000):
        """
        Args:
            n_samples: Number of samples for Monte Carlo integration
        """
        self.n_samples = n_samples

    def estimate(
        self,
        gp: GaussianProcess,
        observed_X: np.ndarray,
        observed_y: np.ndarray,
        bounds: np.ndarray,
    ) -> Tuple[float, float]:
        # Generate uniform samples from the ODD
        n_features = bounds.shape[0]
        samples = np.random.uniform(
            bounds[:, 0], bounds[:, 1], size=(self.n_samples, n_features)
        )

        # Get GP predictions
        mean, std = gp.predict(samples, return_std=True)

        # ASE: average of predicted means
        mean_estimate = np.mean(mean)

        # Uncertainty: average of predicted uncertainties
        # (could also use variance of predictions)
        std_estimate = np.mean(std)

        return mean_estimate, std_estimate


class EmpiricalEstimator(SuccessRateEstimator):
    """
    Simple empirical estimator using observed success rates.

    This is the baseline: just average the observed outcomes.
    """

    def estimate(
        self,
        gp: GaussianProcess,
        observed_X: np.ndarray,
        observed_y: np.ndarray,
        bounds: np.ndarray,
    ) -> Tuple[float, float]:
        mean_estimate = np.mean(observed_y)
        # Standard error of the mean
        std_estimate = np.std(observed_y) / np.sqrt(len(observed_y))
        return mean_estimate, std_estimate


class LUREEstimator(SuccessRateEstimator):
    """
    LURE (Learned Uncertainty for Robust Estimation) mechanism.

    Uses importance weighting based on GP uncertainty to compute
    a weighted average of observed outcomes.
    """

    def __init__(self, n_samples: int = 1000):
        self.n_samples = n_samples

    def estimate(
        self,
        gp: GaussianProcess,
        observed_X: np.ndarray,
        observed_y: np.ndarray,
        bounds: np.ndarray,
    ) -> Tuple[float, float]:
        # Get GP predictions at observed points
        _, std_obs = gp.predict(observed_X, return_std=True)

        # Inverse variance weighting (higher uncertainty = lower weight)
        weights = 1.0 / (std_obs + 1e-6)
        weights = weights / np.sum(weights)

        # Weighted average
        mean_estimate = np.sum(weights * observed_y)

        # Weighted standard error
        weighted_var = np.sum(weights * (observed_y - mean_estimate) ** 2)
        std_estimate = np.sqrt(weighted_var)

        return mean_estimate, std_estimate


# =============================================================================
# State Sampler (from evaluate_variance_vs_rollouts.py)
# =============================================================================


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
            self.z = sampler.reference_pos[2] + sampler.z_offset
        else:
            self.x_range = [-0.1, 0.1]
            self.y_range = [-0.1, 0.1]
            self.z = 0.82

        self._find_state_indices()

        obs = self.env.reset()
        self.template_state = self.env.get_state()

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

    def get_bounds(self) -> np.ndarray:
        """Get bounds for the ODD (object pose space)."""
        return np.array(
            [
                [self.x_range[0], self.x_range[1]],
                [self.y_range[0], self.y_range[1]],
                [0, 2 * np.pi],  # theta
            ]
        )

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


# =============================================================================
# Active Testing Loop
# =============================================================================


class ActiveTester:
    """
    Active testing framework for sample-efficient policy evaluation.

    Uses a GP surrogate model to predict success rates and an acquisition
    function to select maximally informative samples.
    """

    def __init__(
        self,
        gp: GaussianProcess,
        acquisition: AcquisitionFunction,
        estimator: SuccessRateEstimator,
        bounds: np.ndarray,
        n_candidates: int = 1000,
    ):
        """
        Args:
            gp: Gaussian Process surrogate model
            acquisition: Acquisition function for sample selection
            estimator: Success rate estimation mechanism
            bounds: Input bounds, shape (n_features, 2)
            n_candidates: Number of candidate points for acquisition optimization
        """
        self.gp = gp
        self.acquisition = acquisition
        self.estimator = estimator
        self.bounds = bounds
        self.n_candidates = n_candidates

        # History
        self.X_observed = None
        self.y_observed = None
        self.estimates_history = []  # (mean, std) at each step

    def initialize_incremental(self, X_init: np.ndarray, y_init: np.ndarray):
        """
        Initialize incrementally, recording an estimate after each sample.

        This ensures estimates_history has one entry per sample.

        Args:
            X_init: Initial inputs, shape (n_init, n_features)
            y_init: Initial outcomes, shape (n_init,)
        """
        n_init = len(X_init)

        for i in range(n_init):
            x_new = X_init[i : i + 1]
            y_new = y_init[i : i + 1]

            if self.X_observed is None:
                self.X_observed = x_new
                self.y_observed = y_new
            else:
                self.X_observed = np.vstack([self.X_observed, x_new])
                self.y_observed = np.concatenate([self.y_observed, y_new])

            # Fit GP with all data so far
            self.gp.fit(self.X_observed, self.y_observed)

            # Record estimate after this sample
            mean_est, std_est = self.estimator.estimate(
                self.gp, self.X_observed, self.y_observed, self.bounds
            )
            self.estimates_history.append((mean_est, std_est))

    def select_next_sample(self) -> np.ndarray:
        """
        Select the next sample to evaluate using the acquisition function.

        Returns:
            next_point: Selected point, shape (n_features,)
        """
        # Generate candidate points
        n_features = self.bounds.shape[0]
        candidates = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], size=(self.n_candidates, n_features)
        )

        # Evaluate acquisition function
        acq_values = self.acquisition(candidates, self.gp)

        # Select best candidate
        best_idx = np.argmax(acq_values)
        return candidates[best_idx]

    def update(self, x_new: np.ndarray, y_new: float):
        """
        Update the GP with a new observation and record the estimate.

        Args:
            x_new: New input, shape (n_features,)
            y_new: New outcome (0 or 1)
        """
        x_new = np.atleast_2d(x_new)
        y_new = np.atleast_1d(y_new)

        # Update observations
        if self.X_observed is None:
            self.X_observed = x_new
            self.y_observed = y_new
        else:
            self.X_observed = np.vstack([self.X_observed, x_new])
            self.y_observed = np.concatenate([self.y_observed, y_new])

        # Update GP
        self.gp.update(x_new, y_new)

        # Record estimate
        mean_est, std_est = self.estimator.estimate(
            self.gp, self.X_observed, self.y_observed, self.bounds
        )
        self.estimates_history.append((mean_est, std_est))

    def get_estimates_history(self) -> np.ndarray:
        """
        Get history of success rate estimates.

        Returns:
            Array of shape (n_steps, 2) with (mean, std) at each step
        """
        return np.array(self.estimates_history)


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_single_state(
    env, policy, sampler, x, y, theta, horizon, n_trials=1
) -> float:
    """
    Evaluate policy success at a single object state.

    Args:
        env: Environment
        policy: Policy to evaluate
        sampler: StateSampler for creating state dicts
        x, y, theta: Object pose
        horizon: Rollout horizon
        n_trials: Number of trials (for stochastic policies)

    Returns:
        Success rate (0 to 1)
    """
    state_dict = sampler.create_state_dict(x, y, theta)
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

    return np.mean(trial_successes)


def run_active_testing_seed(
    seed: int,
    checkpoint_path: str,
    horizon: int,
    total_budget: int,
    init_samples: int,
    n_trials: int,
    device_id: Optional[int],
    fixed_robot_qpos: Optional[np.ndarray],
    fixed_gripper_qpos: Optional[np.ndarray],
    gp_length_scales: np.ndarray,
    estimator_type: str = "ase",
    ase_n_samples: int = 1000,
    acquisition_type: str = "max_variance",
    save_gp_steps: bool = False,
    gp_steps_dir: Optional[Path] = None,
) -> dict:
    """
    Run active testing loop for a single seed.

    Args:
        save_gp_steps: If True, save GP visualization at each step
        gp_steps_dir: Directory to save GP step visualizations

    Returns:
        Dictionary with results including estimates history and sampled states
    """
    # Suppress stdout/stderr in worker
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = TorchUtils.get_torch_device(try_to_use_cuda=(device_id is not None))

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=checkpoint_path, device=device, verbose=False
    )

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, render=False, render_offscreen=False, verbose=False
    )

    sampler = StateSampler(
        env, fixed_robot_qpos=fixed_robot_qpos, fixed_gripper_qpos=fixed_gripper_qpos
    )
    bounds = sampler.get_bounds()

    # Create GP and components
    gp = GaussianProcess(length_scales=gp_length_scales)

    if acquisition_type == "max_variance":
        acquisition = MaxVarianceAcquisition()
    elif acquisition_type == "ucb":
        acquisition = UCBAcquisition(beta=2.0)
    elif acquisition_type == "lcb":
        acquisition = LCBAcquisition(beta=2.0)
    else:
        acquisition = MaxVarianceAcquisition()

    if estimator_type == "ase":
        estimator = ASEEstimator(n_samples=ase_n_samples)
    elif estimator_type == "empirical":
        estimator = EmpiricalEstimator()
    elif estimator_type == "lure":
        estimator = LUREEstimator(n_samples=ase_n_samples)
    else:
        estimator = ASEEstimator(n_samples=ase_n_samples)

    tester = ActiveTester(gp, acquisition, estimator, bounds)

    # Phase 1: Random initialization
    init_states = sampler.sample_random(n_samples=init_samples, check_collision=True)
    init_results = np.zeros(init_samples)

    for i, (x, y, theta) in enumerate(init_states):
        init_results[i] = evaluate_single_state(
            env, policy, sampler, x, y, theta, horizon, n_trials
        )

    tester.initialize_incremental(init_states, init_results)

    # Track all sampled states
    all_states = [init_states.copy()]
    all_results = [init_results.copy()]
    acquisition_values = []

    # Save GP visualizations during initialization if requested
    if save_gp_steps and gp_steps_dir is not None:
        for step in range(1, init_samples + 1):
            states_so_far = init_states[:step]
            results_so_far = init_results[:step]
            gp_temp = GaussianProcess(length_scales=gp_length_scales)
            gp_temp.fit(states_so_far, results_so_far)

            create_gp_visualization(
                states_so_far,
                results_so_far,
                bounds,
                gp_temp,
                gp_steps_dir / f"gp_step_{step:03d}.png",
                init_samples,
                step=step,
            )

    # Phase 2: Active sampling
    for i in range(total_budget - init_samples):
        # Select next sample
        next_point = tester.select_next_sample()
        x, y, theta = next_point

        # Ensure collision-free (resample if needed)
        attempts = 0
        while not sampler._check_collision(x, y, theta) and attempts < 100:
            # Perturb slightly and try again
            next_point = tester.select_next_sample()
            x, y, theta = next_point
            attempts += 1

        if attempts >= 100:
            # Fall back to random sample
            fallback = sampler.sample_random(n_samples=1, check_collision=True)
            x, y, theta = fallback[0]
            next_point = fallback[0]

        # Evaluate
        result = evaluate_single_state(
            env, policy, sampler, x, y, theta, horizon, n_trials
        )

        # Update tester
        tester.update(next_point, result)

        all_states.append(next_point.reshape(1, -1))
        all_results.append(np.array([result]))

        # Save GP visualization at this step if requested
        if save_gp_steps and gp_steps_dir is not None:
            step = init_samples + i + 1
            states_so_far = np.vstack(all_states)
            results_so_far = np.concatenate(all_results)
            gp_temp = GaussianProcess(length_scales=gp_length_scales)
            gp_temp.fit(states_so_far, results_so_far)

            create_gp_visualization(
                states_so_far,
                results_so_far,
                bounds,
                gp_temp,
                gp_steps_dir / f"gp_step_{step:03d}.png",
                init_samples,
                step=step,
            )

    # Also run random baseline for comparison
    random_states = sampler.sample_random(n_samples=total_budget, check_collision=True)
    random_results = np.zeros(total_budget)

    for i, (x, y, theta) in enumerate(random_states):
        random_results[i] = evaluate_single_state(
            env, policy, sampler, x, y, theta, horizon, n_trials
        )

    # Compute random baseline estimates (cumulative mean)
    random_estimates = np.zeros((total_budget, 2))
    for i in range(total_budget):
        random_estimates[i, 0] = np.mean(random_results[: i + 1])
        random_estimates[i, 1] = np.std(random_results[: i + 1]) / np.sqrt(i + 1)

    return {
        "seed": seed,
        "active_estimates": tester.get_estimates_history(),
        "active_states": np.vstack(all_states),
        "active_results": np.concatenate(all_results),
        "random_estimates": random_estimates,
        "random_states": random_states,
        "random_results": random_results,
        "bounds": bounds,
    }


def run_active_testing_worker(args):
    """Worker function for multiprocessing."""
    return run_active_testing_seed(*args)


# =============================================================================
# Visualization
# =============================================================================


def create_comparison_plot(
    active_estimates_all: np.ndarray,
    random_estimates_all: np.ndarray,
    output_path: Path,
    init_samples: int,
):
    """
    Create comparison plot showing active vs random sampling performance.

    Args:
        active_estimates_all: Shape (n_seeds, total_budget, 2) - (mean, std)
        random_estimates_all: Shape (n_seeds, total_budget, 2) - (mean, std)
        output_path: Path to save plot
        init_samples: Number of initial random samples
    """
    n_seeds, total_budget, _ = active_estimates_all.shape
    rollout_counts = np.arange(1, total_budget + 1)

    # Compute statistics across seeds
    active_means = active_estimates_all[:, :, 0]  # (n_seeds, total_budget)
    random_means = random_estimates_all[:, :, 0]

    active_mean_of_means = np.mean(active_means, axis=0)
    active_std_of_means = np.std(active_means, axis=0)

    random_mean_of_means = np.mean(random_means, axis=0)
    random_std_of_means = np.std(random_means, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Mean estimates comparison
    ax = axes[0, 0]
    ax.plot(
        rollout_counts,
        active_mean_of_means,
        "b-",
        linewidth=2,
        label="Active (GP-based)",
    )
    ax.fill_between(
        rollout_counts,
        active_mean_of_means - active_std_of_means,
        active_mean_of_means + active_std_of_means,
        alpha=0.3,
        color="blue",
    )
    ax.plot(
        rollout_counts, random_mean_of_means, "r-", linewidth=2, label="Random Baseline"
    )
    ax.fill_between(
        rollout_counts,
        random_mean_of_means - random_std_of_means,
        random_mean_of_means + random_std_of_means,
        alpha=0.3,
        color="red",
    )
    ax.axvline(
        init_samples,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Init phase ({init_samples} samples)",
    )
    ax.set_xlabel("Number of Rollouts", fontsize=12)
    ax.set_ylabel("Estimated Success Rate", fontsize=12)
    ax.set_title(
        "Success Rate Estimate: Active vs Random", fontsize=14, fontweight="bold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, total_budget)
    ax.set_ylim(0, 1)

    # Plot 2: Variance comparison
    ax = axes[0, 1]
    ax.plot(
        rollout_counts,
        active_std_of_means,
        "b-",
        linewidth=2,
        label="Active (GP-based)",
    )
    ax.plot(
        rollout_counts, random_std_of_means, "r-", linewidth=2, label="Random Baseline"
    )
    ax.axvline(init_samples, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of Rollouts", fontsize=12)
    ax.set_ylabel("Std Dev of Estimate Across Seeds", fontsize=12)
    ax.set_title(
        "Estimation Variance: Active vs Random", fontsize=14, fontweight="bold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, total_budget)

    # Plot 3: Individual seed trajectories (active)
    ax = axes[1, 0]
    for seed_idx in range(min(n_seeds, 10)):
        ax.plot(
            rollout_counts,
            active_means[seed_idx],
            alpha=0.5,
            linewidth=1,
            label=f"Seed {seed_idx}" if seed_idx < 5 else None,
        )
    ax.plot(rollout_counts, active_mean_of_means, "k-", linewidth=3, label="Mean")
    ax.axvline(init_samples, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of Rollouts", fontsize=12)
    ax.set_ylabel("Estimated Success Rate", fontsize=12)
    ax.set_title(
        "Active Sampling: Individual Seed Trajectories", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, total_budget)
    ax.set_ylim(0, 1)

    # Plot 4: Convergence comparison (absolute difference from final estimate)
    ax = axes[1, 1]
    # Use final estimate as "ground truth"
    active_final = active_mean_of_means[-1]
    random_final = random_mean_of_means[-1]

    active_convergence = np.abs(active_mean_of_means - active_final)
    random_convergence = np.abs(random_mean_of_means - random_final)

    ax.plot(
        rollout_counts, active_convergence, "b-", linewidth=2, label="Active (GP-based)"
    )
    ax.plot(
        rollout_counts, random_convergence, "r-", linewidth=2, label="Random Baseline"
    )
    ax.axvline(init_samples, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of Rollouts", fontsize=12)
    ax.set_ylabel("|Estimate - Final Estimate|", fontsize=12)
    ax.set_title("Convergence to Final Estimate", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, total_budget)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_gp_visualization(
    active_states: np.ndarray,
    active_results: np.ndarray,
    bounds: np.ndarray,
    gp: GaussianProcess,
    output_path: Path,
    init_samples: int,
    step: Optional[int] = None,
):
    """
    Create visualization of GP predictions and sampled points.

    Args:
        active_states: Sampled states, shape (n_samples, 3)
        active_results: Evaluation results, shape (n_samples,)
        bounds: Input bounds
        gp: Fitted GP
        output_path: Path to save plot
        init_samples: Number of initial samples
        step: Current evaluation step (for title), None for final plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Create grid for GP predictions
    n_grid = 50
    x_grid = np.linspace(bounds[0, 0], bounds[0, 1], n_grid)
    y_grid = np.linspace(bounds[1, 0], bounds[1, 1], n_grid)
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)

    # Predict at grid points (fixing theta to mean)
    theta_mean = np.pi
    grid_points = np.column_stack(
        [X_mesh.ravel(), Y_mesh.ravel(), np.full(n_grid * n_grid, theta_mean)]
    )

    gp.fit(active_states, active_results)
    mean_pred, std_pred = gp.predict(grid_points, return_std=True)
    mean_pred = mean_pred.reshape(n_grid, n_grid)
    std_pred = std_pred.reshape(n_grid, n_grid)

    # Plot 1: GP mean prediction
    ax = axes[0]
    im = ax.contourf(
        X_mesh, Y_mesh, mean_pred, levels=20, cmap="RdYlGn", vmin=0, vmax=1
    )
    ax.scatter(
        active_states[:init_samples, 0],
        active_states[:init_samples, 1],
        c="blue",
        s=50,
        marker="o",
        edgecolors="white",
        linewidth=1,
        label="Init samples",
    )
    ax.scatter(
        active_states[init_samples:, 0],
        active_states[init_samples:, 1],
        c="red",
        s=50,
        marker="x",
        linewidth=2,
        label="Active samples",
    )
    ax.set_xlabel("X position (m)", fontsize=12)
    ax.set_ylabel("Y position (m)", fontsize=12)
    title_suffix = f" - Step {step}" if step is not None else ""
    ax.set_title(
        f"GP Mean Prediction (theta={theta_mean:.2f}){title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    plt.colorbar(im, ax=ax, label="Predicted Success Rate")

    # Plot 2: GP uncertainty
    ax = axes[1]
    im = ax.contourf(X_mesh, Y_mesh, std_pred, levels=20, cmap="YlOrRd")
    ax.scatter(
        active_states[:init_samples, 0],
        active_states[:init_samples, 1],
        c="blue",
        s=50,
        marker="o",
        edgecolors="white",
        linewidth=1,
    )
    ax.scatter(
        active_states[init_samples:, 0],
        active_states[init_samples:, 1],
        c="black",
        s=50,
        marker="x",
        linewidth=2,
    )
    ax.set_xlabel("X position (m)", fontsize=12)
    ax.set_ylabel("Y position (m)", fontsize=12)
    ax.set_title(
        f"GP Uncertainty (Std Dev){title_suffix}", fontsize=14, fontweight="bold"
    )
    plt.colorbar(im, ax=ax, label="Std Dev")

    # Plot 3: Actual evaluation results
    ax = axes[2]
    scatter = ax.scatter(
        active_states[:, 0],
        active_states[:, 1],
        c=active_results,
        s=80,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.set_xlabel("X position (m)", fontsize=12)
    ax.set_ylabel("Y position (m)", fontsize=12)
    ax.set_title(
        f"Actual Evaluation Results{title_suffix}", fontsize=14, fontweight="bold"
    )
    plt.colorbar(scatter, ax=ax, label="Success Rate")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================


def run_evaluation(args):
    """Main evaluation function."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Active Testing for Sample-Efficient Policy Evaluation")
    print("=" * 70)

    # Load environment to get constraints
    print("\n[1/5] Loading environment...")
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

    # Extract fixed robot state if needed
    if args.fix_robot_state:
        fixed_robot_qpos = sampler.robot_qpos.copy()
        fixed_gripper_qpos = sampler.gripper_qpos.copy()
        print(f"  Mode: FIXED robot state")
    else:
        fixed_robot_qpos = None
        fixed_gripper_qpos = None
        print(f"  Mode: RANDOMIZED robot state")

    # GP hyperparameters
    gp_length_scales = np.array(
        [args.gp_length_scale_xy, args.gp_length_scale_xy, args.gp_length_scale_theta]
    )
    print(f"  GP length scales: {gp_length_scales}")
    print(f"  Acquisition function: {args.acquisition}")
    print(f"  Estimator: {args.estimator}")

    # Prepare worker arguments
    print(f"\n[2/5] Preparing {args.n_seeds} seed evaluations...")
    print(f"  Total budget: {args.total_budget}")
    print(f"  Initial random samples: {args.init_samples}")
    print(f"  Workers: {args.num_workers}")

    # Create GP steps directory if saving visualizations (only for first seed)
    gp_steps_dir = None
    if args.save_gp_steps:
        gp_steps_dir = output_dir / "gp_steps"
        gp_steps_dir.mkdir(exist_ok=True)
        print(f"  Saving GP visualizations to: {gp_steps_dir}")

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
            args.total_budget,
            args.init_samples,
            args.n_trials,
            device_assignment[i % args.num_workers],
            fixed_robot_qpos,
            fixed_gripper_qpos,
            gp_length_scales,
            args.estimator,
            args.ase_n_samples,
            args.acquisition,
            args.save_gp_steps and i == 0,  # Only save for first seed
            gp_steps_dir,
        )
        for i, seed in enumerate(seeds)
    ]

    # Run evaluations
    print(f"\n[3/5] Running active testing...")
    t_start = time.time()

    if args.num_workers == 1:
        results = [run_active_testing_worker(arg) for arg in worker_args]
    else:
        with Pool(processes=args.num_workers) as pool:
            results = []
            for i, result in enumerate(
                pool.imap_unordered(run_active_testing_worker, worker_args)
            ):
                results.append(result)
                elapsed = time.time() - t_start
                print(
                    f"  [{i + 1}/{args.n_seeds}] Seed {result['seed']} completed ({elapsed:.1f}s)"
                )

    t_elapsed = time.time() - t_start

    # Sort by seed
    results.sort(key=lambda x: x["seed"])

    # Extract data
    active_estimates_all = np.array([r["active_estimates"] for r in results])
    random_estimates_all = np.array([r["random_estimates"] for r in results])
    active_states_all = [r["active_states"] for r in results]
    active_results_all = [r["active_results"] for r in results]
    random_states_all = [r["random_states"] for r in results]
    random_results_all = [r["random_results"] for r in results]
    bounds = results[0]["bounds"]

    print(f"  Completed in {t_elapsed:.1f}s")

    # Create plots
    print(f"\n[4/5] Creating visualizations...")

    create_comparison_plot(
        active_estimates_all,
        random_estimates_all,
        output_dir / "active_vs_random.png",
        args.init_samples,
    )
    print(f"  Saved comparison plot")

    # Create GP visualization for first seed
    gp = GaussianProcess(length_scales=gp_length_scales)
    create_gp_visualization(
        active_states_all[0],
        active_results_all[0],
        bounds,
        gp,
        output_dir / "gp_visualization.png",
        args.init_samples,
    )
    print(f"  Saved GP visualization")

    # Save data
    print(f"\n[5/5] Saving results...")
    np.savez(
        output_dir / "active_testing_results.npz",
        active_estimates_all=active_estimates_all,
        random_estimates_all=random_estimates_all,
        active_states_all=np.array(active_states_all, dtype=object),
        active_results_all=np.array(active_results_all, dtype=object),
        random_states_all=np.array(random_states_all, dtype=object),
        random_results_all=np.array(random_results_all, dtype=object),
        bounds=bounds,
        seeds=np.array(seeds),
    )

    # Save config
    config = vars(args).copy()
    config["evaluation_time_seconds"] = t_elapsed
    config["seeds_used"] = seeds
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Seeds evaluated: {args.n_seeds}")
    print(f"Total budget: {args.total_budget}")
    print(f"Initial samples: {args.init_samples}")
    print(f"Evaluation time: {t_elapsed:.1f}s")

    # Final estimates comparison
    active_final = np.mean(active_estimates_all[:, -1, 0])
    active_final_std = np.std(active_estimates_all[:, -1, 0])
    random_final = np.mean(random_estimates_all[:, -1, 0])
    random_final_std = np.std(random_estimates_all[:, -1, 0])

    print(f"\nFinal success rate estimates:")
    print(f"  Active (ASE): {active_final:.3f} +/- {active_final_std:.3f}")
    print(f"  Random:       {random_final:.3f} +/- {random_final_std:.3f}")

    # Statistics at different sample counts
    for n_samples in [args.init_samples, args.total_budget // 2, args.total_budget]:
        if n_samples <= args.total_budget:
            idx = n_samples - 1
            active_rates = active_estimates_all[:, idx, 0]
            random_rates = random_estimates_all[:, idx, 0]
            print(f"\nAt {n_samples} samples:")
            print(
                f"  Active: {np.mean(active_rates):.3f} +/- {np.std(active_rates):.3f}"
            )
            print(
                f"  Random: {np.mean(random_rates):.3f} +/- {np.std(random_rates):.3f}"
            )

    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Active testing for sample-efficient policy evaluation"
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained policy checkpoint (.pth)",
    )

    # Budget and sampling
    parser.add_argument(
        "--total_budget",
        type=int,
        default=100,
        help="Total number of evaluations (rollouts)",
    )
    parser.add_argument(
        "--init_samples",
        type=int,
        default=10,
        help="Number of initial random samples for GP initialization",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="Number of seeds for repeated experiments",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=1,
        help="Number of trials per state (for stochastic policies)",
    )

    # GP hyperparameters
    parser.add_argument(
        "--gp_length_scale_xy",
        type=float,
        default=0.15,
        help="GP length scale for x and y dimensions",
    )
    parser.add_argument(
        "--gp_length_scale_theta",
        type=float,
        default=1.0,
        help="GP length scale for theta dimension",
    )

    # Acquisition and estimation
    parser.add_argument(
        "--acquisition",
        type=str,
        default="max_variance",
        choices=["max_variance", "ucb", "lcb"],
        help="Acquisition function for active sampling",
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default="ase",
        choices=["ase", "empirical", "lure"],
        help="Success rate estimation mechanism",
    )
    parser.add_argument(
        "--ase_n_samples",
        type=int,
        default=1000,
        help="Number of samples for ASE Monte Carlo integration",
    )

    # Environment
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Rollout horizon (default: from checkpoint)",
    )
    parser.add_argument(
        "--fix_robot_state",
        action="store_true",
        help="Fix robot state across evaluations",
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
        help="GPU IDs to use",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="active_testing_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--save_gp_steps",
        action="store_true",
        help="Save GP visualization at every evaluation step (only for first seed)",
    )

    args = parser.parse_args()
    run_evaluation(args)
