# Active Level Set Estimation (LSE) for Policy Evaluation

This directory contains tools for performing Active Level Set Estimation to evaluate policy robustness across different initial state configurations in robosuite environments.

## Overview

**Problem**: You have a trained policy and want to understand which initial object placements lead to success vs. failure.

**Solution**: Use Active LSE to efficiently explore the state space and learn the decision boundary (level set) between successful and failed states.

**Key Features**:
- ✅ Respects robosuite's placement constraints automatically
- ✅ Validates states with collision checking
- ✅ Provides constraint-aware acquisition functions
- ✅ Works seamlessly with robomimic policies

## Quick Start

### 1. Basic Usage

```python
from active_lse_sampler import ConstraintAwareSampler
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils

# Load environment
env, _ = FileUtils.env_from_checkpoint(ckpt_path='model.pth')

# Create constraint-aware sampler
sampler = ConstraintAwareSampler(env)

# Sample valid states
x, y, theta = sampler.sample_uniform()

# Create initial state
state_dict = sampler.create_initial_state(x, y, theta)

# Evaluate policy
obs = env.reset_to(state_dict)
# ... run rollout ...
```

### 2. Run Complete Active LSE

```bash
python examples/active_lse_example.py \
    --checkpoint /path/to/model.pth \
    --n_initial 20 \
    --n_iterations 50 \
    --acquisition uncertainty \
    --n_trials 5 \
    --output_dir results/lift_lse
```

## Key Components

### `active_lse_sampler.py`

**`ConstraintAwareSampler`** - Main class for state space sampling

**Key Methods**:
- `sample_uniform(n_samples)` - Sample uniformly from valid region
- `sample_grid(nx, ny, ntheta)` - Generate regular grid
- `is_valid_state(x, y, theta, check_physics=True)` - Validate state
- `create_initial_state(x, y, theta)` - Create state dict for `reset_to()`

**Constraints Extracted from Robosuite**:
- `x_range`: Valid X positions (e.g., [-0.03, 0.03] for Lift)
- `y_range`: Valid Y positions (e.g., [-0.03, 0.03])
- `z`: Object height (e.g., 0.81m for table)
- `rotation_axis`: Typically 'z' for planar rotation

### `examples/active_lse_example.py`

Complete Active LSE implementation with:
- Policy evaluation at custom initial states
- Simple GP-based level set model
- Multiple acquisition functions (uncertainty, boundary sampling)
- Visualization of learned level sets

## State Space

### Representation

States are represented as `(x, y, θ)`:
- **x, y**: Object position in world frame (meters)
- **θ**: Object rotation around z-axis (radians, [0, 2π))

### Constraints

Constraints are automatically extracted from robosuite's `placement_initializer`:

```python
sampler = ConstraintAwareSampler(env)
print(sampler)
# ConstraintAwareSampler(
#   x_range=[-0.03, 0.03],
#   y_range=[-0.03, 0.03],
#   z=0.810,
#   rotation_axis='z'
# )
```

### Validation

Two levels of validation:

1. **Bounds checking** (fast):
   ```python
   valid = sampler.is_valid_state(x, y, theta, check_physics=False)
   ```

2. **Physics validation** (slower, checks collisions):
   ```python
   valid = sampler.is_valid_state(x, y, theta, check_physics=True)
   ```

## Initial State Structure

The full MuJoCo state vector has the following structure (for Lift task):

```python
state = [
    # qpos (16 elements)
    robot_joint_0,        # [0:7]   - 7 robot joint angles
    robot_joint_1,
    ...
    gripper_joint_0,      # [7:9]   - 2 gripper joints
    gripper_joint_1,
    object_x,             # [9:12]  - 3D object position
    object_y,
    object_z,
    object_quat_w,        # [12:16] - Object orientation (quaternion)
    object_quat_x,
    object_quat_y,
    object_quat_z,
    
    # qvel (15 elements)
    robot_vel_0,          # [16:22] - Robot joint velocities
    ...
    gripper_vel_0,        # [23:24] - Gripper velocities
    gripper_vel_1,
    object_vel_x,         # [25:27] - Object linear velocity
    object_vel_y,
    object_vel_z,
    object_angvel_x,      # [28:30] - Object angular velocity
    object_angvel_y,
    object_angvel_z,
    
    time                  # [31]    - Simulation time
]
```

**Key Points**:
- Robot state: indices [0:9] (joints + gripper)
- Object position: indices [9:12]
- Object orientation: indices [12:16]
- Total dimension: 32 for Lift task

## Acquisition Functions

### 1. Max Uncertainty

Select states with highest prediction uncertainty:
```python
next_state = acquisition_max_uncertainty(model, sampler, n_candidates=100)
```

**Best for**: Exploratory phase, reducing overall uncertainty

### 2. Boundary Sampling

Select states closest to decision boundary (p=0.5):
```python
next_state = acquisition_boundary_sampling(model, sampler, n_candidates=100)
```

**Best for**: Refining the level set boundary

### 3. Uniform Random

Baseline: sample uniformly from valid region:
```python
next_state = sampler.sample_uniform()
```

## Example Results

After running Active LSE, you'll get:

```
active_lse_results/
├── config.json                    # Experiment configuration
├── results.npz                    # All query data
├── state_space_bounds.png         # Visualization of valid region
├── level_set_initial.png          # Level set after initial sampling
├── level_set_iter_010.png         # Intermediate results
├── level_set_iter_020.png
├── ...
└── level_set_final.png            # Final learned level set
```

## Advanced Usage

### Custom Robot Configuration

Fix robot initial pose for consistent experiments:

```python
# Define fixed robot configuration
default_robot_qpos = np.array([0, 0.2, 0, -2.6, 0, 2.95, 0.8])

sampler = ConstraintAwareSampler(env, robot_qpos=default_robot_qpos)
```

### Custom Acquisition Function

Implement your own acquisition strategy:

```python
def my_acquisition(model, sampler, **kwargs):
    """Custom acquisition function."""
    # Your logic here
    candidates = sampler.sample_uniform(n_samples=100)
    scores = compute_my_score(candidates)
    best_idx = np.argmax(scores)
    return candidates[best_idx]
```

### Batch Evaluation

Evaluate multiple states in parallel:

```python
states = sampler.sample_uniform(n_samples=10)

for state in states:
    x, y, theta = state
    state_dict = sampler.create_initial_state(x, y, theta)
    success_rate = evaluator.evaluate_at_state(state_dict, n_trials=5)
    model.add_data(state, success_rate)
```

## Integration with Different Tasks

The sampler automatically adapts to different robosuite tasks:

```python
# Works with any robosuite task that has placement_initializer
tasks = ["Lift", "PickPlace", "Stack", "NutAssembly", ...]

for task in tasks:
    env = create_env(task)
    sampler = ConstraintAwareSampler(env)
    # Constraints automatically extracted from task
    print(f"{task}: x_range={sampler.x_range}, y_range={sampler.y_range}")
```

## FAQ

### Q: How do I know if a state is valid?

Use `is_valid_state()` with `check_physics=True`:
```python
valid = sampler.is_valid_state(x, y, theta, check_physics=True)
```

### Q: Can I change the valid region?

Yes, but you need to modify robosuite's environment definition. Alternatively, create a custom sampler:
```python
from robosuite.utils.placement_samplers import UniformRandomSampler

custom_sampler = UniformRandomSampler(
    name="CustomSampler",
    x_range=[-0.1, 0.1],  # Wider range
    y_range=[-0.1, 0.1],
    ...
)
# Replace env's placement_initializer
env.base_env.placement_initializer = custom_sampler
```

### Q: Does robosuite sampling randomize robot state?

Yes! Robosuite adds initialization noise to robot joints during `reset()`. This is why we fix `robot_qpos` in the sampler to isolate object placement variability.

### Q: How many trials per state should I use?

Depends on policy stochasticity:
- **Deterministic policies**: 1-3 trials
- **Stochastic policies**: 5-10 trials
- **High variance**: 10+ trials

### Q: Can I use this with real robot data?

The constraint extraction works with simulation only. For real robots, you'd need to:
1. Define bounds manually based on workspace limits
2. Implement custom collision checking
3. Possibly use sim-to-real transfer techniques

## References

- **Robomimic**: https://robomimic.github.io/
- **Robosuite**: https://robosuite.ai/
- **Active Learning**: Settles, B. (2009). Active Learning Literature Survey
- **Level Set Estimation**: Bryan, B., et al. (2005). Active Learning for Identifying Function Threshold Boundaries

## Citation

If you use this code in your research, please cite:

```bibtex
@software{robomimic_active_lse,
  title={Active Level Set Estimation for Robomimic},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## Troubleshooting

### Issue: "No placement_initializer found"

Some environments may not have a placement sampler. Fallback defaults are used, but you should verify constraints manually.

### Issue: States fail physics validation

The object may be colliding with the robot. Try:
1. Checking if position is too close to robot base
2. Adjusting z-height
3. Using `sampler.sample_rejection()` to filter invalid samples

### Issue: Low success rates everywhere

Check:
1. Is the policy trained on this task?
2. Are initial states too far from training distribution?
3. Is the horizon long enough?

## License

Same as robomimic (MIT License)