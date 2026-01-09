# Evaluate Policy Across Initial Object Placements

Simple script to evaluate a trained robomimic policy across randomly sampled initial object positions and generate a success rate heatmap.

## Quick Start

```bash
python robomimic/scripts/evaluate_initial_states.py \
    --checkpoint /path/to/model.pth \
    --n_samples 100 \
    --n_trials 5 \
    --output_dir results/my_evaluation
```

## What It Does

1. **Loads** your trained policy from a checkpoint
2. **Samples** N random initial object placements (x, y, θ) from the valid region
3. **Validates** each placement using collision detection (optional)
4. **Evaluates** the policy at each state with multiple trials
5. **Outputs** a heatmap showing success rate vs. object position

## Key Features

✅ **Uses robosuite's constraints** - Automatically extracts valid x/y ranges from the environment  
✅ **Collision checking** - Validates states to avoid robot-object collisions  
✅ **Fixed robot state** - Keeps robot configuration constant to isolate object placement effects  
✅ **Simple visualization** - Generates scatter plot heatmap of results  

## Arguments

### Required
- `--checkpoint PATH` - Path to trained policy checkpoint (.pth file)

### Sampling
- `--n_samples N` (default: 100) - Number of random states to evaluate
- `--skip_collision_check` - Skip physics validation (faster but may be invalid)
- `--max_collision_attempts N` (default: 100) - Max tries per sample

### Evaluation
- `--n_trials N` (default: 5) - Trials per state to estimate success rate
- `--horizon N` (default: from checkpoint) - Rollout horizon

### Output
- `--output_dir PATH` (default: "eval_results") - Where to save results
- `--seed N` (default: 42) - Random seed for reproducibility

## Example Usage

### Basic evaluation with 50 samples
```bash
python robomimic/scripts/evaluate_initial_states.py \
    --checkpoint bc_trained_models/test/20260108164048/models/model_epoch_50.pth \
    --n_samples 50 \
    --n_trials 3
```

### Fast evaluation (skip collision check)
```bash
python robomimic/scripts/evaluate_initial_states.py \
    --checkpoint model.pth \
    --n_samples 200 \
    --skip_collision_check \
    --n_trials 1
```

### High-fidelity evaluation
```bash
python robomimic/scripts/evaluate_initial_states.py \
    --checkpoint model.pth \
    --n_samples 100 \
    --n_trials 10 \
    --output_dir detailed_results
```

## Output Files

After running, you'll find in the output directory:

```
eval_results/
├── success_heatmap.png       # Visualization of results
├── evaluation_results.npz    # Raw data (states, success_rates, etc.)
└── config.json               # Configuration used for this run
```

### Loading Results

```python
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load('eval_results/evaluation_results.npz')
states = data['states']          # (N, 3) array of [x, y, theta]
success_rates = data['success_rates']  # (N,) array of success rates
x_range = data['x_range']
y_range = data['y_range']

# Your analysis here
print(f"Mean success rate: {success_rates.mean():.3f}")
print(f"States where policy fails: {states[success_rates < 0.5]}")
```

## Understanding the Output

### Heatmap Interpretation

- **Left plot**: Scatter plot of success rate vs. (x, y) position
  - Green dots = high success rate
  - Red dots = low success rate
  - Each dot is averaged over θ rotations and multiple trials

- **Right plot**: Histogram of success rate distribution
  - Shows how often you see different success rates
  - Red line = mean success rate

### What to Look For

1. **Spatial patterns**: Are there regions where the policy always fails?
2. **Success distribution**: Is success rate bimodal (mostly 0 or 1) or continuous?
3. **Edge effects**: Does performance degrade near workspace boundaries?

## How It Works

### State Sampling

The sampler:
1. Extracts valid (x, y) bounds from robosuite's `placement_initializer`
2. Samples (x, y, θ) uniformly from this region
3. Checks for collisions by:
   - Setting the state in MuJoCo
   - Running `sim.forward()` to compute contacts
   - Rejecting if robot (non-gripper) collides with object

### State Structure

Each state is represented as:
```python
state = {
    'states': [32,]  # Full MuJoCo state vector:
                     #   [0:7]   - Robot joint angles (FIXED)
                     #   [7:9]   - Gripper joints (FIXED)
                     #   [9:12]  - Object position (x, y, z)
                     #   [12:16] - Object quaternion (from θ)
                     #   [16:31] - Velocities (all zeros)
                     #   [31]    - Time (zero)
}
```

**Key point**: Robot configuration is FIXED across all evaluations to isolate the effect of object placement.

## Tips

### For faster results:
- Use `--skip_collision_check` (but verify states are reasonable)
- Reduce `--n_trials` to 1-3
- Use smaller `--n_samples`

### For more accurate results:
- Increase `--n_trials` to 10+
- Use more `--n_samples` (200-500)
- Keep collision checking enabled

### For debugging:
- Start with `--n_samples 10` to test quickly
- Check `config.json` to see what constraints were used
- Verify x_range and y_range match your expectations

## Common Issues

**"Failed to find valid sample after 100 attempts"**
- The valid region may be very small or heavily constrained
- Try increasing `--max_collision_attempts`
- Or use `--skip_collision_check` to see if any samples work

**"All success rates are 0 or 1"**
- Normal for deterministic policies on easy/hard states
- Increase `--n_samples` to find more boundary cases
- Policy may need more training

**"Results look random"**
- Increase `--n_trials` to get better estimates
- Check if your policy is actually trained
- Verify the environment matches training

## Next Steps: Active LSE

Once you have baseline random sampling results, you can:

1. **Identify interesting regions** - Where does success rate vary most?
2. **Focus sampling** - Sample more densely near decision boundaries
3. **Implement acquisition functions** - Use uncertainty or boundary sampling
4. **Build a model** - Fit GP or NN to predict success rate
5. **Active learning loop** - Iteratively query informative states

The `active_lse_sampler.py` and `active_lse_example.py` files provide this functionality.

## Citation

If you use this in your research:

```bibtex
@software{robomimic_state_eval,
  title={Initial State Evaluation for Robomimic},
  author={Stanford ASL},
  year={2024}
}
```
