# Quick Start: Evaluate Policy Across Initial States

Simple guide to evaluate your trained robomimic policy across different initial object placements.

## TL;DR

```bash
# Run evaluation
python robomimic/scripts/evaluate_initial_states.py \
    --checkpoint /path/to/model.pth \
    --n_samples 100 \
    --n_trials 5 \
    --output_dir my_results

# View results
ls my_results/
# → success_heatmap.png, evaluation_results.npz, config.json
```

## What You Get

A heatmap showing which initial object positions lead to success/failure:

- **Green dots** = Policy succeeds
- **Red dots** = Policy fails
- **Yellow dots** = Mixed results

## Full Example

```bash
# Evaluate a Lift policy with 50 random initial object placements
python robomimic/scripts/evaluate_initial_states.py \
    --checkpoint bc_trained_models/test/20260108164048/models/model_epoch_50.pth \
    --n_samples 50 \
    --n_trials 5 \
    --output_dir lift_evaluation \
    --seed 42
```

Output:
```
======================================================================
Evaluating Policy Across Initial Object Placements
======================================================================

[1/5] Loading policy and environment...
  ✓ Policy loaded from bc_trained_models/.../model_epoch_50.pth
  ✓ Environment: Lift
  ✓ Rollout horizon: 400

[2/5] Initializing state sampler...
  ✓ Valid region: x ∈ [-0.03, 0.03], y ∈ [-0.03, 0.03]
  ✓ Object height: z = 0.810
  ✓ Fixed robot config: [0. 0.2 0.] ...

[3/5] Sampling 50 initial states...
  ✓ Sampled 50 valid states in 2.3s

[4/5] Evaluating policy (5 trials per state)...
  [ 10/ 50] x=-0.015, y= 0.022, θ= 1.83 | success=0.80 | avg=0.756 | 3.2s
  [ 20/ 50] x= 0.008, y=-0.011, θ= 4.71 | success=1.00 | avg=0.810 | 3.1s
  [ 30/ 50] x= 0.021, y= 0.005, θ= 2.45 | success=0.60 | avg=0.773 | 3.3s
  [ 40/ 50] x=-0.019, y=-0.028, θ= 5.89 | success=0.40 | avg=0.735 | 3.2s
  [ 50/ 50] x= 0.002, y= 0.018, θ= 0.92 | success=1.00 | avg=0.748 | 3.1s

[5/5] Creating visualizations...
  ✓ Saved heatmap to lift_evaluation/success_heatmap.png
  ✓ Saved data to lift_evaluation/evaluation_results.npz
  ✓ Saved config to lift_evaluation/config.json

======================================================================
EVALUATION SUMMARY
======================================================================
Total states evaluated:  50
Total rollouts:          250

Success Rate Statistics:
  Mean:   0.748
  Std:    0.234
  Min:    0.200
  Max:    1.000
  Median: 0.800

Results saved to: lift_evaluation
======================================================================
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | *required* | Path to trained policy (.pth file) |
| `--n_samples` | 100 | Number of random states to evaluate |
| `--n_trials` | 5 | Trials per state (for stochastic policies) |
| `--output_dir` | "eval_results" | Where to save results |
| `--seed` | 42 | Random seed |
| `--skip_collision_check` | False | Skip physics validation (faster) |
| `--horizon` | from checkpoint | Rollout horizon |

## What It Does

### 1. **Extracts Constraints**
Automatically reads valid object placement region from robosuite:
- For Lift task: typically x,y ∈ [-0.03, 0.03] (6cm × 6cm square)

### 2. **Samples & Validates**
- Randomly samples (x, y, θ) for object placement
- Checks for collisions (robot arm vs. object)
- Rejects invalid states

### 3. **Fixes Robot State**
- Keeps robot joint angles constant across all evaluations
- Only varies object placement
- Isolates effect of initial object pose

### 4. **Evaluates Policy**
- Runs N trials per state
- Computes success rate (% of trials that succeed)
- Average over trials accounts for stochasticity

## Understanding Results

### The Heatmap

**Left plot**: Success rate vs. (x, y) position
```
     Y
     ^
     |  🟢 🟢 🟡 🔴
     |  🟢 🟢 🟡 🔴
     |  🟢 🟡 🔴 🔴
     +-----------> X
```

**Right plot**: Distribution histogram
- Shows how often you see each success rate
- Unimodal = consistent performance
- Bimodal = clear success/failure regions

### Common Patterns

**✅ Good policy**: Green everywhere, mean > 0.8
**⚠️ Okay policy**: Green center, red edges, mean ~0.5-0.8
**❌ Poor policy**: Red everywhere, mean < 0.3
**🤔 Interesting**: Clear spatial patterns (e.g., fails on left, succeeds on right)

## Analyze Results

Load the raw data:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load results
data = np.load('my_results/evaluation_results.npz')
states = data['states']              # (N, 3) array: [x, y, theta]
success_rates = data['success_rates'] # (N,) array: success rate per state

# Find failure states
failures = states[success_rates < 0.5]
print(f"Policy fails at {len(failures)} states:")
print(failures)

# Plot custom visualization
plt.scatter(states[:, 0], states[:, 1], c=success_rates, cmap='RdYlGn')
plt.colorbar(label='Success Rate')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.show()
```

## Advanced: Re-visualize Results

```bash
# Create more detailed plots
python robomimic/scripts/visualize_evaluation_results.py \
    --results my_results/evaluation_results.npz \
    --output_dir detailed_plots
```

Generates:
- `heatmap.png` - Main success rate visualization
- `theta_analysis.png` - Success vs. rotation angle
- `spatial_analysis.png` - 4-panel detailed analysis

## Tips

### Start Small
```bash
# Quick test with 10 samples
python robomimic/scripts/evaluate_initial_states.py \
    --checkpoint model.pth \
    --n_samples 10 \
    --n_trials 1
```

### Production Run
```bash
# High-quality evaluation
python robomimic/scripts/evaluate_initial_states.py \
    --checkpoint model.pth \
    --n_samples 200 \
    --n_trials 10 \
    --output_dir final_results
```

### Faster (Less Accurate)
```bash
# Skip collision checking
python robomimic/scripts/evaluate_initial_states.py \
    --checkpoint model.pth \
    --n_samples 100 \
    --skip_collision_check
```

## Troubleshooting

**"Failed to find valid sample"**
- Valid region may be small
- Increase `--max_collision_attempts 500`
- Or use `--skip_collision_check`

**"All success rates are 1.0 or 0.0"**
- Normal for deterministic policies on easy/hard states
- Increase `--n_samples` to find boundary cases

**"Mean success rate is 0.0"**
- Policy may not be trained
- Check checkpoint path is correct
- Verify environment matches training

## Next Steps

Once you have baseline results:

1. **Identify patterns** - Where does the policy fail?
2. **Hypothesis** - Why? (e.g., edge effects, rotation sensitivity)
3. **Improve** - Retrain with more data from failure regions
4. **Advanced** - Use Active LSE to efficiently find decision boundaries

See `EVALUATE_STATES_README.md` for full documentation.

---

**Questions?** Check the detailed README or robomimic documentation.