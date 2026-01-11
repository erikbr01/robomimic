# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

robomimic is a Python framework for robot learning from demonstration, providing offline learning algorithms (imitation learning and offline RL) for robot manipulation datasets. Version 0.5.0.

## Common Commands

### Installation
```bash
pip install -e .                    # Install from source (recommended)
pip install robosuite               # Required simulator for most examples
```

### Training
```bash
# Main training entry point
python robomimic/scripts/train.py --config <config.json>

# With overrides
python robomimic/scripts/train.py --config config.json --dataset /path/to/data.hdf5 --name my_exp

# Debug mode (quick run)
python robomimic/scripts/train.py --config config.json --debug
```

### Evaluation
```bash
# Run trained policy
python robomimic/scripts/run_trained_agent.py --agent /path/to/model.pth --n_rollouts 50

# Evaluate across initial states (generates heatmaps)
python robomimic/scripts/evaluate_initial_states.py --checkpoint /path/to/model.pth --n_samples 100 --n_trials 5
```

### Testing
```bash
cd tests && bash test.sh            # Run full test suite
python tests/test_bc.py             # Run single algorithm test
python examples/train_bc_rnn.py --debug  # Quick installation verification
```

## Architecture

### Core Module Structure

```
robomimic/
├── algo/           # Learning algorithms (BC, BCQ, CQL, TD3-BC, IQL, HBC, Diffusion Policy)
├── config/         # Configuration system with per-algorithm configs
├── models/         # Neural network modules (policy nets, value nets, encoders)
├── envs/           # Environment wrappers (robosuite, gym)
├── scripts/        # Training, evaluation, data processing scripts
├── utils/          # Utilities (dataset loading, training loops, logging)
└── exps/templates/ # JSON config templates
```

### Key Design Patterns

**Algorithm Factory**: `algo_factory(algo_name, config, ...)` creates algorithm instances. Each algorithm registers itself via `@register_algo_factory_func(algo_name)` decorator in `robomimic/algo/algo.py`.

**Config System**: Dict-based `Config` class with attribute access and locking. Each algorithm has its own config class (e.g., `BCConfig`) that registers via metaclass. Use `config.values_unlocked()` context manager to modify locked configs.

**Config Registry**: `config_factory(algo_name)` in `robomimic/config/__init__.py` creates configs by algorithm name.

### Data Flow

1. Config JSON loaded and parsed (`config_factory`)
2. Dataset loaded via `SequenceDataset` (`robomimic/utils/dataset.py`)
3. Algorithm instantiated (`algo_factory`)
4. Training loop in `TrainUtils.run_epoch()` (`robomimic/utils/train_utils.py`)
5. Periodic rollouts in environment for validation

### Supported Algorithms

- `bc` - Behavioral Cloning (vanilla, GMM, VAE, RNN, Transformer variants)
- `bcq`, `cql`, `td3_bc`, `iql` - Offline RL
- `hbc` - Hierarchical BC
- `diffusion_policy` - Diffusion-based policy
- `iris`, `gl` - Goal-conditioned learning

## Configuration

Configs are JSON files with sections: `experiment`, `train`, `algo`, `observation`, `meta`. Templates in `robomimic/exps/templates/`. Key settings:

- `train.data` - Dataset path(s) (string or list of dicts with `path` keys)
- `train.cuda` - Use GPU (set `false` for MPS on Mac)
- `experiment.rollout.enabled` - Run evaluation rollouts during training
- `algo.rnn.enabled` / `algo.transformer.enabled` - Enable sequence models

## Macros

Global settings in `robomimic/macros.py`. Create `robomimic/macros_private.py` for user-specific overrides (not committed):
```bash
python robomimic/scripts/setup_macros.py
```

## Dataset Format

HDF5 files with structure:
- `/data/demo_X/obs/*` - Observations per timestep
- `/data/demo_X/actions` - Action sequences
- `/mask/train`, `/mask/valid` - Train/val splits

## Code and Reply Style

Be concise and clear in your code and replies. Avoid unnecessary verbosity and repetition, particularly for exception use and fallbacks.
