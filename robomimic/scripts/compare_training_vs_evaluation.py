#!/usr/bin/env python
"""
Compare training data initial states with evaluation results.

This script:
1. Loads training data from HDF5 dataset
2. Extracts initial object states (x, y, theta) from each demonstration
3. Loads evaluation results (states and success rates)
4. Creates visualizations showing:
   - Training data distribution overlaid on success rate heatmap
   - Whether training covered failure regions
   - Comparison of training vs evaluation coverage

Usage:
    python compare_training_vs_evaluation.py \
        --dataset datasets/lift/ph/low_dim_v15.hdf5 \
        --eval_results full_evaluation_parallel_larger/evaluation_results.npz \
        --output_dir training_vs_eval_comparison
"""

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def extract_initial_states_from_dataset(dataset_path):
    """
    Extract initial object states from training dataset.

    Args:
        dataset_path: Path to HDF5 dataset

    Returns:
        states: Array of shape (n_demos, 3) with [x, y, theta]
    """
    states = []

    with h5py.File(dataset_path, 'r') as f:
        demos = list(f['data'].keys())

        for demo in demos:
            # Get first observation
            obs_group = f[f'data/{demo}/obs']

            # Extract object state
            if 'object' in obs_group:
                obj = obs_group['object'][0]  # First timestep
                x, y, z = obj[:3]

                # Extract orientation if available
                if len(obj) >= 7:
                    # Quaternion in obs format: [qx, qy, qz, qw]
                    qx, qy, qz, qw = obj[3:7]
                    # Convert to theta (z-axis rotation)
                    theta = 2 * np.arctan2(qz, qw)
                    theta = theta % (2 * np.pi)  # Normalize to [0, 2π)
                else:
                    theta = 0.0
            else:
                print(f"Warning: 'object' observation not found in {demo}")
                continue

            states.append([x, y, theta])

    return np.array(states)


def plot_training_vs_evaluation(training_states, eval_states, success_rates,
                                x_range, y_range, output_path):
    """
    Create comprehensive comparison plot.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Success heatmap with training data overlay (x-y)
    ax = fig.add_subplot(gs[0:2, 0:2])

    # Background: evaluation results
    scatter = ax.scatter(eval_states[:, 0], eval_states[:, 1],
                        c=success_rates, s=100, cmap='RdYlGn',
                        vmin=0, vmax=1, alpha=0.6, edgecolors='gray',
                        linewidth=0.5, label='Evaluation', zorder=1)

    # Overlay: training data
    ax.scatter(training_states[:, 0], training_states[:, 1],
              c='blue', s=50, marker='x', linewidths=2,
              alpha=0.8, label='Training Data', zorder=2)

    ax.set_xlabel('X position (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y position (m)', fontsize=12, fontweight='bold')
    ax.set_title('Training Data vs. Evaluation Results (X-Y)',
                fontsize=14, fontweight='bold')
    ax.set_xlim(x_range[0] - 0.01, x_range[1] + 0.01)
    ax.set_ylim(y_range[0] - 0.01, y_range[1] + 0.01)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=10)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Success Rate (Evaluation)', fontsize=11)

    # Draw boundary
    rect = patches.Rectangle(
        (x_range[0], y_range[0]),
        x_range[1] - x_range[0],
        y_range[1] - y_range[0],
        linewidth=2, edgecolor='black', facecolor='none',
        linestyle='--', alpha=0.5
    )
    ax.add_patch(rect)

    # Plot 2: Theta distribution comparison
    ax = fig.add_subplot(gs[0, 2])

    bins = np.linspace(0, 2*np.pi, 16)
    ax.hist(training_states[:, 2], bins=bins, alpha=0.6,
           label='Training', color='blue', edgecolor='black')
    ax.hist(eval_states[:, 2], bins=bins, alpha=0.6,
           label='Evaluation', color='orange', edgecolor='black')

    ax.set_xlabel('θ (radians)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Rotation Distribution', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 2*np.pi)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Success rate vs theta (with training coverage)
    ax = fig.add_subplot(gs[1, 2])

    # Evaluation results
    ax.scatter(eval_states[:, 2], success_rates, alpha=0.5, s=30,
              c='gray', label='Eval Results')

    # Show training data density
    theta_bins = np.linspace(0, 2*np.pi, 9)
    train_hist, _ = np.histogram(training_states[:, 2], bins=theta_bins)
    bin_centers = (theta_bins[:-1] + theta_bins[1:]) / 2

    ax2 = ax.twinx()
    ax2.bar(bin_centers, train_hist, width=2*np.pi/8*0.8,
           alpha=0.3, color='blue', label='Train Density')
    ax2.set_ylabel('Training Data Count', fontsize=10, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ax.set_xlabel('θ (radians)', fontsize=11)
    ax.set_ylabel('Success Rate', fontsize=11)
    ax.set_title('Success vs. Rotation\n(with Training Coverage)',
                fontsize=12, fontweight='bold')
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    # Plot 4: X position distributions
    ax = fig.add_subplot(gs[2, 0])
    ax.hist(training_states[:, 0], bins=20, alpha=0.6,
           label='Training', color='blue', edgecolor='black')
    ax.hist(eval_states[:, 0], bins=20, alpha=0.6,
           label='Evaluation', color='orange', edgecolor='black')
    ax.set_xlabel('X position (m)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('X Position Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 5: Y position distributions
    ax = fig.add_subplot(gs[2, 1])
    ax.hist(training_states[:, 1], bins=20, alpha=0.6,
           label='Training', color='blue', edgecolor='black')
    ax.hist(eval_states[:, 1], bins=20, alpha=0.6,
           label='Evaluation', color='orange', edgecolor='black')
    ax.set_xlabel('Y position (m)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Y Position Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 6: Coverage analysis
    ax = fig.add_subplot(gs[2, 2])

    # Analyze coverage in failure regions
    failure_mask = success_rates < 0.5
    success_mask = success_rates >= 0.5

    n_failure_regions = failure_mask.sum()
    n_success_regions = success_mask.sum()

    # Find closest training sample to each eval state
    from scipy.spatial.distance import cdist

    # Normalize theta for distance computation
    train_normalized = training_states.copy()
    train_normalized[:, 2] = np.stack([np.cos(training_states[:, 2]),
                                       np.sin(training_states[:, 2])], axis=1).mean(axis=1)
    eval_normalized = eval_states.copy()
    eval_normalized[:, 2] = np.stack([np.cos(eval_states[:, 2]),
                                      np.sin(eval_states[:, 2])], axis=1).mean(axis=1)

    distances = cdist(eval_normalized, train_normalized)
    min_distances = distances.min(axis=1)

    # Coverage threshold (within 2cm)
    coverage_threshold = 0.02
    covered_mask = min_distances < coverage_threshold

    failure_covered = (failure_mask & covered_mask).sum()
    failure_uncovered = (failure_mask & ~covered_mask).sum()
    success_covered = (success_mask & covered_mask).sum()
    success_uncovered = (success_mask & ~covered_mask).sum()

    categories = ['Failure\nRegions', 'Success\nRegions']
    covered = [failure_covered, success_covered]
    uncovered = [failure_uncovered, success_uncovered]

    x_pos = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, covered, width,
                   label='Covered by Training', color='green', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, uncovered, width,
                   label='Not Covered', color='red', alpha=0.7)

    ax.set_ylabel('Number of States', fontsize=11)
    ax.set_title('Training Coverage Analysis\n(within 2cm)',
                fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def print_coverage_statistics(training_states, eval_states, success_rates):
    """Print detailed statistics about coverage."""
    print("\n" + "=" * 70)
    print("TRAINING VS EVALUATION COVERAGE ANALYSIS")
    print("=" * 70)

    print(f"\nDataset Statistics:")
    print(f"  Training demonstrations: {len(training_states)}")
    print(f"  Evaluation states:       {len(eval_states)}")

    print(f"\nTraining Data Coverage:")
    print(f"  X range: [{training_states[:, 0].min():.4f}, {training_states[:, 0].max():.4f}]")
    print(f"  Y range: [{training_states[:, 1].min():.4f}, {training_states[:, 1].max():.4f}]")
    print(f"  θ range: [{training_states[:, 2].min():.4f}, {training_states[:, 2].max():.4f}] rad")

    print(f"\nEvaluation Coverage:")
    print(f"  X range: [{eval_states[:, 0].min():.4f}, {eval_states[:, 0].max():.4f}]")
    print(f"  Y range: [{eval_states[:, 1].min():.4f}, {eval_states[:, 1].max():.4f}]")
    print(f"  θ range: [{eval_states[:, 2].min():.4f}, {eval_states[:, 2].max():.4f}] rad")

    # Analyze theta distribution in success/failure regions
    failure_mask = success_rates < 0.5
    success_mask = success_rates >= 0.5

    print(f"\nSuccess/Failure Analysis:")
    print(f"  Evaluation states that fail:    {failure_mask.sum()} ({failure_mask.mean()*100:.1f}%)")
    print(f"  Evaluation states that succeed: {success_mask.sum()} ({success_mask.mean()*100:.1f}%)")

    if failure_mask.sum() > 0:
        print(f"\nFailure Region Characteristics:")
        print(f"  Mean θ: {eval_states[failure_mask, 2].mean():.3f} rad ({np.degrees(eval_states[failure_mask, 2].mean()):.1f}°)")
        print(f"  Std θ:  {eval_states[failure_mask, 2].std():.3f} rad")

    if success_mask.sum() > 0:
        print(f"\nSuccess Region Characteristics:")
        print(f"  Mean θ: {eval_states[success_mask, 2].mean():.3f} rad ({np.degrees(eval_states[success_mask, 2].mean()):.1f}°)")
        print(f"  Std θ:  {eval_states[success_mask, 2].std():.3f} rad")

    # Check if training covered failure regions
    print(f"\nTraining Data in Failure Regions:")

    # Define failure theta range (adjust based on your observations)
    # From earlier analysis: failures around 2.5-3.0 rad
    failure_theta_mask = (eval_states[:, 2] >= 2.0) & (eval_states[:, 2] <= 3.5)
    failure_theta_states = eval_states[failure_theta_mask]

    if len(failure_theta_states) > 0:
        failure_rate_in_range = success_rates[failure_theta_mask].mean()
        print(f"  States with θ ∈ [2.0, 3.5] rad: {len(failure_theta_states)}")
        print(f"  Mean success rate in range: {failure_rate_in_range:.3f}")

        train_in_failure_range = ((training_states[:, 2] >= 2.0) &
                                 (training_states[:, 2] <= 3.5)).sum()
        print(f"  Training samples in this range: {train_in_failure_range} ({train_in_failure_range/len(training_states)*100:.1f}%)")

    print("=" * 70 + "\n")


def main(args):
    print("=" * 70)
    print("Comparing Training Data vs. Evaluation Results")
    print("=" * 70)

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    print(f"\n[1/3] Loading training data from {args.dataset}...")
    training_states = extract_initial_states_from_dataset(args.dataset)
    print(f"  ✓ Loaded {len(training_states)} demonstrations")

    # Load evaluation results
    print(f"\n[2/3] Loading evaluation results from {args.eval_results}...")
    eval_data = np.load(args.eval_results)
    eval_states = eval_data['states']
    success_rates = eval_data['success_rates']
    x_range = tuple(eval_data['x_range'])
    y_range = tuple(eval_data['y_range'])
    print(f"  ✓ Loaded {len(eval_states)} evaluation states")

    # Print statistics
    print_coverage_statistics(training_states, eval_states, success_rates)

    # Create visualizations
    print("[3/3] Creating visualizations...")
    plot_training_vs_evaluation(
        training_states, eval_states, success_rates,
        x_range, y_range,
        output_dir / 'training_vs_evaluation.png'
    )

    print(f"\n✓ Results saved to {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare training data distribution with evaluation results"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to training dataset HDF5 file"
    )
    parser.add_argument(
        "--eval_results",
        type=str,
        required=True,
        help="Path to evaluation results NPZ file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training_vs_eval_comparison",
        help="Directory to save comparison plots"
    )

    args = parser.parse_args()
    main(args)
