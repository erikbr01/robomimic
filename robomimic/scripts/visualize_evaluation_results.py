#!/usr/bin/env python
"""
Visualize results from evaluate_initial_states.py

This script loads saved evaluation results and creates various visualizations
to analyze policy performance across initial object placements.

Usage:
    python visualize_evaluation_results.py --results eval_results/evaluation_results.npz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def create_heatmap(states, success_rates, x_range, y_range, output_path, title_suffix=""):
    """Create scatter heatmap of success rates."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Scatter plot
    ax = axes[0]
    scatter = ax.scatter(states[:, 0], states[:, 1], c=success_rates,
                        s=100, cmap='RdYlGn', vmin=0, vmax=1,
                        edgecolors='black', linewidth=0.5, alpha=0.8)

    ax.set_xlabel('X position (m)', fontsize=12)
    ax.set_ylabel('Y position (m)', fontsize=12)
    ax.set_title(f'Success Rate vs. Object Position{title_suffix}', fontsize=14, fontweight='bold')
    ax.set_xlim(x_range[0] - 0.01, x_range[1] + 0.01)
    ax.set_ylim(y_range[0] - 0.01, y_range[1] + 0.01)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Success Rate', fontsize=11)

    # Draw boundary
    rect = patches.Rectangle(
        (x_range[0], y_range[0]),
        x_range[1] - x_range[0],
        y_range[1] - y_range[0],
        linewidth=2, edgecolor='blue', facecolor='none',
        linestyle='--', label='Valid region'
    )
    ax.add_patch(rect)
    ax.legend()

    # Plot 2: Histogram
    ax = axes[1]
    ax.hist(success_rates, bins=20, edgecolor='black', color='skyblue', alpha=0.7)
    ax.set_xlabel('Success Rate', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution of Success Rates{title_suffix}', fontsize=14, fontweight='bold')
    ax.axvline(success_rates.mean(), color='red', linestyle='--',
              linewidth=2, label=f'Mean: {success_rates.mean():.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_theta_analysis(states, success_rates, output_path):
    """Analyze success rate vs. rotation angle."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Success rate vs theta
    ax = axes[0]
    ax.scatter(states[:, 2], success_rates, alpha=0.5, s=50)
    ax.set_xlabel('Rotation θ (radians)', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Success Rate vs. Object Rotation', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 2*np.pi)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax.grid(True, alpha=0.3)

    # Plot 2: Binned analysis
    ax = axes[1]
    n_bins = 8
    theta_bins = np.linspace(0, 2*np.pi, n_bins + 1)
    bin_indices = np.digitize(states[:, 2], theta_bins) - 1

    bin_means = []
    bin_stds = []
    bin_centers = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means.append(success_rates[mask].mean())
            bin_stds.append(success_rates[mask].std())
            bin_centers.append((theta_bins[i] + theta_bins[i+1]) / 2)

    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    bin_centers = np.array(bin_centers)

    ax.bar(bin_centers, bin_means, width=2*np.pi/n_bins * 0.8,
           yerr=bin_stds, capsize=5, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Rotation θ (radians)', fontsize=12)
    ax.set_ylabel('Mean Success Rate', fontsize=12)
    ax.set_title('Success Rate by Rotation Bin', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 2*np.pi)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_spatial_analysis(states, success_rates, x_range, y_range, output_path):
    """Create detailed spatial analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # Plot 1: Success vs X position
    ax = axes[0, 0]
    ax.scatter(states[:, 0], success_rates, alpha=0.5, s=30)
    ax.set_xlabel('X position (m)', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Success Rate vs. X Position', fontsize=12, fontweight='bold')
    ax.set_xlim(x_range)
    ax.axhline(success_rates.mean(), color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Plot 2: Success vs Y position
    ax = axes[0, 1]
    ax.scatter(states[:, 1], success_rates, alpha=0.5, s=30)
    ax.set_xlabel('Y position (m)', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Success Rate vs. Y Position', fontsize=12, fontweight='bold')
    ax.set_xlim(y_range)
    ax.axhline(success_rates.mean(), color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Plot 3: Success vs distance from center
    ax = axes[1, 0]
    center_x = (x_range[0] + x_range[1]) / 2
    center_y = (y_range[0] + y_range[1]) / 2
    distances = np.sqrt((states[:, 0] - center_x)**2 + (states[:, 1] - center_y)**2)
    ax.scatter(distances, success_rates, alpha=0.5, s=30)
    ax.set_xlabel('Distance from Center (m)', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Success Rate vs. Distance from Center', fontsize=12, fontweight='bold')
    ax.axhline(success_rates.mean(), color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Plot 4: 2D histogram
    ax = axes[1, 1]
    from scipy.stats import binned_statistic_2d
    nx, ny = 10, 10
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        states[:, 0], states[:, 1], success_rates,
        statistic='mean', bins=[nx, ny],
        range=[[x_range[0], x_range[1]], [y_range[0], y_range[1]]]
    )

    im = ax.imshow(stat.T, origin='lower', cmap='RdYlGn', vmin=0, vmax=1,
                   extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                   aspect='equal')
    ax.set_xlabel('X position (m)', fontsize=12)
    ax.set_ylabel('Y position (m)', fontsize=12)
    ax.set_title('Binned Success Rate (2D)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Mean Success Rate')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary_statistics(states, success_rates):
    """Print detailed summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\nNumber of states evaluated: {len(states)}")

    print(f"\nSuccess Rate Statistics:")
    print(f"  Mean:      {success_rates.mean():.4f}")
    print(f"  Std Dev:   {success_rates.std():.4f}")
    print(f"  Min:       {success_rates.min():.4f}")
    print(f"  Max:       {success_rates.max():.4f}")
    print(f"  Median:    {np.median(success_rates):.4f}")
    print(f"  25th %-ile: {np.percentile(success_rates, 25):.4f}")
    print(f"  75th %-ile: {np.percentile(success_rates, 75):.4f}")

    # Success thresholds
    print(f"\nStates by Success Rate:")
    print(f"  Always succeed (≥0.95):  {(success_rates >= 0.95).sum():4d} ({(success_rates >= 0.95).mean()*100:.1f}%)")
    print(f"  Often succeed (≥0.75):   {(success_rates >= 0.75).sum():4d} ({(success_rates >= 0.75).mean()*100:.1f}%)")
    print(f"  Sometimes (0.25-0.75):   {((success_rates >= 0.25) & (success_rates < 0.75)).sum():4d} ({((success_rates >= 0.25) & (success_rates < 0.75)).mean()*100:.1f}%)")
    print(f"  Rarely succeed (<0.25):  {(success_rates < 0.25).sum():4d} ({(success_rates < 0.25).mean()*100:.1f}%)")
    print(f"  Always fail (≤0.05):     {(success_rates <= 0.05).sum():4d} ({(success_rates <= 0.05).mean()*100:.1f}%)")

    # Spatial statistics
    print(f"\nSpatial Coverage:")
    print(f"  X range: [{states[:, 0].min():.4f}, {states[:, 0].max():.4f}]")
    print(f"  Y range: [{states[:, 1].min():.4f}, {states[:, 1].max():.4f}]")
    print(f"  θ range: [{states[:, 2].min():.4f}, {states[:, 2].max():.4f}] rad")

    # Best/worst states
    print(f"\nBest States (highest success rate):")
    best_indices = np.argsort(success_rates)[-3:][::-1]
    for i, idx in enumerate(best_indices):
        x, y, theta = states[idx]
        sr = success_rates[idx]
        print(f"  {i+1}. x={x:7.4f}, y={y:7.4f}, θ={theta:6.3f} | success={sr:.3f}")

    print(f"\nWorst States (lowest success rate):")
    worst_indices = np.argsort(success_rates)[:3]
    for i, idx in enumerate(worst_indices):
        x, y, theta = states[idx]
        sr = success_rates[idx]
        print(f"  {i+1}. x={x:7.4f}, y={y:7.4f}, θ={theta:6.3f} | success={sr:.3f}")

    print("=" * 70 + "\n")


def main(args):
    print("=" * 70)
    print("Visualizing Evaluation Results")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from {args.results}...")
    data = np.load(args.results)

    states = data['states']
    success_rates = data['success_rates']
    x_range = tuple(data['x_range'])
    y_range = tuple(data['y_range'])

    print(f"  ✓ Loaded {len(states)} states")

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Output directory: {output_dir}")

    # Print statistics
    if not args.no_stats:
        print_summary_statistics(states, success_rates)

    # Create visualizations
    print("Creating visualizations...")

    if not args.no_heatmap:
        create_heatmap(states, success_rates, x_range, y_range,
                      output_dir / 'heatmap.png')

    if not args.no_theta:
        create_theta_analysis(states, success_rates,
                            output_dir / 'theta_analysis.png')

    if not args.no_spatial:
        create_spatial_analysis(states, success_rates, x_range, y_range,
                              output_dir / 'spatial_analysis.png')

    print(f"\n✓ All visualizations saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize evaluation results"
    )

    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to evaluation_results.npz file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--no_heatmap",
        action="store_true",
        help="Skip heatmap generation"
    )
    parser.add_argument(
        "--no_theta",
        action="store_true",
        help="Skip theta analysis"
    )
    parser.add_argument(
        "--no_spatial",
        action="store_true",
        help="Skip spatial analysis"
    )
    parser.add_argument(
        "--no_stats",
        action="store_true",
        help="Skip printing statistics"
    )

    args = parser.parse_args()
    main(args)
