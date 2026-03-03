#!/usr/bin/env python3
"""
Histogram Comparison Tool
Compares 2-5 overall_statistics.csv files and creates overlay histogram plots
with average differences displayed in the legend.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from pathlib import Path

# Color palette for up to 5 datasets
COLORS = [
    ('red',   'darkred'), #nuscenes
    ('steelblue',  'midnightblue'), #nuscenes_day
    ('blue',  'darkblue'), #seed4d
    ('green', 'darkgreen'), #secogan
    ('orange','darkorange'), #flux
    ('#F0E442', '#B5A800'),
    ('purple','indigo'),
    ('plum',       'purple'),
    ('green', 'darkgreen'),
    ('darkorange', 'saddlebrown'),
    ('blue',  'darkblue'),
    ('green', 'darkgreen'),
    ('orange','darkorange'),
    ('steelblue',  'midnightblue'),
    ('mediumseagreen', 'seagreen'),
    ('coral',      'firebrick'),
    ('goldenrod',  'darkgoldenrod'),
]


def load_histogram_data(csv_file):
    """Load histogram data from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        required_columns = ['metric', 'average', 'bin_center', 'frequency']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")
        return df
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        return None


def extract_histogram_data(df, metric):
    """Extract histogram data for a specific metric"""
    metric_data = df[df['metric'] == metric].copy()
    if metric_data.empty:
        return None
    metric_data = metric_data.sort_values('bin_center')
    return {
        'bin_centers': metric_data['bin_center'].values,
        'frequencies': metric_data['frequency'].values,
        'bin_starts':  metric_data['bin_start'].values,
        'bin_ends':    metric_data['bin_end'].values,
        'average':     metric_data['average'].iloc[0],
        'std':         metric_data['std'].iloc[0],
        'total_samples': metric_data['total_samples'].iloc[0],
    }


def create_comparison_plots(csv_files, labels, output_dir):
    """Load all datasets and create per-metric and combined plots."""
    datasets = []
    for csv_file, label in zip(csv_files, labels):
        print(f"Loading data from {csv_file}...")
        df = load_histogram_data(csv_file)
        if df is None:
            return False
        datasets.append((label, df))

    # Find common metrics across all datasets
    common_metrics = set(datasets[0][1]['metric'].unique())
    for _, df in datasets[1:]:
        common_metrics &= set(df['metric'].unique())

    if not common_metrics:
        print("Error: No common metrics found across all files")
        return False

    print(f"Found {len(common_metrics)} common metrics: {sorted(common_metrics)}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Individual metric plots
    success_count = 0
    for metric in sorted(common_metrics):
        print(f"Creating comparison plot for {metric}...")
        out_file = output_path / f"comparison_{metric}.png"
        if _plot_metric(datasets, metric, out_file):
            print(f"  Saved: {out_file}")
            success_count += 1

    # Combined plot
    print("Creating combined comparison plot...")
    _plot_combined(datasets, output_path / "comparison_all_metrics.png")

    print(f"\nComparison complete! Created {success_count} individual plots and 1 combined plot.")
    print(f"Results saved to: {output_path}")
    return True


def _plot_metric(datasets, metric, output_path):
    """Create an individual comparison plot for one metric (shared x-axis, separate y-axes)."""
    hists = [(label, extract_histogram_data(df, metric)) for label, df in datasets]
    if any(h is None for _, h in hists):
        print(f"  Warning: missing data for '{metric}' in one or more files")
        return False

    n = len(hists)
    fig, ax_main = plt.subplots(figsize=(13, 7))

    # We'll normalise frequencies to [0,1] per dataset so they're visually comparable
    # on a single y-axis (avoids the messy n-axis approach beyond 2 datasets).
    axes_extra = []  # keep for 2-dataset dual-axis mode
    use_shared_axis = n > 2

    if not use_shared_axis:
        ax2 = ax_main.twinx()
        axes_extra = [ax_main, ax2]

    avgs = []
    for i, (label, hist) in enumerate(hists):
        color, dark_color = COLORS[i]
        bin_widths = hist['bin_ends'] - hist['bin_starts']

        if use_shared_axis:
            # Normalise so all datasets fit on the same axis
            freqs_norm = hist['frequencies'] / hist['frequencies'].sum() if hist['frequencies'].sum() > 0 else hist['frequencies']
            ax_main.bar(hist['bin_centers'], freqs_norm,
                        width=bin_widths, alpha=0.45, color=color,
                        edgecolor=dark_color, linewidth=0.8,
                        label=f'{label} (n={hist["total_samples"]})')
            ax_main.axvline(hist['average'], color=dark_color, linestyle='--', linewidth=2.5,
                            label=f'{label} Avg: {hist["average"]:.2f}')
        else:
            ax = axes_extra[i]
            ax.bar(hist['bin_centers'], hist['frequencies'],
                   width=bin_widths, alpha=0.55, color=color,
                   edgecolor=dark_color, linewidth=1,
                   label=f'{label} (n={hist["total_samples"]})')
            ax.axvline(hist['average'], color=dark_color, linestyle='--', linewidth=2.5,
                       label=f'{label} Avg: {hist["average"]:.2f}')
            ax.set_ylabel(f'{label} Frequency', fontsize=11, color=dark_color)
            ax.tick_params(axis='y', labelcolor=dark_color)

        avgs.append(hist['average'])

    # Title with pairwise deltas relative to first dataset
    delta_parts = [f'Δ({label} vs {hists[0][0]}) = {abs(avg - avgs[0]):.2f}'
                   for (label, _), avg in zip(hists[1:], avgs[1:])]
    metric_title = metric.replace('_', ' ').title()
    ax_main.set_title(f'{metric_title} Distribution Comparison\n' + ' | '.join(delta_parts),
                      fontsize=13, fontweight='bold')
    ax_main.set_xlabel('Value', fontsize=12)

    if use_shared_axis:
        ax_main.set_ylabel('Normalised Frequency', fontsize=12)
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(loc='best', fontsize=10)
    else:
        ax_main.set_ylabel(f'{hists[0][0]} Frequency', fontsize=11, color=COLORS[0][1])
        ax_main.tick_params(axis='y', labelcolor=COLORS[0][1])
        ax_main.grid(True, alpha=0.3)
        lines, lbls = ax_main.get_legend_handles_labels()
        lines2, lbls2 = axes_extra[1].get_legend_handles_labels()
        ax_main.legend(lines + lines2, lbls + lbls2, loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return True


def _plot_combined(datasets, output_path):
    """Create 2×3 combined subplot for all 6 standard metrics."""
    metrics = ['brightness', 'sharpness', 'vignetting',
               'compression_artifacts', 'noise', 'chromatic_aberration']
    metric_titles = ['Brightness', 'Sharpness', 'Vignetting',
                     'Compression Artifacts', 'Noise', 'Chromatic Aberration']

    n_datasets = len(datasets)
    use_shared_axis = n_datasets > 2

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle('Image Quality Metrics Comparison: ' + ' vs '.join(l for l, _ in datasets),
                 fontsize=16, fontweight='bold', y=0.97)

    axes_flat = axes.flatten()

    for i, (metric, metric_title) in enumerate(zip(metrics, metric_titles)):
        ax1 = axes_flat[i]

        hists = [(label, extract_histogram_data(df, metric)) for label, df in datasets]
        if any(h is None for _, h in hists):
            ax1.text(0.5, 0.5, f'No data for {metric_title}',
                     ha='center', va='center', transform=ax1.transAxes)
            continue

        avgs = [h['average'] for _, h in hists]
        delta_parts = [f'Δ{label}={abs(avg - avgs[0]):.2f}'
                       for (label, _), avg in zip(hists[1:], avgs[1:])]

        if not use_shared_axis:
            ax2 = ax1.twinx()
            sub_axes = [ax1, ax2]

        for j, (label, hist) in enumerate(hists):
            color, dark_color = COLORS[j]
            bin_widths = hist['bin_ends'] - hist['bin_starts']

            if use_shared_axis:
                freqs_norm = (hist['frequencies'] / hist['frequencies'].sum()
                              if hist['frequencies'].sum() > 0 else hist['frequencies'])
                ax1.bar(hist['bin_centers'], freqs_norm,
                        width=bin_widths, alpha=0.40, color=color,
                        edgecolor=dark_color, linewidth=0.7)
                ax1.axvline(hist['average'], color=dark_color,
                            linestyle='--', linewidth=1.8)
            else:
                sub_ax = sub_axes[j]
                sub_ax.bar(hist['bin_centers'], hist['frequencies'],
                           width=bin_widths, alpha=0.55, color=color,
                           edgecolor=dark_color, linewidth=0.8)
                sub_ax.axvline(hist['average'], color=dark_color,
                               linestyle='--', linewidth=1.8)
                sub_ax.set_ylabel(label, fontsize=8, color=dark_color)
                sub_ax.tick_params(axis='y', labelcolor=dark_color, labelsize=7)

        title_text = (f'{metric_title}\n'
                      + ' | '.join(f'{l}: {h["average"]:.2f}' for l, h in hists) + '\n'
                      + ' | '.join(delta_parts))
        ax1.set_title(title_text, fontweight='bold', fontsize=9)
        ax1.set_xlabel('Value', fontsize=8)
        ax1.tick_params(axis='x', labelsize=7)

        if use_shared_axis:
            ax1.set_ylabel('Norm. Frequency', fontsize=8)
            ax1.tick_params(axis='y', labelsize=7)
        else:
            ax1.set_ylabel(datasets[0][0], fontsize=8, color=COLORS[0][1])
            ax1.tick_params(axis='y', labelcolor=COLORS[0][1], labelsize=7)

        ax1.grid(True, alpha=0.3)

    # Global legend
    legend_elements = []
    for j, (label, _) in enumerate(datasets):
        color, dark_color = COLORS[j]
        legend_elements += [
            plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.55,
                           edgecolor=dark_color, label=f'{label} (bars)'),
            plt.Line2D([0], [0], color=dark_color, linestyle='--',
                       linewidth=2, label=f'{label} avg'),
        ]
    if use_shared_axis:
        legend_elements.append(
            plt.Line2D([0], [0], color='black', linewidth=0,
                       label='Y-axis: normalised frequency (sums to 1)')
        )
    legend_elements.append(
        plt.Line2D([0], [0], color='black', linewidth=0,
                   label='Δ = absolute difference vs first dataset')
    )

    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=min(4, n_datasets * 2 + 1),
               fontsize=10, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.92, hspace=0.38, wspace=0.30)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved combined plot: {output_path}")

    # Print summary table
    print(f"\nSummary of Average Differences (vs {datasets[0][0]}):")
    for metric in metrics:
        hists = [(label, extract_histogram_data(df, metric)) for label, df in datasets]
        if any(h is None for _, h in hists):
            continue
        base_avg = hists[0][1]['average']
        metric_name = metric.replace('_', ' ').title()
        deltas = '  '.join(f'Δ{label}={abs(h["average"] - base_avg):.2f}'
                           for label, h in hists[1:])
        avgs_str = '  '.join(f'{label}={h["average"]:.2f}' for label, h in hists)
        print(f"  {metric_name:22s}: {avgs_str}  |  {deltas}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Compare 2–5 histogram CSV files and create overlay plots'
    )
    parser.add_argument('csv_files', nargs='+',
                        help='2–5 CSV files (overall_statistics.csv format)')
    parser.add_argument('output_dir',
                        help='Output directory for comparison plots')
    parser.add_argument('--labels', nargs='+',
                        help='Labels for each dataset (must match number of CSV files)')
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Separate last positional arg as output_dir, rest as csv_files
    # (argparse can't natively split nargs='+' before a positional, so we handle it)
    all_positional = args.csv_files
    output_dir = args.output_dir

    csv_files = all_positional  # already correctly parsed

    if len(csv_files) < 2 or len(csv_files) > 5:
        print("Error: Please provide between 2 and 5 CSV files.")
        sys.exit(1)

    for f in csv_files:
        if not Path(f).exists():
            print(f"Error: File '{f}' does not exist!")
            sys.exit(1)

    # Build labels
    if args.labels:
        if len(args.labels) != len(csv_files):
            print("Error: Number of --labels must match number of CSV files.")
            sys.exit(1)
        labels = args.labels
    else:
        labels = [Path(f).stem for f in csv_files]

    print("Histogram Comparison Tool")
    print("=" * 60)
    for label, f in zip(labels, csv_files):
        print(f"  {label}: {f}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    success = create_comparison_plots(csv_files, labels, output_dir)

    if success:
        print("\nComparison completed successfully!")
    else:
        print("\nComparison failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()