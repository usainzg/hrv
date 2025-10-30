"""
HRV Comparison Script - Enhanced Styling with Shaded Normal Range
Compares two methods of recording HRV values with trend analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 10)


def load_hrv_data(filepath):
    """
    Load HRV data from CSV file
    Expected columns: date, method1, method2
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def calculate_rolling_baseline_with_std(data, window_days):
    """
    Calculate rolling baseline (moving average) and standard deviation for given window
    """
    rolling_mean = data.rolling(window=window_days, min_periods=1).mean()
    rolling_std = data.rolling(window=window_days, min_periods=1).std()
    return rolling_mean, rolling_std


def plot_hrv_styled(df, method_name, ax, color_daily, color_7d, color_60d, color_shade):
    """
    Create styled HRV plot with shaded normal range for a single method
    """
    # Plot daily values as bars
    ax.bar(df['date'], df[f'{method_name}'], 
           color=color_daily, alpha=0.3, width=0.8, 
           label='Daily HRV value', zorder=1)
    
    # Plot 7-day baseline
    ax.plot(df['date'], df[f'{method_name}_7d'], 
            color=color_7d, linewidth=2.5, marker='o', markersize=4,
            label='7 days HRV baseline', zorder=3)
    
    # Plot 60-day baseline with shaded area (±1 std)
    ax.plot(df['date'], df[f'{method_name}_60d'], 
            color=color_60d, linewidth=2, linestyle='--',
            label='Normal values (60 days)', zorder=2)
    
    # Shaded area for normal range (±1 standard deviation)
    upper_bound = df[f'{method_name}_60d'] + df[f'{method_name}_60d_std']
    lower_bound = df[f'{method_name}_60d'] - df[f'{method_name}_60d_std']
    
    ax.fill_between(df['date'], lower_bound, upper_bound, 
                     color=color_shade, alpha=0.2, zorder=1,
                     label='Normal range (±1 SD)')
    
    # Styling
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('HRV (ms)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)


def plot_comparison_combined(df):
    """
    Create comprehensive comparison plot with both methods
    """
    fig, axes = plt.subplots(3, 1, figsize=(20, 16))
    
    # Color schemes
    method1_colors = {
        'daily': '#BDBDBD',
        '7d': '#2196F3',
        '60d': '#90CAF9',
        'shade': '#BBDEFB'
    }
    
    method2_colors = {
        'daily': '#BDBDBD',
        '7d': '#F57C00',
        '60d': '#FFB74D',
        'shade': '#FFE0B2'
    }
    
    # Plot 1: Method 1
    plot_hrv_styled(df, 'method1', axes[0], 
                    method1_colors['daily'], method1_colors['7d'], 
                    method1_colors['60d'], method1_colors['shade'])
    axes[0].set_title('Heart Rate Variability - Method 1', 
                      fontsize=16, fontweight='bold', pad=20)
    
    # Plot 2: Method 2
    plot_hrv_styled(df, 'method2', axes[1], 
                    method2_colors['daily'], method2_colors['7d'], 
                    method2_colors['60d'], method2_colors['shade'])
    axes[1].set_title('Heart Rate Variability - Method 2', 
                      fontsize=16, fontweight='bold', pad=20)
    
    # Plot 3: Combined comparison
    ax = axes[2]
    
    # Method 1 - Daily bars (lighter)
    ax.bar(df['date'], df['method1'], 
           color=method1_colors['daily'], alpha=0.2, width=0.8, 
           label='Method 1 - Daily', zorder=1)
    
    # Method 2 - Daily bars (even lighter, slightly offset)
    ax.bar(df['date'], df['method2'], 
           color=method2_colors['daily'], alpha=0.2, width=0.6, 
           label='Method 2 - Daily', zorder=1)
    
    # Method 1 - 7-day baseline
    ax.plot(df['date'], df['method1_7d'], 
            color=method1_colors['7d'], linewidth=2.5, marker='o', markersize=4,
            label='Method 1 - 7d baseline', zorder=4, alpha=0.8)
    
    # Method 2 - 7-day baseline
    ax.plot(df['date'], df['method2_7d'], 
            color=method2_colors['7d'], linewidth=2.5, marker='s', markersize=4,
            label='Method 2 - 7d baseline', zorder=4, alpha=0.8)
    
    # Method 1 - 60-day baseline with shaded area
    ax.plot(df['date'], df['method1_60d'], 
            color=method1_colors['60d'], linewidth=2, linestyle='--',
            label='Method 1 - Normal (60d)', zorder=3)
    
    upper_bound_m1 = df['method1_60d'] + df['method1_60d_std']
    lower_bound_m1 = df['method1_60d'] - df['method1_60d_std']
    ax.fill_between(df['date'], lower_bound_m1, upper_bound_m1, 
                     color=method1_colors['shade'], alpha=0.15, zorder=1,
                     label='Method 1 - Normal range (±1 SD)')
    
    # Method 2 - 60-day baseline with shaded area
    ax.plot(df['date'], df['method2_60d'], 
            color=method2_colors['60d'], linewidth=2, linestyle='--',
            label='Method 2 - Normal (60d)', zorder=3)
    
    upper_bound_m2 = df['method2_60d'] + df['method2_60d_std']
    lower_bound_m2 = df['method2_60d'] - df['method2_60d_std']
    ax.fill_between(df['date'], lower_bound_m2, upper_bound_m2, 
                     color=method2_colors['shade'], alpha=0.15, zorder=1,
                     label='Method 2 - Normal range (±1 SD)')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('HRV (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Heart Rate Variability - Complete Comparison (Both Methods)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)
    
    # Format x-axis for all plots
    for ax in axes:
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_comparison_overlay(df):
    """
    Create a single overlay plot with both methods for direct comparison
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Color schemes with better contrast
    method1_colors = {
        'daily': '#9E9E9E',
        '7d': '#1976D2',
        '60d': '#64B5F6',
        'shade': '#BBDEFB'
    }
    
    method2_colors = {
        'daily': '#757575',
        '7d': '#E65100',
        '60d': '#FF9800',
        'shade': '#FFE0B2'
    }
    
    # Method 1 - 60-day baseline with shaded area (plot first, so it's in background)
    upper_bound_m1 = df['method1_60d'] + df['method1_60d_std']
    lower_bound_m1 = df['method1_60d'] - df['method1_60d_std']
    ax.fill_between(df['date'], lower_bound_m1, upper_bound_m1, 
                     color=method1_colors['shade'], alpha=0.2, zorder=1,
                     label='Method 1 - Normal range (±1 SD)')
    
    # Method 2 - 60-day baseline with shaded area
    upper_bound_m2 = df['method2_60d'] + df['method2_60d_std']
    lower_bound_m2 = df['method2_60d'] - df['method2_60d_std']
    ax.fill_between(df['date'], lower_bound_m2, upper_bound_m2, 
                     color=method2_colors['shade'], alpha=0.2, zorder=1,
                     label='Method 2 - Normal range (±1 SD)')
    
    # Daily bars for both methods (very subtle)
    width = 0.4
    dates_numeric = np.arange(len(df['date']))
    ax.bar(df['date'], df['method1'], 
           color=method1_colors['daily'], alpha=0.25, width=width, 
           label='Method 1 - Daily', zorder=2)
    ax.bar(df['date'], df['method2'], 
           color=method2_colors['daily'], alpha=0.25, width=width*0.7, 
           label='Method 2 - Daily', zorder=2)
    
    # 60-day baselines (dotted lines)
    ax.plot(df['date'], df['method1_60d'], 
            color=method1_colors['60d'], linewidth=2, linestyle=':',
            label='Method 1 - Normal (60d)', zorder=3, alpha=0.7)
    ax.plot(df['date'], df['method2_60d'], 
            color=method2_colors['60d'], linewidth=2, linestyle=':',
            label='Method 2 - Normal (60d)', zorder=3, alpha=0.7)
    
    # 7-day baselines (solid lines with markers - most prominent)
    ax.plot(df['date'], df['method1_7d'], 
            color=method1_colors['7d'], linewidth=3, marker='o', markersize=5,
            label='Method 1 - 7d baseline', zorder=5, alpha=0.9)
    ax.plot(df['date'], df['method2_7d'], 
            color=method2_colors['7d'], linewidth=3, marker='s', markersize=5,
            label='Method 2 - 7d baseline', zorder=5, alpha=0.9)
    
    # Styling
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('HRV (ms)', fontsize=14, fontweight='bold')
    ax.set_title('Heart Rate Variability - Complete Method Comparison', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend with better organization
    handles, labels = ax.get_legend_handles_labels()
    # Reorder legend items for clarity
    order = [4, 5, 0, 1, 6, 7, 2, 3]  # Group by type
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
              loc='upper left', fontsize=10, framealpha=0.95, ncol=2)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def calculate_statistics(df):
    """
    Calculate correlation and other statistics between methods
    """
    correlation = df[['method1', 'method2']].corr().iloc[0, 1]
    
    # Bland-Altman analysis
    mean_values = (df['method1'] + df['method2']) / 2
    diff_values = df['method1'] - df['method2']
    mean_diff = diff_values.mean()
    std_diff = diff_values.std()
    
    stats_dict = {
        'correlation': correlation,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'limits_of_agreement': (mean_diff - 1.96 * std_diff, mean_diff + 1.96 * std_diff)
    }
    
    return stats_dict, mean_values, diff_values


def print_summary_statistics(df, stats_dict):
    """
    Print summary statistics
    """
    print("\n" + "="*60)
    print("HRV COMPARISON SUMMARY STATISTICS")
    print("="*60)
    
    print("\n--- METHOD 1 ---")
    print(f"Mean: {df['method1'].mean():.2f} ms")
    print(f"Std Dev: {df['method1'].std():.2f} ms")
    print(f"Min: {df['method1'].min():.2f} ms")
    print(f"Max: {df['method1'].max():.2f} ms")
    
    print("\n--- METHOD 2 ---")
    print(f"Mean: {df['method2'].mean():.2f} ms")
    print(f"Std Dev: {df['method2'].std():.2f} ms")
    print(f"Min: {df['method2'].min():.2f} ms")
    print(f"Max: {df['method2'].max():.2f} ms")
    
    print("\n--- COMPARISON ---")
    print(f"Correlation: {stats_dict['correlation']:.4f}")
    print(f"Mean Difference: {stats_dict['mean_difference']:.2f} ms")
    print(f"Std of Differences: {stats_dict['std_difference']:.2f} ms")
    print(f"Limits of Agreement: [{stats_dict['limits_of_agreement'][0]:.2f}, {stats_dict['limits_of_agreement'][1]:.2f}]")
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(df['method1'].dropna(), df['method2'].dropna())
    print(f"\nPaired t-test: t = {t_stat:.4f}, p = {p_value:.4f}")
    if p_value < 0.05:
        print("→ Significant difference between methods (p < 0.05)")
    else:
        print("→ No significant difference between methods (p >= 0.05)")
    
    print("\n" + "="*60 + "\n")


def main():
    """
    Main execution function
    """
    # Load data
    print("Loading HRV data...")
    filepath = 'hrv_data.csv'  # Update this with your file path
    df = load_hrv_data(filepath)
    
    print(f"Loaded {len(df)} days of data")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Calculate rolling baselines with standard deviations
    print("\nCalculating rolling baselines and standard deviations...")
    df['method1_7d'], _ = calculate_rolling_baseline_with_std(df['method1'], 7)
    df['method2_7d'], _ = calculate_rolling_baseline_with_std(df['method2'], 7)
    df['method1_60d'], df['method1_60d_std'] = calculate_rolling_baseline_with_std(df['method1'], 60)
    df['method2_60d'], df['method2_60d_std'] = calculate_rolling_baseline_with_std(df['method2'], 60)
    
    # Calculate statistics
    print("Calculating statistics...")
    stats_dict, mean_values, diff_values = calculate_statistics(df)
    
    # Print summary
    print_summary_statistics(df, stats_dict)
    
    # Create visualizations
    print("Generating plots...")
    
    # Three-panel comparison (separate + combined)
    fig1 = plot_comparison_combined(df)
    output_filename1 = 'hrv_comparison_three_panel.png'
    plt.savefig(output_filename1, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Three-panel plot saved as '{output_filename1}'")
    
    # Single overlay comparison
    fig2 = plot_comparison_overlay(df)
    output_filename2 = 'hrv_comparison_overlay.png'
    plt.savefig(output_filename2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Overlay plot saved as '{output_filename2}'")
    
    # Save processed data
    output_csv = 'hrv_comparison_processed.csv'
    df.to_csv(output_csv, index=False)
    print(f"✓ Processed data saved as '{output_csv}'")
    
    plt.show()


if __name__ == "__main__":
    main()