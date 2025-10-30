import marimo

__generated_with = "0.17.3"
app = marimo.App()


@app.cell
def _():
    """
    HRV Comparison Script - Enhanced Styling with Horizontal Banded Shaded Area
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
        Expected columns: date, suunto,hrv4training
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


    def create_horizontal_bands_fill(ax, dates, baseline, std_dev, color, alpha=0.2, zorder=1):
        """
        Create horizontal banded shading that follows the baseline
        Uses step-like horizontal fills for each time period
        """
        for i in range(len(dates) - 1):
            if pd.notna(baseline.iloc[i]) and pd.notna(std_dev.iloc[i]):
                upper = baseline.iloc[i] + std_dev.iloc[i]
                lower = baseline.iloc[i] - std_dev.iloc[i]

                # Create horizontal band from current date to next date
                ax.fill_between([dates.iloc[i], dates.iloc[i+1]],
                               lower, upper,
                               color=color, alpha=alpha, zorder=zorder,
                               step='post', linewidth=0)

        # Handle the last point
        if len(dates) > 0 and pd.notna(baseline.iloc[-1]) and pd.notna(std_dev.iloc[-1]):
            upper = baseline.iloc[-1] + std_dev.iloc[-1]
            lower = baseline.iloc[-1] - std_dev.iloc[-1]
            # Extend the last band slightly
            last_date = dates.iloc[-1]
            extend_date = last_date + pd.Timedelta(days=1)
            ax.fill_between([last_date, extend_date],
                           lower, upper,
                           color=color, alpha=alpha, zorder=zorder,
                           linewidth=0)


    def plot_hrv_styled(df, method_name, ax, color_daily, color_7d, color_60d, color_shade):
        """
        Create styled HRV plot with horizontal banded shaded normal range for a single method
        """
        # Create horizontal banded shading following 60-day baseline
        create_horizontal_bands_fill(ax, df['date'],
                                     df[f'{method_name}_60d'],
                                     df[f'{method_name}_60d_std'],
                                     color_shade, alpha=0.25, zorder=1)

        # Plot 60-day baseline
        ax.plot(df['date'], df[f'{method_name}_60d'],
                color=color_60d, linewidth=2, linestyle='--',
                label='Normal values (60 days)', zorder=2, alpha=0.7)

        # Plot daily values as bars
        ax.bar(df['date'], df[f'{method_name}'],
               color=color_daily, alpha=0.5, width=0.8,
               label='Daily HRV value', zorder=3)

        # Plot 7-day baseline
        ax.plot(df['date'], df[f'{method_name}_7d'],
                color=color_7d, linewidth=2.5, marker='o', markersize=4,
                label='7 days HRV baseline', zorder=4)

        # Dummy plot for legend (shaded area)
        ax.fill_between([], [], [], color=color_shade, alpha=0.25,
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
        suunto_colors = {
            'daily': '#A5D6A7',
            '7d': '#1976D2',
            '60d': '#64B5F6',
            'shade': '#BBDEFB'
        }

        hrv4training_colors = {
            'daily': '#A5D6A7',
            '7d': '#F57C00',
            '60d': '#FFB74D',
            'shade': '#FFE0B2'
        }

        # Plot 1:Suunto
        plot_hrv_styled(df, 'suunto', axes[0],
                        suunto_colors['daily'], suunto_colors['7d'],
                        suunto_colors['60d'], suunto_colors['shade'])
        axes[0].set_title('Heart Rate Variability - Suunto',
                          fontsize=16, fontweight='bold', pad=20)

        # Plot 2:HRV4Training
        plot_hrv_styled(df, 'hrv4training', axes[1],
                        hrv4training_colors['daily'], hrv4training_colors['7d'],
                        hrv4training_colors['60d'], hrv4training_colors['shade'])
        axes[1].set_title('Heart Rate Variability - HRV4Training',
                          fontsize=16, fontweight='bold', pad=20)

        # Plot 3: Combined comparison
        ax = axes[2]

        # Suunto - Horizontal banded shading
        create_horizontal_bands_fill(ax, df['date'],
                                     df['suunto_60d'],
                                     df['suunto_60d_std'],
                                     suunto_colors['shade'], alpha=0.18, zorder=1)

        # HRV4Training - Horizontal banded shading
        create_horizontal_bands_fill(ax, df['date'],
                                     df['hrv4training_60d'],
                                     df['hrv4training_60d_std'],
                                     hrv4training_colors['shade'], alpha=0.18, zorder=1)

        # 60-day baselines
        ax.plot(df['date'], df['suunto_60d'],
                color=suunto_colors['60d'], linewidth=2, linestyle='--',
                label='Suunto - Normal (60d)', alpha=0.6, zorder=2)
        ax.plot(df['date'], df['hrv4training_60d'],
                color=hrv4training_colors['60d'], linewidth=2, linestyle='--',
                label='HRV4Training - Normal (60d)', alpha=0.6, zorder=2)

        # Suunto - Daily bars
        ax.bar(df['date'], df['suunto'],
               color=suunto_colors['daily'], alpha=0.3, width=0.6,
               label='Suunto - Daily', zorder=3)

        # HRV4Training - Daily bars (slightly different opacity)
        ax.bar(df['date'], df['hrv4training'],
               color=hrv4training_colors['daily'], alpha=0.25, width=0.5,
               label='HRV4Training - Daily', zorder=3)

        # Suunto - 7-day baseline
        ax.plot(df['date'], df['suunto_7d'],
                color=suunto_colors['7d'], linewidth=2.5, marker='o', markersize=4,
                label='Suunto - 7d baseline', zorder=5, alpha=0.9)

        # HRV4Training - 7-day baseline
        ax.plot(df['date'], df['hrv4training_7d'],
                color=hrv4training_colors['7d'], linewidth=2.5, marker='s', markersize=4,
                label='HRV4Training - 7d baseline', zorder=5, alpha=0.9)

        # Dummy plots for legend (shaded areas)
        ax.fill_between([], [], [], color=suunto_colors['shade'], alpha=0.18,
                       label='Suunto - Normal range (±1 SD)')
        ax.fill_between([], [], [], color=hrv4training_colors['shade'], alpha=0.18,
                       label='HRV4Training - Normal range (±1 SD)')

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

        # Color schemes
        suunto_colors = {
            'daily': '#A5D6A7',
            '7d': '#1976D2',
            '60d': '#64B5F6',
            'shade': '#BBDEFB'
        }

        hrv4training_colors = {
            'daily': '#A5D6A7',
            '7d': '#E65100',
            '60d': '#FF9800',
            'shade': '#FFE0B2'
        }

        # Suunto - Horizontal banded shading (background)
        create_horizontal_bands_fill(ax, df['date'],
                                     df['suunto_60d'],
                                     df['suunto_60d_std'],
                                     suunto_colors['shade'], alpha=0.2, zorder=1)

        # HRV4Training - Horizontal banded shading (background)
        create_horizontal_bands_fill(ax, df['date'],
                                     df['hrv4training_60d'],
                                     df['hrv4training_60d_std'],
                                     hrv4training_colors['shade'], alpha=0.2, zorder=1)

        # 60-day baselines
        ax.plot(df['date'], df['suunto_60d'],
                color=suunto_colors['60d'], linewidth=2, linestyle=':',
                label='Suunto - Normal (60d)', alpha=0.7, zorder=2)
        ax.plot(df['date'], df['hrv4training_60d'],
                color=hrv4training_colors['60d'], linewidth=2, linestyle=':',
                label='HRV4Training - Normal (60d)', alpha=0.7, zorder=2)

        # Daily bars for both methods
        ax.bar(df['date'], df['suunto'],
               color=suunto_colors['daily'], alpha=0.35, width=0.6,
               label='Suunto - Daily', zorder=3)
        ax.bar(df['date'], df['hrv4training'],
               color=hrv4training_colors['daily'], alpha=0.3, width=0.5,
               label='HRV4Training - Daily', zorder=3)

        # 7-day baselines (most prominent)
        ax.plot(df['date'], df['suunto_7d'],
                color=suunto_colors['7d'], linewidth=3, marker='o', markersize=5,
                label='Suunto - 7d baseline', zorder=5, alpha=0.9)
        ax.plot(df['date'], df['hrv4training_7d'],
                color=hrv4training_colors['7d'], linewidth=3, marker='s', markersize=5,
                label='HRV4Training - 7d baseline', zorder=5, alpha=0.9)

        # Dummy plots for legend (shaded areas)
        ax.fill_between([], [], [], color=suunto_colors['shade'], alpha=0.2,
                       label='Suunto - Normal range (±1 SD)')
        ax.fill_between([], [], [], color=hrv4training_colors['shade'], alpha=0.2,
                       label='HRV4Training - Normal range (±1 SD)')

        # Styling
        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax.set_ylabel('HRV (ms)', fontsize=14, fontweight='bold')
        ax.set_title('Heart Rate Variability - Complete Method Comparison',
                     fontsize=18, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95, ncol=2)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig


    def calculate_statistics(df):
        """
        Calculate correlation and other statistics between methods
        """
        correlation = df[['suunto', 'hrv4training']].corr().iloc[0, 1]

        # Bland-Altman analysis
        mean_values = (df['suunto'] + df['hrv4training']) / 2
        diff_values = df['suunto'] - df['hrv4training']
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
        print(f"Mean: {df['suunto'].mean():.2f} ms")
        print(f"Std Dev: {df['suunto'].std():.2f} ms")
        print(f"Min: {df['suunto'].min():.2f} ms")
        print(f"Max: {df['suunto'].max():.2f} ms")

        print("\n--- METHOD 2 ---")
        print(f"Mean: {df['hrv4training'].mean():.2f} ms")
        print(f"Std Dev: {df['hrv4training'].std():.2f} ms")
        print(f"Min: {df['hrv4training'].min():.2f} ms")
        print(f"Max: {df['hrv4training'].max():.2f} ms")

        print("\n--- COMPARISON ---")
        print(f"Correlation: {stats_dict['correlation']:.4f}")
        print(f"Mean Difference: {stats_dict['mean_difference']:.2f} ms")
        print(f"Std of Differences: {stats_dict['std_difference']:.2f} ms")
        print(f"Limits of Agreement: [{stats_dict['limits_of_agreement'][0]:.2f}, {stats_dict['limits_of_agreement'][1]:.2f}]")

        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(df['suunto'].dropna(), df['hrv4training'].dropna())
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
        df['suunto_7d'], _ = calculate_rolling_baseline_with_std(df['suunto'], 7)
        df['hrv4training_7d'], _ = calculate_rolling_baseline_with_std(df['hrv4training'], 7)
        df['suunto_60d'], df['suunto_60d_std'] = calculate_rolling_baseline_with_std(df['suunto'], 60)
        df['hrv4training_60d'], df['hrv4training_60d_std'] = calculate_rolling_baseline_with_std(df['hrv4training'], 60)

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

    def _main_():
        main()

    _main_()
    return


if __name__ == "__main__":
    app.run()
