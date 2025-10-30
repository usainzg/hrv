import marimo

__generated_with = "0.17.3"
app = marimo.App()


@app.cell
def _():
    """
    HRV Comparison Script
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
    plt.rcParams['figure.figsize'] = (15, 10)


    def load_hrv_data(filepath):
        """
        Load HRV data from CSV file
        Expected columns: date, suunto, hrv4training
        """
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df


    def calculate_rolling_baseline(data, window_days):
        """
        Calculate rolling baseline (moving average) for given window
        """
        return data.rolling(window=window_days, min_periods=1).mean()


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


    def plot_hrv_comparison(df, stats_dict, mean_values, diff_values):
        """
        Create comprehensive visualization of HRV comparison
        """
        fig = plt.figure(figsize=(18, 12))

        # 1. Daily HRV Values
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(df['date'], df['suunto'], marker='o', label='Suunto',
                 linewidth=2, markersize=4, alpha=0.7)
        ax1.plot(df['date'], df['hrv4training'], marker='s', label='HRV4Training',
                 linewidth=2, markersize=4, alpha=0.7)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('HRV (ms)', fontsize=12)
        ax1.set_title('Daily HRV Values', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 2. 7-Day Rolling Baseline
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(df['date'], df['suunto_7d'], label='Suunto (7-day)',
                 linewidth=2.5, alpha=0.8)
        ax2.plot(df['date'], df['hrv4training_7d'], label='HRV4Training (7-day)',
                 linewidth=2.5, alpha=0.8)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('HRV (ms)', fontsize=12)
        ax2.set_title('7-Day Rolling Baseline', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 3. 60-Day Normal Values (Baseline)
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(df['date'], df['suunto_60d'], label='Suunto (60-day)',
                 linewidth=3, alpha=0.8)
        ax3.plot(df['date'], df['hrv4training_60d'], label='HRV4Training (60-day)',
                 linewidth=3, alpha=0.8)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel('HRV (ms)', fontsize=12)
        ax3.set_title('60-Day Normal Baseline', fontsize=14, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 4. All Trends Combined
        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(df['date'], df['suunto'], 'o-', label='Suunto Daily',
                 alpha=0.3, markersize=3)
        ax4.plot(df['date'], df['suunto_7d'], label='Suunto (7d)',
                 linewidth=2, alpha=0.8)
        ax4.plot(df['date'], df['suunto_60d'], label='Suunto (60d)',
                 linewidth=2.5, alpha=0.9)
        ax4.set_xlabel('Date', fontsize=12)
        ax4.set_ylabel('HRV (ms)', fontsize=12)
        ax4.set_title('Suunto - All Trends', fontsize=14, fontweight='bold')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 5. Scatter Plot with Correlation
        ax5 = plt.subplot(3, 2, 5)
        ax5.scatter(df['suunto'], df['hrv4training'], alpha=0.6, s=50)

        # Add regression line
        z = np.polyfit(df['suunto'].dropna(), df['hrv4training'].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['suunto'].min(), df['suunto'].max(), 100)
        ax5.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Fit line')

        # Add identity line
        min_val = min(df['suunto'].min(), df['hrv4training'].min())
        max_val = max(df['suunto'].max(), df['hrv4training'].max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'k--',
                 linewidth=1, alpha=0.5, label='Identity line')

        ax5.set_xlabel('Suunto HRV (ms)', fontsize=12)
        ax5.set_ylabel('HRV4Training HRV (ms)', fontsize=12)
        ax5.set_title(f'Correlation Plot (r = {stats_dict["correlation"]:.3f})',
                      fontsize=14, fontweight='bold')
        ax5.legend(loc='best')
        ax5.grid(True, alpha=0.3)

        # 6. Bland-Altman Plot
        ax6 = plt.subplot(3, 2, 6)
        ax6.scatter(mean_values, diff_values, alpha=0.6, s=50)
        ax6.axhline(stats_dict['mean_difference'], color='red',
                    linestyle='-', linewidth=2, label=f'Mean diff: {stats_dict["mean_difference"]:.2f}')
        ax6.axhline(stats_dict['limits_of_agreement'][0], color='red',
                    linestyle='--', linewidth=1.5,
                    label=f'Lower LoA: {stats_dict["limits_of_agreement"][0]:.2f}')
        ax6.axhline(stats_dict['limits_of_agreement'][1], color='red',
                    linestyle='--', linewidth=1.5,
                    label=f'Upper LoA: {stats_dict["limits_of_agreement"][1]:.2f}')
        ax6.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax6.set_xlabel('Mean of Methods (ms)', fontsize=12)
        ax6.set_ylabel('Difference (Suunto - HRV4Training) (ms)', fontsize=12)
        ax6.set_title('Bland-Altman Plot', fontsize=14, fontweight='bold')
        ax6.legend(loc='best', fontsize=9)
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


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

        # Calculate rolling baselines
        print("\nCalculating rolling baselines...")
        df['suunto_7d'] = calculate_rolling_baseline(df['suunto'], 7)
        df['hrv4training_7d'] = calculate_rolling_baseline(df['hrv4training'], 7)
        df['suunto_60d'] = calculate_rolling_baseline(df['suunto'], 60)
        df['hrv4training_60d'] = calculate_rolling_baseline(df['hrv4training'], 60)

        # Calculate statistics
        print("Calculating statistics...")
        stats_dict, mean_values, diff_values = calculate_statistics(df)

        # Print summary
        print_summary_statistics(df, stats_dict)

        # Create visualizations
        print("Generating plots...")
        fig = plot_hrv_comparison(df, stats_dict, mean_values, diff_values)

        # Save figure
        output_filename = 'hrv_comparison_analysis.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as '{output_filename}'")

        # Save processed data
        output_csv = 'hrv_comparison_processed.csv'
        df.to_csv(output_csv, index=False)
        print(f"Processed data saved as '{output_csv}'")

        plt.show()

    def _main_():
        main()

    _main_()
    return


if __name__ == "__main__":
    app.run()
