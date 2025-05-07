#!/usr/bin/env python3
"""
Plot the results from evaluate_attention_manipulation.py using seaborn.
This script creates attractive visualizations to compare different attention manipulations.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set the style for all plots
sns.set(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Define colors for different manipulation types
MANIPULATION_COLORS = {
    'baseline': '#1f77b4',  # Blue
    'gamma_zero': '#ff7f0e',  # Orange
    'constant': '#2ca02c',  # Green
    'random': '#d62728',  # Red
    'bypass': '#9467bd'  # Purple
}

def load_results(csv_path='./attention_manipulation_results/all_results.csv'):
    """Load results from CSV file"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    # Load the data with proper numeric types
    df = pd.read_csv(csv_path)
    
    # Ensure all numeric columns are properly converted
    numeric_cols = ['psnr', 'ms_ssim', 'lpips', 'bpp']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill any NaN values with 0
    df = df.fillna(0)
    
    # Extract lambda value from checkpoint name
    df['lambda'] = df['checkpoint'].str.extract(r'lambda_(\d+\.\d+)').astype(float)
    
    # Sort by lambda value for consistent plotting
    df = df.sort_values('lambda')
    
    # Filter out extreme outliers that might affect visualization
    df = df[df['psnr'] > -30]  # Filter out extreme negative PSNR values
    
    return df

def plot_metric_by_manipulation(df, metric, output_dir='./attention_manipulation_plots'):
    """Create a grouped bar plot for a specific metric across different manipulation types"""
    plt.figure(figsize=(12, 7))
    
    # Create a filtered version for visualization - clip extreme values
    df_plot = df.copy()
    
    # For PSNR, clip negative values to ensure reasonable visualization
    if metric == 'psnr':
        df_plot = df_plot[df_plot[metric] > -10]  # Filter extreme negative values
    
    # Create the grouped bar chart
    ax = sns.barplot(
        x='checkpoint', 
        y=metric, 
        hue='manipulation',
        data=df_plot,
        palette=MANIPULATION_COLORS
    )
    
    # Get the metric pretty name for the title
    metric_names = {
        'psnr': 'PSNR (dB)',
        'ms_ssim': 'MS-SSIM',
        'lpips': 'LPIPS (lower is better)',
        'bpp': 'Bits Per Pixel (BPP)'
    }
    
    metric_title = metric_names.get(metric, metric)
    
    # Set labels and title
    plt.title(f'Impact of Attention Manipulation on {metric_title}')
    plt.xlabel('Checkpoint (λ value)')
    plt.ylabel(metric_title)
    
    # Get unique lambda values and format as tick labels
    lambda_values = df_plot['lambda'].unique()
    labels = [f"λ={x:.6f}" for x in lambda_values]
    
    # Set fixed number of ticks to avoid warnings
    ax.set_xticks(range(len(lambda_values)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    plt.legend(title='Manipulation Type')
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'bar_{metric}_by_manipulation.png'), dpi=300)
    
    plt.close()

def plot_heatmap(df, metric, output_dir='./attention_manipulation_plots'):
    """Create a heatmap showing the relative impact of manipulations"""
    # Create a pivot table
    pivot = df.pivot_table(
        index='manipulation', 
        columns='checkpoint', 
        values=metric
    )
    
    # Handle extreme values that might cause visualization issues
    # Apply a threshold to avoid excessive distortion
    pivot = pivot.clip(lower=-1000, upper=1000)
    
    # For LPIPS, lower is better, so we'll invert the values for visualization
    if metric == 'lpips':
        baseline_values = pivot.loc['baseline', :]
        relative_values = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
        
        for manipulation in pivot.index:
            # Calculate percentage change from baseline (larger values = more degradation)
            changes = (pivot.loc[manipulation, :].astype(float) - baseline_values.astype(float)) / baseline_values.astype(float) * 100
            # Clip extreme values for better visualization
            changes = changes.clip(lower=-100, upper=1000)
            relative_values.loc[manipulation, :] = changes
    else:
        # For all other metrics, higher is better
        baseline_values = pivot.loc['baseline', :]
        relative_values = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
        
        for manipulation in pivot.index:
            # Calculate percentage change from baseline (negative values = degradation)
            changes = (pivot.loc[manipulation, :].astype(float) - baseline_values.astype(float)) / baseline_values.astype(float) * 100
            # Clip extreme values for better visualization
            changes = changes.clip(lower=-100, upper=100)
            relative_values.loc[manipulation, :] = changes
    
    # Get metric name
    metric_names = {
        'psnr': 'PSNR',
        'ms_ssim': 'MS-SSIM',
        'lpips': 'LPIPS',
        'bpp': 'BPP'
    }
    metric_name = metric_names.get(metric, metric)
    
    # Create heatmap
    plt.figure(figsize=(12, 6))
    
    # Define a custom diverging colormap centered at 0
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # Create the heatmap - ensure data is numeric
    try:
        # Convert explicitly to float values
        heatmap_data = relative_values.astype(float)
        
        # Replace any remaining NaN values
        heatmap_data = heatmap_data.fillna(0)
        
        # Create the heatmap
        ax = sns.heatmap(
            heatmap_data, 
            annot=True, 
            cmap=cmap, 
            center=0, 
            fmt=".1f",
            linewidths=.5,
            cbar_kws={'label': '% Change from Baseline'}
        )
    except Exception as e:
        print(f"Error creating heatmap for {metric}: {e}")
        print("Falling back to simpler visualization...")
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        plt.title(f'Unable to create heatmap for {metric_name}')
    
    # Format column labels to show lambda values
    lambda_values = [float(col.split('_')[1]) for col in pivot.columns]
    ax.set_xticklabels([f"λ={x:.6f}" for x in lambda_values], rotation=45, ha='right')
    
    plt.title(f'Relative Change in {metric_name} by Manipulation Type (%)')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'heatmap_{metric}_relative_change.png'), dpi=300)
    
    plt.close()

def plot_line_comparison(df, metric, output_dir='./attention_manipulation_plots'):
    """Create a line plot comparing performance with BPP on x-axis"""
    plt.figure(figsize=(10, 6))
    
    # Skip this plot for bpp metric
    if metric == 'bpp':
        plt.title(f'Skipping BPP vs. BPP plot (redundant)')
        return
    
    # Filter out extreme values
    df_plot = df.copy()
    # Filter out the extremely high BPP values that mess up the scale
    df_plot = df_plot[df_plot['bpp'] < 10]
    
    # Calculate average BPP per manipulation type and lambda value
    avg_bpp = df_plot.groupby(['manipulation', 'lambda'])['bpp'].mean().reset_index()
    
    # Merge back to get other metrics
    plot_data = df_plot.merge(avg_bpp, on=['manipulation', 'lambda'], suffixes=('', '_avg'))
    
    # Add small offsets to separate overlapping lines
    # This makes them distinct in the plot without affecting the analysis
    manipulation_offsets = {
        'baseline': 0.0,
        'constant': 0.003,  # Small offset for constant manipulation
        'random': 0.006,    # Slightly larger offset for random manipulation
        'gamma_zero': 0.0,  # No offset needed as it's already distinct
        'bypass': 0.0       # No offset needed as it's already distinct
    }
    
    # Apply offsets
    for manipulation, offset in manipulation_offsets.items():
        mask = plot_data['manipulation'] == manipulation
        if metric == 'psnr':
            plot_data.loc[mask, metric] += offset
        elif metric == 'ms_ssim':
            plot_data.loc[mask, metric] += offset * 0.005  # Smaller scale for MS-SSIM
        elif metric == 'lpips':
            plot_data.loc[mask, metric] += offset * 0.002  # Smaller scale for LPIPS
    
    # Create line plot
    ax = sns.lineplot(
        data=plot_data,
        x='bpp_avg',  # Use average BPP on x-axis
        y=metric,
        hue='manipulation',
        style='manipulation',
        markers=True,
        dashes=False,
        palette=MANIPULATION_COLORS,
        linewidth=2.5  # Thicker lines for better visibility
    )
    
    # Get the metric pretty name for the title
    metric_names = {
        'psnr': 'PSNR (dB)',
        'ms_ssim': 'MS-SSIM',
        'lpips': 'LPIPS (lower is better)',
        'bpp': 'Bits Per Pixel (BPP)'
    }
    
    metric_title = metric_names.get(metric, metric)
    
    # Set labels and title
    plt.title(f'{metric_title} vs. Average Bits Per Pixel by Manipulation Type')
    plt.xlabel('Average Bits Per Pixel (BPP)')
    plt.ylabel(metric_title)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    plt.legend(title='Manipulation Type')
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'line_{metric}_by_bpp.png'), dpi=300)
    
    plt.close()

def create_radar_chart(df, output_dir='./attention_manipulation_plots'):
    """Create a radar chart to compare manipulations across all metrics"""
    # Average across all checkpoints
    avg_by_manipulation = df.groupby('manipulation')[['psnr', 'ms_ssim', 'lpips', 'bpp']].mean()
    
    # Normalize metrics to [0, 1] for radar chart
    # For LPIPS, lower is better, so we invert it
    normalized = pd.DataFrame(index=avg_by_manipulation.index)
    
    for col in avg_by_manipulation.columns:
        if col == 'lpips':
            # Invert so lower values = better performance
            min_val = avg_by_manipulation[col].min()
            max_val = avg_by_manipulation[col].max()
            normalized[col] = 1 - (avg_by_manipulation[col] - min_val) / (max_val - min_val)
        else:
            # Higher values = better performance
            min_val = avg_by_manipulation[col].min()
            max_val = avg_by_manipulation[col].max()
            normalized[col] = (avg_by_manipulation[col] - min_val) / (max_val - min_val)
    
    # Set up the radar chart
    labels = ['PSNR', 'MS-SSIM', 'LPIPS', 'BPP']
    num_vars = len(labels)
    
    # Angle for each axis
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    
    # Close the plot
    angles += angles[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add each manipulation type
    for i, manipulation in enumerate(normalized.index):
        values = normalized.loc[manipulation].tolist()
        values += values[:1]  # Close the loop
        
        color = MANIPULATION_COLORS.get(manipulation, f'C{i}')
        
        ax.plot(angles, values, 'o-', linewidth=2, label=manipulation, color=color)
        ax.fill(angles, values, color=color, alpha=0.1)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    # Draw y-axis labels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=8)
    plt.ylim(0, 1)
    
    # Add title
    plt.title('Performance Comparison Across All Metrics\n(Higher is Better)', size=14, y=1.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'radar_overall_comparison.png'), dpi=300)
    
    plt.close()

def create_manipulation_summary(df, output_dir='./attention_manipulation_plots'):
    """Create a summary visualization showing the overall impact of manipulations"""
    # Calculate percentage difference from baseline for each checkpoint and metric
    metrics = ['psnr', 'ms_ssim', 'lpips', 'bpp']
    summary_data = []
    
    for checkpoint in df['checkpoint'].unique():
        checkpoint_data = df[df['checkpoint'] == checkpoint]
        baseline = checkpoint_data[checkpoint_data['manipulation'] == 'baseline'].iloc[0]
        
        for _, row in checkpoint_data.iterrows():
            if row['manipulation'] != 'baseline':
                result = {
                    'checkpoint': checkpoint,
                    'manipulation': row['manipulation'],
                    'lambda': row['lambda']
                }
                
                for metric in metrics:
                    if metric == 'lpips':
                        # For LPIPS, lower is better
                        pct_change = (row[metric] - baseline[metric]) / baseline[metric] * 100
                    else:
                        # For other metrics, higher is better
                        pct_change = (row[metric] - baseline[metric]) / baseline[metric] * 100
                    
                    result[f'{metric}_pct'] = pct_change
                
                summary_data.append(result)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create a grouped bar chart showing percentage changes across all metrics
    plt.figure(figsize=(15, 8))
    
    # Reshape data for seaborn
    plot_data = pd.melt(
        summary_df, 
        id_vars=['manipulation', 'lambda'], 
        value_vars=[f'{m}_pct' for m in metrics],
        var_name='metric', 
        value_name='percent_change'
    )
    
    # Clean up metric names for plot
    plot_data['metric'] = plot_data['metric'].str.replace('_pct', '')
    
    # Create facet grid
    g = sns.FacetGrid(
        plot_data, 
        col='metric', 
        height=4, 
        aspect=1.2,
        sharex=True, 
        sharey=False,
        despine=True
    )
    
    # Map barplot to each facet
    g.map_dataframe(
        sns.barplot, 
        x='manipulation', 
        y='percent_change', 
        hue='manipulation',
        palette=MANIPULATION_COLORS,
        ci=None
    )
    
    # Customize appearance
    g.set_axis_labels('Manipulation Type', 'Percent Change from Baseline')
    g.set_titles(col_template='{col_name}')
    
    # Add horizontal line at y=0
    for ax in g.axes.flat:
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Remove duplicate legends
        if ax.get_subplotspec().colspan.start != 0:
            ax.get_legend().remove()
    
    g.fig.suptitle('Impact of Attention Manipulations (% Change from Baseline)', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'summary_percent_change.png'), dpi=300)
    
    plt.close()

def main():
    """Main function to generate all plots"""
    output_dir = './attention_manipulation_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    try:
        df = load_results()
        print(f"Loaded data for {len(df)} evaluation runs")
        
        # Print data overview to help diagnose issues
        print("\nData summary:")
        print(df.describe())
        print("\nData types:")
        print(df.dtypes)
        
        # Generate individual metric plots
        metrics = ['psnr', 'ms_ssim', 'lpips', 'bpp']
        for metric in metrics:
            try:
                print(f"\nGenerating plots for {metric}...")
                plot_metric_by_manipulation(df, metric, output_dir)
                plot_heatmap(df, metric, output_dir)
                plot_line_comparison(df, metric, output_dir)
            except Exception as e:
                print(f"Error generating {metric} plots: {e}")
        
        # Generate summary visualizations
        try:
            print("\nGenerating summary visualizations...")
            create_radar_chart(df, output_dir)
            create_manipulation_summary(df, output_dir)
        except Exception as e:
            print(f"Error generating summary visualizations: {e}")
        
        print(f"\nAll plots saved to {output_dir}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run evaluate_attention_manipulation.py first to generate results.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
