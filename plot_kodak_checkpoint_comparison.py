#!/usr/bin/env python3
import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

"""
This script reads the pre-computed results from the kodak_checkpoint_comparison_results.json file
and generates side-by-side plots comparing LPIPS vs BPP and PSNR vs BPP for different checkpoints.
"""

def extract_lambda_value(checkpoint_name):
    """Extract the lambda value from checkpoint filename for sorting"""
    # Extract the numeric value between "lambda_" and ".pth.tar"
    lambda_str = checkpoint_name.split('lambda_')[1].split('.pth.tar')[0]
    return float(lambda_str)

def main():
    # Set the style
    sns.set(style="whitegrid", context="paper")
    
    # Load the JSON results file
    results_file = "kodak_checkpoint_comparison_results.json"
    print(f"Reading results from {results_file}")
    
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    
    # Get checkpoint names and sort them by lambda value
    checkpoint_names = list(all_results.keys())
    checkpoint_names.sort(key=extract_lambda_value)
    
    print(f"Found results for {len(checkpoint_names)} checkpoints: {checkpoint_names}")
    
    # Extract data for plotting
    data = []
    for cp in checkpoint_names:
        lambda_val = extract_lambda_value(cp)
        with_caption_bpp = all_results[cp]['with_caption']['avg_bpp']
        with_caption_lpips = all_results[cp]['with_caption']['avg_lpips']
        with_caption_psnr = all_results[cp]['with_caption']['avg_psnr']
        no_caption_bpp = all_results[cp]['no_caption']['avg_bpp']
        no_caption_lpips = all_results[cp]['no_caption']['avg_lpips']
        no_caption_psnr = all_results[cp]['no_caption']['avg_psnr']
        
        data.append({
            'checkpoint': cp,
            'lambda': lambda_val,
            'type': 'With Caption',
            'bpp': with_caption_bpp,
            'lpips': with_caption_lpips,
            'psnr': with_caption_psnr
        })
        
        data.append({
            'checkpoint': cp,
            'lambda': lambda_val,
            'type': 'No Caption',
            'bpp': no_caption_bpp,
            'lpips': no_caption_lpips,
            'psnr': no_caption_psnr
        })
    
    # Convert to DataFrame for easier plotting with seaborn
    df = pd.DataFrame(data)
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: LPIPS vs BPP
    # Create a scatter plot with connecting lines on the first subplot
    sns.scatterplot(
        data=df, 
        x='bpp', 
        y='lpips', 
        hue='type', 
        style='type',
        s=150,  # Marker size
        palette={'With Caption': 'green', 'No Caption': 'blue'},
        ax=ax1
    )
    
    # Connect points with lines for LPIPS plot
    for caption_type in ['With Caption', 'No Caption']:
        subset = df[df['type'] == caption_type].sort_values('bpp')
        ax1.plot(subset['bpp'], subset['lpips'], '-', 
                 color='green' if caption_type == 'With Caption' else 'blue',
                 linewidth=2.5)
    
    # Add checkpoint labels for LPIPS plot
    for cp in checkpoint_names:
        lambda_val = extract_lambda_value(cp)
        with_caption_row = df[(df['checkpoint'] == cp) & (df['type'] == 'With Caption')]
        
        # Only annotate the "With Caption" points to avoid cluttering
        ax1.annotate(
            f"λ={lambda_val}", 
            (with_caption_row['bpp'].values[0], with_caption_row['lpips'].values[0]),
            xytext=(0, -15), 
            textcoords='offset points',
            ha='center', 
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
        )
    
    # Styling for LPIPS plot
    ax1.set_title('LPIPS vs BPP', fontsize=16)
    ax1.set_xlabel('Bits per pixel (BPP)', fontsize=14)
    ax1.set_ylabel('LPIPS (lower is better) ↓', fontsize=14)
    ax1.legend(title='', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Adjust y limits for LPIPS plot to add a little space at the bottom for annotations
    y_min, y_max = ax1.get_ylim()
    ax1.set_ylim(y_min - 0.003, y_max)
    
    # Plot 2: PSNR vs BPP
    # Create a scatter plot with connecting lines on the second subplot
    sns.scatterplot(
        data=df, 
        x='bpp', 
        y='psnr', 
        hue='type', 
        style='type',
        s=150,  # Marker size
        palette={'With Caption': 'green', 'No Caption': 'blue'},
        ax=ax2
    )
    
    # Connect points with lines for PSNR plot
    for caption_type in ['With Caption', 'No Caption']:
        subset = df[df['type'] == caption_type].sort_values('bpp')
        ax2.plot(subset['bpp'], subset['psnr'], '-', 
                 color='green' if caption_type == 'With Caption' else 'blue',
                 linewidth=2.5)
    
    # Add checkpoint labels for PSNR plot
    for cp in checkpoint_names:
        lambda_val = extract_lambda_value(cp)
        with_caption_row = df[(df['checkpoint'] == cp) & (df['type'] == 'With Caption')]
        
        # Only annotate the "With Caption" points to avoid cluttering
        ax2.annotate(
            f"λ={lambda_val}", 
            (with_caption_row['bpp'].values[0], with_caption_row['psnr'].values[0]),
            xytext=(0, -15), 
            textcoords='offset points',
            ha='center', 
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
        )
    
    # Styling for PSNR plot
    ax2.set_title('PSNR vs BPP', fontsize=16)
    ax2.set_xlabel('Bits per pixel (BPP)', fontsize=14)
    ax2.set_ylabel('PSNR (higher is better) ↑', fontsize=14)
    ax2.legend(title='', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Adjust y limits for PSNR plot to add a little space at the bottom for annotations
    y_min, y_max = ax2.get_ylim()
    ax2.set_ylim(y_min - 0.3, y_max)
    
    # Add a common title for the entire figure
    fig.suptitle('TACO Performance on Kodak Dataset: Caption Impact', fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
    
    # Save plot
    plt.savefig('kodak_caption_impact_plot.png', dpi=300, bbox_inches='tight')
    print("Generated plot saved as kodak_caption_impact_plot.png")
    
    # Also export data to CSV for reference
    csv_file = "kodak_caption_impact_data.csv"
    df.to_csv(csv_file, index=False)
    print(f"Data exported to CSV: {csv_file}")
    
    # Show the plot
    plt.show()
    
    # Also print the numerical results
    print("\nNumerical Results:")
    print("-" * 100)
    print(f"{'Checkpoint':<20} {'Caption Type':<15} {'LPIPS':<10} {'PSNR':<10} {'BPP':<10}")
    print("-" * 100)
    
    for cp in checkpoint_names:
        print(f"{cp:<20} {'With Caption':<15} {all_results[cp]['with_caption']['avg_lpips']:.4f} {all_results[cp]['with_caption']['avg_psnr']:.4f} {all_results[cp]['with_caption']['avg_bpp']:.4f}")
        print(f"{cp:<20} {'No Caption':<15} {all_results[cp]['no_caption']['avg_lpips']:.4f} {all_results[cp]['no_caption']['avg_psnr']:.4f} {all_results[cp]['no_caption']['avg_bpp']:.4f}")
        print("-" * 100)

if __name__ == "__main__":
    main()
