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
and generates a plot comparing LPIPS vs BPP for different checkpoints.
"""

def extract_lambda_value(checkpoint_name):
    """Extract the lambda value from checkpoint filename for sorting"""
    # Extract the numeric value between "lambda_" and ".pth.tar"
    lambda_str = checkpoint_name.split('lambda_')[1].split('.pth.tar')[0]
    return float(lambda_str)

def main():
    # Set the style
    sns.set(style="whitegrid", context="paper")
    plt.figure(figsize=(10, 6))
    
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
        no_caption_bpp = all_results[cp]['no_caption']['avg_bpp']
        no_caption_lpips = all_results[cp]['no_caption']['avg_lpips']
        
        data.append({
            'checkpoint': cp,
            'lambda': lambda_val,
            'type': 'With Caption',
            'bpp': with_caption_bpp,
            'lpips': with_caption_lpips
        })
        
        data.append({
            'checkpoint': cp,
            'lambda': lambda_val,
            'type': 'No Caption',
            'bpp': no_caption_bpp,
            'lpips': no_caption_lpips
        })
    
    # Convert to DataFrame for easier plotting with seaborn
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Create a scatter plot with connecting lines
    sns.scatterplot(
        data=df, 
        x='bpp', 
        y='lpips', 
        hue='type', 
        style='type',
        s=150,  # Marker size
        palette={'With Caption': 'green', 'No Caption': 'blue'}
    )
    
    # Connect points with lines
    for caption_type in ['With Caption', 'No Caption']:
        subset = df[df['type'] == caption_type].sort_values('bpp')
        plt.plot(subset['bpp'], subset['lpips'], '-', 
                 color='green' if caption_type == 'With Caption' else 'blue',
                 linewidth=2.5)
    
    # Add checkpoint labels
    for cp in checkpoint_names:
        lambda_val = extract_lambda_value(cp)
        with_caption_row = df[(df['checkpoint'] == cp) & (df['type'] == 'With Caption')]
        
        # Only annotate the "With Caption" points to avoid cluttering
        plt.annotate(
            f"λ={lambda_val}", 
            (with_caption_row['bpp'].values[0], with_caption_row['lpips'].values[0]),
            xytext=(0, -15), 
            textcoords='offset points',
            ha='center', 
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
        )
    
    # Styling
    plt.title('TACO Performance on Kodak Dataset: Caption Impact', fontsize=16)
    plt.xlabel('Bits per pixel (BPP)', fontsize=14)
    plt.ylabel('LPIPS (lower is better) ↓', fontsize=14)
    plt.legend(title='', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Adjust y limits to add a little space at the bottom for annotations
    y_min, y_max = plt.ylim()
    plt.ylim(y_min - 0.003, y_max)
    
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
    print("-" * 80)
    print(f"{'Checkpoint':<20} {'Caption Type':<15} {'LPIPS':<10} {'BPP':<10}")
    print("-" * 80)
    
    for cp in checkpoint_names:
        print(f"{cp:<20} {'With Caption':<15} {all_results[cp]['with_caption']['avg_lpips']:.4f} {all_results[cp]['with_caption']['avg_bpp']:.4f}")
        print(f"{cp:<20} {'No Caption':<15} {all_results[cp]['no_caption']['avg_lpips']:.4f} {all_results[cp]['no_caption']['avg_bpp']:.4f}")
        print("-" * 80)

if __name__ == "__main__":
    main()
