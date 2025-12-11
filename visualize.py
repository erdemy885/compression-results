import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys

def normalize_column_name(col):
    """Normalizes column names to find intersections."""
    col = col.lower().strip()
    if col == 'ms_ssim':
        return 'msssim'
    return col

def get_rate_column(df):
    """Heuristic to find the bitrate column."""
    # Priority list for rate columns
    candidates = ['bpp_real', 'q_bpp', 'bpp', 'bitrate']
    
    # Lowercase columns for search
    df_cols_lower = [c.lower() for c in df.columns]
    
    for candidate in candidates:
        if candidate in df_cols_lower:
            return df.columns[df_cols_lower.index(candidate)]
    
    # Fallback: look for any column containing 'bpp' that isn't 'target' or 'original'
    for col in df.columns:
        c_low = col.lower()
        if 'bpp' in c_low and 'target' not in c_low and 'original' not in c_low:
            return col
            
    return None

def standardize_df(df):
    """Renames columns to standard names and identifies/renames the rate column to 'bpp'."""
    # 1. Normalize metric names
    new_cols = {}
    for col in df.columns:
        new_cols[col] = normalize_column_name(col)
    df = df.rename(columns=new_cols)
    
    # 2. Identify and rename rate column to 'bpp'
    rate_col = get_rate_column(df)
    if rate_col:
        df = df.rename(columns={rate_col: 'bpp'})
    
    return df

def load_data_from_dir(directory):
    """Recursively loads, standardizes, and concatenates all CSV files in a directory."""
    all_dfs = []
    detected_rate_col = None
    print(f"Scanning {directory}...")
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        # Detect rate column from the first valid file
                        if detected_rate_col is None:
                            detected_rate_col = get_rate_column(df)

                        # Standardize columns before merging to ensure alignment
                        df = standardize_df(df)
                        all_dfs.append(df)
                except Exception as e:
                    print(f"  Warning: Could not read {file_path}: {e}")
    
    if not all_dfs:
        return pd.DataFrame(), None
    
    # Concatenate all found dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True, sort=False)
    print(f"  Total rows for {directory}: {len(combined_df)}")
    return combined_df, detected_rate_col

def main():
    parser = argparse.ArgumentParser(description="Generate Rate-Distortion graphs from two directories of CSV files.")
    parser.add_argument("dir1", help="Path to the first directory")
    parser.add_argument("dir2", help="Path to the second directory")
    parser.add_argument("--output_dir", default=".", help="Directory to save output graphs")
    
    args = parser.parse_args()

    if not os.path.isdir(args.dir1) or not os.path.isdir(args.dir2):
        print("Error: One or both inputs are not directories.")
        sys.exit(1)

    # Load and standardize DataFrames
    df1, rate_col1 = load_data_from_dir(args.dir1)
    df2, rate_col2 = load_data_from_dir(args.dir2)

    if df1.empty or df2.empty:
        print("Error: One or both directories contain no valid CSV data.")
        sys.exit(1)
    
    if 'bpp' not in df1.columns or 'bpp' not in df2.columns:
        print("Error: Could not identify 'bpp' column in one of the aggregated datasets.")
        sys.exit(1)

    # Determine labels based on rate column
    def get_label(rate_col, default_label):
        if not rate_col:
            return default_label
        rc = rate_col.lower()
        if rc == 'q_bpp':
            return 'hific'
        elif rc == 'bpp_real':
            return 'image-gs'
        return default_label

    # Get labels for the legend based on directory names or rate columns
    default_label1 = os.path.basename(os.path.normpath(args.dir1))
    default_label2 = os.path.basename(os.path.normpath(args.dir2))
    
    label1 = get_label(rate_col1, default_label1)
    label2 = get_label(rate_col2, default_label2)

    if label1 == label2:
        label1 += " (1)"
        label2 += " (2)"

    # Find intersection of columns (metrics)
    metrics1 = set(df1.columns)
    metrics2 = set(df2.columns)
    
    shared_metrics = metrics1.intersection(metrics2)
    
    # Filter out non-metric columns
    exclude = {'input_filename', 'output_filename', 'bytes', 'encoding_time', 'flip', 'num_gaussians', 'bpp_original', 'bpp', 'target_bpp'}
    shared_metrics = shared_metrics - exclude

    if not shared_metrics:
        print("No shared metrics found between the two directories.")
        sys.exit(1)

    print(f"Found shared metrics: {', '.join(shared_metrics)}")

    # Create Output Directory
    if args.output_dir != "." and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Plotting
    for metric in shared_metrics:
        plt.figure(figsize=(10, 6))
        
        # Plot scatter points
        # Drop NaNs for the specific metric being plotted
        data1 = df1[['bpp', metric]].dropna()
        data2 = df2[['bpp', metric]].dropna()

        plt.scatter(data1['bpp'], data1[metric], alpha=0.6, label=label1, marker='o')
        plt.scatter(data2['bpp'], data2[metric], alpha=0.6, label=label2, marker='x')

        # Calculate and plot averages
        if not data1.empty:
            plt.scatter(data1['bpp'].mean(), data1[metric].mean(), s=200, c='blue', edgecolors='black', marker='*', label=f"{label1} Mean")
        if not data2.empty:
            plt.scatter(data2['bpp'].mean(), data2[metric].mean(), s=200, c='orange', edgecolors='black', marker='*', label=f"{label2} Mean")

        plt.title(f"Rate-Distortion Comparison: {metric.upper()}")
        plt.xlabel("Bitrate (bpp)")
        plt.ylabel(metric.upper())
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        output_path = os.path.join(args.output_dir, f"comparison_{metric}.png")
        plt.savefig(output_path)
        print(f"Saved graph to {output_path}")
        plt.close()

if __name__ == "__main__":
    main()