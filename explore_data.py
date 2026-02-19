#!/usr/bin/env python3
"""
Data Exploration Script for the Dataset
==========================================

This script explores IMU data from smart ring wearables and keystroke timestamps.
It provides comprehensive statistics and optional visualizations.

Usage:
    python explore_data.py [--visualize] [--save-plots]
"""

import os
import pickle
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_pkl_file(pkl_path):
    """Load and return pickle file contents."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def parse_filename(filename):
    """Parse filename to extract subject, session, and ring side."""
    # Format: 003_005_DIBS-L_corrected.csv or 003_005_Macbook.pkl
    parts = filename.replace('_corrected', '').replace('.csv', '').replace('.pkl', '').split('_')
    if len(parts) >= 3:
        subject = parts[0]
        session = parts[1]
        device = '_'.join(parts[2:])
        ring_side = None
        if 'DIBS-L' in device:
            ring_side = 'L'
        elif 'DIBS-R' in device:
            ring_side = 'R'
        return subject, session, ring_side, device
    return None, None, None, None


def analyze_imu_data(csv_path, visualize=False):
    """Analyze IMU data from CSV file."""
    print(f"\n{'='*60}")
    print(f"Analyzing IMU Data: {os.path.basename(csv_path)}")
    print(f"{'='*60}")
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Basic info
    print(f"\n📊 Basic Statistics:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Time range
    if 'Effective Timestamp' in df.columns:
        time_col = 'Effective Timestamp'
    elif 'Time Stamp' in df.columns:
        time_col = 'Time Stamp'
    else:
        time_col = None
    
    if time_col:
        time_min = df[time_col].min()
        time_max = df[time_col].max()
        duration = time_max - time_min
        print(f"\n⏱️  Time Range:")
        print(f"  Start: {datetime.fromtimestamp(time_min)}")
        print(f"  End: {datetime.fromtimestamp(time_max)}")
        print(f"  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # Sampling rate
        if len(df) > 1:
            time_diff = df[time_col].diff().dropna()
            avg_sample_rate = 1.0 / time_diff.mean()
            print(f"  Average sampling rate: {avg_sample_rate:.2f} Hz")
    
    # IMU statistics
    imu_cols = ['Accel-x', 'Accel-y', 'Accel-z', 'Gyro-x', 'Gyro-y', 'Gyro-z']
    available_imu_cols = [col for col in imu_cols if col in df.columns]
    
    if available_imu_cols:
        print(f"\n📈 IMU Sensor Statistics:")
        stats = df[available_imu_cols].describe()
        print(stats.to_string())
        
        # Check for missing values
        missing = df[available_imu_cols].isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠️  Missing values:")
            print(missing[missing > 0].to_string())
        else:
            print(f"\n✅ No missing values in IMU data")
    
    # Visualizations
    if visualize and available_imu_cols:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'IMU Data: {os.path.basename(csv_path)}', fontsize=16)
        
        for idx, col in enumerate(available_imu_cols):
            ax = axes[idx // 3, idx % 3]
            
            # Sample data for faster plotting if too large
            sample_size = min(50000, len(df))
            if len(df) > sample_size:
                sample_df = df.sample(n=sample_size, random_state=42).sort_values(time_col if time_col else df.index.name)
            else:
                sample_df = df.sort_values(time_col if time_col else df.index.name)
            
            if time_col:
                ax.plot(sample_df[time_col], sample_df[col], alpha=0.6, linewidth=0.5)
                ax.set_xlabel('Time (timestamp)')
            else:
                ax.plot(sample_df.index, sample_df[col], alpha=0.6, linewidth=0.5)
                ax.set_xlabel('Sample Index')
            
            ax.set_ylabel(col)
            ax.set_title(f'{col} over Time')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    return None


def analyze_keystroke_data(pkl_path, visualize=False):
    """Analyze keystroke timestamp data from PKL file."""
    print(f"\n{'='*60}")
    print(f"Analyzing Keystroke Data: {os.path.basename(pkl_path)}")
    print(f"{'='*60}")
    
    # Load data
    data = load_pkl_file(pkl_path)
    
    print(f"\n📦 Data Structure:")
    print(f"  Keys: {list(data.keys())}")
    
    # Analyze key_times
    if 'key_times' in data:
        key_times = data['key_times']
        print(f"\n⌨️  Keystroke Statistics:")
        print(f"  Unique keys pressed: {len(key_times)}")
        
        # Count total keystrokes
        total_keystrokes = sum(len(times) for times in key_times.values())
        print(f"  Total keystrokes: {total_keystrokes:,}")
        
        # Most frequent keys
        key_counts = {key: len(times) for key, times in key_times.items()}
        sorted_keys = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n  Top 20 most frequent keys:")
        for key, count in sorted_keys[:20]:
            print(f"    '{key}': {count} presses")
        
        # Time range
        all_starts = []
        all_ends = []
        for times_list in key_times.values():
            for time_dict in times_list:
                if 'start' in time_dict and time_dict['start'] is not None:
                    all_starts.append(time_dict['start'])
                if 'end' in time_dict and time_dict['end'] is not None:
                    all_ends.append(time_dict['end'])
        
        if all_starts and all_ends:
            min_time = min(all_starts)
            max_time = max(all_ends)
            duration = max_time - min_time
            print(f"\n  Time range:")
            print(f"    Start: {datetime.fromtimestamp(min_time)}")
            print(f"    End: {datetime.fromtimestamp(max_time)}")
            print(f"    Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # Keystroke duration statistics
        durations = []
        for times_list in key_times.values():
            for time_dict in times_list:
                if 'start' in time_dict and 'end' in time_dict:
                    if time_dict['start'] is not None and time_dict['end'] is not None:
                        durations.append(time_dict['end'] - time_dict['start'])
        
        if durations:
            durations = np.array(durations) * 1000  # Convert to milliseconds
            print(f"\n  Keystroke duration statistics (ms):")
            print(f"    Mean: {durations.mean():.2f} ms")
            print(f"    Median: {np.median(durations):.2f} ms")
            print(f"    Std: {durations.std():.2f} ms")
            print(f"    Min: {durations.min():.2f} ms")
            print(f"    Max: {durations.max():.2f} ms")
    
    # Analyze other data
    for key in ['mouse_moves', 'mouse_clicks', 'mouse_scrolls']:
        if key in data:
            data_list = data[key]
            if isinstance(data_list, (list, dict)):
                count = len(data_list) if hasattr(data_list, '__len__') else 'N/A'
                print(f"\n🖱️  {key}: {count} events")
    
    # Session times
    if 'session_start_times' in data and 'session_end_times' in data:
        print(f"\n📅 Session Information:")
        start_times = data['session_start_times']
        end_times = data['session_end_times']
        
        if isinstance(start_times, dict):
            print(f"  Session starts: {len(start_times)} (by device)")
            print(f"  Session ends: {len(end_times)} (by device)")
            if start_times:
                # Get the earliest start time
                earliest_start = min(start_times.values())
                print(f"  First session start: {datetime.fromtimestamp(earliest_start)}")
            if end_times:
                # Get the latest end time
                latest_end = max(end_times.values())
                print(f"  Last session end: {datetime.fromtimestamp(latest_end)}")
        elif isinstance(start_times, (list, tuple)):
            print(f"  Session starts: {len(start_times)}")
            print(f"  Session ends: {len(end_times)}")
            if start_times:
                print(f"  First session start: {datetime.fromtimestamp(start_times[0])}")
            if end_times:
                print(f"  Last session end: {datetime.fromtimestamp(end_times[-1])}")
    
    # Visualizations
    if visualize and 'key_times' in data:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Keystroke Data: {os.path.basename(pkl_path)}', fontsize=16)
        
        key_times = data['key_times']
        
        # 1. Key frequency bar chart
        ax1 = axes[0, 0]
        key_counts = {key: len(times) for key, times in key_times.items()}
        sorted_keys = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        keys, counts = zip(*sorted_keys) if sorted_keys else ([], [])
        ax1.barh(range(len(keys)), counts)
        ax1.set_yticks(range(len(keys)))
        ax1.set_yticklabels(keys)
        ax1.set_xlabel('Frequency')
        ax1.set_title('Top 20 Most Frequent Keys')
        ax1.invert_yaxis()
        
        # 2. Keystroke duration distribution
        ax2 = axes[0, 1]
        durations = []
        for times_list in key_times.values():
            for time_dict in times_list:
                if 'start' in time_dict and 'end' in time_dict:
                    if time_dict['start'] is not None and time_dict['end'] is not None:
                        durations.append((time_dict['end'] - time_dict['start']) * 1000)
        if durations:
            ax2.hist(durations, bins=50, edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Duration (ms)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Keystroke Duration Distribution')
            ax2.axvline(np.mean(durations), color='r', linestyle='--', label=f'Mean: {np.mean(durations):.1f}ms')
            ax2.legend()
        
        # 3. Keystroke timeline
        ax3 = axes[1, 0]
        all_events = []
        for key, times_list in key_times.items():
            for time_dict in times_list:
                if 'start' in time_dict and time_dict['start'] is not None:
                    all_events.append((time_dict['start'], key))
        all_events.sort()
        
        if all_events:
            # Sample for visualization if too many
            if len(all_events) > 10000:
                sample_events = all_events[::len(all_events)//10000]
            else:
                sample_events = all_events
            
            times, keys = zip(*sample_events) if sample_events else ([], [])
            if times:
                # Normalize times to start from 0
                times = np.array(times)
                times = times - times[0]
                ax3.scatter(times, range(len(times)), alpha=0.5, s=1)
                ax3.set_xlabel('Time (seconds from start)')
                ax3.set_ylabel('Keystroke Index')
                ax3.set_title('Keystroke Timeline')
        
        # 4. Keys per second over time
        ax4 = axes[1, 1]
        if all_events:
            # Calculate typing rate in 10-second windows
            if len(all_events) > 1:
                start_time = all_events[0][0]
                end_time = all_events[-1][0]
                window_size = 10  # seconds
                windows = []
                rates = []
                
                current_time = start_time
                while current_time < end_time:
                    window_end = current_time + window_size
                    count = sum(1 for t, _ in all_events if current_time <= t < window_end)
                    windows.append(current_time - start_time)
                    rates.append(count / window_size)
                    current_time = window_end
                
                if windows:
                    ax4.plot(windows, rates, linewidth=2)
                    ax4.set_xlabel('Time (seconds from start)')
                    ax4.set_ylabel('Keystrokes per second')
                    ax4.set_title('Typing Rate Over Time')
                    ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    return None


def analyze_data_alignment(csv_path, pkl_path, visualize=False):
    """Analyze alignment between IMU data and keystroke data."""
    print(f"\n{'='*60}")
    print(f"Analyzing Data Alignment")
    print(f"{'='*60}")
    
    # Load data
    df = pd.read_csv(csv_path)
    keystroke_data = load_pkl_file(pkl_path)
    
    # Get time ranges
    if 'Effective Timestamp' in df.columns:
        imu_time_col = 'Effective Timestamp'
    elif 'Time Stamp' in df.columns:
        imu_time_col = 'Time Stamp'
    else:
        print("⚠️  Cannot find timestamp column in IMU data")
        return None
    
    imu_start = df[imu_time_col].min()
    imu_end = df[imu_time_col].max()
    
    # Get keystroke time range
    if 'key_times' in keystroke_data:
        all_starts = []
        all_ends = []
        for times_list in keystroke_data['key_times'].values():
            for time_dict in times_list:
                if 'start' in time_dict and time_dict['start'] is not None:
                    all_starts.append(time_dict['start'])
                if 'end' in time_dict and time_dict['end'] is not None:
                    all_ends.append(time_dict['end'])
        
        if all_starts and all_ends:
            key_start = min(all_starts)
            key_end = max(all_ends)
            
            print(f"\n🔄 Time Alignment:")
            print(f"  IMU data range: {datetime.fromtimestamp(imu_start)} to {datetime.fromtimestamp(imu_end)}")
            print(f"  Keystroke range: {datetime.fromtimestamp(key_start)} to {datetime.fromtimestamp(key_end)}")
            
            overlap_start = max(imu_start, key_start)
            overlap_end = min(imu_end, key_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            print(f"  Overlap: {overlap_duration:.2f} seconds ({overlap_duration/60:.2f} minutes)")
            
            if overlap_duration > 0:
                print(f"  ✅ Data has temporal overlap")
            else:
                print(f"  ⚠️  No temporal overlap detected")
            
            # Calculate how many keystrokes fall within IMU range
            keystrokes_in_range = 0
            total_keystrokes = 0
            for times_list in keystroke_data['key_times'].values():
                for time_dict in times_list:
                    if 'start' in time_dict and time_dict['start'] is not None:
                        total_keystrokes += 1
                        if imu_start <= time_dict['start'] <= imu_end:
                            keystrokes_in_range += 1
            
            print(f"\n  Keystroke coverage:")
            print(f"    Total keystrokes: {total_keystrokes:,}")
            print(f"    Keystrokes in IMU range: {keystrokes_in_range:,} ({100*keystrokes_in_range/total_keystrokes:.1f}%)")
    
    return None


def create_combined_imu_keystroke_plot(csv_files_session, pkl_file, subject, session, visualize=False):
    """Create a combined plot showing IMU data with keystrokes overlaid."""
    if not visualize:
        return None
    
    # Load keystroke data
    keystroke_data = load_pkl_file(pkl_file)
    if 'key_times' not in keystroke_data:
        return None
    
    # Collect all keystroke events
    all_events = []
    for key, times_list in keystroke_data['key_times'].items():
        for time_dict in times_list:
            if 'start' in time_dict and time_dict['start'] is not None:
                all_events.append((time_dict['start'], key))
    all_events.sort()
    
    if not all_events:
        return None
    
    # Create figure with subplots for each IMU file
    num_imu_files = len(csv_files_session)
    if num_imu_files == 0:
        return None
    
    fig, axes = plt.subplots(num_imu_files, 1, figsize=(18, 6 * num_imu_files))
    if num_imu_files == 1:
        axes = [axes]
    
    fig.suptitle(f'IMU Data with Keystrokes: Subject {subject}, Session {session}', fontsize=16, y=0.995)
    
    # Get the earliest timestamp from all IMU files for normalization
    earliest_imu_time = None
    for csv_file in csv_files_session:
        df = pd.read_csv(csv_file)
        if 'Effective Timestamp' in df.columns:
            time_col = 'Effective Timestamp'
        elif 'Time Stamp' in df.columns:
            time_col = 'Time Stamp'
        else:
            continue
        file_start = df[time_col].min()
        if earliest_imu_time is None or file_start < earliest_imu_time:
            earliest_imu_time = file_start
    
    if earliest_imu_time is None:
        return None
    
    for idx, csv_file in enumerate(csv_files_session):
        ax = axes[idx]
        
        # Load IMU data
        df = pd.read_csv(csv_file)
        _, _, ring_side, _ = parse_filename(csv_file.name)
        
        if 'Effective Timestamp' in df.columns:
            time_col = 'Effective Timestamp'
        elif 'Time Stamp' in df.columns:
            time_col = 'Time Stamp'
        else:
            continue
        
        # Get full IMU time range (before sampling)
        imu_start_full = df[time_col].min()
        imu_end_full = df[time_col].max()
        imu_start_norm = imu_start_full - earliest_imu_time
        imu_end_norm = imu_end_full - earliest_imu_time
        
        # Drop rows with NaN values in IMU columns before sampling
        imu_cols = ['Accel-x', 'Accel-y', 'Accel-z', 'Gyro-x', 'Gyro-y', 'Gyro-z']
        df_clean = df.dropna(subset=imu_cols)
        
        # Sample data for faster plotting
        sample_size = min(50000, len(df_clean))
        if len(df_clean) > sample_size:
            sample_df = df_clean.sample(n=sample_size, random_state=42).sort_values(time_col)
        else:
            sample_df = df_clean.sort_values(time_col)
        
        # Normalize timestamps to start from 0 (using earliest time as reference)
        imu_times = sample_df[time_col].values
        imu_times_norm = imu_times - earliest_imu_time
        
        # Plot IMU data (all 6 channels)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for col_idx, col in enumerate(imu_cols):
            if col in sample_df.columns:
                # Get values (should already be clean of NaNs, but double-check)
                values = sample_df[col].values
                if len(values) > 0 and not np.all(np.isnan(values)):
                    # Normalize for visualization (z-score)
                    values_clean = values[~np.isnan(values)]
                    times_clean = imu_times_norm[~np.isnan(values)]
                    if len(values_clean) > 0:
                        values_norm = (values_clean - values_clean.mean()) / (values_clean.std() + 1e-8)
                        ax.plot(times_clean, values_norm + col_idx * 3, 
                               label=col, color=colors[col_idx], alpha=0.7, linewidth=0.5)
        
        # Overlay keystrokes
        # Normalize keystroke times to match IMU time range
        keystroke_times = [t for t, _ in all_events]
        keystroke_times_norm = np.array(keystroke_times) - earliest_imu_time
        
        # Filter keystrokes within full IMU time range (not just sampled range)
        valid_mask = (keystroke_times_norm >= imu_start_norm) & (keystroke_times_norm <= imu_end_norm)
        valid_keystroke_times = keystroke_times_norm[valid_mask]
        valid_keys = [k for i, (_, k) in enumerate(all_events) if valid_mask[i]]
        
        if len(valid_keystroke_times) > 0:
            # Plot keystrokes as vertical lines
            y_max = len(imu_cols) * 3 + 2
            for kt, key in zip(valid_keystroke_times, valid_keys):
                # Only show regular character keys to reduce clutter
                if len(key) == 1 or key == 'Key.space':
                    ax.axvline(kt, color='red', alpha=0.4, linewidth=0.8)
        
        ax.set_xlabel('Time (seconds from start)', fontsize=12)
        ax.set_ylabel('Normalized IMU Values (offset by channel)', fontsize=12)
        ax.set_title(f'Ring {ring_side} - IMU Data with {len(valid_keystroke_times)} Keystrokes Overlaid', fontsize=14)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Explore dataset')
    parser.add_argument('--visualize', action='store_true', 
                       help='Generate visualizations')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to files')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing data files (default: data)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Find all CSV and PKL files
    csv_files = sorted(data_dir.glob('*.csv'))
    pkl_files = sorted(data_dir.glob('*.pkl'))
    
    print(f"\n{'='*60}")
    print(f"Dataset Exploration")
    print(f"{'='*60}")
    print(f"\n📁 Data Directory: {data_dir}")
    print(f"  Found {len(csv_files)} CSV files (IMU data)")
    print(f"  Found {len(pkl_files)} PKL files (Keystroke data)")
    
    # Group files by subject and session
    file_groups = defaultdict(list)
    for csv_file in csv_files:
        subject, session, ring_side, device = parse_filename(csv_file.name)
        if subject and session:
            key = (subject, session)
            file_groups[key].append(('csv', csv_file, ring_side))
    
    for pkl_file in pkl_files:
        subject, session, ring_side, device = parse_filename(pkl_file.name)
        if subject and session:
            key = (subject, session)
            file_groups[key].append(('pkl', pkl_file, None))
    
    print(f"\n📋 Found {len(file_groups)} unique subject-session combinations")
    
    # Analyze each file
    figures = []
    
    for (subject, session), files in sorted(file_groups.items()):
        print(f"\n\n{'#'*60}")
        print(f"# Subject {subject}, Session {session}")
        print(f"{'#'*60}")
        
        # Analyze CSV files
        for file_type, file_path, ring_side in files:
            if file_type == 'csv':
                fig = analyze_imu_data(file_path, visualize=args.visualize)
                if fig:
                    figures.append((fig, f"imu_{subject}_{session}_{ring_side}"))
        
        # Analyze PKL files
        for file_type, file_path, _ in files:
            if file_type == 'pkl':
                fig = analyze_keystroke_data(file_path, visualize=args.visualize)
                if fig:
                    figures.append((fig, f"keystroke_{subject}_{session}"))
        
        # Analyze alignment for matching files
        # Sort CSV files by ring side (L first, then R) for consistent plotting
        csv_files_session = sorted([f for t, f, r in files if t == 'csv'], 
                                   key=lambda f: (parse_filename(f.name)[2] or 'Z', f.name))
        pkl_files_session = [f for t, f, _ in files if t == 'pkl']
        
        if csv_files_session and pkl_files_session:
            # Try to match L and R rings with keystroke data
            for csv_file in csv_files_session:
                subject, session, ring_side, _ = parse_filename(csv_file.name)
                for pkl_file in pkl_files_session:
                    analyze_data_alignment(csv_file, pkl_file, visualize=False)
                    break  # Just check one alignment per CSV
            
            # Create combined IMU + keystroke visualization
            if args.visualize and pkl_files_session:
                combined_fig = create_combined_imu_keystroke_plot(
                    csv_files_session, pkl_files_session[0], subject, session, visualize=args.visualize
                )
                if combined_fig:
                    figures.append((combined_fig, f"combined_{subject}_{session}"))
    
    # Summary statistics
    print(f"\n\n{'='*60}")
    print(f"Dataset Summary")
    print(f"{'='*60}")
    
    total_imu_rows = sum(len(pd.read_csv(f)) for f in csv_files)
    print(f"\n📊 Total IMU samples: {total_imu_rows:,}")
    
    # Calculate total duration
    total_duration_seconds = 0
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if 'Effective Timestamp' in df.columns:
            time_col = 'Effective Timestamp'
        elif 'Time Stamp' in df.columns:
            time_col = 'Time Stamp'
        else:
            continue
        duration = df[time_col].max() - df[time_col].min()
        total_duration_seconds += duration
    
    total_duration_minutes = total_duration_seconds / 60
    total_duration_hours = total_duration_minutes / 60
    print(f"⏱️  Total IMU data duration: {total_duration_seconds:.1f} seconds ({total_duration_minutes:.1f} minutes, {total_duration_hours:.2f} hours)")
    
    total_keystrokes = 0
    total_keystroke_duration_seconds = 0
    for pkl_file in pkl_files:
        data = load_pkl_file(pkl_file)
        if 'key_times' in data:
            keystrokes_in_file = sum(len(times) for times in data['key_times'].values())
            total_keystrokes += keystrokes_in_file
            
            # Calculate keystroke time range
            all_starts = []
            all_ends = []
            for times_list in data['key_times'].values():
                for time_dict in times_list:
                    if 'start' in time_dict and time_dict['start'] is not None:
                        all_starts.append(time_dict['start'])
                    if 'end' in time_dict and time_dict['end'] is not None:
                        all_ends.append(time_dict['end'])
            if all_starts and all_ends:
                file_duration = max(all_ends) - min(all_starts)
                total_keystroke_duration_seconds += file_duration
    
    print(f"⌨️  Total keystrokes: {total_keystrokes:,}")
    if total_keystroke_duration_seconds > 0:
        keystroke_duration_minutes = total_keystroke_duration_seconds / 60
        keystroke_duration_hours = keystroke_duration_minutes / 60
        print(f"⏱️  Total keystroke data duration: {total_keystroke_duration_seconds:.1f} seconds ({keystroke_duration_minutes:.1f} minutes, {keystroke_duration_hours:.2f} hours)")
    
    print(f"\n📈 Per-session averages:")
    num_sessions = len(file_groups)
    if num_sessions > 0:
        avg_imu_samples = total_imu_rows / num_sessions
        avg_keystrokes = total_keystrokes / num_sessions
        avg_duration_minutes = total_duration_minutes / num_sessions
        print(f"  Average IMU samples per session: {avg_imu_samples:,.0f}")
        print(f"  Average keystrokes per session: {avg_keystrokes:,.0f}")
        print(f"  Average duration per session: {avg_duration_minutes:.1f} minutes")
    
    # Save or show plots
    if args.visualize:
        if args.save_plots:
            os.makedirs('plots', exist_ok=True)
            for fig, name in figures:
                fig.savefig(f'plots/{name}.png', dpi=150, bbox_inches='tight')
                plt.close(fig)
            print(f"\n💾 Saved {len(figures)} plots to 'plots/' directory")
        else:
            print(f"\n📊 Showing {len(figures)} plots...")
            plt.show()
    
    print(f"\n✅ Exploration complete!")


if __name__ == '__main__':
    main()
