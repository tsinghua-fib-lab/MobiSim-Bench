import pandas as pd
import os
import glob
import json
import numpy as np
from pathlib import Path

def parse_hourly_visits(hourly_str):
    """
    Parses the hourly visits string into a list of integers.
    
    - **Description**:
        - Converts the string representation of hourly visits to a list of integers
        - Handles the format like "[0,0,0,1,2,...]"
    
    - **Args**:
        - `hourly_str` (str): String representation of hourly visits
    
    - **Returns**:
        - `list`: List of hourly visit counts
    """
    try:
        # Remove brackets and split by comma
        hourly_str = hourly_str.strip('[]')
        return [int(x.strip()) for x in hourly_str.split(',')]
    except:
        return []

def calculate_hurricane_mobility_stats():
    """
    Calculates hurricane mobility statistics including total trips and hourly patterns.
    
    - **Description**:
        - Analyzes mobility data before, during, and after hurricane
        - Calculates relative change rates and hourly trip patterns
        - Returns comprehensive statistics for hurricane impact analysis
    
    - **Returns**:
        - `dict`: Dictionary containing all hurricane mobility statistics
    """
    
    # Define time periods with correct day indices
    hurricane_periods = {
        'before': [
            ('2019-08-28.csv', 3),  # day 3 (index 2)
            ('2019-08-29.csv', 4),  # day 4 (index 3) 
            ('2019-08-30.csv', 5)   # day 5 (index 4)
        ],
        'during': [
            ('2019-08-31.csv', 6),  # day 6 (index 5)
            ('2019-09-01.csv', 7)   # day 7 (index 6)
        ],
        'after': [
            ('2019-09-02.csv', 1),  # day 1 (index 0)
            ('2019-09-03.csv', 2),  # day 2 (index 1)
            ('2019-09-04.csv', 3),  # day 3 (index 2)
            ('2019-09-05.csv', 4)   # day 4 (index 3)
        ]
    }
    
    # Dictionary to store results
    results = {
        'total_trips': {},
        'hourly_trips': {},
        'relative_changes': {}
    }
    
    # Process each period
    for period, file_day_pairs in hurricane_periods.items():
        print(f"\n{'='*60}")
        print(f"Processing {period.upper()} hurricane period")
        print(f"{'='*60}")
        
        period_total_trips = []
        period_hourly_trips = []
        
        for file_name, day_index in file_day_pairs:
            try:
                print(f"Reading {file_name} for day {day_index}...")
                df = pd.read_csv(file_name)
                
                # Parse visits_by_day to get the correct day's total trips
                daily_totals = []
                for _, row in df.iterrows():
                    try:
                        visits_by_day_str = str(row['visits_by_day'])
                        visits_by_day = parse_hourly_visits(visits_by_day_str)
                        if len(visits_by_day) >= day_index:
                            daily_totals.append(visits_by_day[day_index - 1])  # Convert to 0-based index
                    except:
                        continue
                
                # Calculate total trips for this specific day
                daily_total = sum(daily_totals)
                period_total_trips.append(daily_total)
                print(f"  Day {day_index} total trips: {daily_total:,}")
                
                # Process hourly data - extract the correct 24-hour segment
                hourly_data = []
                for _, row in df.iterrows():
                    hourly_visits = parse_hourly_visits(str(row['visits_by_each_hour']))
                    if len(hourly_visits) == 168:  # 7 days * 24 hours
                        # Extract the correct 24-hour segment based on day_index
                        start_hour = (day_index - 1) * 24  # Convert to 0-based index
                        end_hour = start_hour + 24
                        day_24_hours = hourly_visits[start_hour:end_hour]
                        hourly_data.append(day_24_hours)
                
                if hourly_data:
                    # Sum across all POIs for each hour (24 hours)
                    hourly_sums = np.sum(hourly_data, axis=0)
                    period_hourly_trips.append(hourly_sums)
                    print(f"  Processed {len(hourly_data)} POIs with 24-hour data for day {day_index}")
                
            except Exception as e:
                print(f"Error processing {file_name} day {day_index}: {str(e)}")
        
        # Calculate averages for the period
        if period_total_trips:
            results['total_trips'][period] = np.mean(period_total_trips)
            print(f"Average total trips for {period}: {results['total_trips'][period]:,.0f}")
        
        if period_hourly_trips:
            # Average across days for each hour (24 hours)
            avg_hourly = np.mean(period_hourly_trips, axis=0)
            results['hourly_trips'][period] = avg_hourly.tolist()
            print(f"Average hourly trips for {period}: {len(avg_hourly)} hours (24-hour pattern)")
    
    # Calculate relative changes
    if 'before' in results['total_trips']:
        before_total = results['total_trips']['before']
        
        if 'during' in results['total_trips']:
            during_total = results['total_trips']['during']
            results['relative_changes']['during_vs_before'] = (during_total / before_total - 1) * 100
        
        if 'after' in results['total_trips']:
            after_total = results['total_trips']['after']
            results['relative_changes']['after_vs_before'] = (after_total / before_total - 1) * 100
    
    return results

def calculate_proportional_stats(stats, target_agents=100):
    """
    Converts absolute statistics to proportional values suitable for agent simulation comparison.
    
    - **Description**:
        - Converts total trips and hourly patterns to proportions
        - Scales data to be comparable with small agent populations
        - Maintains relative patterns while making absolute values suitable for simulation
    
    - **Args**:
        - `stats` (dict): Original hurricane mobility statistics
        - `target_agents` (int): Target number of agents for scaling (default: 100)
    
    - **Returns**:
        - `dict`: Dictionary containing proportional statistics
    """
    
    proportional_stats = {
        'total_trips': {},
        'hourly_trips': {},
        'relative_changes': {},
        'scaling_factor': {},
        'target_agents': target_agents
    }
    
    # Calculate scaling factors based on before period
    if 'before' in stats['total_trips']:
        before_total = stats['total_trips']['before']
        # Scale to target_agents total trips per day
        scaling_factor = target_agents / before_total
        proportional_stats['scaling_factor'] = scaling_factor
        
        print(f"\nScaling factor: {scaling_factor:.6f} (target: {target_agents} agents)")
        
        # Scale total trips
        for period, total in stats['total_trips'].items():
            proportional_stats['total_trips'][period] = total * scaling_factor
        
        # Scale hourly trips
        for period, hourly_data in stats['hourly_trips'].items():
            proportional_stats['hourly_trips'][period] = [h * scaling_factor for h in hourly_data]
        
        # Relative changes remain the same (they're already ratios)
        proportional_stats['relative_changes'] = stats['relative_changes']
    
    return proportional_stats

def print_hurricane_stats(stats):
    """
    Prints formatted hurricane mobility statistics.
    
    - **Description**:
        - Displays comprehensive hurricane impact statistics in a readable format
        - Shows total trips, relative changes, and hourly patterns
    
    - **Args**:
        - `stats` (dict): Hurricane mobility statistics dictionary
    """
    
    print(f"\n{'='*80}")
    print("HURRICANE MOBILITY STATISTICS")
    print(f"{'='*80}")
    
    # Total trips by period
    print("\n1. TOTAL TRIPS BY PERIOD:")
    for period, total in stats['total_trips'].items():
        print(f"   {period.upper()}: {total:,.0f} trips")
    
    # Relative changes
    print("\n2. RELATIVE CHANGES:")
    for change_type, change_rate in stats['relative_changes'].items():
        direction = "increase" if change_rate > 0 else "decrease"
        print(f"   {change_type.replace('_', ' ').title()}: {change_rate:+.2f}% ({direction})")
    
    # Hourly patterns summary
    print("\n3. HOURLY PATTERNS SUMMARY:")
    for period, hourly_data in stats['hourly_trips'].items():
        if hourly_data:
            avg_hourly = np.mean(hourly_data)
            max_hourly = np.max(hourly_data)
            min_hourly = np.min(hourly_data)
            print(f"   {period.upper()}:")
            print(f"     Average hourly trips: {avg_hourly:.1f}")
            print(f"     Max hourly trips: {max_hourly:.0f}")
            print(f"     Min hourly trips: {min_hourly:.0f}")
    
    # Key statistics for the requested parameters
    print(f"\n{'='*80}")
    print("KEY STATISTICS (as requested)")
    print(f"{'='*80}")
    
    if 'during_vs_before' in stats['relative_changes']:
        print(f"1. 飓风时总出行量/飓风前总出行量 相对变化率: {stats['relative_changes']['during_vs_before']:+.2f}%")
    
    if 'after_vs_before' in stats['relative_changes']:
        print(f"2. 飓风后总出行量/飓风前总出行量 相对变化率: {stats['relative_changes']['after_vs_before']:+.2f}%")
    
    if 'before' in stats['hourly_trips']:
        print(f"3. 飓风前小时级出行量 (24小时): {stats['hourly_trips']['before']}")
    
    if 'during' in stats['hourly_trips']:
        print(f"4. 飓风时小时级出行量 (24小时): {stats['hourly_trips']['during']}")
    
    if 'after' in stats['hourly_trips']:
        print(f"5. 飓风后小时级出行量 (24小时): {stats['hourly_trips']['after']}")

def print_proportional_stats(proportional_stats):
    """
    Prints formatted proportional hurricane mobility statistics for agent simulation.
    
    - **Description**:
        - Displays scaled statistics suitable for comparison with agent simulations
        - Shows proportional values that maintain relative patterns
    
    - **Args**:
        - `proportional_stats` (dict): Proportional hurricane mobility statistics
    """
    
    target_agents = proportional_stats.get('target_agents', 100)
    scaling_factor = proportional_stats.get('scaling_factor', 1)
    
    print(f"\n{'='*80}")
    print(f"PROPORTIONAL STATISTICS (Scaled for {target_agents} Agents)")
    print(f"{'='*80}")
    
    # Total trips by period (scaled)
    print("\n1. SCALED TOTAL TRIPS BY PERIOD:")
    for period, total in proportional_stats['total_trips'].items():
        print(f"   {period.upper()}: {total:.2f} trips (scaled)")
    
    # Relative changes (unchanged)
    print("\n2. RELATIVE CHANGES (unchanged):")
    for change_type, change_rate in proportional_stats['relative_changes'].items():
        direction = "increase" if change_rate > 0 else "decrease"
        print(f"   {change_type.replace('_', ' ').title()}: {change_rate:+.2f}% ({direction})")
    
    # Hourly patterns summary (scaled)
    print("\n3. SCALED HOURLY PATTERNS SUMMARY:")
    for period, hourly_data in proportional_stats['hourly_trips'].items():
        if hourly_data:
            avg_hourly = np.mean(hourly_data)
            max_hourly = np.max(hourly_data)
            min_hourly = np.min(hourly_data)
            print(f"   {period.upper()}:")
            print(f"     Average hourly trips: {avg_hourly:.3f}")
            print(f"     Max hourly trips: {max_hourly:.3f}")
            print(f"     Min hourly trips: {min_hourly:.3f}")
    
    # Key statistics for agent simulation
    print(f"\n{'='*80}")
    print("AGENT SIMULATION COMPARISON STATISTICS")
    print(f"{'='*80}")
    
    if 'during_vs_before' in proportional_stats['relative_changes']:
        print(f"1. 飓风时总出行量/飓风前总出行量 相对变化率: {proportional_stats['relative_changes']['during_vs_before']:+.2f}%")
    
    if 'after_vs_before' in proportional_stats['relative_changes']:
        print(f"2. 飓风后总出行量/飓风前总出行量 相对变化率: {proportional_stats['relative_changes']['after_vs_before']:+.2f}%")
    
    if 'before' in proportional_stats['hourly_trips']:
        print(f"3. 飓风前小时级出行量比例 (24小时): {[round(x, 3) for x in proportional_stats['hourly_trips']['before']]}")
    
    if 'during' in proportional_stats['hourly_trips']:
        print(f"4. 飓风时小时级出行量比例 (24小时): {[round(x, 3) for x in proportional_stats['hourly_trips']['during']]}")
    
    if 'after' in proportional_stats['hourly_trips']:
        print(f"5. 飓风后小时级出行量比例 (24小时): {[round(x, 3) for x in proportional_stats['hourly_trips']['after']]}")
    
    print(f"\nScaling factor used: {scaling_factor:.6f}")
    print(f"Target agent population: {target_agents}")

def read_all_csv_files():
    """
    Reads all CSV files in the current directory and provides basic information about each file.
    
    - **Description**:
        - Scans the current directory for all CSV files
        - Reads each file and provides basic statistics and preview
        - Returns a dictionary containing file information and dataframes
    
    - **Returns**:
        - dict: Dictionary containing file information and dataframes for each CSV file
    """
    
    # Get current directory
    current_dir = Path('.')
    
    # Find all CSV files in current directory
    csv_files = list(current_dir.glob('*.csv'))
    
    print(f"Found {len(csv_files)} CSV files in current directory:")
    for file in csv_files:
        print(f"  - {file.name}")
    
    # Dictionary to store file information and dataframes
    file_data = {}
    
    # Read each CSV file
    for csv_file in csv_files:
        print(f"\n{'='*50}")
        print(f"Processing: {csv_file.name}")
        print(f"{'='*50}")
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Store file information
            file_info = {
                'file_path': str(csv_file),
                'file_size_mb': csv_file.stat().st_size / (1024 * 1024),
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'dataframe': df
            }
            
            file_data[csv_file.name] = file_info
            
            # Print basic information
            print(f"File size: {file_info['file_size_mb']:.2f} MB")
            print(f"Rows: {file_info['rows']:,}")
            print(f"Columns: {file_info['columns']}")
            print(f"Column names: {file_info['column_names']}")
            
            # Show data preview
            print(f"\nFirst 5 rows:")
            print(df.head())
            
            # Show data types
            print(f"\nData types:")
            print(df.dtypes)
            
            # Show basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                print(f"\nBasic statistics for numeric columns:")
                print(df[numeric_cols].describe())
            
            # Show missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                print(f"\nMissing values:")
                print(missing_values[missing_values > 0])
            else:
                print(f"\nNo missing values found.")
                
        except Exception as e:
            print(f"Error reading {csv_file.name}: {str(e)}")
            file_data[csv_file.name] = {'error': str(e)}
    
    return file_data

if __name__ == "__main__":
    # Calculate hurricane mobility statistics
    print("Calculating hurricane mobility statistics...")
    hurricane_stats = calculate_hurricane_mobility_stats()
    
    # Print formatted results
    print_hurricane_stats(hurricane_stats)
    
    # Calculate proportional statistics for agent simulation
    print("\n" + "="*80)
    print("CALCULATING PROPORTIONAL STATISTICS FOR AGENT SIMULATION")
    print("="*80)
    
    proportional_stats = calculate_proportional_stats(hurricane_stats, target_agents=100)
    print_proportional_stats(proportional_stats)
    
    # Save both original and proportional results to JSON files
    output_file = 'hurricane_mobility_stats.json'
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_stats = {}
        for key, value in hurricane_stats.items():
            if isinstance(value, dict):
                json_stats[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        json_stats[key][sub_key] = sub_value.tolist()
                    else:
                        json_stats[key][sub_key] = sub_value
            else:
                json_stats[key] = value
        
        json.dump(json_stats, f, indent=2)
    
    # Save proportional statistics
    proportional_output_file = 'hurricane_mobility_proportional_stats.json'
    with open(proportional_output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_proportional_stats = {}
        for key, value in proportional_stats.items():
            if isinstance(value, dict):
                json_proportional_stats[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        json_proportional_stats[key][sub_key] = sub_value.tolist()
                    else:
                        json_proportional_stats[key][sub_key] = sub_value
            else:
                json_proportional_stats[key] = value
        
        json.dump(json_proportional_stats, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Proportional results saved to {proportional_output_file}")
    
    # Optional: Also run the original file reading function
    print(f"\n{'='*80}")
    print("ORIGINAL FILE ANALYSIS")
    print(f"{'='*80}")
    
    # Read all CSV files
    all_data = read_all_csv_files()
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Total files processed: {len(all_data)}")
    
    # Summary of all files
    for filename, info in all_data.items():
        if 'error' not in info:
            print(f"{filename}: {info['rows']:,} rows, {info['columns']} columns, {info['file_size_mb']:.2f} MB")
        else:
            print(f"{filename}: ERROR - {info['error']}")
