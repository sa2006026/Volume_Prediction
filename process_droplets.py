import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

def find_unique_droplets(df, error_margin=10):
    """
    Find unique droplets across all slides by clustering droplets within error_margin pixels
    and finding the maximum diameter for each cluster.
    
    Args:
        df: DataFrame with columns Center_X_px, Center_Y_px, Diameter_μm, slide
        error_margin: Maximum distance in pixels between droplets in the same cluster
        
    Returns:
        DataFrame with unique droplets and their maximum diameters
    """
    if df.empty:
        return pd.DataFrame()

    # Add a unique original index to keep track of droplets
    df = df.reset_index(drop=True)
    df['original_index'] = df.index

    # Sort by diameter in descending order
    df_sorted = df.sort_values(by='Diameter_μm', ascending=False).reset_index(drop=True)

    assigned = np.zeros(len(df_sorted), dtype=bool)
    unique_droplet_results = []

    for i in range(len(df_sorted)):
        if not assigned[i]:
            # Start a new group with the current unassigned droplet
            current_group_indices = [df_sorted.loc[i, 'original_index']]
            assigned[i] = True

            # Iteratively expand the group by finding nearby droplets
            group_expanded = True
            while group_expanded:
                group_expanded = False
                current_group_df = df.loc[current_group_indices]
                
                # Create mask for unassigned droplets in df_sorted
                assigned_orig_indices = set(current_group_df['original_index'].values)
                unassigned_orig_indices = []
                for idx in range(len(df_sorted)):
                    if not assigned[idx]:
                        unassigned_orig_indices.append(df_sorted.loc[idx, 'original_index'])
                
                if not unassigned_orig_indices:
                    break
                    
                unassigned_df = df.loc[unassigned_orig_indices]

                if unassigned_df.empty:
                    break

                # Calculate distances from all droplets in the current group to all unassigned droplets
                distances = cdist(
                    current_group_df[['Center_X_px', 'Center_Y_px']],
                    unassigned_df[['Center_X_px', 'Center_Y_px']],
                    'euclidean'
                )

                # Find unassigned droplets that are close to any droplet in the current group
                min_distances = distances.min(axis=0)
                nearby_mask = min_distances <= error_margin

                if nearby_mask.any():
                    # Get the original_index values for these nearby droplets
                    newly_assigned_original_indices = unassigned_df.iloc[nearby_mask]['original_index'].tolist()

                    # Add them to the current group
                    current_group_indices.extend(newly_assigned_original_indices)

                    # Mark them as assigned in df_sorted
                    for orig_idx in newly_assigned_original_indices:
                        df_sorted_idx = df_sorted[df_sorted['original_index'] == orig_idx].index[0]
                        assigned[df_sorted_idx] = True
                    
                    group_expanded = True

            # Find the max diameter within this group
            final_group_df = df.loc[current_group_indices]
            
            # Drop NaN values before finding max
            final_group_df_clean = final_group_df.dropna(subset=['Diameter_μm'])
            
            if not final_group_df_clean.empty:
                max_diameter_row = final_group_df_clean.loc[final_group_df_clean['Diameter_μm'].idxmax()]

                unique_droplet_results.append({
                    'Center_X_px': max_diameter_row['Center_X_px'],
                    'Center_Y_px': max_diameter_row['Center_Y_px'],
                    'Diameter_μm': max_diameter_row['Diameter_μm'],
                    'slide': max_diameter_row['slide']
                })

    return pd.DataFrame(unique_droplet_results)


def process_all_slides(file_paths, error_margin=10):
    """
    Process all CSV files to find unique droplets with maximum diameters.
    
    Args:
        file_paths: List of paths to CSV files
        error_margin: Maximum distance in pixels between droplets in the same cluster
        
    Returns:
        DataFrame with unique droplets and their maximum diameters
    """
    all_droplets_data = []
    for file_path in file_paths:
        slide_name = file_path.split('/')[-1].replace('.csv', '')
        df = pd.read_csv(file_path)
        df['slide'] = slide_name
        all_droplets_data.append(df)

    combined_df = pd.concat(all_droplets_data, ignore_index=True)
    unique_max_diameters_df = find_unique_droplets(combined_df, error_margin)
    return unique_max_diameters_df

