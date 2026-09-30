import argparse
import laspy
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

from scipy.ndimage import grey_opening, distance_transform_edt, median_filter
from scipy.interpolate import RegularGridInterpolator

def estimate_ground_elevation(points: np.ndarray, cell_size: float = 1.0, window_size: float = 15.0, percentile: float = 5.0) -> np.ndarray:
    """
    Estimates the ground elevation using a Simple Morphological Filter (SMRF) approach.
    1. Creates a minimum elevation grid (using a percentile to ignore negative noise).
    2. Inpaints empty cells using nearest neighbor (distance transform).
    3. Applies morphological opening to remove non-ground objects (buildings, trees).
    4. Interpolates the bare-earth surface back to the original points.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # 1. Discretize into a grid
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    
    cols = int(np.ceil((max_x - min_x) / cell_size)) + 1
    rows = int(np.ceil((max_y - min_y) / cell_size)) + 1
    
    c = np.floor((x - min_x) / cell_size).astype(int)
    r = np.floor((y - min_y) / cell_size).astype(int)
    
    # Combine r and c into a single flat index for grouping
    flat_indices = r * cols + c
    
    # Sort points by Z to easily find percentiles
    sort_idx = np.argsort(z)
    sorted_z = z[sort_idx]
    sorted_flat = flat_indices[sort_idx]
    
    unique_flat, first_indices = np.unique(sorted_flat, return_index=True)
    _, block_counts = np.unique(sorted_flat, return_counts=True)
    
    # Use the Nth percentile for each cell to avoid negative noise points
    percentile_indices = first_indices + (block_counts * (percentile / 100.0)).astype(int)
    cell_z = sorted_z[percentile_indices]
    
    # Map back to 2D grid
    unique_r = unique_flat // cols
    unique_c = unique_flat % cols
    
    max_z = np.max(z)
    min_grid = np.full((rows, cols), max_z + 10.0)
    min_grid[unique_r, unique_c] = cell_z
    
    # 2. Inpaint empty cells
    empty_mask = min_grid > max_z
    if np.all(empty_mask):
        return np.zeros(len(points))
        
    if np.any(empty_mask):
        indices = distance_transform_edt(empty_mask, return_distances=False, return_indices=True)
        min_grid = min_grid[tuple(indices)]
        
    # Apply a small median filter to remove isolated negative noise spikes
    min_grid = median_filter(min_grid, size=3)
        
    # 3. Morphological Opening
    # Convert window size from meters to pixels/cells
    size = int(np.ceil(window_size / cell_size))
    if size > 1:
        bare_earth_grid = grey_opening(min_grid, size=(size, size))
    else:
        bare_earth_grid = min_grid
        
    # 4. Interpolate back to points
    grid_y = min_y + np.arange(rows) * cell_size + cell_size / 2
    grid_x = min_x + np.arange(cols) * cell_size + cell_size / 2
    
    interpolator = RegularGridInterpolator((grid_y, grid_x), bare_earth_grid, bounds_error=False, fill_value=None)
    ground_z = interpolator(np.vstack((y, x)).T)
    
    return ground_z

def detect_powerlines(input_path: str, output_path: str, search_radius: float = 0.05, linearity_threshold: float = 0.85, planarity_threshold: float = 0.2, max_thickness: float = 0.04, min_height: float = 4.5):
    """
    Detects powerlines in a LAS file using PCA (Principal Component Analysis).
    
    Args:
        input_path: Path to the input .las file.
        output_path: Path to save the output .las file containing only the detected lines.
        search_radius: Radius in meters to search for neighbors. 
                       For a 5-30mm radius powerline, 0.05m (50mm) is a good starting point.
        linearity_threshold: Threshold for the linearity metric (0.0 to 1.0).
                             Higher means stricter line detection.
        planarity_threshold: Maximum allowed planarity metric (0.0 to 1.0).
                             Lower means stricter rejection of flat surfaces (like building edges).
        max_thickness: Maximum allowed thickness (radius) of the line in meters.
                       Points belonging to structures thicker than this will be rejected.
    """
    print(f"Loading {input_path}...")
    las = laspy.read(input_path)
    
    # Extract coordinates
    points = np.vstack((las.x, las.y, las.z)).transpose()
    num_points = len(points)
    print(f"Loaded {num_points} points.")
    
    print("Estimating ground elevation...")
    ground_z = estimate_ground_elevation(points)
    height_above_ground = points[:, 2] - ground_z
    
    print("Building KDTree for fast neighborhood search...")
    tree = KDTree(points)
    
    print(f"Computing PCA for each point (search radius: {search_radius}m, min height: {min_height}m)...")
    is_line = np.zeros(num_points, dtype=bool)
    
    # Process in chunks to avoid massive memory usage if we queried all at once
    chunk_size = 100000
    
    for start_idx in tqdm(range(0, num_points, chunk_size), desc="Processing points"):
        end_idx = min(start_idx + chunk_size, num_points)
        chunk_points = points[start_idx:end_idx]
        chunk_heights = height_above_ground[start_idx:end_idx]
        
        # Only query neighbors for points that are high enough
        valid_mask = chunk_heights >= min_height
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            continue
            
        valid_chunk_points = chunk_points[valid_indices]
        
        # Find neighbors within the search radius
        # query_ball_point returns a list of lists containing neighbor indices
        neighbors_list = tree.query_ball_point(valid_chunk_points, r=search_radius)
        
        for i, neighbors in enumerate(neighbors_list):
            if len(neighbors) < 5:
                # Not enough points to form a meaningful line
                continue
                
            # Get the coordinates of the neighbors
            neighbor_coords = points[neighbors]
            
            # Center the points
            centroid = np.mean(neighbor_coords, axis=0)
            centered = neighbor_coords - centroid
            
            # Compute covariance matrix
            cov_matrix = np.dot(centered.T, centered) / (len(neighbors) - 1)
            
            # Compute eigenvalues
            # eigh is optimized for symmetric matrices like covariance matrices
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            
            # Sort eigenvalues in descending order
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            # Avoid division by zero
            if eigenvalues[0] > 1e-6:
                # Linearity metric: (lambda_1 - lambda_2) / lambda_1
                # If lambda_1 is much larger than lambda_2, it's a line.
                linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
                
                # Planarity metric: (lambda_2 - lambda_3) / lambda_1
                # If lambda_2 is much larger than lambda_3, it's a flat surface (like a building edge).
                # For a cylindrical wire, lambda_2 and lambda_3 should be roughly equal, making planarity close to 0.
                planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
                
                # Estimate the thickness (radius) of the structure
                # eigenvalues[1] is the variance along the secondary axis.
                # Standard deviation (approximate radius) is the square root of the variance.
                # We multiply by 2 to get a rough estimate of the full radius capturing most points (2 sigma).
                estimated_radius = 2 * np.sqrt(eigenvalues[1])
                
                if linearity > linearity_threshold and planarity < planarity_threshold and estimated_radius <= max_thickness:
                    original_idx = start_idx + valid_indices[i]
                    is_line[original_idx] = True

    # Filter the original LAS data
    detected_count = np.sum(is_line)
    print(f"\nDetection complete. Found {detected_count} points belonging to lines.")
    
    if detected_count > 0:
        print(f"Saving results to {output_path}...")
        # Create a new LAS file with the same header/format as the input
        out_las = laspy.LasData(las.header)
        out_las.points = las.points[is_line]
        out_las.write(output_path)
        print("Done!")
    else:
        print("No lines detected. Try adjusting the search_radius or lowering the linearity_threshold.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect thin lines (powerlines) in LiDAR data.")
    parser.add_argument("input", help="Input .las file path")
    parser.add_argument("output", help="Output .las file path")
    parser.add_argument("--radius", type=float, default=0.05, 
                        help="Search radius in meters (default: 0.05). Should be slightly larger than the powerline radius.")
    parser.add_argument("--threshold", type=float, default=0.85, 
                        help="Linearity threshold 0.0-1.0 (default: 0.85). Higher is stricter.")
    parser.add_argument("--planarity-threshold", type=float, default=0.2, 
                        help="Maximum allowed planarity 0.0-1.0 (default: 0.2). Filters out flat building edges.")
    parser.add_argument("--max-thickness", type=float, default=0.04, 
                        help="Maximum allowed thickness (radius) of the line in meters (default: 0.04m).")
    parser.add_argument("--min-height", type=float, default=4.5, 
                        help="Minimum height above ground in meters (default: 4.5m).")
    
    args = parser.parse_args()
    detect_powerlines(args.input, args.output, args.radius, args.threshold, args.planarity_threshold, args.max_thickness, args.min_height)
