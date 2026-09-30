import argparse
import laspy
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

def estimate_ground_elevation(points: np.ndarray, block_size: float = 10.0) -> np.ndarray:
    """
    Estimates the ground elevation for each point by finding the lowest point 
    in each 2D block and interpolating.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Discretize into blocks
    min_x, min_y = np.min(x), np.min(y)
    x_blocks = np.floor((x - min_x) / block_size).astype(int)
    y_blocks = np.floor((y - min_y) / block_size).astype(int)
    
    max_y_blocks = np.max(y_blocks) + 1
    block_indices = x_blocks * max_y_blocks + y_blocks
    
    unique_blocks, inverse_indices = np.unique(block_indices, return_inverse=True)
    
    # Find min Z for each block
    min_z_per_block = np.full(len(unique_blocks), np.inf)
    np.minimum.at(min_z_per_block, inverse_indices, z)
    
    # Get the x, y coordinates of the center of each block
    unique_x_blocks = unique_blocks // max_y_blocks
    unique_y_blocks = unique_blocks % max_y_blocks
    
    block_centers_x = min_x + (unique_x_blocks + 0.5) * block_size
    block_centers_y = min_y + (unique_y_blocks + 0.5) * block_size
    block_centers = np.vstack((block_centers_x, block_centers_y)).T
    
    # Build a 2D KDTree of the block centers
    ground_tree = KDTree(block_centers)
    
    # For each point, find the nearest block centers and interpolate
    points_2d = points[:, :2]
    k = min(3, len(unique_blocks))
    
    if k == 0:
        return np.zeros(len(points))
        
    distances, indices = ground_tree.query(points_2d, k=k)
    
    if k == 1:
        if len(unique_blocks) == 1:
            ground_z = np.full(len(points), min_z_per_block[0])
        else:
            ground_z = min_z_per_block[indices]
    else:
        distances = np.maximum(distances, 1e-6)
        weights = 1.0 / (distances ** 2)
        weights /= np.sum(weights, axis=1, keepdims=True)
        ground_z = np.sum(weights * min_z_per_block[indices], axis=1)
        
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
