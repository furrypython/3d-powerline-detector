import argparse
import laspy
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

def detect_powerlines(input_path: str, output_path: str, search_radius: float = 0.05, linearity_threshold: float = 0.85, planarity_threshold: float = 0.2, max_thickness: float = 0.04):
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
    
    print("Building KDTree for fast neighborhood search...")
    tree = KDTree(points)
    
    print(f"Computing PCA for each point (search radius: {search_radius}m)...")
    is_line = np.zeros(num_points, dtype=bool)
    
    # Process in chunks to avoid massive memory usage if we queried all at once
    chunk_size = 100000
    
    for start_idx in tqdm(range(0, num_points, chunk_size), desc="Processing points"):
        end_idx = min(start_idx + chunk_size, num_points)
        chunk_points = points[start_idx:end_idx]
        
        # Find neighbors within the search radius
        # query_ball_point returns a list of lists containing neighbor indices
        neighbors_list = tree.query_ball_point(chunk_points, r=search_radius)
        
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
                    is_line[start_idx + i] = True

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
    
    args = parser.parse_args()
    detect_powerlines(args.input, args.output, args.radius, args.threshold, args.planarity_threshold, args.max_thickness)
