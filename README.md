# 3D Powerline Detector

A simple, PCA-based LiDAR point cloud processor to extract thin lines (like powerlines).

## Recommendations for your "First Try"

1. **Language: Python**
   - **Why?** Python is the absolute best language for *prototyping* spatial and data science algorithms. The ecosystem (`numpy`, `scipy`, `laspy`, `open3d`) is incredibly rich. 
   - While Rust is faster for production, iterating on 3D math and algorithms is much slower in Rust due to compile times and fewer out-of-the-box point cloud libraries. We can always optimize the Python code (using `numba` or C++ extensions) or rewrite it in Rust once the logic is perfected.

2. **Package Manager: `uv`**
   - **Why?** `uv` is blazing fast and seamlessly handles modern Python workflows. 
   - Historically, `conda` was required for complex spatial libraries (like PDAL or GDAL), but modern Python wheels (which `uv` downloads) now cover 99% of our needs out-of-the-box. `uv` keeps your workspace clean and installs dependencies in milliseconds.

## How it works (The Logic)

We use **Principal Component Analysis (PCA)** on the local neighborhood of every point to determine its shape.

1. **Neighborhood Search:** For every point, we find all neighboring points within a specific `radius` (e.g., 50mm / 0.05m) using a fast KD-Tree.
2. **Covariance & PCA:** We compute the covariance matrix of these local points and extract its eigenvalues ($\lambda_1, \lambda_2, \lambda_3$, sorted from largest to smallest).
3. **Linearity Metric:** 
   - $\lambda_1$ represents the spread of points along the main axis.
   - $\lambda_2$ and $\lambda_3$ represent the spread in the orthogonal directions (the thickness).
   - If the points form a thin line (like a powerline), $\lambda_1$ will be much larger than $\lambda_2$.
   - We calculate Linearity: $L = (\lambda_1 - \lambda_2) / \lambda_1$.
   - If $L$ is close to `1.0` (e.g., $> 0.85$), we classify the point as part of a line.
4. **Thickness Filtering:**
   - Because $\lambda_2$ represents the variance across the width of the line, we can estimate the physical radius of the tube as $2 \times \sqrt{\lambda_2}$.
   - We reject any lines where this estimated radius exceeds our `--max-thickness` (e.g., rejecting thick tree trunks while keeping thin wires).

## Usage

1. **Activate the environment** (if not already handled by your IDE):
   ```bash
   source .venv/bin/activate
   ```

2. **Run the detector:**
   ```bash
   python detect_powerlines.py input.las output.las --radius 0.1 --threshold 0.85 --max-thickness 0.04
   ```

### Parameters

- `--radius`: The search radius in meters. This defines the "neighborhood" size to look at when determining shape. It should be comfortably *larger* than your target object. For a 30mm radius powerline, `0.1` (100mm) is a good starting point.
- `--threshold`: The strictness of the line detection (0.0 to 1.0). `0.85` is a good default. Increase it to `0.9` or `0.95` if you are getting too much noise (like tree branches), or decrease it to `0.7` if parts of the powerline are missing.
- `--max-thickness`: The maximum allowed physical thickness (radius) of the detected line in meters. Since powerlines are 5-30mm, `0.04` (40mm) is a good default to filter out thicker cylindrical objects (like tree trunks or thick pipes) while allowing for slight LiDAR noise.

## Next Steps for Improvement

Once you test this simple logic, here are ways we can improve it:
1. **Speed:** Vectorize the PCA computation or use `numba` to parallelize the loop across all CPU cores.
2. **Height Filtering:** Powerlines are usually high off the ground. We can filter out ground points (e.g., lowest 10% of Z coordinates) before running PCA to save massive amounts of compute time.
3. **Region Growing:** Instead of checking every single point, we can find a few "seed" points that are definitely lines, and trace the line outwards.