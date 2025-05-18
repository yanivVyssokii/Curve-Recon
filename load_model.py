import trimesh
import numpy as np
import matplotlib.pyplot as plt

# Load the OFF mesh
mesh = trimesh.load_mesh("dog1.off")

# Load sampled vertex indices from the .txt file
sample_indices = np.loadtxt("dog1-samples.txt", dtype=int)

# Extract corresponding 3D points from the mesh
sample_points = mesh.vertices[sample_indices]

# Print to verify
print(f"Loaded mesh with {len(mesh.vertices)} vertices")
print(f"Loaded {len(sample_indices)} sampled vertices")

def visualize_samples(mesh, points):
    # Visualize the mesh using trimesh's show method
    scene = mesh.scene()
    
    # Create a PointCloud object for the points
    points_mesh = trimesh.points.PointCloud(points)
    
    # Add the points to the scene
    scene.add_geometry(points_mesh)
    
    # Show the scene with the mesh and the points
    scene.show()

# Visualize the mesh and sampled points
visualize_samples(mesh, sample_points)