import trimesh
import numpy as np
import random
import networkx as nx
from trimesh.graph import vertex_adjacency_graph

def dijkstra_geodesic_distance(graph, source_index):
    """
    Compute shortest paths from a source vertex using Dijkstra's algorithm.
    Returns a dict of distances keyed by vertex index.
    """
    return nx.single_source_dijkstra_path_length(graph, source_index)

def non_uniform_sparse_sampling_dijkstra(mesh, num_samples):
    """
    Perform non-uniform sparse sampling using approximate geodesic distances.
    """
    graph = vertex_adjacency_graph(mesh)
    vertices = mesh.vertices
    sample_points = [random.choice(range(len(vertices)))]  # Start with a random vertex

    for _ in range(num_samples - 1):
        all_distances = {}
        for s in sample_points:
            dist = dijkstra_geodesic_distance(graph, s)
            for k, v in dist.items():
                if k not in all_distances or v < all_distances[k]:
                    all_distances[k] = v

        # Choose the point with the maximum distance to the current set
        candidates = [i for i in range(len(vertices)) if i not in sample_points]
        farthest_point = max(candidates, key=lambda i: all_distances.get(i, 0))
        sample_points.append(farthest_point)

    sampled_coords = vertices[sample_points]
    return sample_points, sampled_coords

def visualize_samples(mesh, sample_coords):
    """
    Visualize the mesh and sampled points.
    """
    scene = mesh.scene()
    point_cloud = trimesh.points.PointCloud(sample_coords, colors=[255, 0, 0])  # Red points
    scene.add_geometry(point_cloud)
    scene.show()

if __name__ == "__main__":
    # Load mesh (replace with your own path)
    mesh = trimesh.load_mesh('dog1.off')  # e.g., 'bunny.off'

    # Number of samples
    num_samples = 23

    # Perform non-uniform sparse sampling
    sampled_indices, sampled_coords = non_uniform_sparse_sampling_dijkstra(mesh, num_samples)

    # Optionally save sampled indices to file
    np.savetxt('sampled_indices.txt', sampled_indices, fmt='%d')

    # Visualize
    visualize_samples(mesh, sampled_coords)
