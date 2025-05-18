import trimesh
import numpy as np
import networkx as nx
import pygeodesic.geodesic as geodesic
import plotly.graph_objects as go


def load_mesh(mesh_file):
    mesh = trimesh.load(mesh_file, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Loaded mesh is not a Trimesh object.")
    return mesh

def load_samples(sample_file):
    with open(sample_file, 'r') as f:
        samples = [int(line.strip()) for line in f if line.strip().isdigit()]
    return samples

def compute_geodesic_distances(mesh, sample_indices):
    points = mesh.vertices
    faces = mesh.faces
    geoalg = geodesic.PyGeodesicAlgorithmExact(points, faces)
    distances = np.zeros((len(sample_indices), len(sample_indices)))
    for i, src in enumerate(sample_indices):
        dists, _ = geoalg.geodesicDistances([src], sample_indices)
        distances[i, :] = dists
    return distances

def compute_voronoi_adjacency(mesh, sample_indices):
    geoalg = geodesic.PyGeodesicAlgorithmExact(mesh.vertices, mesh.faces)
    vertex_to_sample = np.full(len(mesh.vertices), -1, dtype=int)
    dists, sources = geoalg.geodesicDistances(sample_indices, list(range(len(mesh.vertices))))
    sample_indices = np.array(sample_indices)
    vertex_to_sample[:] = sample_indices[sources]

    voronoi_adjacency = {int(s): set() for s in sample_indices}
    for face in mesh.faces:
        for i in range(3):
            v0, v1 = face[i], face[(i + 1) % 3]
            s0, s1 = vertex_to_sample[v0], vertex_to_sample[v1]
            if s0 != s1 and s0 in voronoi_adjacency and s1 in voronoi_adjacency:
                voronoi_adjacency[s0].add(int(s1))
                voronoi_adjacency[s1].add(int(s0))
    return voronoi_adjacency

def construct_sig_graph(sample_indices, distances):
    G = nx.Graph()
    num_samples = len(sample_indices)
    nn_distances = []
    for i in range(num_samples):
        temp = distances[i].copy()
        temp[i] = np.inf
        nn_dist = np.min(temp)
        nn_distances.append(nn_dist)
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            if distances[i][j] <= nn_distances[i] + nn_distances[j]:
                G.add_edge(sample_indices[i], sample_indices[j], weight=distances[i][j])
    return G

def construct_dual_voronoi_graph(sample_indices, voronoi_adjacency):
    G = nx.Graph()
    for i in sample_indices:
        for j in voronoi_adjacency.get(i, []):
            if i < j:
                G.add_edge(i, j)
    return G

def construct_sigdv_graph(sample_indices, distances, voronoi_adjacency):
    sig_graph = construct_sig_graph(sample_indices, distances)
    dual_voronoi_graph = construct_dual_voronoi_graph(sample_indices, voronoi_adjacency)
    sigdv_graph = nx.Graph()
    for edge in dual_voronoi_graph.edges():
        if sig_graph.has_edge(*edge):
            weight = distances[sample_indices.index(edge[0])][sample_indices.index(edge[1])]
            sigdv_graph.add_edge(*edge, weight=weight)
    return sigdv_graph

def solve_tsp(G):
    return nx.approximation.traveling_salesman_problem(G, weight='weight')

def make_path_hamiltonian(path):
    seen = set()
    hamiltonian_path = []
    for node in path:
        if node not in seen:
            seen.add(node)
            hamiltonian_path.append(node)
    hamiltonian_path.append(path[-1])  # Append the last node to close the path
    return hamiltonian_path

def find_hamiltonian_cycle_ST(G: nx.Graph, start_vertex: int):
    """
    Finds a Hamiltonian cycle in G starting from start_vertex using
    DFS with pruning.
    
    Returns a tuple (path, total_length, found)
    """
    visited = [False] * G.number_of_nodes()
    stack = [(start_vertex, 0.0, True)]
    path = []
    total_length = 0.0
    found = False

    while stack:
        cur_vertex, cur_len, first_extraction = stack.pop()

        if not first_extraction:
            visited[cur_vertex] = False
            path.pop()
            continue

        if len(path) == G.number_of_nodes() - 1:
            if G.has_edge(cur_vertex, start_vertex):
                path.append(cur_vertex)
                total_length = cur_len + G[cur_vertex][start_vertex].get('weight', 1.0)
                found = True
                return path + [start_vertex], total_length, found
            continue

        if _prune(G, visited, start_vertex, cur_vertex):
            continue

        stack.append((cur_vertex, cur_len, False))
        visited[cur_vertex] = True
        path.append(cur_vertex)

        for neighbor in G.neighbors(cur_vertex):
            if not visited[neighbor]:
                weight = G[cur_vertex][neighbor].get('weight', 1.0)
                stack.append((neighbor, cur_len + weight, True))

    return [], 0.0, False


def _prune(G, visited, start_vertex, cur_vertex):
    # Basic placeholder for pruning heuristic.
    # You can implement more advanced checks based on your needs.
    return False

def tsp_solver_sigbiased(G, distances, sample_indices):
    """
    Hybrid TSP solver:
    - Initialize using DFS-based Hamiltonian cycle (SIGDV bias),
    - Refine using 2-opt optimization.

    Parameters:
    - G: NetworkX graph with edges between sample indices.
    - distances: 2D numpy array of geodesic distances between sample points.
    - sample_indices: List of sample vertex indices in the mesh.

    Returns:
    - List of vertex indices in order of the optimized TSP path.
    """
    import copy
    import networkx as nx

    # Step 1: Relabel sample_indices to range(len(sample_indices)) for internal search
    index_map = {v: i for i, v in enumerate(sample_indices)}
    inv_map = {i: v for v, i in index_map.items()}
    G_sub = nx.relabel_nodes(G.subgraph(sample_indices).copy(), index_map)

    # Step 2: Find Hamiltonian cycle with pruning
    path, _, found = find_hamiltonian_cycle_ST(G_sub, start_vertex=0)
    if not found:
        raise ValueError("Hamiltonian cycle not found")

    # Step 3: 2-opt refinement (on index space)
    def tour_length(p):
        return sum(distances[p[i]][p[i + 1]] for i in range(len(p) - 1)) + distances[p[-1]][p[0]]

    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path)):
                if j - i == 1:
                    continue
                a, b = path[i - 1], path[i]
                c, d = path[j - 1], path[j % len(path)]
                if distances[a][c] + distances[b][d] < distances[a][b] + distances[c][d]:
                    path[i:j] = reversed(path[i:j])
                    improved = True
                    print("Improved path")

    # Step 4: Map back to original mesh vertex indices
    tsp_path = [inv_map[i] for i in path]
    return tsp_path


def visualize_with_toggle(mesh, path_indices, G):
    import numpy as np
    import plotly.graph_objects as go
    from collections import Counter

    # Mesh vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces
    i, j, k = faces.T

    # Mesh plot
    mesh_plot = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i, j=j, k=k,
        color='lightgrey',
        opacity=0.5,
        name='Mesh',
        visible=True
    )

    # Sampled points
    sampled_points = vertices[path_indices]
    points_plot = go.Scatter3d(
        x=sampled_points[:, 0],
        y=sampled_points[:, 1],
        z=sampled_points[:, 2],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Sampled Points',
        visible=True
    )

    # Path plot
    path_points = vertices[path_indices]
    x_lines, y_lines, z_lines = [], [], []
    for i in range(len(path_points) - 1):
        x_lines += [path_points[i, 0], path_points[i + 1, 0], None]
        y_lines += [path_points[i, 1], path_points[i + 1, 1], None]
        z_lines += [path_points[i, 2], path_points[i + 1, 2], None]

    path_plot = go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines',
        line=dict(color='blue', width=4),
        name='Path',
        visible=True
    )

    # Duplicated points
    count = Counter(path_indices)
    double_points = [v for v, c in count.items() if c > 1]
    double_plot = go.Scatter3d(
        x=vertices[double_points, 0],
        y=vertices[double_points, 1],
        z=vertices[double_points, 2],
        mode='markers',
        marker=dict(size=6, color='green'),
        name='Duplicated Points',
        visible=True
    )

    # Graph edges (G is defined on sample_indices)
    graph_x, graph_y, graph_z = [], [], []
    for u, v in G.edges():
        p1, p2 = vertices[u], vertices[v]
        graph_x += [p1[0], p2[0], None]
        graph_y += [p1[1], p2[1], None]
        graph_z += [p1[2], p2[2], None]

    graph_plot = go.Scatter3d(
        x=graph_x, y=graph_y, z=graph_z,
        mode='lines',
        line=dict(color='orange', width=2),
        name='Graph Edges',
        visible=True
    )

    # Figure
    fig = go.Figure(data=[mesh_plot, points_plot, path_plot, double_plot, graph_plot])
    fig.update_layout(
        title='3D Mesh with TSP Path and Graph',
        scene=dict(aspectmode='data'),
        updatemenus=[
            dict(
                type='buttons',
                buttons=[
                    dict(label='Toggle Mesh', method='restyle', args=['visible', [True, False, False, False, False]]),
                    dict(label='Toggle Points', method='restyle', args=['visible', [False, True, False, False, False]]),
                    dict(label='Toggle Path', method='restyle', args=['visible', [False, False, True, False, False]]),
                    dict(label='Toggle Duplicates', method='restyle', args=['visible', [False, False, False, True, False]]),
                    dict(label='Toggle Graph', method='restyle', args=['visible', [False, False, False, False, True]]),
                    dict(label='Show All', method='restyle', args=['visible', [True, True, True, True, True]])
                ],
                direction='down'
            )
        ]
    )
    fig.show()


def visualize(mesh, path_indices):
    path_points = mesh.vertices[path_indices]
    segments = np.array([[path_points[i], path_points[i + 1]] for i in range(len(path_points) - 1)])
    path_entity = trimesh.load_path(segments)
    sampled_points = mesh.vertices[path_indices]
    colors = np.tile([255, 0, 0, 255], (len(sampled_points), 1))
    point_cloud = trimesh.points.PointCloud(sampled_points, colors=colors)
    double_points = [x for x in path_indices if path_indices.count(x) > 1]
    double_cloud = trimesh.points.PointCloud(mesh.vertices[double_points], colors=[0, 255, 0, 255])
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    scene.add_geometry(path_entity)
    scene.add_geometry(point_cloud)
    scene.add_geometry(double_cloud)
    scene.show()


def main():
    mesh_file = 'cat0.off'
    sample_file = 'cat0-samples.txt'
    mesh = load_mesh(mesh_file)
    print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
    sample_indices = load_samples(sample_file)
    print(f"Loaded {len(sample_indices)} sampled vertices.")
    distances = compute_geodesic_distances(mesh, sample_indices)
    print("Computed geodesic distances between sampled points.")
    voronoi_adjacency = compute_voronoi_adjacency(mesh, sample_indices)
    print("Computed Voronoi adjacency.")
    print("Voronoi adjacency:", voronoi_adjacency)
    G = construct_sigdv_graph(sample_indices, distances, voronoi_adjacency) 
    print("Constructed SIGDV graph.")
    print(f"SIGDV graph has {len(G.nodes())} nodes and {len(G.edges())} edges.")
    tsp_path = tsp_solver_sigbiased(G, distances, sample_indices)
    print("Solved TSP on SIGDV graph.")
    print(f"TSP path: {tsp_path}")
    print([x for x in tsp_path if tsp_path.count(x) > 1])
    visualize_with_toggle(mesh, tsp_path, G)
    print("Visualized the mesh and the reconstructed curve.")

if __name__ == '__main__':
    main()
