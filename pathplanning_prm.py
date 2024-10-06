import numpy as np
import pylab as pl
import sys
sys.path.append('osr_examples/scripts/')
import environment_2d
from scipy.spatial import KDTree
import networkx as nx
import time
import os

class PRM(object):
    def __init__(self, env, n_samples=100, k=5):
        self.env = env
        self.n_samples = n_samples
        self.k = k
        self.samples = []
        self.graph = nx.Graph()

    def sample_free(self):
        # Randomly sample points 
        while len(self.samples) < self.n_samples:
            x = np.random.rand() * self.env.size_x
            y = np.random.rand() * self.env.size_y
            if not self.env.check_collision(x, y):
                self.samples.append((x, y))

    def connect_samples(self):
        # Use a KDTree to find nearest neighbors
        tree = KDTree(self.samples)
        for i, sample in enumerate(self.samples):
            # Find k nearest neighbors
            dists, idxs = tree.query(sample, k=self.k + 1)
            for j in idxs[1:]:
                neighbor = self.samples[j]
                if self.check_collision_free(sample, neighbor):
                    self.graph.add_edge(i, j, weight=np.linalg.norm(np.array(sample) - np.array(neighbor)))

    def check_collision_free(self, p1, p2):
        # Check if the path between two points is free of obstacles
        n = 10  # Number of points to check along the line
        for i in range(1, n + 1):
            alpha = i / n
            x = p1[0] * (1 - alpha) + p2[0] * alpha
            y = p1[1] * (1 - alpha) + p2[1] * alpha
            if self.env.check_collision(x, y):
                return False
        return True

    def find_path(self, start, goal):
        # Add start and goal to the list of samples
        self.samples.append(start)
        self.samples.append(goal)
        
        start_idx = len(self.samples) - 2  # Index of the start node
        goal_idx = len(self.samples) - 1    # Index of the goal node

        # Connect start and goal to the roadmap
        self.connect_to_graph(start_idx)
        self.connect_to_graph(goal_idx)

        # Use A* search to find the shortest path
        try:
            path = nx.astar_path(
                self.graph, 
                start_idx, 
                goal_idx, 
                heuristic=lambda u, v: np.linalg.norm(np.array(self.samples[u]) - np.array(self.samples[v]))
            )
            return [self.samples[i] for i in path]
        except nx.NetworkXNoPath:
            return None

    def connect_to_graph(self, point_idx):
        tree = KDTree(self.samples)
        sample = self.samples[point_idx]
        dists, idxs = tree.query(sample, k=self.k + 1)
        for j in idxs[1:]:
            neighbor = self.samples[j]
            if self.check_collision_free(sample, neighbor):
                self.graph.add_edge(point_idx, j, weight=np.linalg.norm(np.array(sample) - np.array(neighbor)))


def generate_environment_instances(env_configs):
    environments = []
    for env_size, obstacle_count in env_configs:
        env = environment_2d.Environment(env_size[0], env_size[1], obstacle_count)
        environments.append(env)
    return environments

environment_configs = [
    ((20, 15), 10)
]

def generate_random_queries(env, n_queries):
    queries = []
    for _ in range(n_queries):
        query = env.random_query()
        if query is not None:
            queries.append(query)
    return queries


# Test the PRM with multiple environment instances and queries
def test_prm_with_environments_and_queries(env_configs, n_samples=100, k=5, n_queries=5, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    
    environments = generate_environment_instances(env_configs)
    
    for env_idx, env in enumerate(environments):
        start_time = time.time()  
        print(f"Testing PRM on environment {env_idx + 1} with size {env.size_x} x {env.size_y} and {len(env.obs)} obstacles.")
        
        env.plot()

        prm = PRM(env, n_samples=n_samples, k=k)
        prm.sample_free()
        prm.connect_samples()

        queries = generate_random_queries(env, n_queries)
        
        for q_idx, q in enumerate(queries):
            x_start, y_start, x_goal, y_goal = q
            print(f"  Query {q_idx + 1}: start=({x_start:.2f}, {y_start:.2f}), goal=({x_goal:.2f}, {y_goal:.2f})")

            env.plot_query(x_start, y_start, x_goal, y_goal)

            start = (x_start, y_start)
            goal = (x_goal, y_goal)

            path = prm.find_path(start, goal)

            if path:
                path = np.array(path)
                pl.plot(path[:, 0], path[:, 1], 'g--', linewidth=2)
            else:
                print(f"No path found for query {q_idx + 1}: start={start}, goal={goal}")

        pl.title(f"Environment size: {env.size_x} x {env.size_y}, Obstacles: {len(env.obs)}")
        plot_filename = os.path.join(save_dir, f'environment_{env_idx + 1}_queries_{n_queries}.png')
        pl.savefig(plot_filename)
        pl.close()  
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Environment {env_idx + 1} processed in {elapsed_time:.2f} seconds.\n")


np.random.seed(24)  
test_prm_with_environments_and_queries(environment_configs, n_samples=100, k=5, n_queries=5)



