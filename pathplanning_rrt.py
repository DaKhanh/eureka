import numpy as np
import pylab as pl
from scipy.spatial import KDTree
import sys
import time  # Added for tracking running time
sys.path.append('osr_examples/scripts/')
import environment_2d
import os

class RRT(object):
    def __init__(self, env, step_size=0.5, max_iters=1000):
        self.env = env
        self.step_size = step_size
        self.max_iters = max_iters
        self.tree = []
        self.edges = []

    def initialize(self, start):
        """Initialize the tree with the start node"""
        self.tree.append(start)

    def nearest_neighbor(self, q_rand):
        """Find the nearest neighbor in the tree to q_rand"""
        tree_kdtree = KDTree(self.tree)
        _, idx = tree_kdtree.query(q_rand)
        return self.tree[idx]

    def steer(self, q_near, q_rand):
        """Move from q_near towards q_rand by a step size"""
        direction = np.array(q_rand) - np.array(q_near)
        length = np.linalg.norm(direction)
        direction = direction / length  # Normalize to unit vector
        q_new = np.array(q_near) + self.step_size * direction

        # Check for collision-free movement
        if not self.env.check_collision(q_new[0], q_new[1]):
            return tuple(q_new)
        return None

    def extend(self, q_rand):
        """Extend the tree towards q_rand"""
        q_near = self.nearest_neighbor(q_rand)
        q_new = self.steer(q_near, q_rand)
        
        if q_new is not None:
            self.tree.append(q_new)
            self.edges.append((q_near, q_new))

    def build_rrt(self, q_start, q_goal):
        """Build the RRT from q_start to q_goal"""
        self.initialize(q_start)
        for _ in range(self.max_iters):
            q_rand = (np.random.rand() * self.env.size_x, np.random.rand() * self.env.size_y)
            self.extend(q_rand)

            # If the goal is reached, stop early
            if np.linalg.norm(np.array(q_rand) - np.array(q_goal)) < self.step_size:
                print("Goal reached!")
                self.extend(q_goal)
                return self.tree, self.edges
        return None, None

    def plot(self):
        """Plot the RRT tree"""
        for edge in self.edges:
            q1, q2 = edge
            pl.plot([q1[0], q2[0]], [q1[1], q2[1]], 'g-', linewidth=2)
        pl.draw()

def generate_environments(env_configs):
    environments = []
    for env_size, obstacle_count in env_configs:
        env = environment_2d.Environment(env_size[0], env_size[1], obstacle_count)
        environments.append(env)
    return environments


environment_configs = [
    ((10, 6), 5),   # Small environment with a few obstacles
    ((15, 10), 10), # Larger environment with more obstacles
    ((20, 15), 20), # Even larger environment
    ((10, 6), 15),  # Small environment with many obstacles
    ((15, 10), 0)   # Large empty environment
]


def generate_random_queries(env, n_queries):
    queries = []
    for _ in range(n_queries):
        x_start, y_start = np.random.rand(2) * [env.size_x, env.size_y]
        x_goal, y_goal = np.random.rand(2) * [env.size_x, env.size_y]
        queries.append((x_start, y_start, x_goal, y_goal))
    return queries


def test_rrt_with_environments_and_queries(n_envs=5, n_queries=2, step_size=0.5, max_iters=1000, save_dir='rrt_plots', single_query=False):
    os.makedirs(save_dir, exist_ok=True)

    environments = generate_environments(environment_configs)
    
    for env_idx, env in enumerate(environments):
        start_time = time.time()  
        
        print(f"Testing RRT on environment {env_idx + 1}")
        pl.clf()
        env.plot() 
        
        rrt = RRT(env, step_size=step_size, max_iters=max_iters)
        
        queries = generate_random_queries(env, n_queries) if not single_query else [generate_random_queries(env, 1)]
        
        for q_idx, q in enumerate(queries):
            x_start, y_start, x_goal, y_goal = q
            env.plot_query(x_start, y_start, x_goal, y_goal)  
            
            start = (x_start, y_start)
            goal = (x_goal, y_goal)
            
            rrt.build_rrt(start, goal)
            rrt.plot()  
            
            print(f"Query {q_idx + 1}: start={start}, goal={goal}")
        
        plot_filename = os.path.join(save_dir, f'environment_{env_idx + 1}_queries_{n_queries if not single_query else 1}.png')
        pl.title(f"Environment {env_idx + 1}: size {env.size_x} x {env.size_y}, Obstacles: {len(env.obs)}")
        pl.savefig(plot_filename)
        pl.close()  
        
        end_time = time.time() 
        elapsed_time = end_time - start_time
        print(f"Environment {env_idx + 1} processed in {elapsed_time:.2f} seconds\n")

np.random.seed(42)
test_rrt_with_environments_and_queries(n_envs=5, n_queries=2, single_query=False)  # For multiple queries
test_rrt_with_environments_and_queries(n_envs=5, single_query=True)  # For single query

