import numpy as np
import pylab as pl
import sys
sys.path.append('osr_examples/scripts/')
import environment_2d
import os

def path_shortcutting(path, env, maxrep=100):
    """
    Optimize the path using the Path Short-Cutting algorithm.
    """
    optimized_path = path[:]
    
    for _ in range(maxrep):
        if len(optimized_path) < 3:
            # Not enough points to shortcut
            break
        
        # Pick two random distinct points along the path
        t1, t2 = sorted(np.random.choice(len(optimized_path), size=2, replace=False))
        
        if t2 - t1 <= 1:
            # Too close, can't shortcut
            continue
        
        # Get the two waypoints q(t1) and q(t2)
        q_t1 = optimized_path[t1]
        q_t2 = optimized_path[t2]
        
        # Check if the straight line between q(t1) and q(t2) is collision-free
        if is_collision_free(q_t1, q_t2, env):
            # If collision-free, replace the path segment with a straight line
            optimized_path = optimized_path[:t1+1] + [q_t2] + optimized_path[t2+1:]
    
    return optimized_path

def is_collision_free(p1, p2, env, num_samples=10):
    """
    Check if the straight line between two points is collision-free.
    """
    # Linear interpolation between p1 and p2
    for alpha in np.linspace(0, 1, num_samples):
        x = p1[0] * (1 - alpha) + p2[0] * alpha
        y = p1[1] * (1 - alpha) + p2[1] * alpha
        if env.check_collision(x, y):
            return False
    return True

np.random.seed(4)
env = environment_2d.Environment(10, 6, 5)
pl.clf()
env.plot()

path = [(0, 0), (1, 1), (2, 3), (4, 5), (6, 6)]

optimized_path = path_shortcutting(path, env)

print("Original Path:", path)
print("Optimized Path:", optimized_path)

original_x, original_y = zip(*path)
optimized_x, optimized_y = zip(*optimized_path)

pl.plot(original_x, original_y, marker='o', label='Original Path', color='blue')
pl.plot(optimized_x, optimized_y, marker='o', label='Optimized Path', color='red')

pl.legend()
pl.title('Path Optimization with Short-Cutting')
pl.xlabel('X-axis')
pl.ylabel('Y-axis')

save_dir = 'path_plots'  
os.makedirs(save_dir, exist_ok=True)  
plot_filename = os.path.join(save_dir, 'path_optimization.png')
pl.savefig(plot_filename)

pl.show()

pl.close()
