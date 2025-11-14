import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from algs import clustering 

SAMPLES_PER_AXIS = 10
SAMPLES_TEST = 10
SAMPLES_VIS = 10

def generate_samples(d, n, ranges=None):
    """
    Generate equidistant sample points in a d-dimensional space defined by the provided ranges.

    Parameters:
    ranges (list of tuples): List of (min, max) tuples for each dimension.
    n (int): Number of points along each dimension.

    Returns:
    np.ndarray: Array of shape (n^d, d) containing all sample points.
    """
    if ranges:
        d = len(ranges)  # Number of dimensions
        linspaces = []

        for min_val, max_val in ranges:
            # Generate n points linearly spaced between min_val and max_val
            linspace_1d = np.linspace(min_val, max_val, n)
            linspaces.append(linspace_1d)

        # Create a meshgrid from the 1D linspaces for d dimensions
        grids = np.meshgrid(*linspaces, indexing='ij')

        # Reshape grids to get a list of points in (n^d, d) shape
        samples = np.stack(grids, axis=-1).reshape(-1, d)

        return samples
    else:
        # Generate 1D linspace for each dimension
        linspace_1d = np.linspace(-0.5, 0.5, n)

        # Create a meshgrid from the 1D linspaces for d dimensions
        grids = np.meshgrid(*[linspace_1d] * d, indexing='ij')

        # Reshape grids to get a list of points in (n^d, d) shape
        samples = np.stack(grids, axis=-1).reshape(-1, d)

        return samples
    
def generate_gaussian_samples(d, n):
    mean = np.zeros(d)
    cov = np.eye(d)
    num_points = n
    points = np.random.multivariate_normal(mean, cov, num_points)
    return points

def func(target_point, trained_points):
    ''' 
    target_point: [d] or [m1, d] array
    trained_points: [m2, d] array'''
    if len(target_point.shape) == 1:
        # [m2, d]
        return np.min(np.linalg.norm(trained_points - target_point[np.newaxis, :], axis=1))
    
    elif len(target_point.shape) == 2:
        # [m1, m2, d]
        return np.min(np.linalg.norm(trained_points[np.newaxis, :, :] - target_point[:, np.newaxis, :], axis = 2), axis = 1)

def objective(trained_points, samples_per_axis=SAMPLES_TEST, ranges = None, sample_points = None):
    ''' 
    trained_points: [m2, d] array
    '''
    
    d = trained_points.shape[1]
    samples = generate_samples(d, samples_per_axis) if sample_points is None else sample_points
    function_values = func(samples, trained_points)
    average_value = np.mean(function_values)
    return average_value

def test_alg_for_dims_general(alg_class, D, K, grid_points_list, num_samples=None, samples_per_axis = SAMPLES_PER_AXIS, PLOT_PFORMANCE=False, ranges = None):
    time_for_dims = []
    performance_for_dims = []
    centroids_for_dims = []
    labels_for_dims = []
    recal_list_for_dims = []
    index = 0
    for d in D:
        grid_points = grid_points_list[index]
        alg = alg_class(grid_points, num_samples=num_samples)
        performance, time, centroids_list, labels_list, recal_list = test_alg(alg, K, ranges = ranges, sample_points=grid_points)
        
        time_for_dims.append(time)
        performance_for_dims.append(performance)
        centroids_for_dims.append(centroids_list)
        labels_for_dims.append(labels_list)
        recal_list_for_dims.append(recal_list)

        index += 1
    return performance_for_dims, time_for_dims, centroids_for_dims, labels_for_dims, recal_list_for_dims

def test_alg(alg, K, ranges=None, sample_points = None):
    time_list = []
    performance_list = []
    centroids_list = []
    labels_list = []
    recal_list = []
    start_time = time.process_time()
    for k in tqdm(range(1, K+1)):
        centroids, labels, recal_data = alg.step()
        performance = objective(centroids, ranges = ranges, sample_points=sample_points)
        performance_list += [performance]
        time_list += [time.process_time() - start_time]
        centroids_list.append(centroids)
        labels_list.append(labels)
        recal_list.append(recal_data)
    return performance_list, time_list, centroids_list, labels_list, recal_list

def test_alg_for_dims(alg_class, D, K, num_samples=None, samples_per_axis = SAMPLES_PER_AXIS, PLOT_PFORMANCE=False, ranges = None):
    time_for_dims = []
    performance_for_dims = []
    centroids_for_dims = []
    labels_for_dims = []
    recal_list_for_dims = []
    for d in D:
        grid_points = generate_samples(d, samples_per_axis, ranges)
        alg = alg_class(grid_points, num_samples=num_samples)
        performance, time, centroids_list, labels_list, recal_list = test_alg(alg, K, ranges = ranges)
        time_for_dims.append(time)
        performance_for_dims.append(performance)
        centroids_for_dims.append(centroids_list)
        labels_for_dims.append(labels_list)
        recal_list_for_dims.append(recal_list)
    return performance_for_dims, time_for_dims, centroids_for_dims, labels_for_dims, recal_list_for_dims

def test_clustering(K, grid_points, sample_points = None):
    time_list = []
    performance_list = []
    centroids_list = []
    labels_list = []
    start_time = time.process_time()
    for k in tqdm(range(1, K+1)):
        centroids, labels = clustering(grid_points, k)
        performance = objective(centroids, sample_points=sample_points)
        performance_list += [performance]
        time_list += [time.process_time() - start_time]
        centroids_list += [centroids]
        labels_list += [labels]
    return performance_list, time_list, centroids_list, labels_list

def test_clustering_for_dims(D, K, samples_per_axis = SAMPLES_PER_AXIS, PLOT=True, data_points = None):
    time_clustering_for_dims = []
    performance_clustering_for_dims = []
    centroids_clustering_for_dims = []
    labels_clustering_for_dims = []
    index = 0
    for d in D:
        grid_points = generate_samples(d, samples_per_axis) if data_points is None else data_points[index]
        performance_clustering, time_clustering, centroids_clustering, labels_clustering = test_clustering(K, grid_points, sample_points=None if data_points is None else data_points[index])

        time_clustering_for_dims.append(time_clustering)
        performance_clustering_for_dims += [performance_clustering]
        centroids_clustering_for_dims += [centroids_clustering]
        labels_clustering_for_dims += [labels_clustering]

        index += 1
    if PLOT:
        plt.figure()
        for i in range(len(time_clustering_for_dims)):
            plt.plot(range(1, K+1, 1), time_clustering_for_dims[i], label=f"{i+D[0]} dimensions")
        plt.xlabel('Step')
        plt.ylabel('CPU Time (s)')
        plt.yscale('log')
        plt.legend()
        plt.grid()
        plt.show()
    return performance_clustering_for_dims, time_clustering_for_dims, centroids_clustering_for_dims, labels_clustering_for_dims

def convert_ndarray_to_list(nested_list):
    """
    Recursively converts all numpy arrays within a nested list structure to Python lists.

    Args:
        nested_list (list): A nested list structure where each element is either a numpy array or a list.

    Returns:
        list: A fully serializable list structure with all numpy arrays converted to lists.
    """
    return [[item.tolist() if isinstance(item, np.ndarray) else item for item in sublist] for sublist in nested_list]

def save_centroids(centroids, file_name):
    ''' 
    centroids: D * K list, where each element is a np.array with shape [k, d]
    '''
    centroids = convert_ndarray_to_list(centroids)
    with open(file_name, "w") as file:
        json.dump(centroids, file)

def convert_list_to_ndarray(nested_list):
    """
    Recursively converts all lists within a nested list structure back to numpy arrays.

    Args:
        nested_list (list): A nested list structure where each inner list represents an array.

    Returns:
        list: A nested list structure with inner lists converted back to numpy arrays.
    """
    return [[np.array(item) if isinstance(item, list) else item for item in sublist] for sublist in nested_list]

def load_centroids(file_name):
    with open(file_name, "r") as file:
        centroids = json.load(file)
        return convert_list_to_ndarray(centroids)

def visualize_2d(centroids_list, ranges = None):
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    for k in range(12):
        i = k//3
        j = k%3
        ax = axes[i,j]
        centroids = centroids_list[k]
        grid_points = generate_samples(2, SAMPLES_VIS, ranges = ranges)
        distances = np.linalg.norm(grid_points[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        ax.scatter(grid_points[:, 0], grid_points[:, 1], c=labels, s=10, cmap='viridis')
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
        ax.set_title(f'K={k}')
    plt.tight_layout()
    plt.show()

def plot_time_different_algs(*args, **kwargs):
    D = kwargs['D']
    K = kwargs['K']
    names = kwargs['arg_names']
    for ii in range(len(D)):
        d = D[ii]
        plt.figure()
        for i in range(len(args)):
            time_alg = args[i]
            plt.plot(range(1, K+1, 1), time_alg[ii], label=names[i])
        plt.xlabel('Step')
        plt.ylabel('CPU Time (s)')
        
        plt.legend()
        plt.grid()
        plt.title(f"{d} dimensions")
        plt.show()

def plot_performance_different_algs(*args, **kwargs):
    D = kwargs['D']
    K = kwargs['K']
    names = kwargs['arg_names']
    for ii in range(len(D)):
        d = D[ii]
        plt.figure()
        for i in range(len(args)):
            performance_alg = args[i]
            plt.plot(range(1, K+1, 1), performance_alg[ii], label=names[i])
        plt.xlabel('Step')
        plt.ylabel('Regret')
        # plt.yscale('log')
        plt.legend()
        plt.grid()
        plt.title(f"{d} dimensions")
        plt.show()

def plot_time_and_performance(*args, **kwargs):
    D = kwargs['D']
    K = kwargs['K']
    names = kwargs['arg_names']
    for ii in range(len(D)):
        d = D[ii]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        for i in range(0, len(args)):
            time_alg = args[i][0][ii]
            performance_alg = args[i][1][ii]

            ax1.plot(range(1, K+1, 1), time_alg, label=names[i])
            ax2.plot(range(1, K+1, 1), performance_alg, label=names[i])
            
        ax1.set_xlabel('Step')
        ax1.set_ylabel('CPU Time')
        ax1.legend()
        ax1.grid()
        ax1.set_yscale('log')

        ax2.set_xlabel('Step')
        ax2.set_ylabel('Regret')
        ax2.legend()
        ax2.grid()
        fig.suptitle(f"{d} dimensions")
        plt.tight_layout()  # Adjusts layout to prevent overlap
        plt.show()

def plot_time_and_performance_baseline(*args, **kwargs):
    D = kwargs['D']
    K = kwargs['K']
    names = kwargs['arg_names']
    for ii in range(len(D)):
        d = D[ii]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        performance_baseline = args[0][1][ii]

        for i in range(1, len(args)):
            time_alg = args[i][0][ii]
            performance_alg = args[i][1][ii]
            performance_alg = np.array(performance_alg) - np.array(performance_baseline)

            ax1.plot(range(1, K+1, 1), time_alg, label=names[i])
            ax2.plot(range(1, K+1, 1), performance_alg, label=names[i])
            
        ax1.set_xlabel('Step')
        ax1.set_ylabel('CPU Time')
        ax1.legend()
        ax1.grid()
        ax1.set_yscale('log')

        ax2.set_xlabel('Step')
        ax2.set_ylabel('Regret')
        ax2.legend()
        ax2.grid()
        fig.suptitle(f"{d} dimensions")
        plt.tight_layout()  # Adjusts layout to prevent overlap
        plt.show()

def plot_time_and_performance_baseline_multi(*args, **kwargs):
    D = kwargs['D']
    K = kwargs['K']
    names = kwargs['arg_names']
    colors = kwargs['colors']
    linestyles = kwargs['linestyles']
    # [which_alg, time/perf, num_run, num_dim, num_step]
    for ii in range(len(D)):
        d = D[ii]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        performance_baseline = np.array(args[0][1])[:, ii, :]

        for i in range(0, len(args)):
            time_alg = np.array(args[i][0])[:, ii, :]
            time_mean = np.mean(time_alg, axis=0)
            time_std = np.std(time_alg, axis=0)
            ax1.plot(range(1, K+1, 1), time_mean, label=names[i], color=colors[i], linestyle=linestyles[i])
            ax1.fill_between(range(1, K+1, 1), time_mean - time_std, time_mean + time_std, alpha=0.1, color=colors[i])

        for i in range(1, len(args)):
            performance_alg = np.array(args[i][1])[:, ii, :]
            performance_alg = performance_alg - performance_baseline
            performance_mean = np.mean(performance_alg, axis=0)
            performance_std = np.std(performance_alg, axis=0)
            ax2.plot(range(1, K+1, 1), performance_mean, label=names[i], color=colors[i], linestyle=linestyles[i])
            ax2.fill_between(range(1, K+1, 1), performance_mean - performance_std, performance_mean + performance_std, alpha=0.1, color=colors[i])
            
        ax1.set_xlabel('Step')
        ax1.set_ylabel('CPU Time')
        ax1.legend()
        ax1.grid()
        ax1.set_yscale('log')

        ax2.set_xlabel('Step')
        ax2.set_ylabel('Regret')
        ax2.legend()
        ax2.grid()
        fig.suptitle(f"{d} dimensions")
        plt.tight_layout()  # Adjusts layout to prevent overlap
        plt.show()

def plot_recal(recal_list, real_D):
    ''' 
    recal_list: [D, K, 4]'''
    recal_list = np.array(recal_list)
    D, K, _ = recal_list.shape
    for d in range(D):
        plt.figure()

        plt.plot(range(1, K+1, 1), recal_list[d, :, 0], label="Number of Candidates")
        plt.plot(range(1, K+1, 1), recal_list[d, :, 1], label="Number of Recalculation")
        plt.plot(range(1, K+1, 1), recal_list[d, :, 2], label="Number of New points")
        plt.plot(range(1, K+1, 1), recal_list[d, :, 3], label="Number of Intersections")
        plt.xlabel('Step')
        plt.ylabel('Number of calculations')
        plt.legend()
        plt.grid()
        plt.title(f"{real_D[d]} dimensions")
        plt.show()

def has_duplicates(list_of_arrays):
    seen = set()
    for arr in list_of_arrays:
        # Convert the list to a tuple so it can be added to a set
        t = tuple(arr)
        if t in seen:
            return True  # Found a duplicate
        seen.add(t)
    return False 

def count_redundant_arrays(list_of_arrays):
    # Convert each array to a tuple
    array_tuples = [tuple(arr) for arr in list_of_arrays]
    
    # Count total arrays
    total = len(array_tuples)
    
    # Count unique arrays by putting them in a set
    unique_count = len(set(array_tuples))
    
    # The difference gives the number of redundant (duplicate) arrays
    return total - unique_count

def get_distance(context1, context2, noise=0):
    ''' 
    context is 3 dimensional np.array
    '''
    return np.linalg.norm(context1 - context2) + noise * np.random.normal(0, 1)

def plot_matrix(matrix, title=None):
        plt.imshow(matrix, 
                cmap='viridis',     # Colormap (try 'hot', 'coolwarm', etc.)
                aspect='auto')      # Aspect ratio; 'auto' to fill the plot

        plt.colorbar(label='Value')    # Add a colorbar to show the scale
        plt.title('2D Matrix Heatmap')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        if title:
                plt.title(title)
        plt.show()

def get_rounded(context):
    return np.vectorize(round)(context)

def get_rounded_list(context_list):
    return [get_rounded(context) for context in context_list]

def sample_integer_3d(list_range, N=1000):
    """
    Sample N integer points (x, y, z) uniformly from the 3D box
    [min_1, max_1] x [min_2, max_2] x [min_3, max_3].
    
    list_range: [min_1, max_1, min_2, max_2, min_3, max_3]
    N: number of points to sample
    
    Returns: (N, 3) array of sampled integer points.
    """
    # Unpack the bounds
    x_min, x_max, y_min, y_max, z_min, z_max = list_range
    
    # Sample each dimension independently
    x = np.random.randint(x_min, x_max + 1, size=N)
    y = np.random.randint(y_min, y_max + 1, size=N)
    z = np.random.randint(z_min, z_max + 1, size=N)
    
    # Stack into a (N, 3) array
    samples = np.column_stack((x, y, z))
    return samples

def gen_samples_3d(context_range):
    return np.array([[i, j, k] for i in range(context_range[0]) for j in range(context_range[1]) for k in range(context_range[2])])

def three_d_to_1d(context, context_range):
    return context[0] * context_range[2] * context_range[1] + context[1]*context_range[2] + context[2]

def get_performance(training_contexts, target_contexts, matrix, context_range):
    ''' 
    training_contexts: [t, 3]
    matrix: [1000, 1000]
    '''
    training_index_list = [three_d_to_1d(context, context_range) for context in training_contexts]
    target_index_list = [three_d_to_1d(context, context_range) for context in target_contexts]
    matrix_used = matrix[training_index_list, :]
    matrix_used = matrix_used[:, target_index_list]
    return np.mean(np.max(matrix_used, axis=0))

def gen_matrix_vectorized(noise=0, weight=None, b=500):
    if weight is None:
        weight = np.abs(np.random.normal(loc=[8, 14, 20], scale=3, size=3))
    N = 1000
    contexts = gen_samples_3d([10, 10, 10])
    # print(contexts.shape)
    difference = np.abs(contexts[:, np.newaxis, :] - contexts[np.newaxis, :, :]) # [1000, 1000, 3]
    distances = difference @ weight # [1000, 1000]
    noise_matrix = noise * np.random.normal(loc=0.0, scale=1.0, size=(N, N))
    distances_with_noise = distances + noise_matrix  # Shape: (1000, 1000)
    matrix = -distances_with_noise + b  # Shape: (1000, 1000)
    return matrix




