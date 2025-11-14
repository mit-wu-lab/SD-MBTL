import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from scipy.optimize import minimize

import torch
import torch.optim as optim
import gpytorch

from itertools import product
from concurrent.futures import ThreadPoolExecutor


SLOPE_CONST = 0.01
WEIGHT = np.array([1.0, 1.0, 1.0])*SLOPE_CONST
ORD = 1


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
    
def nd_index_to_1d(index_nd, length):
    d = len(index_nd)
    index_1d = 0
    for dim_i in range(d):
        index_1d += length**dim_i * index_nd[dim_i]
    return int(index_1d)

def one_d_index_to_nd(index_1d, length, d):
    remain = index_1d
    index_nd = []
    for dim_i in range(d):
        index_nd.append(int(remain%length))
        remain = remain//length
    return index_nd


def get_neighbors(index_nd, length):
    neighbors = [index_nd]
    d = index_nd.shape[0]
    for dim_i in range(d):
        index_i = index_nd[dim_i]
        left = index_nd.copy()
        left[dim_i] -= 1
        right = index_nd.copy()
        right[dim_i] += 1
        if index_i!=0:
            neighbors += [left]
        if index_i!=length-1:
            neighbors += [right]
    return neighbors

def clustering(grid_points, K):
    kmeans_alg = KMeans(n_clusters=K, n_init=10)
    kmeans_alg.fit(grid_points)
    labels = kmeans_alg.predict(grid_points)
    centroids = kmeans_alg.cluster_centers_
    return centroids, labels
    
class sequential_clustering_sampling:
    def __init__(self, X, num_samples = 10, max_iterations=300, tolerance=1e-6, dist_weight=WEIGHT):
        ''' 
        X: [N, d]
        '''
        self.n_clusters = 0
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids_prev = None
        self.labels_prev = None
        self.X = X.astype(float)
        self.N = X.shape[0]
        self.d = X.shape[1]
        self.num_samples = num_samples
        self.dist_weight = dist_weight
        self.remaining_X = self.X[:, :]
    
    def get_nearest_neighbor(self, array_1):
        array_2 = self.remaining_X
        diff = np.abs(array_2 - array_1)  # shape (N, d) due to broadcasting

        # Compute squared distances (sum over axis=1)
        distances = np.sum(diff, axis=1)  # shape (N,)

        # Argmin to find index of smallest distance
        min_idx = np.argmin(distances)

        # Nearest neighbor and its distance
        nearest_neighbor = array_2[min_idx]
        self.remaining_X = np.delete(self.remaining_X, min_idx, axis=0)
        return nearest_neighbor

    def update_weight(self, new_weight):
        self.dist_weight = new_weight

    def get_new_labels_fast_vectorize(self, labels, centroids):
        ''' 
        compare the distance to original centroids (can be stored) and new centroids
        '''
        if self.n_clusters <= 1 or labels is None:
            distances = self._calculate_distances(centroids)
            labels = np.argmin(distances, axis=1)
            return labels

        labels_new = labels.copy()
        labels_new[labels==0] = np.argmin(
                np.linalg.norm(
                    (self.X[labels==0, np.newaxis, :] - centroids) * self.dist_weight, ord=ORD, axis=2
                ), axis=1
            ) 
        mask_other = labels!=0
        len_other = mask_other.sum()
        centroids_0 = np.tile(centroids[np.newaxis, 0, :], (len_other, 1, 1))
        other_labels = labels[mask_other]
        centroids_other = centroids[other_labels, np.newaxis, :]
        centroids_vec = np.concatenate([centroids_other, centroids_0], axis=1)
        ''' 
        replace:
            0: not replace
            1: replace to 0
        '''
        
        replace = np.argmin(
            np.linalg.norm(
                (self.X[mask_other, np.newaxis, :] - centroids_vec) * self.dist_weight, ord=ORD, axis=2
            ), axis=1
        ).astype(bool)
        labels_new[mask_other] = np.where(replace, 0, labels_new[mask_other])
        return labels_new
    
    def fit(self, init_centroids, round_to_nearest=False):
        centroids = init_centroids
        labels = None

        for i in range(self.max_iterations):
            labels = self.get_new_labels_fast_vectorize(labels, centroids)
            # Store the old centroids to check for convergence
            old_centroids = centroids.copy()

            # Update centroids based on the mean of assigned clusters
            points = self.X[labels == 0]
            if points.size > 0:
                centroids[0] = np.mean(points, axis=0)
            # Check for convergence (if centroids do not change)
            if np.all(np.abs(centroids - old_centroids) < self.tolerance):
                break
        if round_to_nearest:
            centroids[0] = np.vectorize(round)(centroids[0])
            labels = self.get_new_labels_fast_vectorize(None, centroids)
        return centroids, labels
    
    def _calculate_distances(self, centroids):
        ''' 
        X: [N, d]
        centroids: [K, d]
        '''
        # dif = self.X[:, np.newaxis] - centroids
        # dif_left = dif.copy()
        # dif_right = dif.copy()
        # dif_left[dif_left>0]=0
        # dif_right[dif_right<0]=0
        # dist = dif_left * self.dist_weight[:3] + dif_right * self.dist_weight[3:]
        # return np.linalg.norm(dist, ord=ORD, axis=2)
        return np.linalg.norm((self.X[:, np.newaxis] - centroids)*self.dist_weight, ord=ORD, axis=2)
    
    def get_new_centroids(self, new_centroid):
        ''' 
        Combine the new centroid with old centroids
        '''
        return np.concatenate([new_centroid, self.centroids_prev], axis = 0)
    
    def sample_candidates(self):
        return np.random.choice(self.N, size=self.num_samples)


    def try_candidates(self, candidates):
        ''' 
        Try each new centroid, test their performance, return the best candidate
        '''
        best_performance = 9999999
        best_centroids = None
        best_labels = None
        for index_1d in candidates:
            # x_candidate: [1, d]
            x_candidate = self.X[index_1d][np.newaxis, :]
            new_centroids = self.get_new_centroids(x_candidate)
            performance, centroids, labels = self.performance(new_centroids)
            if performance < best_performance:
                best_performance = performance
                best_centroids = centroids
                best_labels = labels
        return best_centroids, best_labels
    
    def performance(self, init_centroids, round_to_nearest=False):
        centroids, labels = self.fit(init_centroids, round_to_nearest=round_to_nearest)
        # distance: [N, K]
        distance = self._calculate_distances(centroids)
        min_distance = np.min(distance, axis=1)
        performance = np.sum(min_distance)
        return performance, centroids, labels

    def get_random_init(self):
        random_indices = np.random.choice(self.X.shape[0], 1, replace=False)
        return self.X[random_indices]
    
    def step(self):
        self.n_clusters += 1
        best_centroids = None
        best_labels = None

        if self.n_clusters == 1:
            init_centroids = self.get_random_init()
            best_centroids, best_labels = self.fit(init_centroids)
        else:
            candidates = self.sample_candidates()
            best_centroids, best_labels = self.try_candidates(candidates)
        
        self.centroids_prev = best_centroids
        self.labels_prev = best_labels
        return best_centroids, best_labels, None

class sequential_clustering_general:
    def __init__(self, X, num_samples = None, num_samples_axis = 10, max_iterations=300, tolerance=1e-6, dist_weight=WEIGHT):
        ''' 
        X: [N, d], target tasks
        num_samples/length: number of points per axis
        grid_points: [N^\prime, d], grid points to find the initial points as candidates
        labels_grid_prev: the labels of grid points for the previous round
        '''
        self.n_clusters = 0
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids_prev = None
        self.X = X
        self.N = X.shape[0]
        self.d = X.shape[1]
        self.length = num_samples_axis
        self.range = self.get_range(X)
        self.grid_points = generate_samples(self.d, self.length, ranges=self.range)
        self.labels_grid_prev = None

        # Initialization for the technique using the performance of previous round
        self.prev_chosen_point = None
        self.prev_init_points = None
        self.prev_performance = None
        self.prev_points_per_candidate = None
        self.prev_reduced_performance = None
        self.prev_performance_chosen = None

        self.dist_weight = np.ones(self.d) if dist_weight is None else dist_weight

    def get_range(self, X):
        ''' 
        X: [N, d] array
        '''
        d = X.shape[1]
        return [(
            np.min(X[:, i]),
            np.max(X[:, i])
        ) for i in range(d)]
        
    def get_new_labels_fast_vectorize(self, labels, centroids):
        ''' 
        compare the distance to original centroids (can be stored) and new centroids
        '''
        if self.n_clusters <= 1 or labels is None:
            distances = self._calculate_distances(centroids)
            labels = np.argmin(distances, axis=1)
            return labels

        labels_new = labels.copy()
        labels_new[labels==0] = np.argmin(
                np.linalg.norm(
                    (self.X[labels==0, np.newaxis, :] - centroids)*self.dist_weight, ord=ORD, axis=2
                ), axis=1
            ) 
        mask_other = labels!=0
        len_other = mask_other.sum()
        centroids_0 = np.tile(centroids[np.newaxis, 0, :], (len_other, 1, 1))
        other_labels = labels[mask_other]
        centroids_other = centroids[other_labels, np.newaxis, :]
        centroids_vec = np.concatenate([centroids_other, centroids_0], axis=1)
        ''' 
        replace:
            0: not replace
            1: replace to 0
        '''
        replace = np.argmin(
            np.linalg.norm(
                (self.X[mask_other, np.newaxis, :] - centroids_vec)*self.dist_weight, ord=ORD, axis=2
            ), axis=1
        ).astype(bool)
        labels_new[mask_other] = np.where(replace, 0, labels_new[mask_other])
        return labels_new
    
    def fit(self, init_centroids):
        centroids = init_centroids
        labels = None

        for i in range(self.max_iterations):
            labels = self.get_new_labels_fast_vectorize(labels, centroids)

            # Store the old centroids to check for convergence
            old_centroids = centroids.copy()

            # Update centroids based on the mean of assigned clusters
            points = self.X[labels == 0]
            if points.size > 0:
                centroids[0] = points.mean(axis=0)
            # Check for convergence (if centroids do not change)
            if np.all(np.abs(centroids - old_centroids) < self.tolerance):
                break
        return centroids, labels
    
    def _calculate_distances(self, centroids):
        ''' 
        X: [N, d]
        centroids: [K, d]
        '''
        # Calculate the Euclidean distance from each point to each centroid
        return np.linalg.norm((self.X[:, np.newaxis] - centroids)*self.dist_weight, ord=ORD, axis=2)
    
    def get_new_centroids(self, new_centroid):
        ''' 
        Combine the new centroid with old centroids
        '''
        return np.concatenate([new_centroid, self.centroids_prev], axis = 0)

    def try_candidates(self, candidates):
        ''' 
        Try each new centroid, test their performance, return the best candidate
        '''
        best_performance = 9999999
        best_centroids = None
        best_labels = None
        for index_1d in candidates:
            # x_candidate: [1, d]
            x_candidate = self.X[index_1d][np.newaxis, :]
            new_centroids = self.get_new_centroids(x_candidate)
            performance, centroids, labels = self.performance(new_centroids)
            if performance < best_performance:
                best_performance = performance
                best_centroids = centroids
                best_labels = labels
        return best_centroids, best_labels
    
    def performance(self, init_centroids):
        centroids, labels = self.fit(init_centroids)
        # distance: [N, K]
        distance = self._calculate_distances(centroids)
        min_distance = np.min(distance, axis=1)
        performance = np.sum(min_distance)
        return performance, centroids, labels

    def get_random_init(self):
        random_indices = np.random.choice(self.X.shape[0], 1, replace=False)
        return self.X[random_indices]
    
    def intersect(self, point1, point2):
        nodes_list = self.prev_init_points
        n1 = nodes_list.index(point1)
        n2 = nodes_list.index(point2)
        # points1 and points2: 1d array containing indexes of cluster members
        points1 = self.prev_points_per_candidate[n1]
        points2 = self.prev_points_per_candidate[n2]
        common_elements = np.intersect1d(points1, points2)
        return common_elements.size > 0
    
    def get_performance_for_candidate(self, point):
        ''' 
        point is an 1d index
        '''
        performance = None
        cluster_points = None

        RECALCULATE = False
        NEW_POINT = (not (point in self.prev_init_points)) if self.prev_init_points is not None else False
        INTERSECT = self.intersect(point, self.prev_chosen_point) if (self.prev_init_points is not None and not NEW_POINT) else False
        
        if (
            self.prev_init_points is None
            or NEW_POINT
            or INTERSECT
        ):
            RECALCULATE = True
            # Get the initial point from grid points
            grid_candidate = self.grid_points[point][np.newaxis, :]
            new_centroids = self.get_new_centroids(grid_candidate)
            performance, _, labels = self.performance(new_centroids)
            cluster_points = np.where(labels == 0)
        else:
            prev_index = self.prev_init_points.index(point)
            performance = self.prev_performance[prev_index] - self.prev_reduced_performance
            cluster_points = self.prev_points_per_candidate[prev_index]
        return performance, cluster_points, [RECALCULATE, NEW_POINT, INTERSECT]

    def find_candidates(self):
        '''
        labels: [length^d]
        for each item in the labels:
            if on the end boundaries: continue
            find the neighbors
            get the number of colors in the neighbors
            get the number of boundaries nearby
            if the sum is greater than d+1, include it in the list
            if satisfy the criteria, include in the list
        return the list of 1d indexes
        '''

        len_labels=len(self.labels_grid_prev)
        candidates = []
        for index_1d in range(len_labels):
            index_nd = np.array(one_d_index_to_nd(index_1d, self.length, self.d))
            num_boundaries = np.count_nonzero(index_nd == 0) + np.count_nonzero(index_nd == self.length-1)
            
            # Array that contains 1d index of neighbors
            neighbors_nd = get_neighbors(index_nd, self.length)
            neighbors = [nd_index_to_1d(neig, self.length) for neig in neighbors_nd]
            
            color_array = self.labels_grid_prev[neighbors]
            num_colors = len(np.unique(color_array))

            if num_boundaries + num_colors >= self.d+1:
                candidates += [index_1d]
        return candidates
    
    def get_labels_grid_prev(self, centroids):
        distance = np.linalg.norm((self.grid_points[:, np.newaxis] - centroids)*self.dist_weight, ord=ORD, axis=2)
        return np.argmin(distance, axis=1)
    
    def step(self):
        self.n_clusters += 1 

        best_centroids = None
        best_labels = None
        performance_chosen = None

        num_candidates = 0
        num_recalculate = 0
        num_new_points = 0
        num_intersection = 0

        if self.n_clusters == 1:
            self.centroids = self.get_random_init()
            performance_chosen, best_centroids, best_labels = self.performance(self.centroids)
        else:
            # init_points: 1d array
            init_points = self.find_candidates() 
            num_candidates = len(init_points)
            performance = [0]*num_candidates
            points_per_candidate = [None]*num_candidates

            for n in range(num_candidates):
                point = init_points[n]
                performance[n], points_per_candidate[n], flags = self.get_performance_for_candidate(point)
                
                num_recalculate += int(flags[0])
                num_new_points += int(flags[1])
                num_intersection += int(flags[2])
            
            # Get the best candidate as chosen point, recalculate its performance
            chosen_point_index = np.argmin(np.array(performance))
            chosen_point = init_points[chosen_point_index]
            grid_candidate = self.grid_points[chosen_point][np.newaxis, :]

            centroids_init = self.get_new_centroids(grid_candidate)
            performance_chosen, best_centroids, best_labels = self.performance(centroids_init)

            self.prev_points_per_candidate = points_per_candidate
            self.prev_init_points = init_points
            self.prev_chosen_point = chosen_point
            self.prev_performance = performance
            self.prev_reduced_performance = self.prev_performance_chosen - performance_chosen

        self.centroids_prev = best_centroids
        self.labels_grid_prev = self.get_labels_grid_prev(best_centroids)
        self.prev_performance_chosen = performance_chosen
        return best_centroids, best_labels, [num_candidates, num_recalculate, num_new_points, num_intersection]

class random_select:
    def __init__(self, X, num_samples=None):
        self.X = X
        self.length = X.shape[0]
        self.candidates = list(range(self.length))
        self.chosen_indices = []
    def step(self):
        index = np.random.choice(self.candidates)
        self.candidates.remove(index)
        self.chosen_indices.append(index)
        return self.X[self.chosen_indices, :], None, None
    def update_weight(self, new_weight):
        pass

class sequential_clustering_sampling_fixed(sequential_clustering_sampling):
    ''' 
    SCC algorithm with accleration techniques
    '''
    def __init__(self, X, num_samples = 10, max_iterations=300, tolerance=1e-6):
        super().__init__(X, num_samples, max_iterations, tolerance)
        ''' 
        Additionally store the following variables in the previous step:
            self.prev_chosen_point: the chosen point to train
            self.prev_init_points: initial points
            self.prev_performance: performance list for all initial points
            self.prev_points_per_candidate: clusters for all initial points
            self.prev_reduced_performance: reduced performance
            self.prev_performance_chosen: performance of the chosen point
        '''
        self.prev_chosen_point = None
        self.prev_init_points = None
        self.prev_performance = None
        self.prev_points_per_candidate = None
        self.prev_optimized_candidate = None
        self.prev_reduced_performance = None
        self.prev_performance_chosen = None
        
        # self.candidates = np.random.choice(self.X.shape[0], num_samples, replace=False).tolist()
        self.candidates = list(range(self.X.shape[0]))

        
    def find_candidates(self):
        return self.candidates
    
    def get_performance_for_candidate(self, point):
        ''' 
        Technique 2 to accelerate clustering. If not intersect with previous training task, then use the information in k-1 round.
        '''
        performance = None
        cluster_points = None

        RECALCULATE = False
        NEW_POINT = (not (point in self.prev_init_points)) if self.prev_init_points is not None else False
        INTERSECT = self.intersect(point, self.prev_chosen_point) if (self.prev_init_points is not None and not NEW_POINT) else False

        if (
            self.prev_init_points is None
            or NEW_POINT
            or INTERSECT
        ):
            RECALCULATE = True
            # Get the initial point from grid points
            grid_candidate = self.X[point][np.newaxis, :]
            new_centroids = self.get_new_centroids(grid_candidate)
            
            # performance, centroids, labels = self.performance(new_centroids, round_to_nearest=True)
            performance, centroids, labels = self.performance(new_centroids)
            cluster_points = np.where(labels == 0)
            optimized_candidate = centroids[0]
        else:
            prev_index = self.prev_init_points.index(point)
            performance = self.prev_performance[prev_index] - self.prev_reduced_performance
            cluster_points = self.prev_points_per_candidate[prev_index]
            optimized_candidate = self.prev_optimized_candidate[prev_index]
        return performance, cluster_points, optimized_candidate, [RECALCULATE, NEW_POINT, INTERSECT]
    
    def intersect(self, point1, point2):
        ''' 
        Test if the cluster of point 1 intersect with point 2 in the previous round
        '''
        nodes_list = self.prev_init_points
        n1 = nodes_list.index(point1)
        n2 = nodes_list.index(point2)
        # points1 and points2: 1d array containing indexes of cluster members
        points1 = self.prev_points_per_candidate[n1]
        points2 = self.prev_points_per_candidate[n2]
        common_elements = np.intersect1d(points1, points2)
        return common_elements.size > 0
    
    def step(self):
        self.n_clusters += 1 

        best_centroids = None
        best_labels = None
        performance_chosen = None

        num_candidates = 0
        num_recalculate = 0
        num_new_points = 0
        num_intersection = 0

        if self.n_clusters == 1:
            self.centroids = self.get_random_init()
            performance_chosen, best_centroids, best_labels = self.performance(self.centroids, round_to_nearest=True)
        else:
            init_points = self.find_candidates() 
            num_candidates = len(init_points)
            performance = [0]*num_candidates
            points_per_candidate = [None]*num_candidates
            optimized_candidate = [None]*num_candidates   
            for n in range(num_candidates):
                point = init_points[n]
                performance[n], points_per_candidate[n], optimized_candidate[n], flags = self.get_performance_for_candidate(point)
                
                num_recalculate += int(flags[0])
                num_new_points += int(flags[1])
                num_intersection += int(flags[2])
                
            chosen_point_index = np.argmin(np.array(performance))
            chosen_point = init_points[chosen_point_index]
            x_candidate = self.X[chosen_point][np.newaxis, :]

            centroids_init = self.get_new_centroids(x_candidate)
            performance_chosen, best_centroids, best_labels = self.performance(centroids_init, round_to_nearest=True)

            self.prev_points_per_candidate = points_per_candidate
            self.prev_optimized_candidate = optimized_candidate
            self.prev_init_points = init_points
            self.prev_chosen_point = chosen_point
            self.prev_performance = performance
            self.prev_reduced_performance = self.prev_performance_chosen - performance_chosen

        self.centroids_prev = best_centroids
        self.labels_prev = best_labels
        self.prev_performance_chosen = performance_chosen
        return best_centroids, best_labels, [num_candidates, num_recalculate, num_new_points, num_intersection]

def get_weight(M, training_contexts, target_contexts, lr=1e-4, num_iter=1000, convergence_threshold=1e-3, init_weight=list(WEIGHT), max_reward=500):
    N_source = training_contexts.shape[0]
    N_target = target_contexts.shape[0]
    differences = np.abs((training_contexts[:, np.newaxis, :] - target_contexts[np.newaxis, : :]))
    differences = np.concatenate([np.ones([N_source, N_target, 1]), differences], axis=2)
    w = get_slope(M, differences)
    return -w[1:], [w[0]]

def get_weight_nonconstant_J(M, training_contexts, target_contexts, lr=1e-4, num_iter=1000, convergence_threshold=1e-3, init_weight:list=[1, 1, 1], J_training=None):
    ''' 
    Learn the distance weight
    '''
    M = torch.from_numpy(M).float()  # Ensure the tensor has the correct dtype
    training_contexts = torch.from_numpy(training_contexts).float()
    target_contexts = torch.from_numpy(target_contexts).float()
    N = M.shape[1]
    w = torch.tensor(init_weight, dtype=torch.float32, requires_grad=True)
    J_training_matrix = torch.from_numpy(np.tile(J_training[:, np.newaxis], N)).float()

    optimizer = optim.Adam([w], lr=lr)
    differences = (training_contexts.unsqueeze(1) - target_contexts.unsqueeze(0)).abs()
    prev_w = w.clone().detach()
    for iteration in range(num_iter):
        optimizer.zero_grad()
        l1_norms = differences @ w
        R = - l1_norms + J_training_matrix  # shape (k, N)
        loss = ((M - R)**2).sum()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta_w = torch.norm(w - prev_w).item()
            if delta_w < convergence_threshold:
                break
            prev_w = w.clone()
    return w.detach().numpy()


def get_weight_left_right(M, training_contexts, target_contexts, lr=1e-4, num_iter=5000, convergence_threshold=1e-3, init_weight:list=[-1, -1, -1, 1, 1, 1], max_reward=500):
    M = torch.from_numpy(M).float()  # Ensure the tensor has the correct dtype
    training_contexts = torch.from_numpy(training_contexts).float()
    target_contexts = torch.from_numpy(target_contexts).float()
    d = training_contexts.shape[1]
    w = torch.tensor(init_weight, dtype=torch.float32, requires_grad=True)
    b = torch.tensor([max_reward], dtype=torch.float32, requires_grad=True)
    #  optimizer = optim.Adam([w, b], lr=lr, weight_decay=0.001)  # Try smaller lr
    differences = training_contexts.unsqueeze(1) - target_contexts.unsqueeze(0)
    differences_left = torch.where(differences > 0, torch.tensor(0.0), differences)
    differences_right = torch.where(differences < 0, torch.tensor(0.0), differences)
    
    optimizer = optim.Adam([w, b], lr=lr)
    prev_w = w.clone().detach()
    prev_b = b.clone().detach()
    
    
    for iteration in range(num_iter):
        optimizer.zero_grad()
        l1_norms = differences_left @ w[:3] + differences_right @ w[3:]
        R = - l1_norms + b  # shape (N, M)
        loss = ((M - R)**2).sum()
        loss.backward()
        optimizer.step()
        # Check for convergence
        with torch.no_grad():
            delta_w = torch.norm(w - prev_w).item()
            delta_b = torch.norm(b-prev_b).item()
            if delta_w < convergence_threshold and delta_b<convergence_threshold:
                break
            prev_w = w.clone()
            prev_b = b.clone()
    
    return np.maximum(w.detach().numpy(), 0), b.detach().numpy()

class optimizer_solution:
    def __init__(self, X, num_samples=None, dist_weight=WEIGHT):
        self.X = X.astype(float)
        self.N = X.shape[0]
        self.d = X.shape[1]
        self.dist_weight = dist_weight
        self.centroids = np.empty((0, self.d))

    def update_weight(self, new_weight):
        # self.dist_weight = np.maximum(new_weight, 0.001)
        self.dist_weight = new_weight

    def objective_function(self, new_centroid):
        # new_centroid: np.array, [d]
        # centroids: np.array, [K, d]
        centroids = np.concatenate([new_centroid[np.newaxis, :], self.centroids], axis=0)
        # dist: np.array, [N, K]

        # dif = self.X[:, np.newaxis] - centroids
        # dif_left = dif.copy()
        # dif_right = dif.copy()
        # dif_left[dif_left>0]=0
        # dif_right[dif_right<0]=0
        # dist = dif_left * self.dist_weight[:3] + dif_right * self.dist_weight[3:]
        # dist = np.linalg.norm(dist, ord=ORD, axis=2)

        dist = np.linalg.norm((self.X[:, np.newaxis]-centroids)*self.dist_weight, ord=ORD, axis=2)
        loss = np.sum(np.min(dist, axis=1))
        return loss

    def step(self):
        init_point = self.X[np.random.choice(self.N), :]
        result = minimize(self.objective_function, init_point, method='L-BFGS-B', options={'maxiter': 2000}).x
        result = np.vectorize(round)(result)
        self.centroids = np.concatenate([result[np.newaxis, :], self.centroids], axis=0)
        return self.centroids, None, None

class GP_SCC(sequential_clustering_sampling_fixed):
    def __init__(self, X, num_samples, max_iterations=300, tolerance=1e-6):
        ''' 
        x: [k, 3] array
        y: [k] array
        X: [N, 3] array
        gp_mean/std: [N] array
        '''
        super().__init__(X, num_samples, max_iterations, tolerance)
        noise_std = 0.001
        n_restarts_optimizer = 15
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        self.gaussian_process = GaussianProcessRegressor(
            kernel=kernel, 
            alpha=noise_std**2, 
            n_restarts_optimizer=n_restarts_optimizer
        )
        self.y = np.array([])
        # self.weight_penalty = 0.0043
        # x: [k, 3] array
        # y: [k] array
        # X: [N, 3] array
        # gp_mean/std: [N] array

    def update_GP(self, y_k):
        self.y = np.insert(self.y, 0, y_k)
        x = self.centroids_prev
        self.gaussian_process.fit(x, self.y)
    
    # def get_penalty(self, gp_pred, changed_points, gp_std):
    #     ''' 
    #     gp_pred: predicted J of the chosen point
    #     changed_points: the cluster points that changed the labels
    #     self.labels_prev: previous labels, [N] array
    #     self.y: real J of previous training tasks, [k] array
    #     '''
    #     num_points = changed_points.shape[0] 
    #     prev_labels_changed_points = self.labels_prev[changed_points]
    #     prev_J_list = self.y[prev_labels_changed_points]
    #     # print("prev_J_mean:", np.mean(prev_J_list), "gp_pred:", gp_pred)

    #     dif = self.X[changed_points] - self.centroids_prev[prev_labels_changed_points]
    #     dif_left = dif.copy()
    #     dif_right = dif.copy()
    #     dif_left[dif_left>0]=0
    #     dif_right[dif_right<0]=0
    #     dist = dif_left @ self.dist_weight[:3] + dif_right @ self.dist_weight[3:]

    #     # dist = (np.abs(self.X[changed_points] - self.centroids_prev[prev_labels_changed_points])) @ self.dist_weight
        
    #     ''' 
    #     J = - sum(min_dist[n]) + (J_pred_mean + beta * J_pred_std) * num_points - sum(J[n] - min_dist[n])
    #     J_new = np.mean(np.max(J_pred_mean + beta * J_pred_std - J[n] - min_dist[n], 0))
    #     '''
    #     prev_J_sum = np.sum(prev_J_list)
    #     beta = np.sqrt(np.log(self.n_clusters+1))
    #     improved_J = (gp_pred + beta * gp_std) * num_points - prev_J_sum + np.sum(dist)
    #     return improved_J
    
    def acquisition_func(self, J_pred_mean, J_pred_std, x_k, points):
        ''' 
        J_new = np.sum(np.max(J_pred_mean + beta * J_pred_std - min_dist_new[n] - (J_prev[n] - min_dist_prev[n]), 0))
        '''
        num_points = points.shape[0] 
        prev_labels_changed_points = self.labels_prev[points]
        J_prev = self.y[prev_labels_changed_points]
        min_dist_prev = (np.abs(self.X[points] - self.centroids_prev[prev_labels_changed_points])) @ self.dist_weight
        
        beta = np.sqrt(np.log(self.n_clusters))
        beta=0
        J_now = J_pred_mean + beta * J_pred_std
        min_dist_now = (np.abs(self.X[points] - x_k)) @ self.dist_weight
        acquisition = np.sum(np.maximum(
            0, 
            (J_now - min_dist_now) - (J_prev - min_dist_prev)
        ))
        return -acquisition
    
    def acquisition_func_all_points(self, J_pred_mean, J_pred_std, x_k, points):
        ''' 
        TODO: calculate the increased J of all points
        '''
        J_prev = self.y[self.labels_prev]
        min_dist_prev = (np.abs(self.X - self.centroids_prev[self.labels_prev])) @ self.dist_weight
        
        beta = np.sqrt(np.log(self.n_clusters+1))
        J_now = J_pred_mean + beta * J_pred_std
        min_dist_now = (np.abs(self.X - x_k)) @ self.dist_weight
        acquisition = np.sum(np.maximum(
            0, 
            (J_now - min_dist_now) - (J_prev - min_dist_prev)
        ))
        return -acquisition

    def add_GP_penalty(self, optimized_candidate, points_per_candidate):
        optimized_candidate = np.array(optimized_candidate)
        num_candidates = optimized_candidate.shape[0]
        gp_mean, gp_std = self.gaussian_process.predict(optimized_candidate, return_std=True)
        return [self.acquisition_func(gp_mean[i], gp_std[i], optimized_candidate[i, :], points_per_candidate[i][0]) for i in range(num_candidates)]


    # def add_GP_penalty(self, performance, optimized_candidate, points_per_candidate):
        
    #     optimized_candidate = np.array(optimized_candidate)
    #     gp_mean, gp_std = self.gaussian_process.predict(optimized_candidate, return_std=True)
    #     return [performance[i] - self.get_penalty(gp_mean[i], points_per_candidate[i][0], gp_std[i]) for i in range(len(performance))]

    def step(self):
        self.n_clusters += 1 

        best_centroids = None
        best_labels = None
        performance_chosen = None

        num_candidates = 0
        num_recalculate = 0
        num_new_points = 0
        num_intersection = 0

        if self.n_clusters == 1:
            self.centroids = self.get_random_init()
            performance_chosen, best_centroids, best_labels = self.performance(self.centroids, round_to_nearest=True)
        else:
            # init_points: [num_candidates]
            # point: init_points[n]
            # x_{k,i}: self.X[point][np.new_axis, :]
            init_points = self.find_candidates() 
            num_candidates = len(init_points)
            performance = [0]*num_candidates
            points_per_candidate = [None]*num_candidates
            optimized_candidate = [None]*num_candidates   
            for n in range(num_candidates):
                point = init_points[n]
                performance[n], points_per_candidate[n], optimized_candidate[n], flags = self.get_performance_for_candidate(point)
                
                num_recalculate += int(flags[0])
                num_new_points += int(flags[1])
                num_intersection += int(flags[2])

            # Add the GP penalty here
            performance_with_GP = self.add_GP_penalty(optimized_candidate, points_per_candidate)
            chosen_point_index = np.argmin(np.array(performance_with_GP))
            chosen_point = init_points[chosen_point_index]
            x_candidate = self.X[chosen_point][np.newaxis, :]

            centroids_init = self.get_new_centroids(x_candidate)
            performance_chosen, best_centroids, best_labels = self.performance(centroids_init, round_to_nearest=True)

            self.prev_points_per_candidate = points_per_candidate
            self.prev_optimized_candidate = optimized_candidate
            self.prev_init_points = init_points
            self.prev_chosen_point = chosen_point
            self.prev_performance = performance
            self.prev_reduced_performance = self.prev_performance_chosen - performance_chosen

        self.centroids_prev = best_centroids
        self.labels_prev = best_labels
        self.prev_performance_chosen = performance_chosen
        return best_centroids, best_labels, [num_candidates, num_recalculate, num_new_points, num_intersection]

class OnlineGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(OnlineGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Initialize the model with training data
def initialize_gp(train_x, train_y):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = OnlineGPModel(train_x, train_y, likelihood)
    return model, likelihood

def train_gp(model, likelihood, train_x, train_y, num_iter=500, lr=0.1, tol=1e-5):
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    prev_loss = float('inf')  # Initialize previous loss as a very large number

    for i in range(num_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        # Check the change in loss
        loss_change = abs(prev_loss - loss.item())
        if loss_change < tol:  # If the change in loss is smaller than the tolerance, stop
            # print(f"Converged at iteration {i+1}. Loss change: {loss_change:.6f}")
            break
        
        prev_loss = loss.item()  # Update previous loss

    return model, likelihood

# Predict mean and standard deviation
def predict_gp(model, likelihood, test_x):
    model.eval()
    likelihood.eval()
    
    with torch.no_grad():
        pred_dist = likelihood(model(test_x))
        mean = pred_dist.mean
        std = pred_dist.stddev
    return mean, std

class GPMBTL:
    def __init__(self, X, num_samples, dist_weight=SLOPE_CONST*np.array([-1, -1, -1, 1, 1, 1])):
        self.X = X.astype(float)
        self.d = X.shape[1]
        self.N = X.shape[0]
        self.dist_weight_GP = dist_weight
        self.k=0
        self.GP_model, self.GP_likelihood = initialize_gp(torch.empty((0, 3)), torch.empty((0,)))
        self.x_train = np.empty((0, self.d))
        self.x_indices = np.empty((0,), dtype=int)
        self.y_train = np.empty((0,))
        self.J_matrix = np.empty((0, self.N))

    def update_GP(self, J_row):
        self.J_matrix = np.concatenate([J_row[np.newaxis, :], self.J_matrix], axis=0)
        x_index = self.x_indices[0]
        yk = J_row[x_index]
        self.y_train = np.insert(self.y_train, 0, yk)
        self.GP_model.set_train_data(torch.from_numpy(self.x_train), torch.from_numpy(self.y_train), strict=False)
        train_gp(self.GP_model, self.GP_likelihood, torch.from_numpy(self.x_train), torch.from_numpy(self.y_train))

        # Add weight learning here 
        self.dist_weight_GP = self.get_slope_from_J()

    def get_dist(self):
        diff = self.X[:, np.newaxis, :] - self.X[np.newaxis, :, :]
        diff_left = diff.copy()
        diff_right = diff.copy()
        diff_left[diff_left>0]=0
        diff_right[diff_right<0]=0
        return -(diff_left @ self.dist_weight_GP[:3] + diff_right @ self.dist_weight_GP[3:])

    def find_xk(self):
        ''' 
        self.J_matrix: [k, N]
        gp_mean / gp_std: [N_candidates] array
        dist_matrix: [N_candidates, N] array
        v_obs_j_all: [N] array
        val: [N_candidates, N] array
        acquisition: [N_candidates] array
        '''
        gp_mean, gp_std = predict_gp(self.GP_model, self.GP_likelihood, torch.from_numpy(self.X).float())
        gp_mean = gp_mean.numpy()
        gp_std = gp_std.numpy()
        lambdas = np.sqrt(np.log(self.k))
        # dist_matrix = np.abs(self.X[:, np.newaxis, :] - self.X[np.newaxis, :, :]) @ self.dist_weight

        dist_matrix = self.get_dist()

        v_obs_j_all = np.max(self.J_matrix, axis=0)
        val = gp_mean[:, np.newaxis] + lambdas * gp_std[:, np.newaxis] - dist_matrix - v_obs_j_all[np.newaxis, :]
        val = np.maximum(val, 0.0)
        acquisition = np.mean(val, axis=1)
        xk_index = self.argmax_acquisition(acquisition)
        xk = self.X[xk_index]
        return xk, xk_index

    def argmax_acquisition(self, acquisition):
        sorted_idx = np.argsort(-acquisition)
        xk_index = None
        for idx in sorted_idx:
            if idx not in self.x_indices:
                xk_index = idx
                break
        return xk_index 

    def find_xk_heuristic(self):
        '''Choose the task with low generalized performance'''
        acquisition = -np.max(self.J_matrix, axis=0)
        xk_index = self.argmax_acquisition(acquisition)
        xk = self.X[xk_index]
        return xk, xk_index

    def get_slope_from_J(self):
        ''' 
        x_train: [k, d] array
        X: [N, d] array
        J_matrix / J_pred: [k, N] array
        B1 / B2: [k, N, d]
        B_whole: [k, N, 1+2d]
        '''
        x_train = self.x_train
        X = self.X
        J_matrix = self.J_matrix

        k, d = x_train.shape
        N = X.shape[0]
        B = x_train[:, np.newaxis, :] - X[np.newaxis, :, :]
        B1 = B.copy()
        B2 = B.copy()
        B0 = np.ones([k, N, 1])
        B1[B1>0] = 0
        B2[B2<0] = 0
        B_whole = np.concatenate([B0, B1, B2], axis=2)
        w = get_slope(J_matrix, B_whole)
        w = w[1:]
        return w

    def step(self):
        self.k += 1
        if self.k == 1:
            xk = np.round(np.mean(self.X, axis=0))
            xk_index = np.where((self.X==xk).all(axis=1))[0][0]
        elif self.k == 2 or self.k == 3:
            xk, xk_index = self.find_xk_heuristic()
        else:
            xk, xk_index = self.find_xk()
        self.x_train = np.concatenate([xk[np.newaxis, :], self.x_train], axis=0)
        self.x_indices = np.insert(self.x_indices, 0, xk_index)
        return self.x_train, None, None

def get_slope(A, B):
    k, N, d = B.shape
    A_flat = A.flatten()
    B_flat = B.reshape(k * N, d)
    # use pinv instead of inv to avoid singular matrix error
    w = np.linalg.pinv(B_flat.T @ B_flat) @ B_flat.T @ A_flat
    return w

def column_norm(matrix):
    div = (np.max(matrix, axis=0) - np.min(matrix, axis=0))
    return (matrix - np.min(matrix, axis=0) ) / div

def subtract_mean(matrix):
    return matrix - np.mean(matrix, axis=0)

class SDMBTL(GPMBTL, sequential_clustering_sampling_fixed):
    ''' 
    Modification:
    Learn weight for GP-MBTL (1, 6d weight; 2, update weight in slope criteria)
    '''
    def __init__(self, X, num_samples = 10, max_iterations=300, tolerance=1e-6):
        sequential_clustering_sampling_fixed.__init__(self, X, num_samples, max_iterations, tolerance)
        GPMBTL.__init__(self, X, num_samples)
        self.constant = True
        self.positive_slope = True
    
    def constant_criteria(self):
        J_matrix = self.J_matrix
        J_matrix = subtract_mean(J_matrix)
        std_row_mean = np.mean(np.std(J_matrix, axis=1))
        std = np.std(J_matrix[range(len(self.x_indices)), self.x_indices])
        ratio = std / std_row_mean
        return ratio<1
    
    def slope_criteria(self):
        ''' 
        x_train: [k, d] array
        X: [N, d] array
        J_matrix / J_pred: [k, N] array
        B1 / B2: [k, N, d]
        B_whole: [k, N, 1+2d]
        '''
        # update
        x_train = self.x_train
        X = self.X
        J_matrix = self.J_matrix 
        k, d = x_train.shape
        N = X.shape[0]
        B = x_train[:, np.newaxis, :] - X[np.newaxis, :, :]
        B1 = B.copy()
        B2 = B.copy()
        B0 = np.ones([k, N, 1])
        B1[B1>0] = 0
        B2[B2<0] = 0
        B_whole = np.concatenate([B0, B1, B2], axis=2)
        w = get_slope(J_matrix, B_whole)
        w = w[1:]
        self.dist_weight_GP = w

        # Detect
        x_train = self.x_train
        X = self.X
        J_matrix = self.J_matrix 
        J_matrix = subtract_mean(J_matrix)
        k, d = x_train.shape
        N = X.shape[0]
        B = x_train[:, np.newaxis, :] - X[np.newaxis, :, :]
        B1 = B.copy()
        B2 = B.copy()
        B0 = np.ones([k, N, 1])
        B1[B1>0] = 0
        B2[B2<0] = 0
        B_whole = np.concatenate([B0, B1, B2], axis=2)
        w = get_slope(J_matrix, B_whole)
        w = w[1:]
        w_bool = w>0
        w_diff_sign = w_bool[:3] ^ w_bool[3:]
        return np.mean(w_diff_sign.astype(int))>0.5

    def detect(self):
        self.constant = self.constant_criteria()
        self.positive_slope = self.slope_criteria()

    def update_GP(self, J_row):
        self.J_matrix = np.concatenate([J_row[np.newaxis, :], self.J_matrix], axis=0)
        x_index = self.x_indices[0]
        yk = J_row[x_index]
        self.y_train = np.insert(self.y_train, 0, yk)

        if self.k>5:
            self.detect()
            if not (self.constant and self.positive_slope):
                self.GP_model.set_train_data(torch.from_numpy(self.x_train), torch.from_numpy(self.y_train), strict=False)
                train_gp(self.GP_model, self.GP_likelihood, torch.from_numpy(self.x_train), torch.from_numpy(self.y_train))

    def update_M_from_GP(self):
        self.n_clusters += 1
        # Update variables in SCC
        self.x_train
        distance = self._calculate_distances(self.x_train)
        labels = np.argmin(distance, axis=1)
        min_distance = np.min(distance, axis=1)
        performance = np.sum(min_distance)

        self.centroids_prev = self.x_train
        self.labels_prev = labels
        self.prev_performance_chosen = performance

    def update_GP_from_M(self):
        self.k += 1
        self.x_train = self.centroids_prev
        xk = self.x_train[0, :]
        xk_index = np.where((self.X==xk).all(axis=1))[0][0]
        self.x_indices = np.insert(self.x_indices, 0, xk_index)

    def step(self):
        # Detect if assumptions are satisfied by the J_matrix we have now
        print("Constant:", self.constant, "slope:", self.positive_slope)
        if self.constant and self.positive_slope:
            return self.step_MMBTL()
        else:
            return self.step_GPMBTL()
    
    def step_MMBTL(self):
        self.n_clusters += 1 

        best_centroids = None
        best_labels = None
        performance_chosen = None

        num_candidates = 0

        if self.n_clusters == 1:
            self.centroids = self.get_random_init()
            performance_chosen, best_centroids, best_labels = self.performance(self.centroids, round_to_nearest=True)
        else:
            init_points = self.find_candidates() 
            num_candidates = len(init_points)
            performance = [0]*num_candidates
            points_per_candidate = [None]*num_candidates
            optimized_candidate = [None]*num_candidates   
            for n in range(num_candidates):
                point = init_points[n]
                performance[n], points_per_candidate[n], optimized_candidate[n], flags = self.get_performance_for_candidate(point)
                
            chosen_point_index = np.argmin(np.array(performance))
            chosen_point = init_points[chosen_point_index]
            x_candidate = self.X[chosen_point][np.newaxis, :]

            centroids_init = self.get_new_centroids(x_candidate)
            performance_chosen, best_centroids, best_labels = self.performance(centroids_init, round_to_nearest=True)

            self.prev_points_per_candidate = points_per_candidate
            self.prev_optimized_candidate = optimized_candidate
            self.prev_init_points = init_points
            self.prev_chosen_point = chosen_point
            self.prev_performance = performance
            self.prev_reduced_performance = self.prev_performance_chosen - performance_chosen

        self.centroids_prev = best_centroids
        self.labels_prev = best_labels
        self.prev_performance_chosen = performance_chosen
        self.update_GP_from_M()

        # The second return value means whether Hybrid uses SC-MBTL
        return best_centroids, True, None

    def step_GPMBTL(self):
        self.k += 1
        if self.k == 1:
            xk = np.round(np.mean(self.X, axis=0))
            xk_index = np.where((self.X==xk).all(axis=1))[0][0]
        elif self.k == 2 or self.k == 3:
            xk, xk_index = self.find_xk_heuristic()
        else:
            xk, xk_index = self.find_xk()
        self.x_train = np.concatenate([xk[np.newaxis, :], self.x_train], axis=0)
        self.x_indices = np.insert(self.x_indices, 0, xk_index)
        self.update_M_from_GP()

        # The second return value means whether Hybrid uses SC-MBTL
        return self.x_train, False, None
    
    def step_random(self):
        self.k+=1
        mask = ~np.isin(np.arange(self.N), self.x_indices)
        # Use the mask to filter out the elements
        remaining_indices = np.arange(self.N)[mask]
        xk_index = np.random.choice(remaining_indices)
        xk = self.X[xk_index]

        self.x_train = np.concatenate([xk[np.newaxis, :], self.x_train], axis=0)
        self.x_indices = np.insert(self.x_indices, 0, xk_index)
        self.update_M_from_GP()
        return self.x_train, None, None
class random_mountain(SDMBTL):
    def step(self):
        ''' 
        M-MBTL is deigned to solve problems with constant 
        '''
        if self.constant and self.positive_slope:
            return self.step_MMBTL()
        else:
            return self.step_random()

class random_GP(SDMBTL):
    def step(self):
        ''' 
        GP-MBTL is designed to solve the problems that have positive slopes
        '''
        if self.positive_slope:
            return self.step_GPMBTL()
        else:
            return self.step_random()