import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.optim as optim
import gpytorch

SLOPE_CONST = 0.01
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

def find_intersections_old_version(labels, length, d):
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

    len_labels=len(labels)
    candidates = []
    for index_1d in range(len_labels):
        index_nd = np.array(one_d_index_to_nd(index_1d, length, d))
        if (index_nd==(length-1)).any():
            continue

        num_boundaries = np.count_nonzero(index_nd == 0) + np.count_nonzero(index_nd == length-2)
        
        # Array that contains 1d index of neighbors
        neighbors = []
        for dim_i in range(d):
            neighbor_i_nd = index_nd.copy()
            neighbor_i_nd[dim_i] += 1
            neighbor_i_1d = nd_index_to_1d(neighbor_i_nd, length)
            neighbors += [neighbor_i_1d]
        # neighbors_nd = get_neighbors(index_nd)
        # neighbors = [nd_index_to_1d(neig, length) for neig in neighbors_nd]
        
        color_array = labels[neighbors]
        num_colors = len(np.unique(color_array))

        if num_boundaries + num_colors >= d+1:
            candidates += [index_1d]
    return candidates

def find_intersections(labels, length, d):
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

    len_labels=len(labels)
    candidates = []
    for index_1d in range(len_labels):
        index_nd = np.array(one_d_index_to_nd(index_1d, length, d))
        num_boundaries = np.count_nonzero(index_nd == 0) + np.count_nonzero(index_nd == length-1)
        
        # Array that contains 1d index of neighbors
        neighbors_nd = get_neighbors(index_nd, length)
        neighbors = [nd_index_to_1d(neig, length) for neig in neighbors_nd]
        
        color_array = labels[neighbors]
        num_colors = len(np.unique(color_array))

        if num_boundaries + num_colors >= d+1:
            candidates += [index_1d]
    return candidates

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
    def __init__(self, X, num_samples = 10, max_iterations=300, tolerance=1e-6, dist_weight=None):
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
        if dist_weight is None:
            dist_weight = np.array([1.0]*self.d)*SLOPE_CONST
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
    M-MBTL algorithm with acceleration techniques
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
        
        self.candidates = np.random.choice(self.X.shape[0], 400, replace=False).tolist() ##-#
        # self.candidates = list(range(self.X.shape[0]))

        
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

class MBTL:
    ''' 
    GP-MBTL algorithm
    '''
    def __init__(self, X, num_samples, dist_weight=None):
        self.X = X.astype(float)
        self.d = X.shape[1]
        self.N = X.shape[0]
        if dist_weight is None:
            dist_weight = np.array([-1.0]*self.d+[1.0]*self.d)*SLOPE_CONST
        self.dist_weight_GP = dist_weight
        self.k=0
        self.GP_model, self.GP_likelihood = initialize_gp(torch.empty((0, self.d)), torch.empty((0,)))
        self.x_train = np.empty((0, self.d))
        self.x_indices = np.empty((0,), dtype=int)
        self.y_train = np.empty((0,))
        self.J_matrix = np.empty((0, self.N))

    def update_GP(self, J_row):
        self.J_matrix = np.concatenate([J_row[np.newaxis, :], self.J_matrix], axis=0)
        x_index = self.x_indices[0]
        yk = J_row[x_index]
        self.y_train = np.insert(self.y_train, 0, yk)
        self.GP_model.set_train_data(torch.from_numpy(self.x_train).float(), torch.from_numpy(self.y_train).float(), strict=False)
        train_gp(self.GP_model, self.GP_likelihood, torch.from_numpy(self.x_train).float(), torch.from_numpy(self.y_train).float())

        # Add weight learning here 
        self.dist_weight_GP = self.get_slope_from_J()

    def get_dist(self):
        diff = self.X[:, np.newaxis, :] - self.X[np.newaxis, :, :]
        diff_left = diff.copy()
        diff_right = diff.copy()
        diff_left[diff_left>0]=0
        diff_right[diff_right<0]=0
        return -(diff_left @ self.dist_weight_GP[:self.d] + diff_right @ self.dist_weight_GP[self.d:])

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
    w = np.linalg.pinv(B_flat.T @ B_flat) @ B_flat.T @ A_flat
    return w

def column_norm(matrix):
    div = (np.max(matrix, axis=0) - np.min(matrix, axis=0))
    return (matrix - np.min(matrix, axis=0) ) / div

def subtract_mean(matrix):
    return matrix - np.mean(matrix, axis=0)
    
class Combined_alg_mountain(MBTL, sequential_clustering_sampling_fixed):
    ''' 
    M/GP-MBTL (Ours) algorithm
    '''
    def __init__(self, X, num_samples = 10, max_iterations=300, tolerance=1e-6):
        sequential_clustering_sampling_fixed.__init__(self, X, num_samples, max_iterations, tolerance)
        MBTL.__init__(self, X, num_samples)
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
        w_diff_sign = w_bool[:self.d] ^ w_bool[self.d:]
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

    def update_SCC_from_MBTL(self):
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

    def update_MBTL_from_SCC(self):
        self.k += 1
        self.x_train = self.centroids_prev
        xk = self.x_train[0, :]
        xk_index = np.where((self.X==xk).all(axis=1))[0][0]
        self.x_indices = np.insert(self.x_indices, 0, xk_index)

    def step(self):
        # Detect if assumptions are satisfied by the J_matrix we have now
        if self.constant and self.positive_slope:
            return self.step_SCC()
        else:
            return self.step_MBTL()
    
    def step_SCC(self):
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
        self.update_MBTL_from_SCC()

        # The second return value means whether Hybrid uses SC-MBTL
        return best_centroids, True, None

    def step_MBTL(self):
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
        self.update_SCC_from_MBTL()

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
        self.update_SCC_from_MBTL()
        return self.x_train, None, None

class random_mountain(Combined_alg_mountain):
    ''' 
    M-MBTL + random algorithm
    '''
    def step(self):
        ''' 
        M-MBTL is deigned to solve problems with constant 
        '''
        if self.constant and self.positive_slope:
            return self.step_SCC()
        else:
            return self.step_random()

class random_GP(Combined_alg_mountain):
    ''' 
    GP-MBTL + random algorithm
    '''
    def step(self):
        ''' 
        GP-MBTL is designed to solve the problems that have positive slopes
        '''
        if self.positive_slope:
            return self.step_MBTL()
        else:
            return self.step_random()