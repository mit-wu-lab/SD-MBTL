from algs import * 
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

K = 216 # 20 steps
D = 3 # 3 dimensions
SLOPE_CONST = 0.01

def independent_training(target_contexts, matrix, context_range):
    target_index_list = [three_d_to_1d(context, context_range) for context in target_contexts]
    matrix_used = matrix[target_index_list, :]
    matrix_used = matrix_used[:, target_index_list]
    performance = np.trace(matrix_used) / matrix_used.shape[0]
    return [performance for _ in range(K)], [0 for _ in range(K)], None

def oracle_greedy(target_contexts, matrix, context_range, return_x_indices=False):
    target_index_list = [three_d_to_1d(context, context_range) for context in target_contexts]
    index_length = matrix.shape[0]
    remaining_indices = list(range(index_length))
    performance_list = []
    choice_list = []
    matrix = matrix[:, target_index_list]

    for k in tqdm(range(K)):
        performance_candidates = []
        for i in range(len(remaining_indices)):
            new_element = remaining_indices[i]
            new_choice_list = choice_list + [new_element]
            matrix_used = matrix[new_choice_list, :].copy()
            performance = np.mean(np.max(matrix_used, axis=0))
            performance_candidates.append(performance)
        best_performance = max(performance_candidates)
        best_i = performance_candidates.index(best_performance)
        best_index = remaining_indices[best_i]
        remaining_indices.remove(best_index)
        performance_list.append(best_performance)
        choice_list.append(best_index)
    if return_x_indices:
        return choice_list
    return performance_list, [0 for _ in range(K)], None

def stochastic_oracle(target_contexts, matrices, context_range):
    ''' 
    Stochastic oracle, take the average over all trials
    '''
    n_trial = matrices.shape[0]
    matrix_mean = np.mean(matrices, axis=0)
    x_train_indices = oracle_greedy(target_contexts, matrix_mean, context_range, return_x_indices=True)
    
    performance_list_trials = []
    time_list_trials = []
    for trial in range(n_trial):
        performance_list = []
        time_list = [0 for _ in range(K)]
        matrix_trial = matrices[trial, :, :]
        for k in range(K):
            matrix_used = matrix_trial[x_train_indices[:k+1], :].copy()
            performance = np.mean(np.max(matrix_used, axis=0))
            performance_list.append(performance)
        performance_list_trials.append(performance_list)
        time_list_trials.append(time_list)
    return performance_list_trials, time_list_trials

def myopic_oracle(target_contexts, matrices, context_range):
    ''' 
    Myopic Oracle
    '''
    n_trial = matrices.shape[0]
    N = matrices.shape[1]
    # matrix_mean = np.mean(matrices, axis=0)

    x_train_indices = np.empty([0,]).astype(int)
    remaining_indices = np.array(list(range(N))).astype(int)
    for k in range(K):
        performance_candidates = np.full([N,], -np.inf)
        for candidate in remaining_indices:
            x_train_indices_new = np.insert(x_train_indices, 0, candidate)
            ''' 
            matrices: [num_matrices, training_contexts, target_contexts]
            1, select the training indices on axis 1
            2, take the maximum over axis 1 to get the maximum generalization performance on training tasks
            3, take the mean over axis 0 and 2, which means the average on both target tasks and three matrices
            '''
            performance = np.mean(np.max(matrices[:, x_train_indices_new, :], axis=1))
            performance_candidates[candidate] = performance
        xk = np.argmax(performance_candidates)
        x_train_indices = np.insert(x_train_indices, 0, xk)
        # Delete the element at the found index
        remaining_indices = np.delete(remaining_indices, np.where(remaining_indices == xk))
    x_train_indices = x_train_indices[::-1]
    
    performance_list_trials = []
    time_list_trials = []
    for trial in range(n_trial):
        performance_list = []
        time_list = [0 for _ in range(K)]
        matrix_trial = matrices[trial, :, :]
        for k in range(K):
            matrix_used = matrix_trial[x_train_indices[:k+1], :].copy()
            performance = np.mean(np.max(matrix_used, axis=0))
            performance_list.append(performance)
        performance_list_trials.append(performance_list)
        time_list_trials.append(time_list)
    return performance_list_trials, time_list_trials

def Gaussian_process(target_contexts, matrix, context_range, weight=SLOPE_CONST*np.array([1, 1, 1])):
    ''' 
    Uses GPyTorch as the new package for Gaussian process
    TODO: 
        1, Change the slope input
        2, Use non-constant slope learning
        3, Learn the slope for GP
    '''
    time_start = time.process_time()
    time_list = []

    acquisition_function = 'UCB-log'

    source_lenpole_list = list(range(context_range[2]))
    source_masscart_list = list(range(context_range[1]))
    source_masspole_list = list(range(context_range[0]))
    var1_list = np.array(source_lenpole_list)
    var2_list = np.array(source_masscart_list)
    var3_list = np.array(source_masspole_list)

    # 3D grid of tasks
    var1, var2, var3 = np.meshgrid(var1_list, var2_list, var3_list)
    tasks = [(v1, v2, v3) for v1 in var1_list for v2 in var2_list for v3 in var3_list]
    num_tasks = len(tasks)

    # Flattened X for GP predictions
    X = np.column_stack((var1.ravel(), var2.ravel(), var3.ravel()))

    # Setup containers
    num_transfer_steps = K  # from your original code

    J_gttl_gp = np.full((len(var1_list), len(var2_list), len(var3_list), num_transfer_steps), -np.inf)

    mean_prediction = np.zeros((len(var1_list), len(var2_list), len(var3_list)))
    std_prediction = np.zeros((len(var1_list), len(var2_list), len(var3_list)))

    J_transfer = np.full((num_tasks, num_transfer_steps), -np.inf)
    V_obs_tmp = np.full((num_tasks, num_tasks, num_transfer_steps), -np.inf)

    ttl_deltas = np.zeros((num_transfer_steps, 3))
    # Will hold chosen tasks as a single integer index in the flattened space
    ttl_deltas_sqz = np.zeros(num_transfer_steps, dtype=int)

    GP_model, GP_likelihood = initialize_gp(torch.empty((0, 3)), torch.empty((0,)))

    for k in tqdm(range(num_transfer_steps)):

        # Determine which new task is selected for step k
        if k == 0:
            # Middle in each dimension
            tmp1 = (var1_list.max() + var1_list.min()) / 2
            tmp2 = (var2_list.max() + var2_list.min()) / 2
            tmp3 = (var3_list.max() + var3_list.min()) / 2
        elif k == 1 or k == 2:
            # Sort by previously obtained J_transfer
            acquisition = -J_transfer[:, k - 1]
            sorted_idx = np.argsort(-acquisition)
            # Pick first idx not already in ttl_deltas_sqz
            for sidx in sorted_idx:
                if sidx not in ttl_deltas_sqz[:k]:
                    tmp = sidx
                    # decode the (v1, v2, v3) from the flattened index
                    tmp1 = var1_list[tmp % len(var1_list)]
                    tmp2 = var2_list[(tmp // len(var1_list)) % len(var2_list)]
                    tmp3 = var3_list[(tmp // (len(var1_list) * len(var2_list))) % len(var3_list)]
                    break
        else:
                
            x = ttl_deltas[:k]
            y = []
            for j in range(k):
                i1 = np.where(var1_list == ttl_deltas[j, 0])[0][0]
                i2 = np.where(var2_list == ttl_deltas[j, 1])[0][0]
                i3 = np.where(var3_list == ttl_deltas[j, 2])[0][0]
                y.append(J_gttl_gp[i1, i2, i3, j])
            y = np.array(y)

            GP_model.set_train_data(torch.from_numpy(x), torch.from_numpy(y), strict=False)
            train_gp(GP_model, GP_likelihood, torch.from_numpy(x), torch.from_numpy(y))
            gp_mean, gp_std = predict_gp(GP_model, GP_likelihood, torch.from_numpy(X))
            gp_mean = gp_mean.numpy()
            gp_std = gp_std.numpy()

            # shape them back to 3D
            mean_prediction = gp_mean.reshape(len(var1_list), len(var2_list), len(var3_list))
            std_prediction = gp_std.reshape(len(var1_list), len(var2_list), len(var3_list))

            # Choose acquisition function
            if acquisition_function == 'EI':
                acquisition_3d = mean_prediction
            elif acquisition_function == 'UCB':
                acquisition_3d = mean_prediction + 1.96 * std_prediction
            elif acquisition_function == 'UCB-log':
                # Vectorized approach for the custom function
                lambdas = np.sqrt(np.log(k+1))  # single value repeated
                # Flatten mean/std for easier indexing
                mean_flat = mean_prediction.ravel()
                std_flat = std_prediction.ravel()

                all_indices = np.arange(num_tasks)
                new_acquisition_flat = np.zeros(num_tasks)

                tasks_arr = np.array(tasks)  # shape (num_tasks, 3)
                
                dist_matrix = np.abs(tasks_arr[:, None, :] - tasks_arr[None, :, :]) @ weight
                
                # revised here
                v_obs_j_all = V_obs_tmp[:, :, k-1].max(axis=0)  # shape (num_tasks,)
                # v_obs_j_all = V_obs_tmp[:, :, k-1].max(axis=0) 
                
                # print(mean_flat[:10])
                for i_idx in range(num_tasks):
                    ''' 
                    value from source i to all target j = mean_i + lambda * std_i - dist_i_j - v_j
                    '''
                    val_ij = (mean_flat[i_idx] 
                              + lambdas*std_flat[i_idx] 
                              - dist_matrix[i_idx]
                              - v_obs_j_all)
                    # clip at 0, then average
                    new_acquisition_flat[i_idx] = np.mean(np.maximum(val_ij, 0.0))

                # Reshape back
                acquisition_3d = new_acquisition_flat.reshape(len(var1_list), len(var2_list), len(var3_list))
            else:
                raise ValueError('Invalid acquisition function')

            # Now pick the best in the 3D array
            acquisition = acquisition_3d.ravel()  # flattened
            sorted_idx = np.argsort(-acquisition)
            for sidx in sorted_idx:
                if sidx not in ttl_deltas_sqz[:k]:
                    tmp = sidx
                    tmp1 = var1_list[tmp % len(var1_list)]
                    tmp2 = var2_list[(tmp // len(var1_list)) % len(var2_list)]
                    tmp3 = var3_list[(tmp // (len(var1_list) * len(var2_list))) % len(var3_list)]
                    break

        # Store chosen deltas
        chosen_i = (np.abs(var1_list - tmp1)).argmin()
        chosen_j = (np.abs(var2_list - tmp2)).argmin()
        chosen_b = (np.abs(var3_list - tmp3)).argmin()
        ttl_deltas[k] = [var1_list[chosen_i], var2_list[chosen_j], var3_list[chosen_b]]

        
        # Convert (chosen_i, chosen_j, chosen_b) to a single integer index in [0..num_tasks-1]
        chosen_flat_idx = (
            chosen_i
            + chosen_j * len(var1_list)
            + chosen_b * len(var1_list) * len(var2_list)
        )
        ttl_deltas_sqz[k] = chosen_flat_idx
        # print(f"k={k}, xk=({chosen_i, chosen_j, chosen_b}), index={chosen_flat_idx}")

        # 1) Vectorized update of J_gttl_gp[:,:,:,k]
        sub_matrix = matrix[chosen_flat_idx, :].reshape(
            len(var1_list), len(var2_list), len(var3_list)
        )
        if k == 0:
            # Direct assignment
            J_gttl_gp[..., k] = sub_matrix
        else:
            # Elementwise max with previous step
            J_gttl_gp[..., k] = np.maximum(sub_matrix, J_gttl_gp[..., k - 1])

        # 2) Vectorized update of J_transfer[:, k]
        if k == 0:
            J_transfer[:, k] = matrix[chosen_flat_idx, :]
        else:
            J_transfer[:, k] = np.maximum(matrix[chosen_flat_idx, :], J_transfer[:, k - 1])

        chosen_mask = np.isin(np.arange(num_tasks), ttl_deltas_sqz[: k + 1])
        V_obs_tmp[chosen_mask, :, k] = matrix[chosen_mask, :]
        
        time_list.append(time.process_time() - time_start)
        
    return list(J_gttl_gp.mean(axis=0).mean(axis=0).mean(axis=0)), time_list, None
        
def test_sequential_type_algorithm(K, alg_class, target_contexts, matrix, context_range, learn_weight=False, lr=None, num_iter=None, convergence_threshold=None, num_samples=None):
    ''' 
    Test sequential algorithms that is written in algs.py
    '''
    performance_list = []
    time_list = []
    time_start = time.process_time()
    alg_switch_list = []

    alg = alg_class(target_contexts, num_samples=num_samples)

    for k in tqdm(range(1, K+1)):
        centroids, alg_switch, _ = alg.step()

        training_contexts = get_rounded_list(centroids)
        training_indices = [three_d_to_1d(context, context_range) for context in training_contexts]
        
        if hasattr(alg, "update_GP"):
            alg.update_GP(matrix[training_indices[0], :])

        performance = get_performance(training_contexts, target_contexts, matrix, context_range)
        performance_list.append(performance)
        time_list.append(time.process_time() - time_start)
        alg_switch_list.append(alg_switch)
    return performance_list, time_list, alg_switch_list  

def test_alg_tabular(alg_class, K, target_contexts, matrix, context_range):
    lr=5e-4
    convergence_threshold = 1e-6
    num_iter=30000
    if alg_class == "independent_training":
        return independent_training(target_contexts, matrix, context_range)
    elif alg_class == "oracle_greedy":
        return oracle_greedy(target_contexts, matrix, context_range)
    elif alg_class == "Gaussian_process":
        return Gaussian_process(target_contexts, matrix, context_range)
    else:
        return test_sequential_type_algorithm(
            K,
            sequential_clustering_sampling_fixed if (
                alg_class == "without_learning_weight"
                or alg_class == "without_random_restart"
            ) else alg_class, 
            target_contexts, 
            matrix, 
            context_range, 
            learn_weight=True if (
                (callable(alg_class) and alg_class.__name__ == "sequential_clustering_sampling_fixed")
                or alg_class == "without_random_restart"
            ) else False,
            lr=lr, 
            num_iter=num_iter, 
            convergence_threshold=convergence_threshold, 
            num_samples=1 if alg_class == "without_random_restart" else target_contexts.shape[0],
        )

def run_experiments(
    alg_list,       # e.g. [sequential_clustering_sampling_fixed, random_select, "independent_training"]
    K,              # horizon length
    N_trial=3,      # number of experiments/trials
    data_dir="data",  # where your .npy reward matrices are located
    REAL_DATA=None,
    names = None,
    context_range = None,
    repeats = None
):
    """
    Runs multiple trials of the given algorithms, computes statistics, and plots the results.

    Parameters:
    -----------
    alg_list : list
        A list of algorithms to be tested. Each entry can be either a function reference
        or a string (for example: "independent_training").
    K : int
        The horizon length (number of steps).
    N_trial : int
        Number of times to repeat the experiment (default = 3).
    data_dir : str
        Directory path where the transfer_reward_matrix_trial{trial}.npy files are stored.
    n_samples : int
        Number of samples to generate for each trial (via gen_samples_3d).

    Returns:
    --------
    perf_data : dict
        A dictionary where keys are algorithm names (strings) and values are NumPy arrays
        of shape (N_trial, K) containing performance data for each trial and each step.
    """
    # Dictionary to store performance data for each algorithm across trials
    # Key will be a string (algorithm name), value will be a list of 1D arrays (length K).
    perf_data = {}
    time_data = {}
    name_plot = {}
    alg_switch_data = []

    # Initialize dict entries for all algorithms in alg_list
    for i in range(len(alg_list)):
        alg = alg_list[i]
        if callable(alg):
            # If it's a function, use its __name__ as the key
            alg_name = alg.__name__
        else:
            # If it's a string, use it directly
            alg_name = str(alg)
        perf_data[alg_name] = []
        time_data[alg_name] = []
        name_plot[alg_name] = names[i]
    # Generate your samples (adapt this call as needed)
    samples = gen_samples_3d(context_range)
    
    # Run experiments for N_trial times
    min_value = np.inf
    max_value = -np.inf
    
    # Assume that we already known the max and min value of the environment
    for trial in range(N_trial):
        matrix=None
        if REAL_DATA == "cartpole":
            matrix_path = f"{data_dir}/transfer_reward_matrix_trial{trial}.npy"
            matrix = np.load(matrix_path)
        elif REAL_DATA == "intersectionZoo":
            matrix_path = f"{data_dir}/transfer_reward_array_intersectionzoo_trial{trial}.npy"
            matrix = np.load(matrix_path)
        elif REAL_DATA == "bipedal-walker":
            matrix_path = f"{data_dir}/transfer_reward_bipedalwalker_matrix_8_trial{trial}.npy"
            matrix = np.load(matrix_path)
        elif REAL_DATA == "crop":
            matrix_path = f"{data_dir}/transfer_reward_matrix_crop_new_trial{trial}.npy"
            matrix = np.load(matrix_path)
            # matrix = np.where(np.isnan(matrix), np.nanmean(matrix), matrix)
        elif "syn" in REAL_DATA:
            matrix_path = f"{data_dir}/transfer_reward_matrix_{REAL_DATA}_trial{trial}.npy"
            matrix = np.load(matrix_path)
        else:
            matrix = gen_matrix_vectorized(noise=3)
        max_value_trial = np.max(matrix)
        min_value_trial = np.min(matrix)
        max_value = np.maximum(max_value, max_value_trial)
        min_value = np.minimum(min_value, min_value_trial)
    

    matrix_list = []

    for trial in range(N_trial):
        # Load the reward matrix
        if REAL_DATA == "cartpole":
            matrix_path = f"{data_dir}/transfer_reward_matrix_trial{trial}.npy"
            matrix = np.load(matrix_path)
        elif REAL_DATA == "intersectionZoo":
            matrix_path = f"{data_dir}/transfer_reward_array_intersectionzoo_trial{trial}.npy"
            matrix = np.load(matrix_path)
        elif REAL_DATA == "bipedal-walker":
            matrix_path = f"{data_dir}/transfer_reward_bipedalwalker_matrix_8_trial{trial}.npy"
            matrix = np.load(matrix_path)
        elif REAL_DATA == "crop":
            matrix_path = f"{data_dir}/transfer_reward_matrix_crop_new_trial{trial}.npy"
            matrix = np.load(matrix_path)
            # matrix = np.where(np.isnan(matrix), np.nanmean(matrix), matrix)
        elif "syn" in REAL_DATA:
            matrix_path = f"{data_dir}/transfer_reward_matrix_{REAL_DATA}_trial{trial}.npy"
            matrix = np.load(matrix_path)
        else:
            matrix = gen_matrix_vectorized(noise=3)
        # Normalize the transfer matrix
        matrix = (matrix - min_value)/(max_value - min_value)
        matrix_list.append(matrix)

        # Test each algorithm in the list
        for alg in alg_list:
            if callable(alg):
                # If alg is a function reference
                alg_name = alg.__name__
            else:
                # If alg is a string
                alg_name = str(alg)
            print("Testing "+alg_name)
            num_repeats = repeats[alg_name]
            for repeat_i in range(num_repeats):
                seed = (trial+1)*(repeat_i+1)
                np.random.seed(seed)
                trial_perf_list = []
                trial_time_list = []
                # Run test_alg_tabular on the chosen algorithm
                if (alg_name=="SDMBTL"):
                    perf, time_alg, alg_switch = test_alg_tabular(alg, K, samples, matrix, context_range)
                    alg_switch_data.append(alg_switch)
                else:
                    perf, time_alg, _ = test_alg_tabular(alg, K, samples, matrix, context_range)
                trial_perf_list.append(perf)
                trial_time_list.append(time_alg)

            trial_perf_list = np.mean(np.array(trial_perf_list), axis=0)
            trial_time_list = np.mean(np.array(trial_time_list), axis=0)

            perf_data[alg_name].append(trial_perf_list)
            time_data[alg_name].append(trial_time_list)

    # New sequential oracle
    matrix_list = np.array(matrix_list)
    perf_data["stochastic_oracle"], time_data["stochastic_oracle"] = stochastic_oracle(samples, matrix_list, context_range)
    name_plot["stochastic_oracle"] = "Stochastic Oracle"

    perf_data["myopic_oracle"], time_data["myopic_oracle"] = myopic_oracle(samples, matrix_list, context_range)
    name_plot["myopic_oracle"] = "Myopic Oracle"

    return perf_data, time_data, np.array(alg_switch_data)

def run_bootstrap_experiments(
    alg_list,       # e.g. [sequential_clustering_sampling_fixed, random_select, "independent_training"]
    K,              # horizon length
    N_real_trial=3,
    N_trial=100,      # number of experiments/trials
    data_dir="data",  # where your .npy reward matrices are located
    REAL_DATA=None,
    names = None,
    context_range = None,
    repeats = None
):
    """
    Runs multiple trials of the given algorithms, computes statistics, and plots the results.

    Parameters:
    -----------
    alg_list : list
        A list of algorithms to be tested. Each entry can be either a function reference
        or a string (for example: "independent_training").
    K : int
        The horizon length (number of steps).
    N_trial : int
        Number of times to repeat the experiment (default = 3).
    data_dir : str
        Directory path where the transfer_reward_matrix_trial{trial}.npy files are stored.
    n_samples : int
        Number of samples to generate for each trial (via gen_samples_3d).

    Returns:
    --------
    perf_data : dict
        A dictionary where keys are algorithm names (strings) and values are NumPy arrays
        of shape (N_trial, K) containing performance data for each trial and each step.
    """

    # Dictionary to store performance data for each algorithm across trials
    # Key will be a string (algorithm name), value will be a list of 1D arrays (length K).
    perf_data = {}
    time_data = {}
    name_plot = {}
    alg_switch_data = []

    # Initialize dict entries for all algorithms in alg_list
    for i in range(len(alg_list)):
        alg = alg_list[i]
        if callable(alg):
            # If it's a function, use its __name__ as the key
            alg_name = alg.__name__
        else:
            # If it's a string, use it directly
            alg_name = str(alg)
        perf_data[alg_name] = []
        time_data[alg_name] = []
        name_plot[alg_name] = names[i]
        
    # Generate your samples (adapt this call as needed)
    samples = gen_samples_3d(context_range)
    
    # Run experiments for N_trial times
    min_value = np.inf
    max_value = -np.inf
    
    # Assume that we already known the max and min value of the environment
    for trial in range(N_real_trial, N_real_trial+N_trial):
        matrix=None
        if REAL_DATA == "cartpole":
            matrix_path = f"{data_dir}/transfer_reward_matrix_trial{trial}.npy"
            matrix = np.load(matrix_path)
        elif REAL_DATA == "intersectionZoo":
            matrix_path = f"{data_dir}/transfer_reward_array_intersectionzoo_trial{trial}.npy"
            matrix = np.load(matrix_path)
        elif REAL_DATA == "bipedal-walker":
            matrix_path = f"{data_dir}/transfer_reward_bipedalwalker_matrix_8_trial{trial}.npy"
            matrix = np.load(matrix_path)
        elif REAL_DATA == "crop":
            matrix_path = f"{data_dir}/transfer_reward_matrix_crop_new_trial{trial}.npy"
            matrix = np.load(matrix_path)
        elif "syn" in REAL_DATA:
            matrix_path = f"{data_dir}/transfer_reward_matrix_{REAL_DATA}_trial{trial}.npy"
            matrix = np.load(matrix_path)
        else:
            assert False, "Unknown REAL_DATA"
        max_value_trial = np.max(matrix)
        min_value_trial = np.min(matrix)
        max_value = np.maximum(max_value, max_value_trial)
        min_value = np.minimum(min_value, min_value_trial)
    

    matrix_list = []

    for trial in range(N_real_trial, N_real_trial+N_trial):
        # Load the reward matrix
        if REAL_DATA == "cartpole":
            matrix_path = f"{data_dir}/transfer_reward_matrix_trial{trial}.npy"
            matrix = np.load(matrix_path)
        elif REAL_DATA == "intersectionZoo":
            matrix_path = f"{data_dir}/transfer_reward_array_intersectionzoo_trial{trial}.npy"
            matrix = np.load(matrix_path)
        elif REAL_DATA == "bipedal-walker":
            matrix_path = f"{data_dir}/transfer_reward_bipedalwalker_matrix_8_trial{trial}.npy"
            matrix = np.load(matrix_path)
        elif REAL_DATA == "crop":
            matrix_path = f"{data_dir}/transfer_reward_matrix_crop_new_trial{trial}.npy"
            matrix = np.load(matrix_path)
        elif "syn" in REAL_DATA:
            matrix_path = f"{data_dir}/transfer_reward_matrix_{REAL_DATA}_trial{trial}.npy"
            matrix = np.load(matrix_path)
        else:
            assert False, "Unknown REAL_DATA"
        # Normalize the transfer matrix
        matrix = (matrix - min_value)/(max_value - min_value)
        matrix_list.append(matrix)

        # Test each algorithm in the list
        for alg in alg_list:
            if callable(alg):
                # If alg is a function reference
                alg_name = alg.__name__
            else:
                # If alg is a string
                alg_name = str(alg)
            print("Testing "+alg_name)
            num_repeats = repeats[alg_name]
            for repeat_i in range(num_repeats):
                seed = (trial+1)*(repeat_i+1)
                np.random.seed(seed)
                trial_perf_list = []
                trial_time_list = []
                # Run test_alg_tabular on the chosen algorithm
                if (alg_name=="SDMBTL"):
                    perf, time_alg, alg_switch = test_alg_tabular(alg, K, samples, matrix, context_range)
                    alg_switch_data.append(alg_switch)
                else:
                    perf, time_alg, _ = test_alg_tabular(alg, K, samples, matrix, context_range)
                trial_perf_list.append(perf)
                trial_time_list.append(time_alg)

            trial_perf_list = np.mean(np.array(trial_perf_list), axis=0)
            trial_time_list = np.mean(np.array(trial_time_list), axis=0)

            perf_data[alg_name].append(trial_perf_list)
            time_data[alg_name].append(trial_time_list)

    # New sequential oracle
    matrix_list = np.array(matrix_list)
    perf_data["stochastic_oracle"], time_data["stochastic_oracle"] = stochastic_oracle(samples, matrix_list, context_range)
    name_plot["stochastic_oracle"] = "Stochastic Oracle"

    perf_data["myopic_oracle"], time_data["myopic_oracle"] = myopic_oracle(samples, matrix_list, context_range)
    name_plot["myopic_oracle"] = "Myopic Oracle"

    return perf_data, time_data, np.array(alg_switch_data)

import argparse

# get argument of environment
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="cartpole", help="Environment to run the experiment")
parser.add_argument("--K", type=int, default=50, help="Decision rounds")
parser.add_argument("--noise", type=int, default=0, help="noise")
parser.add_argument("--xweight", type=str, default=None, help="Weight for the x-axis")
parser.add_argument("--yweight", type=str, default=None, help="Weight for the y-axis")
parser.add_argument("--weightleft", type=str, default=None, help="Weight for the left side")
parser.add_argument("--weightright", type=str, default=None, help="Weight for the right side")
args = parser.parse_args()

# Default parameter
algs_to_test = [
    SDMBTL,
    GPMBTL,
    "without_learning_weight",
    # "without_random_restart",
    random_select,
    "independent_training",
    random_mountain,
    random_GP
]

names = [
    "SD-MBTL",
    "GP-MBTL",
    "M-MBTL",
    "Random",
    "Independent training",
    "M-MBTL + random",
    "GP-MBTL + random"
]

repeats = {
    "SD-MBTL": 1,
    "GP-MBTL": 1,
    "M-MBTL": 1,
    "random_select": 50,
    "independent_training": 1,
    "random_mountain": 50,
    "random_GP": 50
}

if args.env == "cartpole":
    if args.K >100:
        K = 15
    else:
        K = args.K

    # Run the experiment
    results_cartpole, time_cartpole, alg_switch_cartpole = run_bootstrap_experiments(
        alg_list=algs_to_test,
        K=K,
        N_real_trial=3,
        N_trial=100,
        data_dir="data",  # folder with your .npy files
        REAL_DATA="cartpole",
        names = names,
        context_range=[10, 10, 9],
        repeats = repeats
    )

    np.savez(f"results/cartpole_boot_new_{K}_slope={SLOPE_CONST}.npz", results_cartpole)
    np.savez(f"results/cartpole_boot_new_time_{K}_slope={SLOPE_CONST}.npz", time_cartpole)
    np.save(f"results/cartpole_boot_new_algswitch_{K}_slope={SLOPE_CONST}.npy", alg_switch_cartpole)
elif args.env == "walker":
    if args.K > 256:
        K = 15
    else:
        K = args.K

    # Run the experiment
    results_walker, time_walker, alg_switch_walker = run_bootstrap_experiments(
        alg_list=algs_to_test,
        K=K,
        N_real_trial=3,
        N_trial=100,
        data_dir="data",  # folder with your .npy files
        REAL_DATA="bipedal-walker",
        names = names,
        context_range=[8, 8, 8],
        repeats = repeats
    )
    np.savez(f"results/walker_boot_new_{K}_slope={SLOPE_CONST}.npz", results_walker)
    np.savez(f"results/walker_boot_new_time_{K}_slope={SLOPE_CONST}.npz", time_walker)
    np.save(f"results/walker_boot_new_algswitch_{K}_slope={SLOPE_CONST}.npy", alg_switch_walker)
elif args.env == "intersectionZoo":
    if args.K > 216:
        K = 15
    else:
        K = args.K

    # Run the experiment
    results_intersectionZoo, time_intersectionZoo, alg_switch_intersectionZoo = run_bootstrap_experiments(
        alg_list=algs_to_test,
        K=K,
        N_real_trial=3,
        N_trial=100,
        data_dir="data",  # folder with your .npy files
        REAL_DATA="intersectionZoo",
        names = names,
        context_range=[6, 6, 6],
        repeats = repeats
    )

    np.savez(f"results/intersectionZoo_boot_new_{K}_slope={SLOPE_CONST}.npz", results_intersectionZoo)
    np.savez(f"results/intersectionZoo_boot_new_time_{K}_slope={SLOPE_CONST}.npz", time_intersectionZoo)
    np.save(f"results/intersectionZoo_boot_new_algswitch_{K}_slope={SLOPE_CONST}.npy", alg_switch_intersectionZoo)
elif args.env == "crop":
    if args.K > 216:
        K = 15
    else:
        K = args.K

    # Run the experiment
    results_crop, time_crop, alg_switch_crop = run_bootstrap_experiments(
        alg_list=algs_to_test,
        K=K,
        N_real_trial=3,
        N_trial=100,
        data_dir="data",  # folder with your .npy files
        REAL_DATA="crop",
        names = names,
        context_range=[6, 6, 6],
        repeats = repeats
    )
    np.savez(f"results/crop_boot_new_{K}_slope={SLOPE_CONST}.npz", results_crop)
    np.savez(f"results/crop_boot_new_time_{K}_slope={SLOPE_CONST}.npz", time_crop)
    np.save(f"results/crop_boot_new_algswitch_{K}_slope={SLOPE_CONST}.npy", alg_switch_crop)
elif args.env == "synt_g":
    if args.K > 200:
        K = 15
    else:
        K = args.K
    noise = args.noise
    x_weight = args.xweight
    y_weight = args.yweight
    left_weight = args.weightleft
    right_weight = args.weightright
    # parser doesn't get the value starts with "-".
    if left_weight == "3-3-3":
        left_weight = "-3-3-3"

    print(f"synt_g_noise{noise}_x_weight{x_weight}_y_weight{y_weight}_dist_left{left_weight}_dist_right{right_weight}")
    # Run the experiment
    results_syn_g, time_syn_g, alg_switch_syn_g = run_bootstrap_experiments(
        alg_list=algs_to_test,
        K=K,
        N_real_trial=0,
        N_trial=100,
        data_dir="data",  # folder with your .npy files
        REAL_DATA=f"synt_g_noise{noise}_x_weight{x_weight}_y_weight{y_weight}_dist_left{left_weight}_dist_right{right_weight}",
        names = names,
        context_range=[8, 8, 8],
        repeats = repeats
    )
    np.savez(f"results/synt_g_noise{noise}_x_weight{x_weight}_y_weight{y_weight}_dist_left{left_weight}_dist_right{right_weight}_new_{K}_slope={SLOPE_CONST}.npz", results_syn_g)
    np.savez(f"results/synt_g_noise{noise}_x_weight{x_weight}_y_weight{y_weight}_dist_left{left_weight}_dist_right{right_weight}_new_time_{K}_slope={SLOPE_CONST}.npz", time_syn_g)
    np.save(f"results/synt_g_noise{noise}_x_weight{x_weight}_y_weight{y_weight}_dist_left{left_weight}_dist_right{right_weight}_new_algswitch_{K}_slope={SLOPE_CONST}.npy", alg_switch_syn_g)
else:
    # assert
    assert False, "Invalid environment"
