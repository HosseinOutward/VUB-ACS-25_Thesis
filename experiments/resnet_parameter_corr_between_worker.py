import gzip
import numpy as np
import os
import torch
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_selection import mutual_info_regression


# todo: for generating samples of training, dont use multiple entirely separate attempts.
#    instead of having multiple entirely separate attempts, from start to end,
#    have a single attempt but repeat a single batch/epoch multiple times on one state_dict
#    and then settle with the last one and continue the training, repeating for each batch/epoch.
#    this way we ensure the attempts are comparable and no need for divergence checking

# todo: after above, compute similarities (and others) during training,
#    and save them to disk, so that we can analyze them later.


def _calculate_mi_for_pair(v1: np.ndarray, v2: np.ndarray) -> float:
    sample_count = len(v1)
    n_neigh = max(1, min(3, sample_count - 1))
    mi_val = mutual_info_regression(
        v1.reshape(-1, 1), v2, n_neighbors=n_neigh, random_state=42)[0]
    mi_val = max(0.0, mi_val)  # Ensure non-negative MI

    return mi_val


# def get_similarity_metrics(list_of_pairs_of_elements: np.ndarray) -> np.ndarray:
#     pair_count, _, sample_count = list_of_pairs_of_elements.shape
#     results = get_similarity_metrics_gpu(list_of_pairs_of_elements)
#     p_corr, mean_dist, std_dist, val_mean = results[:, 0], results[:, 2], results[:, 3]
#
#     # --- 5. Cosine Similarity (Vectorized) ---
#     # Formula: dot(v1, v2) / (norm(v1) * norm(v2))
#     dot_product = np.sum(v1_all * v2_all, axis=1)  # shape [pair_count]
#     norm_v1 = np.linalg.norm(v1_all, axis=1)  # shape [pair_count]
#     norm_v2 = np.linalg.norm(v2_all, axis=1)  # shape [pair_count]
#     denominator_cos = norm_v1 * norm_v2
#
#     cos_sim = np.zeros(pair_count, dtype=np.float64)
#     both_zero_norm_mask = (norm_v1 == 0) & (norm_v2 == 0)
#     cos_sim[both_zero_norm_mask] = 1.0
#     non_zero_denominator_mask = denominator_cos != 0
#     cos_sim[non_zero_denominator_mask] = (
#             dot_product[non_zero_denominator_mask] / denominator_cos[non_zero_denominator_mask])
#
#     # --- 6. Mutual Information (Parallelized) ---
#     with warnings.catch_warnings():
#         # Suppress sklearn warnings if any occur during MI calculation in parallel
#         warnings.simplefilter("ignore")
#         mi_values = Parallel(n_jobs=-1, backend='loky')(
#             delayed(_calculate_mi_for_pair)(
#                 list_of_pairs_of_elements[i, 0, :], list_of_pairs_of_elements[i, 1, :])
#             for i in range(pair_count)
#         )
#     mi_values = np.array(mi_values)  # shape [pair_count]
#     entropy = ??
#
#
#     # --- Combine Results ---
#     results = np.stack([p_corr, mean_dist, std_dist, val_mean,
#       cos_sim, mi_values], axis=1)  # shape [pair_count, 6]
#
#     return results


def get_similarity_metrics_gpu(list_of_pairs_of_elements: np.ndarray) -> np.ndarray:
    pair_count, _, sample_count = list_of_pairs_of_elements.shape

    # Separate the two sets of vectors across all pairs
    v1_all = torch.tensor(list_of_pairs_of_elements[:, 0, :]).to('cuda')
    v2_all = torch.tensor(list_of_pairs_of_elements[:, 1, :]).to('cuda')

    # --- 1. Pearson Correlation ---
    # Formula: cov(v1, v2) / (std(v1) * std(v2))
    v1_mean = torch.mean(v1_all, dim=1, keepdim=True)  # shape [pair_count, 1]
    v2_mean = torch.mean(v2_all, dim=1, keepdim=True)  # shape [pair_count, 1]
    cov = torch.mean((v1_all - v1_mean) * (v2_all - v2_mean), dim=1)  # shape [pair_count]

    v1_std = torch.std(v1_all, dim=1)  # shape [pair_count]
    v2_std = torch.std(v2_all, dim=1)  # shape [pair_count]

    denominator = v1_std * v2_std
    p_corr = torch.full(denominator.shape,
        np.nan, dtype=v1_all.dtype, device='cuda')
    non_zero_std_mask = denominator != 0
    p_corr[non_zero_std_mask] = cov[non_zero_std_mask] / denominator[non_zero_std_mask]

    # --- 2, 3, 4. Distance Mean and Std ---
    temp_all = torch.abs(v1_all - v2_all)  # shape [pair_count, sample_count]
    mean_dist = torch.mean(temp_all, dim=1)  # shape [pair_count]
    std_dist = torch.std(temp_all, dim=1)  # shape [pair_count]
    val_mean = (torch.mean(torch.abs(v1_all), dim=1) +
                torch.mean(torch.abs(v2_all), dim=1))/2

    # --- Combine Results ---
    results = torch.stack([p_corr, mean_dist/val_mean, std_dist/val_mean, val_mean], dim=1)  # shape [pair_count, 4]

    return results.cpu().numpy()


def _load_and_flatten(args):
    (train_attempt, batch_idx, layer_names,
     path_to_files, curr_round, current_epoch, single_worker) = args
    filename = f"_round_{curr_round}_epoch_{current_epoch}_batch_{batch_idx}_gradients.pt.gz"
    to_np = lambda x, i: x[i].numpy().ravel().copy()  # Add .copy() to create independent arrays

    p0 = path_to_files[train_attempt] + f"worker_{0}" + filename
    with gzip.open(p0, "rb") as f:
        g0 = torch.load(f, map_location="cpu")

    if single_worker:
        return {k: to_np(g0, i) for i, k in enumerate(layer_names)}

    p1 = path_to_files[train_attempt] + f"worker_{1}" + filename
    with gzip.open(p1, "rb") as f:
        g1 = torch.load(f, map_location="cpu")
    return {k: (to_np(g0, i), to_np(g1, i)) for i, k in enumerate(layer_names)}


def load_grad_files(train_att_batch_idx_sample_steps, layer_names, path_to_files,
                    curr_round, current_epoch=None, single_worker=False):
    sample_dict = {k: [] for k in layer_names}

    if train_att_batch_idx_sample_steps.shape[1]==2:
        assert current_epoch is not None
        jobs = [(ta, bi, layer_names, path_to_files, curr_round,
             current_epoch, single_worker) for ta, bi in train_att_batch_idx_sample_steps]
    else:
        jobs = [(ta, bi, layer_names, path_to_files, curr_round,
             ep, single_worker) for ta, ep, bi in train_att_batch_idx_sample_steps]

    with ProcessPoolExecutor(max_workers=os.cpu_count()-2) as pool:
        for res in pool.map(_load_and_flatten, jobs, chunksize=1):
            for k in layer_names:
                sample_dict[k].append(res[k])
    return sample_dict


if __name__ == "__main__":
    train_attempt_count, worker_count, round_count, epoch_count, batch_count = 6, 2, 2, 30, 17

    # train_attempt_count, worker_count, round_count, epoch_count, batch_count = 2, 2, 1, 6, 3

    path_to_files = [f"exp_data/gradients_resnet/gradients_resnet_t{i}/"
                     for i in range(train_attempt_count)]

    with open(path_to_files[-1] + f"_grad_namings.txt", "rb") as f:
        layer_names = f.read().decode("utf-8").replace("\r", '').split("\n")[:-1]

    time_steps = np.array(np.meshgrid(
        range(round_count), range(epoch_count))).T.reshape(-1, 2)
    sample_steps = np.array(np.meshgrid(
        range(train_attempt_count), range(batch_count))).T.reshape(-1, 2)

    result = {k: [] for k in layer_names}
    for curr_round, current_epoch in time_steps:
        print(f"\nRound {curr_round}, Epoch {current_epoch} ------------")

        # indexing: {layer_name}, sample_id, worker_id, element_id
        sample_dict = load_grad_files(sample_steps, layer_names, path_to_files, curr_round, current_epoch)

        sample_dict = {k: np.array(sample_dict[k]).transpose(2, 1, 0) for k in layer_names}

        print("      - reading disk done; calculating similarity metrics...")
        for i, k in enumerate(layer_names):
            if (i + 1) % 30 == 0:
                print(f"          > getting sim vec for layer {i + 1}/{len(layer_names)}")
            result[k].append(get_similarity_metrics_gpu(sample_dict[k]))
    result = {k: np.array(result[k]) for k in layer_names}

    print("similarity metrics calculated; saving results...")
    # save results to disk
    for k in layer_names:
        print('          > saving sim vec for layer', k)
        # todo parallelize saving
        temp = f"exp_data/resnet_parameter_corr_between_worker/param_sim_vec_{k}.pt.gz"
        with gzip.open(temp, "wb") as f:
            torch.save(result[k], f)
