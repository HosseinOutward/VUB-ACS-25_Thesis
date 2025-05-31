import gzip
import numpy as np
import os
import torch
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics.pairwise import cosine_similarity


def _calculate_metrics_for_pair(v1, v2):
    # 1. Pearson Correlation --------
    p_corr = np.corrcoef(v1, v2)[0, 1]

    # 2. Mutual Information --------
    n_neigh = max(1, min(3, len(v1) - 1))
    mi_val = mutual_info_regression(
        v1.reshape(-1, 1), v2, n_neighbors=n_neigh, random_state=42)[0]
    mi_val = max(0, mi_val)

    # 3. Distance and std --------
    temp = v1 - v2
    mean_dist = np.mean(temp)
    std_dist = np.std(temp)

    # 4. Cosine Similarity --------
    cos_sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]

    return np.array([p_corr, mi_val, mean_dist, std_dist, cos_sim])


def get_similarity_metrics(list_of_pairs_of_elements: np.ndarray):
    # list_of_pairs_of_elements.shape = [pair_count, 2, sample_count]
    results = [
        _calculate_metrics_for_pair(v1, v2)
        for v1, v2 in list_of_pairs_of_elements
    ]
    return np.array(results)


def _load_and_flatten(args):
    (train_attempt, batch_idx, layer_names,
        path_to_files, curr_round, current_epoch) = args
    filename = f"_round_{curr_round}_epoch_{current_epoch}_batch_{batch_idx}_gradients.pt.gz"
    p0 = path_to_files[train_attempt] + f"worker_{0}" + filename
    p1 = path_to_files[train_attempt] + f"worker_{1}" + filename
    with gzip.open(p0, "rb") as f:
        g0 = torch.load(f, map_location="cpu")
    with gzip.open(p1, "rb") as f:
        g1 = torch.load(f, map_location="cpu")
    to_np = lambda x, i: x[i].numpy().ravel()
    return {k: (to_np(g0, i), to_np(g1, i)) for i, k in enumerate(layer_names)}


if __name__ == "__main__":
    train_attempt_count, worker_count, round_count, epoch_count, batch_count = 4, 2, 2, 30, 17

    path_to_files = [f"experiments/exp_data/gradients_resnet/gradients_resnet_t{i}/" for i in range(4)]
    filename = f"_round_{0}_epoch_{0}_batch_{0}_gradients.pt.gz"

    with gzip.open(path_to_files[0] + f"worker_{0}" + filename, "rb") as f:
        temp1 = torch.load(f)
        layer_counts = len(temp1)
        per_layer_ele_count = [len(t) for t in temp1]

    with open(path_to_files[0] + f"_grad_namings.txt", "rb") as f:
        layer_names = f.read().decode("utf-8").replace("\r",'').split("\n")[:-1]

    result = {k:[] for k in layer_names}
    time_steps = np.array(np.meshgrid(
        range(round_count), range(epoch_count))).T.reshape(-1, 2)
    sample_steps = np.array(np.meshgrid(
        range(train_attempt_count), range(batch_count))).T.reshape(-1, 2)
    for curr_round, current_epoch in time_steps:
        print(f"\nRound {curr_round}, Epoch {current_epoch} ------------")
        sample_dict = {k:[] for k in layer_names}

        jobs = [(ta, bi, layer_names, path_to_files, curr_round, current_epoch)
                    for ta, bi in sample_steps]
        # Iterate directly over pool.map without as_completed.
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
            for res in pool.map(_load_and_flatten, jobs, chunksize=1):
                for k in layer_names:
                    sample_dict[k].append(res[k])

        sample_dict = {k:np.array(sample_dict[k]).transpose(2,1,0) for k in sample_dict.keys()}

        print("      - reading disk done; calculating similarity metrics...")
        for i, k in enumerate(sample_dict.keys()):
            if (i+1) % 10 == 0:
                print(f"          > getting sim vec for layer {i+1}/{len(sample_dict.keys())}")
            result[k].append(get_similarity_metrics(sample_dict[k]))
    result = {k: np.array(result[k]) for k in layer_names}

    # save results to disk
    temp = "experiments/resnet_parameter_corr_between_worker.pt.gz"
    with gzip.open(temp, "wb") as f:
        torch.save(result, f)