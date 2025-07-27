import numpy as np
import torch

from components.broadcast_components.WZ_models.wz_quant_ANN import get_real_bin_prob


def prep_data(y, side_info_data, normalize=True):
    temp = np.percentile(y, [0.001, 99.999])
    temp = ((y >= temp[0]) * (y <= temp[1]))
    y = y[temp]
    side_info_data = [a[temp] for a in side_info_data]
    y_argsort = np.argsort(y)

    if normalize:
        temp = np.percentile(y, [0.001, 99.999])
        y = ((y - temp[0]) / (temp[1] - temp[0]) * 2 - 1).astype(np.float32)
        side_info_data = [((a - temp[0]) / (temp[1] - temp[0]) * 2 - 1).astype(np.float32) for a in side_info_data]

    # %%
    side_info_variance, noise_variance = [
        np.mean([np.var(temp[np.random.randint(0, len(y), len(y) // 1000)]) for _ in range(10000)])
            for temp in [side_info_data[0], y]  ]
    noise_variance -= side_info_variance
    return y, side_info_data, side_info_variance, noise_variance


#%%
def get_metrics(y, side_info_data, wz_quantizer, val_indices=None):
    if val_indices is None:
        val_indices = np.arange(len(y))

    y_test = y[val_indices]
    si_test = [a[val_indices] for a in side_info_data]

    deunified_bins_list = wz_quantizer.encoding_process(y_test)
    bins = wz_quantizer.wz_pl_model.unify_bins(deunified_bins_list)
    y_pred = wz_quantizer.decoding_process(bins, si_test, len(y_test))

    practical_pu = get_real_bin_prob(bins, wz_quantizer.bin_count)[0]
    prior, softcodes = wz_quantizer.get_prior_and_softcodes(bins, si_test, 100_000)

    mse = np.mean((y_test - y_pred)**2)
    mspe = mse / np.mean((y)**2) * 100

    real_bit_rate = torch.mean(-torch.log2(practical_pu + 1e-12))
    prior_bit_rate = torch.mean(-torch.log2(prior + 1e-12))
    softcodes_bit_rate = torch.mean(-torch.log2(softcodes + 1e-12))

    return mse, mspe, real_bit_rate, prior_bit_rate, softcodes_bit_rate


#%%
def bound_lines(y, side_info_variance, noise_variance, mape_flag = False):
    cond_var_x_y = (noise_variance*side_info_variance)/(noise_variance+side_info_variance)
    mse=np.linspace(1e-5, cond_var_x_y, 1000)[:-1]

    assert np.all((0<mse) * (mse<cond_var_x_y))

    lattice_db = 1.53
    bit_rate_wz_bound = 1/2 * np.log2(cond_var_x_y/mse)
    if mape_flag:
        denom = np.mean(y**2)
        mse = mse/denom*100
        lattice_db = 10 * np.log10(denom + 10**(lattice_db/10))
    return bit_rate_wz_bound, 10 * np.log10(mse), lattice_db