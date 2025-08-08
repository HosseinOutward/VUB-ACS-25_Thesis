import numpy as np
import torch

from components.broadcast_components.WZ_models.wz_quant_RNN import get_real_bin_prob
from components.broadcast_components.WZ_models.WZ_quantizer import WZQuantizer


def get_data_var(y, side_info_data):
    noise_variance = np.mean([np.var(y[np.random.randint(0, len(y), len(y) // 1000)]) for _ in range(10000)])
    side_info_variance=0
    if len(side_info_data) != 0:
        side_info_variance = np.mean([
            np.var(side_info_data[0][np.random.randint(0, len(y), len(y) // 1000)]) for _ in range(10000)])

    noise_variance -= side_info_variance
    return side_info_variance, noise_variance


#%%
def get_metrics(y, side_info_data, wz_quantizer:WZQuantizer):
    val_indices = np.arange(len(y))
    if wz_quantizer.val_indices is not None:
        val_indices = wz_quantizer.val_indices

    y_test = y[val_indices]
    si_test = [a[val_indices] for a in side_info_data]

    deunified_bins_list, extra_enc_data = wz_quantizer.encoding_process(y_test)
    y_pred = wz_quantizer.decoding_process(deunified_bins_list, si_test, encoding_extra_data=extra_enc_data)

    if len(side_info_data) == 0:
        si_test = [y_test.copy()*0]
    temp_si = [] if wz_quantizer.wz_pl_model.coding_model.marginal else si_test
    prior, softcodes = wz_quantizer.get_prior_and_softcodes(y_test, temp_si, 100_000)

    practical_pu = [get_real_bin_prob(b, wz_quantizer.wz_pl_model.bins_per_plane)[0]
                        for b in deunified_bins_list.to(int)]

    mse = np.mean((y_test - y_pred)**2)
    mspe = mse / np.mean(y**2) * 100

    real_bit_rate, prior_bit_rate, softcodes_bit_rate = 0, 0, 0
    for i in range(len(practical_pu)):
        real_bit_rate += torch.mean(-torch.log2(practical_pu[i] + 1e-12))
        prior_bit_rate += torch.mean(-torch.log2(prior[i] + 1e-12))
        softcodes_bit_rate += torch.mean(-torch.log2(softcodes[i] + 1e-12))

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
        lattice_mse = 10**(lattice_db/10)
        lattice_db = 10 * np.log10(lattice_mse/denom)
    return bit_rate_wz_bound, 10 * np.log10(mse), lattice_db