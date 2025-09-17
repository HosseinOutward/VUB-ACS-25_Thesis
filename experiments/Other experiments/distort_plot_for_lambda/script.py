if __name__ == '__main__':
    import numpy as np
    import torch
    from components.broadcast_components.WZ_models.wz_quant_ANN import WZQuantizer, get_real_bin_prob
    from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN

    # %%
    torch.set_float32_matmul_precision('medium')

    # %%
    import logging

    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    import warnings

    warnings.filterwarnings("ignore", message="Starting from v1.9.0, `tensorboardX` has been removed")
    warnings.filterwarnings("ignore", message="You defined a `validation_step` but have no `val_dataloader`")
    warnings.filterwarnings("ignore", message="Consider setting `persistent_workers=True` in 'train_dataloader'")
    warnings.filterwarnings("ignore", message="The 'val_dataloader' does not have")

    # %%

    noise_power = 0.1

    # %%
    side_info_data = np.random.normal(0, 1, 1_000_000, ).astype(np.float32)
    y = side_info_data + np.random.normal(0, np.sqrt(noise_power), 1_000_000, ).astype(np.float32)

    side_info_data = [side_info_data]
    temp = np.percentile(y, [0.0003, 99.9997])
    temp = ((y >= temp[0]) * (y <= temp[1]))
    y = y[temp]
    side_info_data = [a[temp] for a in side_info_data]

    # temp = np.percentile(y, [0.001, 99.999])
    # y = ((y - temp[0]) / (temp[1] - temp[0]) * 2 - 1).astype(np.float32)

    y_argsort = np.argsort(y)

    def test_model(ld, plane_setting='44'):
        num_planes = len(plane_setting)
        bins_per_plane = [int(a) for a in plane_setting]
        assert all(bins_per_plane[0] == a for a in bins_per_plane), \
            "All planes must have the same number of bins"
        bins_per_plane = bins_per_plane[0]
        wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=1, lr=8e-4,
                                         num_planes=num_planes, bins_per_plane=bins_per_plane,
                                         tau=5, reconst_ld=ld).to(torch.float32)
        wz_quantizer = WZQuantizer(wz_model, train_sample_size=200_000,
                                   count_side_info_data=1, enable_progress_bar=False)

        wz_quantizer.train_model(y, side_info_data, epoch=180, batch_size=1_000)

        deunified_bins_list = wz_quantizer.encoding_process(y)
        bins = wz_quantizer.wz_pl_model.unify_bins(deunified_bins_list)
        y_pred = wz_quantizer.decoding_process(bins, side_info_data, len(y))

        return deunified_bins_list, bins, y_pred, wz_quantizer.bin_count


    # %%

    ld_list = np.arange(0.5, 7.01, 0.5)
    plane_setting = '44'

    #%%
    mae_db_list, mape_db_list, bit_rate_list = [], [], []
    for ld in ld_list:
        print(ld)
        _, bins, y_pred, bin_count = test_model(ld, plane_setting)

        bit_rate = np.mean(-np.log2(get_real_bin_prob(bins, bin_count)[0].numpy() + 1e-12))
        mae_db = 10 * np.log10(np.mean(np.abs(y - y_pred)))
        mape_db = 10 * np.log10(np.mean(np.abs(y - y_pred)) / np.mean(np.abs(y)) * 100)

        mae_db_list.append(mae_db)
        mape_db_list.append(mape_db)
        bit_rate_list.append(bit_rate)


    res = np.array([ld_list, bit_rate_list, mae_db_list, mape_db_list])

    # write to file
    np.savez('lambda_distortion_data_44_0,1.npz', res)
