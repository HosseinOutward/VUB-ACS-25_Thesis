if __name__ == "__main__":
    import logging
    import warnings
    import torch
    import torchvision.transforms as transforms

    from components.other_utilities.models_to_train import ResNetPLModel
    from components.FL_sim import FLSimulator
    from components.other_utilities.datasets import FasterSVHN
    from components.broadcast_components.broadcasting_process.WZ_broadcast import WZBroadcastProtocol
    from components.broadcast_components.broadcasting_process.broadcast_reporting_utilities import BroadcastMetricGatheringUtilities
    from components.broadcast_components.WZ_models.wz_quant_ANN import WZQuantizer
    from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
    from components.other_utilities.user_logger import UnifiedLoggingClass

    #%%
    torch.set_float32_matmul_precision('high')

    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", "LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]")
    warnings.filterwarnings("ignore", "You defined a `validation_step` but")
    warnings.filterwarnings("ignore", "Starting from v1.9.0, `tensorboardX` has been")
    warnings.filterwarnings("ignore", "The number of training batches")
    warnings.filterwarnings("ignore", "`Trainer.fit` stopped: ")

    #%%
    data_folder = r'../../data'
    dataset = [
        FasterSVHN(
            root=data_folder+'/SVHN', split=s,
            transform=transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4377, 0.4438, 0.4728],
                    std=[0.1980, 0.2010, 0.1970]
                ),
            ])
        ) for s in ['train', 'test']]

    #%%
    worker_count = 5
    batch_size = 7_500

    # *****************

    user_logger = UnifiedLoggingClass(worker_count, name='new_prot', runs_reporting_folder='reports of runs/')
    # ****
    wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=3, bins_per_plane=4,
                                     tau=1.5, reconst_ld=600, lr=4e-3, ).to(torch.float32)
    wz_model.load_state_dict(torch.load(f'{data_folder}/basicRNN_3plane_4bins_state.pt', map_location='cpu'))
    # ****
    base_quantizer = WZQuantizer(wz_model, train_sample_size=100_000,
            count_side_info_data=0, enable_progress_bar=False, user_logger=user_logger)
    broadcast_prot_base = WZBroadcastProtocol(worker_count, base_quantizer)
    broadcast_prot = BroadcastMetricGatheringUtilities(broadcast_prot_base, user_logger=user_logger)

    # *****************

    model = ResNetPLModel(num_classes=10, resnet_version='resnet18', lr=0.005,)
    model.load_state_dict(torch.load(f'{data_folder}/resnet18_svhn.pth', map_location='cpu'))

    # *****************
    sim = FLSimulator(
        pl_model=model, num_agents=worker_count, communication_rounds=50, client_epochs_per_round=10,
        batch_size=batch_size, dataset_train=dataset[0], dataset_test=dataset[1],
        aggregation_method='fedavg', non_iid_sampling=False, user_logger=user_logger)
    # ****
    sim.run_simulation(broadcast_prot)