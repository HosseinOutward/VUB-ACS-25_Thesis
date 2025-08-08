proto_choices = ['no_proto', 'all_out', 'hybrid', 'no_proto_only_global', 'simple', 'worker-side', 'balanced_hybrid']
if __name__ == "__main__":
    import argparse
    import logging
    import warnings
    import torch
    import torchvision.transforms as transforms

    from components.other_utilities.models_to_train import ResNetPLModel
    from components.FL_sim import FLSimulator
    from components.other_utilities.datasets import FasterSVHN
    from components.broadcast_components.broadcasting_process.broadcast_reporting_utilities import BroadcastMetricGatheringUtilities
    from components.broadcast_components.WZ_models.wz_quant_ANN import WZQuantizer
    from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
    from components.other_utilities.user_logger import UnifiedLoggingClass

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run FL simulation with different protocols')
    parser.add_argument('--protocol', type=str, choices=proto_choices,)
    parser.add_argument('--global_quant', type=str, default=False)

    args = parser.parse_args()

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


            # limit_count = 10,


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
    batch_size = 15_000

    # *****************

    user_logger = UnifiedLoggingClass(worker_count, runs_reporting_folder='reports of runs/', name=args.protocol)

    broadcast_prot = None
    if args.protocol != 'no_proto':
        wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=2, bins_per_plane=16, tau=1.5,
                                     reconst_ld=400, lr=1e-3, tau_rate=10, marginal=True).to(torch.float32)
        wz_model.load_state_dict(torch.load(f'{data_folder}/basicRNN_2plane_4bins_state.pt', map_location='cpu'))

        base_quantizer = WZQuantizer(wz_model, train_sample_size=200_000,
                count_side_info_data=0, enable_progress_bar=False, user_logger=user_logger)

        if args.protocol=='all_out':
            from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import WZServerTrainingPerRoundProtocol
            broadcast_prot_base = WZServerTrainingPerRoundProtocol(worker_count, base_quantizer)
        elif args.protocol=='hybrid':
            from components.broadcast_components.broadcasting_process.HybridWZBroadcastProtocol import HybridWZBroadcastProtocol
            broadcast_prot_base = HybridWZBroadcastProtocol(worker_count, base_quantizer)
        elif args.protocol=='worker-side':
            from components.broadcast_components.broadcasting_process.WorkersideTraining import WorkersideTrainingProtocol
            broadcast_prot_base = WorkersideTrainingProtocol(worker_count, base_quantizer)
        elif args.protocol=='simple':
            from components.broadcast_components.broadcasting_process.SingleTimeTrainingProtocol import SingleTimeTrainingProtocol
            broadcast_prot_base = SingleTimeTrainingProtocol(worker_count, base_quantizer)
        elif args.protocol=='balanced_hybrid':
            from components.broadcast_components.broadcasting_process.HybridBalanced import BalancedHybridProtocol
            broadcast_prot_base = BalancedHybridProtocol(worker_count, base_quantizer)
        elif args.protocol=='no_proto_only_global':
            from components.broadcast_components.broadcasting_process.OnlyGlobalModel import OnlyGlobalModel
            broadcast_prot_base = OnlyGlobalModel(worker_count, base_quantizer)
        else:
            raise ValueError(f'Unknown protocol: {args.protocol}')

        broadcast_prot = BroadcastMetricGatheringUtilities(broadcast_prot_base, user_logger=user_logger)

        if args.global_quant:
            broadcast_prot.no_global_quantization = True

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