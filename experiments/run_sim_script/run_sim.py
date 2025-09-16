proto_choices = [
    'no_proto',  # 0
    'all_out', 'balanced_hybrid',  # 1 2
    'hybrid', 'no_proto_only_global',  # 3 4
    'simple', 'worker-side',  # 5 6
    'worker-side-with-error-accum',  # 7
    'MarginalOnly',  # 8
    'cancer',  # 9
    'cancer-small-update',  # 10
    'cancer_1bit',  # 11
    'cancer-small-update_1bit',  # 12
    *['conventional_'+r+rr for r in ['round', 'sign'] for rr in ['','_dsc']],  # 13, 14, 15, 16
    'non-wz-cancer',
]
proto_combo = [str(i) for i in range(0, len(proto_choices))]
proto_combo += [''.join([str(i), str(j)])
                for i in range(0, len(proto_choices)) for j in range(0, len(proto_choices)) if i != j]
proto_combo += [''.join([str(i), str(j), str(k)])
                for i in range(0, len(proto_choices))
                for j in range(0, len(proto_choices))
                for k in range(0, len(proto_choices))
                if i != j and i != k and j != k]
proto_choices += proto_combo

if __name__ == "__main__":
    import gc
    import argparse
    import logging
    import warnings
    import torch
    import traceback
    import torchvision.transforms as transforms

    from components.other_utilities.models_to_train import ResNetPLModel
    from components.FL_sim import FLSimulator
    from components.other_utilities.datasets import FasterSVHN
    from torchvision.datasets import ImageNet
    from components.broadcast_components.broadcasting_process.broadcast_reporting_utilities import \
        BroadcastMetricGatheringUtilities
    from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import QuantizerWithDataPrep
    from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
    from components.other_utilities.user_logger import UnifiedLoggingClass

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run FL simulation with different protocols')
    parser.add_argument('--protocol', type=str, choices=proto_choices, )
    parser.add_argument('--no_global_quant', type=str, default=True)
    parser.add_argument('--no_outlier_handling', type=str, default=False)
    parser.add_argument('--no_normalization', type=str, default=False)
    parser.add_argument('--dataset_name', type=str, default='SVHN')
    parser.add_argument('--given_name', type=str, default=None)
    parser.add_argument('--force_no_dsc', type=str, default=False)

    args = parser.parse_args()

    # %%
    torch.set_float32_matmul_precision('high')

    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", "LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]")
    warnings.filterwarnings("ignore", "You defined a `validation_step` but")
    warnings.filterwarnings("ignore", "Starting from v1.9.0, `tensorboardX` has been")
    warnings.filterwarnings("ignore", "The number of training batches")
    warnings.filterwarnings("ignore", "`Trainer.fit` stopped: ")

    # %%
    data_folder = r'data'
    # data_folder = r'../../data'

    if args.dataset_name == 'SVHN':
        dataset = [
            FasterSVHN(

                # limit_count = 10000,

                root=data_folder + '/SVHN', split=s,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4377, 0.4438, 0.4728],
                        std=[0.1980, 0.2010, 0.1970]
                    ),
                ])
            ) for s in ['train', 'test']]
        num_classes = 10

    elif args.dataset_name == 'imagenet':
        dataset = [
            ImageNet(
                root=data_folder + '/Imagenet', split=s,
                download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4377, 0.4438, 0.4728],
                        std=[0.1980, 0.2010, 0.1970]
                    ),
                ])
            ) for s in ['train', 'test']]
        num_classes = 1000  # ImageNet has 1000 classes

    # temp = [t for _, t in dataset[0]]
    # assert len(np.unique(temp)) == num_classes


    # %%
    def f(proto_name):
        worker_count = 5
        batch_size = 5000

        # *****************
        complete_name = proto_name
        if args.no_global_quant != False:
            complete_name += '_no_global_quant'
        if args.no_outlier_handling != False:
            complete_name += '_no_outlier_handling'
        if args.no_normalization != False:
            complete_name += '_no_normalization'
        if args.force_no_dsc != False:
            complete_name += '_no_dsc'

        if args.given_name is None:
            args.given_name = complete_name

        user_logger = UnifiedLoggingClass(worker_count, runs_reporting_folder='reports of runs/', name=args.given_name)
        temp = args.given_name if args.given_name==complete_name else f'{args.given_name} ({complete_name})'
        print(f'Running protocol {temp}', '  |  ', temp)

        broadcast_prot = None
        if proto_name != 'no_proto':
            wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=2, bins_per_plane=16, tau=1.3,
                                             reconst_ld=400, lr=1e-3, tau_rate=10, marginal=True).to(torch.float32)
            wz_model.load_state_dict(torch.load(f'{data_folder}/basicRNN_2plane_4bins_state.pt', map_location='cpu'))

            base_quantizer = QuantizerWithDataPrep(wz_model, train_sample_size=300_000,
                                                   count_side_info_data=0, enable_progress_bar=False,
                                                   user_logger=user_logger)

            if proto_name == 'all_out':
                from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import \
                    WZServerTrainingPerRoundProtocol
                broadcast_prot_base = WZServerTrainingPerRoundProtocol(worker_count, base_quantizer)
            elif proto_name == 'hybrid':
                from components.broadcast_components.broadcasting_process.HybridWZBroadcastProtocol import \
                    HybridWZBroadcastProtocol
                broadcast_prot_base = HybridWZBroadcastProtocol(worker_count, base_quantizer)
            elif proto_name == 'worker-side':
                from components.broadcast_components.broadcasting_process.WorkersideTraining import \
                    WorkersideTrainingProtocol
                broadcast_prot_base = WorkersideTrainingProtocol(worker_count, base_quantizer)
            elif proto_name == 'simple':
                from components.broadcast_components.broadcasting_process.SingleTimeTrainingProtocol import \
                    SingleTimeTrainingProtocol
                broadcast_prot_base = SingleTimeTrainingProtocol(worker_count, base_quantizer)
            elif proto_name == 'balanced_hybrid':
                from components.broadcast_components.broadcasting_process.HybridBalanced import BalancedHybridProtocol
                broadcast_prot_base = BalancedHybridProtocol(worker_count, base_quantizer)
            elif proto_name == 'no_proto_only_global':
                from components.broadcast_components.broadcasting_process.OnlyGlobalModel import OnlyGlobalModel
                broadcast_prot_base = OnlyGlobalModel(worker_count, base_quantizer)
            elif proto_name == 'worker-side-with-error-accum':
                from components.broadcast_components.broadcasting_process.WorkersideTrainingWithAccumError import \
                    WorkersideTrainingWithAccumErrorProtocol
                broadcast_prot_base = WorkersideTrainingWithAccumErrorProtocol(worker_count, base_quantizer)
            elif proto_name == 'non-wz-cancer':
                from components.broadcast_components.WZ_models.Learned_non_wz_quantizer import LearnedNonWZQuantizer
                base_quantizer = LearnedNonWZQuantizer(wz_model, train_sample_size=300_000,
                            count_side_info_data=0, enable_progress_bar=False, user_logger=user_logger)
                from components.broadcast_components.broadcasting_process.CancerProt import CancerProtocol
                broadcast_prot_base = CancerProtocol(worker_count, base_quantizer,
                                                     binary_quantization=('1bit' in proto_name),
                                                     small_update=('small-update' in proto_name))
            elif 'cancer' in proto_name:
                from components.broadcast_components.broadcasting_process.CancerProt import CancerProtocol
                broadcast_prot_base = CancerProtocol(worker_count, base_quantizer,
                                                     binary_quantization=('1bit' in proto_name),
                                                     small_update=('small-update' in proto_name))
            elif 'conventional_' in proto_name:
                from components.broadcast_components.broadcasting_process.ConventionalQuantizerProtocol import \
                    RoundDSCProtocol, SignDSCProtocol, RoundBasicProtocol, SignBasicProtocol
                if 'dsc' in proto_name:
                    temp = [RoundDSCProtocol, SignDSCProtocol][int('sign' in proto_name)]
                else:
                    temp = [RoundBasicProtocol, SignBasicProtocol][int('sign' in proto_name)]
                broadcast_prot_base = temp(worker_count, base_quantizer)
            else:
                raise ValueError(f'Unknown protocol: {proto_name}')

            if args.no_global_quant != False:
                broadcast_prot_base.no_global_quantization = True
            else:
                # Add global error correction quantizer if global quantization is not disabled
                temp = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=1,
                                             bins_per_plane=8, lr=1e-3, marginal=True).to(torch.float32)
                temp.load_state_dict(torch.load(f'{data_folder}/basicRNN_global_correction.pt', map_location='cpu'))
                temp = QuantizerWithDataPrep(temp, train_sample_size=200_000, count_side_info_data=0,
                                             enable_progress_bar=True, vec_slices=None)
                broadcast_prot_base.global_wz_basic_quantizer = temp

            if args.no_outlier_handling != False:
                wz_model.no_outlier_normalization = True

            if args.no_normalization != False:
                wz_model.no_normalization = True

            if args.force_no_dsc != False:
                broadcast_prot_base.force_no_sw = True

            broadcast_prot = BroadcastMetricGatheringUtilities(broadcast_prot_base, user_logger=user_logger)

        # *****************

        model = ResNetPLModel(num_classes=num_classes, resnet_version='resnet18', lr=0.001, )
        model.load_state_dict(torch.load(f'{data_folder}/resnet18_svhn.pth', map_location='cpu'))

        # *****************
        sim = FLSimulator(
            pl_model=model, num_agents=worker_count, communication_rounds=80, client_epochs_per_round=20,
            batch_size=batch_size, dataset_train=dataset[0], dataset_test=dataset[1],
            aggregation_method='fedavg', non_iid_sampling=False, user_logger=user_logger)
        # ****
        sim.run_simulation(broadcast_prot)


    if args.protocol not in proto_combo:
        gc.collect()
        torch.cuda.empty_cache()
        f(args.protocol)
    else:
        for i in range(len(args.protocol)):
            try:
                f(proto_choices[int(args.protocol[i:i + 1])])
            except Exception as e:
                print(f'\n     ***************\n'
                      f'Error in protocol {proto_choices[int(args.protocol[i:i + 1])]}: {e}\n'
                      f'\n     ***************\n')
                # print the stack trace
                traceback.print_exc()