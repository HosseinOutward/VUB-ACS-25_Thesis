import numpy as np
import torch
import os
import pickle
import gzip
from components.other_utilities.models_to_train import ResNetPLModel
from components.FL_sim import FLSimulator
from components.other_utilities.datasets import FasterSVHN
from torchvision import transforms

# %%
w0_accu_grads_list = None
w1_accu_grads_list = None
res_per_step = None
batch_count = None
client_epochs_per_round = None

min_window_size = 10
def compute_corr(worker_id, round_number, accum_grads,):
    max_window_size = int(batch_count*client_epochs_per_round*1.2)
    flatten_vec = torch.concatenate([v.flatten().cpu() for v in accum_grads.values()]).to(torch.float16)
    if worker_id == 0:
        w0_accu_grads_list.append(flatten_vec)
        return
    elif worker_id == 1:
        w1_accu_grads_list.append(flatten_vec)

    if len(w1_accu_grads_list) < min_window_size:
        return

    if len(w1_accu_grads_list) > max_window_size:
        # replace the oldest vector with none to save memory
        w0_accu_grads_list[len(w1_accu_grads_list)-(max_window_size+1)] = None
        w1_accu_grads_list[-(max_window_size+1)] = None

    temp = len(w1_accu_grads_list) - min(len(w1_accu_grads_list), max_window_size)
    vectors_to_check = torch.stack([
        torch.stack(w0_accu_grads_list[temp:len(w1_accu_grads_list)]).T,
        torch.stack(w1_accu_grads_list[temp:]).T,
    ])
    vectors_to_check = torch.transpose(vectors_to_check, 0, 1)

    # compute the correlation for each weight between workers
    corr_list = torch.func.vmap(torch.corrcoef)(vectors_to_check)[:, 0, 1].to(torch.float32)
    temp = torch.isnan(corr_list)
    summed_non_nan_corr = corr_list[~temp].abs().mean().cpu().numpy()

    res_per_step.append((summed_non_nan_corr, round_number, temp.sum()/len(corr_list)))

    # write the last computed correlation to disk
    # iterate between 2 files to avoid corruption
    file_name = f'corr_res_{len(res_per_step)%2}.pkl.gz'
    with gzip.open(file_name, 'wb', compresslevel=1) as f:
        pickle.dump(res_per_step, f)


class CorrResNetPLModel(ResNetPLModel):
    def on_before_optimizer_step(self, optimizer):
        super().on_before_optimizer_step(optimizer)

        worker_id = self.current_step_info['worker_id']
        curr_round = self.current_step_info['curr_round']
        current_epoch = self.current_step_info['current_epoch']
        batch_idx = self.current_step_info['batch_idx']

        # write the accum grad to disk
        accum_grad = {k: v.detach().clone() for k, v in self.accu_param_grads.items()}
        round_number = curr_round + (current_epoch + batch_idx / batch_count)/client_epochs_per_round
        compute_corr(worker_id, round_number, accum_grad)

    def clone(self, copy=None):
        if copy is None:
            copy = self.__class__(num_classes=self.num_classes, lr=self.lr, resnet_version=self.resnet_version)
        return super(ResNetPLModel, self).clone(copy=copy)


# %%
__batch_size__ = 7_500  # 3 batches per w = 73000/2worker/13000  -  np.ceil(len(dataset[0])/batch_size/2)
__epoch_count__ = 10
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    import logging
    import warnings
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logger = logging.getLogger('pytorch_lightning.utilities.rank_zero')
    logger.setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", "LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]")
    warnings.filterwarnings("ignore", "The 'train_dataloader' does")
    warnings.filterwarnings("ignore", "You defined a `validation_step` but")
    warnings.filterwarnings("ignore", "Starting from v1.9.0, `tensorboardX` has been")
    warnings.filterwarnings("ignore", "The number of training batches")
    warnings.filterwarnings("ignore", "`Trainer.fit` stopped: ")

    print('Starting the script...')
    # %%
    dataset = [
        FasterSVHN(
            root='../../data/SVHN', split=s,
            transform=transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4377, 0.4438, 0.4728],
                    std=[0.1980, 0.2010, 0.1970]
                ),
            ])
        ) for s in ['train', 'test']]

    # dataset = [torch.utils.data.Subset(d, list(range(100))) for d in dataset]
    # for i in range(10):
    #     for d in dataset:
    #         d.dataset.labels[i]=i

    # %%
    w0_accu_grads_list = []
    w1_accu_grads_list = []
    res_per_step = []


    batch_count = np.ceil(len(dataset[0])/__batch_size__/2)

    # %%
    model = CorrResNetPLModel(num_classes=10, resnet_version='resnet18', lr=0.005, )
    # model.load_state_dict(torch.load('data/resnet18_svhn.pth', map_location='cpu'))

    # *****************
    sim = FLSimulator(
        pl_model=model, num_agents=2, communication_rounds=50, client_epochs_per_round=__epoch_count__,
        batch_size=__batch_size__, dataset_train=dataset[0], dataset_test=dataset[1],
        aggregation_method='fedavg', non_iid_sampling=False, user_logger=None)
    # ****
    sim.run_simulation()

