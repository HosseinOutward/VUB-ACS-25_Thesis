import gc
import gzip
import os
from types import LambdaType
from pytorch_lightning.loggers import CSVLogger
from typing import Optional, Callable, Dict, List, Iterator, Any
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.optim import Optimizer
from torch.utils.data import Sampler
from torchvision.datasets import VisionDataset


class FederatedModelWrapper(pl.LightningModule):
    def __init__(self,
                 applied_on_grads_before_optim: Callable[[pl.LightningModule, Dict], None] = lambda x, y: None):
        super(FederatedModelWrapper, self).__init__()
        self.applied_on_grads_before_optim = applied_on_grads_before_optim
        self.args_for_f_on_grad = {k: None for k in
                                   ['worker_id', 'curr_round', 'current_epoch', 'batch_idx']}

        self.accu_param_grads = None

    def training_step(self, batch, batch_idx):
        self.args_for_f_on_grad['batch_idx'] = batch_idx

    def on_before_optimizer_step(self, optimizer: Optimizer):
        self.args_for_f_on_grad['current_epoch'] = self.current_epoch

        for name, param in self.named_parameters():
            if not param.requires_grad or param.grad is None: continue
            self.accu_param_grads[name] += param.grad.detach().clone()

        self.applied_on_grads_before_optim(self, **self.args_for_f_on_grad)

        return super().on_before_optimizer_step(optimizer)

    def on_train_start(self):
        self.accu_param_grads = {
            k: torch.zeros_like(v.data, device=v.device)
            for k, v in self.named_parameters() if v.requires_grad
        }
        super().on_train_start()

    def clone(self, copy=None):
        copy.load_state_dict(self.state_dict())
        copy.args_for_f_on_grad = self.args_for_f_on_grad
        copy.applied_on_grads_before_optim = self.applied_on_grads_before_optim

        return copy


# def dirichlet_split(dataset, num_clients, alpha=0.5):
#     labels = np.array([y for _, y in dataset])
#     n_classes = labels.max() + 1
#     client_indices = [[] for _ in range(num_clients)]
#
#     for c in range(n_classes):
#         idx_c = np.where(labels == c)[0]
#         np.random.shuffle(idx_c)
#         proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
#         proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
#         splits = np.split(idx_c, proportions)
#         for client_id, idx in enumerate(splits):
#             client_indices[client_id].extend(idx.tolist())
#
#     return [torch.utils.data.Subset(dataset, idxs) for idxs in client_indices]


def report_metric(model, dataloader, name=None, rank='ALL'):
    # todo utilize self.log with logger instead of this nonsense
    print(f"         {name} loss: ", end='')

    org_logging_dis_flag = model.logging_disabled
    model.logging_disabled = True

    if name == 'train':
        print(f'({rank=}) ', end='')
        dataloader.sampler.set_agent_partition(rank)

    with torch.no_grad():
        model.eval()
        model.to('cuda')

        temp = [
            model.step_with_custom_logs(
                None, (batch[0].to('cuda'), batch[1].to('cuda')), 0)
            for i, batch in enumerate(dataloader)
        ]
        temp = list(zip(*[(t.detach().cpu().numpy() for t in tt) for tt in temp]))

        loss_log = np.mean(temp[0])
        auc_log = np.mean(temp[1])
        print(f"{loss_log:.3f}, {name} auc: {auc_log:.3f}")

        model.to('cpu')

    model.logging_disabled = org_logging_dis_flag


def fedavg(grad_dict_per_agent, sample_size_per_agent):
    print("Aggregating gradients using FedAvg...")

    total_sample_count = sum(sample_size_per_agent)
    agent_weights = map(lambda x: x / total_sample_count, sample_size_per_agent)

    first_worker_grads = grad_dict_per_agent[0]
    layer_with_grad_names = first_worker_grads.keys()
    aggregated_grads = {
        k: torch.zeros_like(first_worker_grads[k], device=first_worker_grads[k].device)
        for k in layer_with_grad_names
    }

    for worker_grad_dict, weight in zip(grad_dict_per_agent, agent_weights):
        for k, v_grad_tensor in worker_grad_dict.items():
            aggregated_grads[k].add_(v_grad_tensor.to(aggregated_grads[k].device), alpha=weight)

    return aggregated_grads


# todo: See issue in file - generating samples while training if needed
#       experiments/resnet_parameter_corr_between_worker.py, line 9
# todo: instead of simply saving, run a custom function on the gradients
def save_grads_f_applied_on_grads(fl_model: FederatedModelWrapper,
                                  save_folder_path, worker_id, curr_round, current_epoch, batch_idx):
    file_path = (save_folder_path +
                 f"worker_{worker_id}_round_"
                 f"{curr_round}_epoch_{current_epoch}"
                 f"_batch_{batch_idx}_gradients.pt.gz")

    with gzip.open(file_path, "wb", compresslevel=1) as f:
        latest_parameters_grad = [
            param.grad.detach().clone()
            for _, param in fl_model.named_parameters()
            if param.requires_grad
        ]
        torch.save(latest_parameters_grad, f)

    # Save the names of the parameters
    if not os.path.exists(save_folder_path + '_grad_namings.txt'):
        with open(save_folder_path + '_grad_namings.txt', "w") as f:
            latest_parameters_grad_names = [
                name
                for name, _ in fl_model.named_parameters()
                if _.requires_grad
            ]
            for name in latest_parameters_grad_names:
                f.write(name + "\n")


# ---------------------------------------------------------------------------------
# todo single dataloader causes thread lock overhead (40% of the run time)
#  a change of offset, even once, causes thread to check the entire dataset each time
class CustomSampler(Sampler):
    def __init__(self, dataset_len, partitions_count, shuffle_whole,
                 shuffle_in_partition, non_iid_flag=False, seed=42):
        assert non_iid_flag is False, "Currently only IID data is supported"
        super().__init__()

        # Store only dataset length, not dataset reference
        self.dataset_len = dataset_len
        self.partitions_count = partitions_count
        self.shuffle_whole = shuffle_whole
        self.shuffle_in_partition = shuffle_in_partition
        self.seed = seed
        self.offset = 0

        # Pre-compute partition sizes to avoid recalculation
        base_size = self.dataset_len // self.partitions_count
        remainder = self.dataset_len % self.partitions_count
        self.size_of_partition = {
            i: base_size + (1 if i < remainder else 0)
            for i in range(self.partitions_count)
        }

    def get_whole_shuffle_idx(self):
        if self.shuffle_whole:
            rng = np.random.RandomState(self.seed)
            return rng.permutation(self.dataset_len)
        else:
            return np.arange(self.dataset_len)

    def __iter__(self):
        shuffle_whole_idx = self.get_whole_shuffle_idx()
        if self.offset == 'ALL':
            return iter(shuffle_whole_idx)

        # Generate indices for the current partition
        indices = shuffle_whole_idx[self.offset::self.partitions_count]

        # Shuffle within partition using numpy (faster and pickles better)
        if self.shuffle_in_partition:
            rng = np.random.RandomState(self.seed + self.offset)
            rng.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        if self.offset == 'ALL':
            return self.dataset_len
        return self.size_of_partition[self.offset]


# ---------------------------------------------------------------------------------
class Agent:
    def __init__(self,
                 agent_id: int,
                 local_data_size: int,
                 pre_send_preprocess: Optional[Callable[[Dict, int], Dict]] = None):
        self.local_model: FederatedModelWrapper | None = None

        self.agent_id = agent_id

        self.pre_send_preprocess = pre_send_preprocess

        self.data_size = local_data_size

    def train(self, train_dataloader, test_dataloader, epochs, round_s):
        assert isinstance(self.local_model, FederatedModelWrapper), \
            "Local model is not set. Call set_local_models() before training."

        # set up the DataLoader parameters for this agent
        self.local_model.args_for_f_on_grad['curr_round'] = round_s

        # train the model
        logger = False
        if not self.local_model.logging_disabled:
            logger = CSVLogger(save_dir="../experiments/exp_data/run_stats",
                               name=f"agent_{self.agent_id}_round_{round_s}", )

        trainer = Trainer(
            max_epochs=epochs,
            accelerator='cuda',
            logger=logger,
            log_every_n_steps=1,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False, )
        trainer.fit(self.local_model, train_dataloader, test_dataloader)

        # remove model from GPU memory
        self.local_model.eval()
        self.local_model.to('cpu')

    def get_worker_broadcast(self):
        grad_dict = self.local_model.accu_param_grads
        broadcast_data = self.pre_send_preprocess(grad_dict, self.agent_id)
        return broadcast_data


# ---------------------------------------------------------------------------------
class FLSimulator:
    def __init__(self, num_agents: int,
                 communication_rounds: int, client_epochs_per_round: int,
                 batch_size: int, dataset_train: VisionDataset, dataset_test: VisionDataset,
                 pl_model: FederatedModelWrapper, aggregation_method='fedavg', non_iid_sampling=False,
                 pre_send_process: Optional[Callable[[Dict, int], Any]] = lambda x, i: x,
                 server_rec_process: Optional[Callable[[Any, int, int, Dict, List], Dict]] = lambda x, i, j, s, z: x):
        if LambdaType in [type(pre_send_process), type(server_rec_process)]:
            assert type(pre_send_process) is LambdaType and type(server_rec_process) is LambdaType

        self.global_model = pl_model

        self.num_agents = num_agents
        self.communication_rounds = communication_rounds
        self.client_epochs_per_round = client_epochs_per_round
        self.aggregation_method = aggregation_method

        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.batch_count = batch_size

        self.server_optimizer = self.global_model.configure_optimizers()

        self.server_rec_process = server_rec_process

        self.train_sampler = CustomSampler(len(self.dataset_train), self.num_agents,
                False, False, non_iid_flag=non_iid_sampling, )

        self.model_shape_dict = {k: v.shape
                for k, v in self.global_model.named_parameters() if v.requires_grad}

        self.agents = [Agent(
            agent_id, self.train_sampler.size_of_partition[agent_id], pre_send_process
        ) for agent_id in range(num_agents)]

    def _set_local_models(self):
        # todo find a way for the optimizer configurations to work after grad aggregation
        for ag in self.agents:
            ag.local_model = self.global_model.clone()
            ag.local_model.args_for_f_on_grad['worker_id'] = ag.agent_id

    # todo doesnt work
    def _aggregate_models(self, grad_dict_per_agent: Optional[List[Dict]] = None):
        sample_size_per_agent = [agent.data_size for agent in self.agents]

        if self.aggregation_method != 'fedavg':
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")
        aggregated_grads = fedavg(grad_dict_per_agent, sample_size_per_agent)

        self.server_optimizer.zero_grad()
        for name, param in self.global_model.named_parameters():
            if param.requires_grad:
                assert name in aggregated_grads, \
                    f"Parameter {name} not found in aggregated gradients"

                grad_to_apply = aggregated_grads[name].detach().clone().to(param.device)
                param.grad = grad_to_apply
        self.server_optimizer.step()

    def do_train_global_model_and_set_local_model(self, shared_train_loader, num_epochs):
        print(f'training global model on all data before simulation starts for {num_epochs} epochs')
        # self.shared_train_loader.sampler.offset = 'ALL'
        self.train_sampler.set_agent_partition('ALL')
        trainer = Trainer(
            max_epochs=num_epochs,
            accelerator='cuda',
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False, )
        trainer.fit(self.global_model, shared_train_loader)

        self._set_local_models()

    def run_simulation(self, post_training_report=True, pre_training_global_epochs=5):
        shared_train_loader = torch.utils.data.DataLoader(
            self.dataset_train, batch_size=self.batch_count, sampler=self.train_sampler,
            num_workers=10, persistent_workers=True)

        shared_test_loader = torch.utils.data.DataLoader(
            self.dataset_test, batch_size=self.batch_count * 3, shuffle=False,
            num_workers=2, persistent_workers=True)

        # pre_training_global_epochs -----------------------------------------------
        if pre_training_global_epochs != 0:
            self.do_train_global_model_and_set_local_model(
                shared_train_loader, num_epochs=pre_training_global_epochs)

        # cycle through communication rounds -----------------------------------------------
        for round_s in range(self.communication_rounds):
            print(f"\nround {round_s + 1}/{self.communication_rounds} --------------------")

            # ------------- report the global model loss and accuracy on entire test set
            print("  - reporting global model metrics")
            report_metric(self.global_model, shared_test_loader, 'test')
            report_metric(self.global_model, shared_train_loader, 'train')

            # ------------- Train each agent for the number of epochs
            grad_dict_per_agent = []
            self._set_local_models()
            for ag_id, ag in enumerate(self.agents):
                print(f"     > training agent {ag_id + 1}/{len(self.agents)}")

                # self.shared_train_loader.sampler.offset = ag_id
                self.train_sampler.set_agent_partition(ag_id)
                ag.train(shared_train_loader, shared_test_loader,
                         epochs=self.client_epochs_per_round, round_s=round_s)

                encoded_ag_broadcast = ag.get_worker_broadcast()

                decoded_agent_broadcast = self.server_rec_process(
                    encoded_ag_broadcast,
                    ag_id, self.num_agents, self.model_shape_dict,
                    grad_dict_per_agent, )

                grad_dict_per_agent.append(decoded_agent_broadcast)

                if post_training_report:
                    report_metric(ag.local_model, shared_test_loader, 'test')
                    report_metric(ag.local_model, shared_train_loader, 'train', rank=ag_id)

            # ------------- Aggregate pl_models from all agents
            self._aggregate_models(grad_dict_per_agent)

        self._set_local_models()

        # final global model report -----------------------------------------------
        print("\nfinal global model metrics")
        report_metric(self.global_model, shared_test_loader, 'test')
        report_metric(self.global_model, shared_train_loader, 'train')

        del shared_train_loader
        gc.collect()
        torch.cuda.empty_cache()
