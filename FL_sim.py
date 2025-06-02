import gzip
import os

import lz4.frame
from typing import Optional, Callable, Dict, List, Iterator, Union
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.optim import Optimizer
from torch.utils.data import Sampler
from torchvision.datasets import VisionDataset


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


def report_metric(model, dataloader, name=None, rank=-1):
    print(f"         {name} loss: ", end='')

    if name == 'train':
        dataloader.sampler.offset = rank

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

        self.applied_on_grads_before_optim(self, self.args_for_f_on_grad)

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


# todo: See issue in file - generating samples while training if needed
#       experiments/resnet_parameter_corr_between_worker.py, line 9
# todo: instead of simply saving, run a custom function on the gradients
def save_grads_f_applied_on_grads(fl_model: FederatedModelWrapper, args_for_f_on_grad: Dict):
    if args_for_f_on_grad.get('save_folder_path') is None:
        raise ValueError("save_folder_path must be provided in args_for_f_on_grad")

    save_folder_path, worker_id, curr_round, current_epoch, batch_idx = \
        (args_for_f_on_grad[k] for k in
         ['save_folder_path', 'worker_id', 'curr_round', 'current_epoch', 'batch_idx'])

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


class CustomSampler(Sampler):
    def __init__(self, dataset, partitions_count, shuffle_whole,
                 shuffle_in_partition, non_iid_flag=False, seed=42):
        super().__init__()

        # todo: custom sampler for non-IID data
        assert non_iid_flag is False, "Currently only IID data is supported"

        self.dataset = dataset
        self.shuffle_in_partition = shuffle_in_partition

        self.offset = -1
        self.seed = seed

        self.shuffle_whole_idx = np.arange(len(self.dataset))
        if shuffle_whole:
            g = torch.Generator()
            g.manual_seed(self.seed)
            self.shuffle_whole_idx = torch.randperm(
                len(dataset)).numpy()

        self.partitions_count = partitions_count

        self.size_of_partition = {
            i: len(self.shuffle_whole_idx[i::self.partitions_count])
            for i in range(self.partitions_count)}

    def __iter__(self) -> Iterator[int]:
        # Generate indices for the current partition
        indices = self.shuffle_whole_idx[self.offset::self.partitions_count]

        # shuffle the indices within the partition
        if self.shuffle_in_partition:
            in_part_idx = torch.randperm(len(self)).numpy()
            indices = indices[in_part_idx]

        return iter(indices)

    def __len__(self) -> int:
        return self.size_of_partition[self.offset]


class Agent:
    def __init__(self, agent_id: int, model: FederatedModelWrapper,
                 shared_train_loader: torch.utils.data.DataLoader,
                 pre_send_preprocess: Optional[Callable[[Dict], Dict]] = None):
        self.agent_id = agent_id
        self.local_data_train = shared_train_loader

        self.pre_send_preprocess = pre_send_preprocess

        self.data_size = self.local_data_train.sampler.size_of_partition[agent_id]

        self.local_model = model.clone()
        self.local_model.args_for_f_on_grad['worker_id'] = agent_id

    def train(self, epochs, round_s):
        # set up the DataLoader parameters for this agent
        self.local_model.args_for_f_on_grad['curr_round'] = round_s

        self.local_data_train.sampler.offset = self.agent_id

        # train the model
        trainer = Trainer(
            max_epochs=epochs,
            accelerator='cuda',
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False, )
        trainer.fit(self.local_model, self.local_data_train)

        # remove model from GPU memory
        self.local_model.eval()
        self.local_model.to('cpu')

    def get_accum_grads(self):
        grad_dict = self.local_model.accu_param_grads

        if self.pre_send_preprocess is not None:
            grad_dict = self.pre_send_preprocess(grad_dict)

        return grad_dict


class FLSimulator:
    def __init__(self, num_agents: int,
                 communication_rounds: int, client_epochs_per_round: int,
                 batch_size: int, dataset_train: VisionDataset, dataset_test: VisionDataset,
                 pl_model: FederatedModelWrapper, aggregation_method='fedavg',
                 non_iid_flag=False, pre_send_process=None,
                 server_rec_process: Optional[Callable[[List[Dict]], List[Dict]]] = None):

        self.num_agents = num_agents
        self.communication_rounds = communication_rounds
        self.client_epochs_per_round = client_epochs_per_round
        self.aggregation_method = aggregation_method

        # Create a shared train DataLoader outside agents
        self.test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=batch_size*3, shuffle=False,
            num_workers=2, pin_memory=True, persistent_workers=True)

        sampler: CustomSampler = CustomSampler(dataset_train, num_agents,
                                               True, True, non_iid_flag=non_iid_flag)
        self.shared_train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, sampler=sampler,
            num_workers=10, pin_memory=True, persistent_workers=True)

        self.server_rec_process = server_rec_process \
            if server_rec_process is not None else lambda x: x

        self.global_model = pl_model

        self.agents = [Agent(
            agent_id, self.global_model,
            self.shared_train_loader, pre_send_process
        ) for agent_id in range(num_agents)]

        self.server_optimizer = self.global_model.configure_optimizers()

    # todo doesnt work
    def _aggregate_models(self):
        sample_size_per_agent = [agent.data_size for agent in self.agents]
        grad_dict_per_agent = [agent.get_accum_grads() for agent in self.agents]
        grad_dict_per_agent = self.server_rec_process(grad_dict_per_agent)

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

    def run_simulation(self):
        # cycle through communication rounds
        for round_s in range(self.communication_rounds):
            print(f"\nround {round_s + 1}/{self.communication_rounds}"
                  " --------------------")

            # report the global model loss and accuracy on entire test set
            print("  - reporting global model metrics")
            report_metric(self.global_model, self.test_loader, 'test')
            report_metric(self.global_model, self.shared_train_loader, 'train',
                          rank=np.random.randint(0, self.num_agents - 1))

            # Train each agent for the number of epochs
            for i, agent in enumerate(self.agents):
                print(f"     > training agent {i + 1}/{len(self.agents)}")
                agent.train(epochs=self.client_epochs_per_round, round_s=round_s)

                report_metric(agent.local_model, self.test_loader, 'test')
                report_metric(agent.local_model, self.shared_train_loader, 'train', rank=i)

            # Aggregate pl_models from all agents
            self._aggregate_models()

            # Load the aggregated global model to each agent's local model
            for agent in self.agents:
                # todo find a way for the optimizer configurations to work after grad aggregation
                agent.local_model.load_state_dict(self.global_model.state_dict())
                # agent.local_model = self.global_model.clone()

        print("\nfinal global model metrics")
        report_metric(self.global_model, self.test_loader, 'test')
        report_metric(self.global_model, self.shared_train_loader, 'train',
                      rank=np.random.randint(0, self.num_agents - 1))
