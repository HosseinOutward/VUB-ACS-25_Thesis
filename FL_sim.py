import gzip
from typing import Optional, Callable, Dict, List
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.optim import Optimizer
from torchvision.datasets import VisionDataset
from torchvision.datasets.samplers import DistributedSampler


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


def fedavg(state_dicts, num_samples):
    total = sum(num_samples)
    global_state = {}
    for k in state_dicts[0]:
        global_state[k] = sum(sd[k] * (n / total) for sd, n in zip(state_dicts, num_samples))
    return global_state

def report_metric(model, dataloader, name=None, rank=0):
    with torch.no_grad():
        model.eval()
        if name=='train': dataloader.sampler.rank = rank

        temp = [model.step_with_custom_logs(None, batch, 0)
                    for i, batch in enumerate(dataloader)]
        temp = list(zip(*[(t.detach().cpu().numpy() for t in tt) for tt in temp]))

        loss_log = np.mean(temp[0])
        auc_log = np.mean(temp[1])
        print(f"         {name} loss: {loss_log:.3f}, {name} auc: {auc_log:.3f}")


class FederatedModelWrapper(pl.LightningModule):
    def __init__(self, record_gradients: bool = False):
        super(FederatedModelWrapper, self).__init__()
        self.record_gradients = record_gradients
        self.latest_parameters = None
        self.worker_id = None
        self.curr_round=None

    def on_before_optimizer_step(self, optimizer: Optimizer):
        """
        This is the hook that is called before the optimizer step.
        """
        self.latest_parameters = []
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            self.latest_parameters.append([name, param.grad.cpu().detach().numpy()])

        # Save the latest gradients as a compressed file
        if self.record_gradients:
            filename = (f"experiments/exp_data/gradients_resnet/"
                        f"worker_{self.worker_id}_round_{self.curr_round}_epoch_{self.current_epoch}"
                        f"_batch_{self.batch_idx}_gradients.pt.gz")
            with gzip.open(filename, "wb") as f:
                torch.save(self.latest_parameters, f)

        return super().on_before_optimizer_step(optimizer)

    def clone(self, copy=None):
        copy.record_gradients = self.record_gradients
        return copy


class Agent:
    def __init__(self, agent_id: int, model: FederatedModelWrapper,
                 shared_train_loader: torch.utils.data.DataLoader,
                 pre_send_preprocess: Optional[Callable[[Dict], Dict]] = None):
        self.agent_id = agent_id
        self.pre_send_preprocess = pre_send_preprocess
        self.local_data_train = shared_train_loader
        self.data_size = len(shared_train_loader.sampler)

        self.local_model = model.clone()
        self.local_model.worker_id = agent_id

        self.last_train_param_change = None

    def train(self, epochs, round_s):
        start_weight = {k: v.clone() for k, v in self.local_model.state_dict().items()}

        # set up the DataLoader parameters for this agent
        self.local_data_train.sampler.rank = self.agent_id
        self.local_model.curr_round = round_s

        # train the model
        trainer = Trainer(
            max_epochs=epochs,
            accelerator='cuda',
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False,)
        trainer.fit(self.local_model, self.local_data_train)

        # remove model from GPU memory
        self.local_model.eval()

        # calculate the change in parameters after training
        temp = self.local_model.state_dict()
        self.last_train_param_change = {
            k: temp[k] - start_weight[k].to(temp[k].device).to(temp[k].dtype)
            for k in temp}

    def get_accum_grads(self):
        # Return the accumulated gradients after processing it
        res = self.last_train_param_change
        if self.pre_send_preprocess is not None:
            res = self.pre_send_preprocess(self.last_train_param_change)
        return res


class FLSimulator:
    def __init__(self, num_agents: int, communication_rounds: int, client_epochs_per_round: int,
                 batch_size: int, dataset_train: VisionDataset, dataset_test: VisionDataset,
                 pl_model: FederatedModelWrapper, aggregation_method='fedavg',
                 iid_data=False, pre_send_process=None,
                 server_rec_process: Optional[Callable[[List[Dict]], List[Dict]]] = None):
        assert iid_data is True, "Currently only IID data is supported"

        self.num_agents = num_agents
        self.communication_rounds = communication_rounds
        self.client_epochs_per_round = client_epochs_per_round
        self.aggregation_method = aggregation_method

        # Create a shared train DataLoader outside agents
        self.test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True, persistent_workers=True)

        # todo: custom sampler for non-IID data
        # todo: fix how sample counts (len) is calulated for fedavg
        sampler = DistributedSampler(dataset_train,
                        num_replicas=num_agents, rank=-1, shuffle=True)
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

    def _aggregate_models(self):
        if self.aggregation_method != 'fedavg':
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")

        # fedavg
        grads = fedavg(
            self.server_rec_process([agent.get_accum_grads() for agent in self.agents]),
            [agent.data_size for agent in self.agents]
        )
        state_dict = self.global_model.state_dict()
        for k in grads:
            state_dict[k] += grads[k].to(state_dict[k].dtype)
        self.global_model.load_state_dict(state_dict)

    def run_simulation(self):
        # cycle through communication rounds
        for round_s in range(self.communication_rounds):
            print(f"\nround {round_s + 1}/{self.communication_rounds}"
                  " --------------------")

            # report the global model loss and accuracy on entire test set
            print("  - reporting global model metrics")
            report_metric(self.global_model, self.test_loader, 'test')
            report_metric(self.global_model, self.shared_train_loader, 'train')

            # Train each agent for the number of epochs
            for i, agent in enumerate(self.agents):
                print(f"     > training agent {i + 1}/{len(self.agents)}")
                agent.train(epochs=self.client_epochs_per_round, round_s=round_s)
                report_metric(agent.local_model, self.test_loader, 'test')
                report_metric(agent.local_model, self.shared_train_loader, 'train', rank=i)

            # Aggregate pl_models from all agents
            self._aggregate_models()
            for agent in self.agents:
                agent.local_model.load_state_dict(self.global_model.state_dict())

        print("\nfinal global model metrics")
        report_metric(self.global_model, self.test_loader, 'test')
        report_metric(self.global_model, self.shared_train_loader, 'train')
