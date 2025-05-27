from typing import Optional, Callable, Dict, List
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchvision.datasets import VisionDataset


def dirichlet_split(dataset, num_clients, alpha=0.5):
    labels = np.array([y for _, y in dataset])
    n_classes = labels.max() + 1
    client_indices = [[] for _ in range(num_clients)]

    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, proportions)
        for client_id, idx in enumerate(splits):
            client_indices[client_id].extend(idx.tolist())

    return [torch.utils.data.Subset(dataset, idxs) for idxs in client_indices]


def partition_per_worker(dataset_train, agent_id, num_agents, shuffled_ids, iid_data):
    if iid_data is False:
        raise
        return dirichlet_split(dataset_train, num_agents, agent_id)

    else:
        assert shuffled_ids is not None, "shuffled_ids must be provided for IID partitioning"

        res = []
        data_per_agent = len(dataset_train) // num_agents
        start_idx = agent_id * data_per_agent
        end_idx = start_idx + data_per_agent
        res.append(torch.utils.data.Subset(
            dataset_train, shuffled_ids[start_idx:end_idx]))
        return res


def fedavg(state_dicts, num_samples):
    total = sum(num_samples)
    global_state = {}
    for k in state_dicts[0]:
        # weighted sum over clients
        global_state[k] = sum(sd[k] * (n / total) for sd, n in zip(state_dicts, num_samples))
    return global_state


class Agent:
    def __init__(self, agent_id: int, model: pl.LightningModule, batch_size,
                 dataset_train, dataset_test, pre_send_preprocess=None):
        self.agent_id = agent_id
        self.local_model = model.clone()
        self.pre_send_preprocess = pre_send_preprocess
        self.local_data_train = dataset_train
        self.local_data_test = dataset_test

        self.train_loader = torch.utils.data.DataLoader(
            self.local_data_train, batch_size=batch_size, shuffle=True,
            num_workers=8, pin_memory=True, persistent_workers=True)

        self.start_weight = None

    def train(self, epochs):
        self.start_weight = {k: v.clone() for k, v in self.local_model.state_dict().items()}

        trainer = Trainer(
            max_epochs=epochs,
            accelerator='gpu',
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        trainer.fit(self.local_model, self.train_loader)  # , self.test_loader)

    def get_accum_grads(self):
        res = self.local_model.state_dict()
        res = {k: res[k] - self.start_weight[k].to(res[k].device).to(res[k].dtype)
               for k in res}
        if self.pre_send_preprocess is not None:
            res = self.pre_send_preprocess(res)
        return res


class FLSimulator:
    def __init__(self, num_agents: int, communication_rounds: int, client_epochs_per_round: int,
                 batch_size: int, dataset_train: VisionDataset, dataset_test: VisionDataset,
                 pl_model: pl.LightningModule, aggregation_method='fedavg',
                 iid_data=False, pre_send_process=None,
                 server_rec_process: Optional[Callable[[List[Dict]], List[Dict]]] = None):
        self.num_agents = num_agents
        self.communication_rounds = communication_rounds
        self.client_epochs_per_round = client_epochs_per_round
        self.aggregation_method = aggregation_method

        self.test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=batch_size, shuffle=False,
                num_workers=2, pin_memory=True, persistent_workers=True)

        self.server_rec_process = server_rec_process \
            if server_rec_process is not None else lambda x: x

        shuffled_ids = np.random.permutation(len(dataset_train))

        self.global_model = pl_model

        self.agents = [Agent(
            agent_id, self.global_model, batch_size,
            *partition_per_worker(dataset_train, agent_id, num_agents, shuffled_ids, iid_data),
            pre_send_process
        ) for agent_id in range(num_agents)]

    def _aggregate_models(self):
        if self.aggregation_method != 'fedavg':
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")

        # fedavg
        grads = fedavg(
            self.server_rec_process([agent.get_accum_grads() for agent in self.agents]),
            [len(agent.local_data_train) for agent in self.agents]
        )
        state_dict = self.global_model.state_dict()
        for k in grads:
            state_dict[k] += grads[k].to(state_dict[k].dtype)

        self.global_model.load_state_dict(state_dict)

    def run_simulation(self):
        for round_s in range(self.communication_rounds):
            print(f"round {round_s + 1}/{self.communication_rounds}")

            # report the global model loss and accuracy on entire test set
            with torch.no_grad():
                loss = np.average([self.global_model.validation_step(batch, i).detach().cpu().numpy()
                                   for i, batch in enumerate(self.test_loader)])
            print(f"         loss: {loss}")

            for agent in self.agents:
                agent.train(epochs=self.client_epochs_per_round)

            # Aggregate pl_models from all agents
            self._aggregate_models()

            # Update each agent's local model
            for agent in self.agents:
                agent.local_model.load_state_dict(self.global_model.state_dict())

