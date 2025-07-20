import gc
from typing import Optional, Dict, List, Any
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import Sampler
from torchvision.datasets import VisionDataset
from components.other_utilities.user_logger import UnifiedLoggingClass


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


#%%
class RawBroadcastProtocol:
    def __init__(self, *args, **kwargs):
        pass

    def to_server_prep_data_for_transfer(self, agent_id, grad_dict, encoder_data_sent_by_server):
        return (grad_dict, )

    def to_worker_prep_data_for_transfer(self, agent_id):
        return

    # %%
    def reconstruction_process(self, agent_id, worker_broadcast_data, worker_count, global_model_dims):
        return worker_broadcast_data[0]

    def model_transfer_to_worker_from_server(self, server_model_state_dict):
        return server_model_state_dict, None


#%%
class FederatedModelWrapper(pl.LightningModule):
    def __init__(self):
        super(FederatedModelWrapper, self).__init__()

        self.current_step_info: Dict[str, Any] =\
            {k: None for k in ['worker_id', 'curr_round', 'current_epoch', 'batch_idx']}
        self.accu_param_grads = None

    def applied_on_grads_before_optimizer(self, worker_id, curr_round, current_epoch, batch_idx, *args, **kwargs):
        # self.accu_param_grads
        pass

    def get_loss_etc(self, batch) -> (torch.Tensor, List[Any]):
        raise NotImplementedError

    def _log_metrics(self, loss, etc, stage: str):
        raise NotImplementedError

    def on_before_optimizer_step(self, *args, **kwargs):
        self.current_step_info['current_epoch'] = self.current_epoch

        for name, param in self.named_parameters():
            if not param.requires_grad or param.grad is None: continue
            self.accu_param_grads[name] += param.grad.detach().clone()

        self.applied_on_grads_before_optimizer(**self.current_step_info)

        return super().on_before_optimizer_step(*args, **kwargs)

    def on_train_start(self):
        self.accu_param_grads = {
            k: torch.zeros_like(v.data, device=v.device)
            for k, v in self.named_parameters() if v.requires_grad
        }
        super().on_train_start()

    def training_step(self, batch, batch_idx):
        self.current_step_info['batch_idx'] = batch_idx

        loss, etc = self.get_loss_etc(batch)
        self._log_metrics(loss, etc, stage='train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, etc = self.get_loss_etc(batch)
        self._log_metrics(loss, etc, stage='val')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0, weight_decay=0)
        return optimizer

    def clone(self, copy=None):
        copy.load_state_dict(self.state_dict())
        copy.current_step_info = self.current_step_info
        return copy


#%%
class CustomSampler(Sampler):
    def __init__(self, dataset_len, partitions_count, shuffle_whole,
                 shuffle_in_partition, replacement=True, non_iid_flag=False, seed=42):
        assert non_iid_flag is False, "Currently only IID data is supported"
        super().__init__()

        # Store only dataset length, not dataset reference
        self.dataset_len = dataset_len
        self.partitions_count = partitions_count
        self.shuffle_whole = shuffle_whole
        self.shuffle_in_partition = shuffle_in_partition
        self.replacement = replacement
        self.seed = seed
        self.offset = 0

        # Pre-compute partition sizes to avoid recalculation
        base_size = self.dataset_len // self.partitions_count
        remainder = self.dataset_len % self.partitions_count
        self.size_of_partition = {
            i: base_size + (1 if i < remainder else 0)
            for i in range(self.partitions_count)
        }

    def set_agent_partition(self, rank: int|str):
        if rank!=self.offset:
            self.offset = rank

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

        if self.replacement:
            indices = np.random.choice(indices, size=self.size_of_partition[self.offset], replace=True)

        return iter(indices)

    def __len__(self) -> int:
        if self.offset == 'ALL':
            return self.dataset_len
        return self.size_of_partition[self.offset]


#%%
class Agent:
    def __init__(self, agent_id: int, local_data_size: int,):
        self.local_model: FederatedModelWrapper | None = None

        self.agent_id = agent_id
        self.data_size = local_data_size

    def train(self, train_dataloader, test_dataloader, epochs, user_logger):
        assert isinstance(self.local_model, FederatedModelWrapper), \
            "Local model is not set. Call set_local_models() before training."

        # train the model
        trainer = Trainer(
            max_epochs=epochs,
            accelerator='gpu',
            logger=user_logger,
            log_every_n_steps=1,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False, )
        trainer.fit(self.local_model, train_dataloader, test_dataloader)

        # remove model from GPU memory
        self.local_model.eval()
        self.local_model.to('cpu')

    def get_worker_broadcast(self, encoder_data_sent_by_server, to_server_prep_data_for_transfer):
        grad_dict = self.local_model.accu_param_grads
        broadcast_data = to_server_prep_data_for_transfer(self.agent_id, grad_dict, encoder_data_sent_by_server)
        return broadcast_data


#%%
class FLSimulator:
    def __init__(self, num_agents: int, communication_rounds: int, client_epochs_per_round: int,
                 batch_size: int, dataset_train: VisionDataset, dataset_test: VisionDataset,
                 pl_model: FederatedModelWrapper, aggregation_method='fedavg',
                 non_iid_sampling=False, user_logger:UnifiedLoggingClass = None):

        self.user_logger = user_logger

        self.global_model = pl_model

        self.num_agents = num_agents
        self.communication_rounds = communication_rounds
        self.client_epochs_per_round = client_epochs_per_round
        self.aggregation_method = aggregation_method

        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.batch_count = batch_size

        self.server_optimizer = self.global_model.configure_optimizers()

        self.train_sampler = CustomSampler(len(self.dataset_train), self.num_agents,
                True, True, non_iid_flag=non_iid_sampling, )

        self.model_shape_dict = {k: v.shape
                for k, v in self.global_model.named_parameters() if v.requires_grad}

        self.agents = [Agent(
            agent_id, self.train_sampler.size_of_partition[agent_id],
        ) for agent_id in range(num_agents)]

    def _set_local_models(self, model_transfer_to_worker_from_server):
        for ag in self.agents:
            model = self.global_model.clone()
            temp = model_transfer_to_worker_from_server(model.state_dict())[0]
            model.load_state_dict(temp)
            ag.local_model = model
            ag.local_model.current_step_info['worker_id'] = ag.agent_id

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

    def _agent_training(self, ag_id, broadcast_prot, shared_train_loader, shared_test_loader):
        print(f"      > training agent {ag_id + 1}/{len(self.agents)}")

        gc.collect()
        torch.cuda.empty_cache()

        self.train_sampler.set_agent_partition(ag_id)

        # training
        logger = self.user_logger.get_agent_csv_logger() if self.user_logger else None
        self.agents[ag_id].train(shared_train_loader, shared_test_loader,
                                 epochs=self.client_epochs_per_round, user_logger=logger)

        # Broadcast the model to the server
        print(f"             + initiating broadcast")
        server_data_sent_to_worker = broadcast_prot.to_worker_prep_data_for_transfer(ag_id)

        current_ag_encoding_function = broadcast_prot.to_server_prep_data_for_transfer
        encoded_ag_broadcast = self.agents[ag_id].get_worker_broadcast(server_data_sent_to_worker, current_ag_encoding_function)

        decoded_agent_broadcast = broadcast_prot.reconstruction_process(
            ag_id, encoded_ag_broadcast, self.num_agents, self.model_shape_dict, )

        return decoded_agent_broadcast

    def _log_report(self, model, shared_train_loader, shared_test_loader, round_s, agent_id):
        print(f"         - train> loss: ", end='')

        self.train_sampler.set_agent_partition(agent_id if agent_id != 'global' else 'ALL')

        temp = self._get_model_metrics(model, shared_train_loader)
        train_metrics = {'loss':temp[0], 'auc':temp[1]}
        print(f"{train_metrics['loss']:.2f}, auc: {train_metrics['auc']:.2f}    |    ", end='')

        temp = self._get_model_metrics(model, shared_test_loader)
        test_metrics = {'loss':temp[0], 'auc':temp[1]}
        print(f"test> loss: {test_metrics['loss']:.2f}, auc: {test_metrics['auc']:.2f}")

        if self.user_logger:
            self.user_logger.fl_sim_log(round_s, agent_id, train_metrics, test_metrics)

    def _get_model_metrics(self, model:FederatedModelWrapper, dataloader):
        loss, auc = 0, 0
        data_count = 0
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                b_loss, (b_auc,) = model.get_loss_etc(batch)
                loss += b_loss.item() * len(batch[0])
                auc += b_auc * len(batch[0])
                data_count += len(batch[0])
        return loss/data_count, auc/data_count

    def run_simulation(self, broadcast_prot=RawBroadcastProtocol(), ):
        shared_train_loader = torch.utils.data.DataLoader(
            self.dataset_train, batch_size=self.batch_count, sampler=self.train_sampler,
            num_workers=10, persistent_workers=True)

        shared_test_loader = torch.utils.data.DataLoader(
            self.dataset_test, batch_size=self.batch_count * 3, shuffle=False,
            num_workers=5, persistent_workers=True)

        # cycle through communication rounds -----------------------------------------------
        for round_s in range(self.communication_rounds):
            print(f"\nround {round_s + 1}/{self.communication_rounds} --------------------")

            print("  - reporting global model metrics")
            self._log_report(self.global_model, shared_train_loader,
                             shared_test_loader, round_s, 'global')
            print("")

            # Train each agent for the number of epochs
            self._set_local_models(broadcast_prot.model_transfer_to_worker_from_server)
            current_round_grad_list = []
            for ag_id, ag in enumerate(self.agents):
                if self.user_logger:
                    self.user_logger.set_aid_rid(ag_id, round_s)

                ag.local_model.current_step_info['curr_round'] = round_s

                agent_grad = self._agent_training(
                    ag_id, broadcast_prot, shared_train_loader, shared_test_loader)
                # while True:
                #     try:
                #         agent_grad = self._agent_training(
                #             ag_id, round_s, broadcast_prot, shared_train_loader, shared_test_loader)
                #         break
                #     except Exception as e:
                #         # todo reset back to before training
                #         print(f"Error during training or broadcasting for agent {ag_id}: {e}")

                current_round_grad_list.append(agent_grad)
                self._log_report(ag.local_model, shared_train_loader, shared_test_loader, round_s, ag_id)

            # Aggregate pl_models from all agents
            self._aggregate_models(current_round_grad_list)

        self._set_local_models(broadcast_prot.model_transfer_to_worker_from_server)

        # final global model report -----------------------------------------------
        print("\nfinal global model metrics")
        self._log_report(self.global_model, shared_train_loader, shared_test_loader, 'end', 'global')

        #  -----------------------------------------------
        del shared_train_loader, shared_test_loader
        gc.collect()
        torch.cuda.empty_cache()


def _main_test():
    torch.set_float32_matmul_precision('high')
    import logging
    import warnings
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", "LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]")
    warnings.filterwarnings("ignore", "The 'train_dataloader' does")
    warnings.filterwarnings("ignore", "You defined a `validation_step` but")
    warnings.filterwarnings("ignore", "Starting from v1.9.0, `tensorboardX` has been")
    warnings.filterwarnings("ignore", "The number of training batches")
    warnings.filterwarnings("ignore", "`Trainer.fit` stopped: ")

    class SimpleModel(FederatedModelWrapper):
        def __init__(self):
            super().__init__()
            self.lr = 0.001
            self.model = torch.nn.Sequential(
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(), torch.nn.Linear(32, 1))
        def get_loss_etc(self, batch):
            x, y = batch
            logits = self.model(x)
            temp = (torch.sigmoid(logits), y)
            loss = torch.nn.functional.mse_loss(*temp)
            loss_detach = loss.detach()
            loss = loss if loss < 1000 else loss/loss_detach * 1000

            acc = torch.mean((torch.abs((
                torch.sigmoid(logits.detach()) > 0.5).float() - temp[1].float())<0.0001).float()).item()
            return loss, (acc, )
        def _log_metrics(self, loss, etc, stage: str):
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
            self.log(f"{stage}_acc", etc[0], on_step=True, on_epoch=False, prog_bar=False)
        def clone(self, copy=None):
            return super(SimpleModel, self).clone(copy=SimpleModel())

    dataset = torch.utils.data.TensorDataset(
        torch.concatenate([torch.randn(100000, 64)*0.5+0.5,torch.randn(100000, 64)*0.5-1]),
        torch.concatenate(        [torch.ones((100000,1)),         torch.zeros((100000,1))]),
    )
    dataset_test = torch.utils.data.TensorDataset(
        torch.concatenate([torch.randn(10000, 64)*0.5+0.5,torch.randn(10000, 64)*0.5-1]),
        torch.concatenate(        [torch.ones((10000,1)),         torch.zeros((10000,1))]),
    )

    # run a single epoch of training before starting the simulation
    model = SimpleModel()
    model.train()
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=True, num_workers=0, persistent_workers=False)
    trainer = Trainer(max_epochs=1, accelerator='gpu', enable_progress_bar=False,
                      enable_checkpointing=False, enable_model_summary=False)
    trainer.fit(model, train_loader)

    return model, dataset, dataset_test

if __name__ == "__main__":
    # Example usage of the FLSimulator with a simple model with no broadcast protocol or logger

    model, dataset, dataset_test = _main_test()
    # *****************
    sim = FLSimulator(
        pl_model=model, num_agents=5, communication_rounds=3, client_epochs_per_round=10,
        batch_size=10000, dataset_train=dataset, dataset_test=dataset_test,
        aggregation_method='fedavg', non_iid_sampling=False, user_logger=None)
    sim.run_simulation()
