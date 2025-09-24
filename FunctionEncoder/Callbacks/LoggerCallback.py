from typing import Union

import torch
from torch.utils.tensorboard import SummaryWriter

from FunctionEncoder import FunctionEncoder, BaseDataset
from FunctionEncoder.Callbacks.BaseCallback import BaseCallback
from stable_baselines3.common.logger import configure


class LoggerCallback(BaseCallback):

    def __init__(self,
                 model:FunctionEncoder,
                 testing_dataset:BaseDataset,
                 logdir: Union[str, None] = None,
                 prefix="test",
                 ):        
        super(LoggerCallback, self).__init__()
        self.logger = configure(logdir)
        self.total_epochs = 0
        self.model = model
        self.testing_dataset = testing_dataset

    def on_step(self, locals:dict):
        with torch.no_grad():
            function_encoder = self.model
            for phase in ["train", "eval"]:
                # sample testing data
                example_xs, example_ys, query_xs, query_ys, info = self.testing_dataset.sample(phase=phase)

                # compute representation
                y_hats = function_encoder.predict_from_examples(example_xs, example_ys, query_xs, method="least_squares")

                y_hats = y_hats.reshape(-1, y_hats.shape[-1])
                query_ys = query_ys.reshape(-1, query_ys.shape[-1])

                diff = query_ys - y_hats
                # L2 norm along the last axis (features)
                l2_error_per_sample = torch.norm(diff, p=2, dim=-1)  # [1800]
                
                # L1 norm along the last axis (features)
                l1_error_per_sample = torch.norm(diff, p=1, dim=-1)  # [1800]
                
                # Mean squared error along the last axis
                mse_error_per_sample = torch.mean(diff**2, dim=-1)  # [1800]
                
                # Root mean squared error along the last axis
                rmse_error_per_sample = torch.sqrt(mse_error_per_sample)  # [1800]
                
                # Total errors (sum across all samples)
                total_l2_error = torch.sum(l2_error_per_sample)
                total_l1_error = torch.sum(l1_error_per_sample)
                total_mse_error = torch.sum(mse_error_per_sample)
                total_rmse_error = torch.sum(rmse_error_per_sample)
                
                # Mean errors (mean across all samples)
                mean_l2_error = torch.mean(l2_error_per_sample)
                mean_l1_error = torch.mean(l1_error_per_sample)
                mean_mse_error = torch.mean(mse_error_per_sample)
                mean_rmse_error = torch.mean(rmse_error_per_sample)

                self.total_epochs += 1
                self.logger.record(f"{phase}_mse", mean_mse_error.item())
                self.logger.record(f"{phase}_l2_error", mean_l2_error.item())
                self.logger.record(f"{phase}_l1_error", mean_l1_error.item())
                self.logger.record(f"{phase}_rmse_error", mean_rmse_error.item())
                self.logger.record(f"{phase}_total_l2_error", total_l2_error.item())
                self.logger.record(f"{phase}_total_l1_error", total_l1_error.item())
                self.logger.record(f"{phase}_total_mse_error", total_mse_error.item())
                self.logger.record(f"{phase}_total_rmse_error", total_rmse_error.item())
            self.logger.dump(self.total_epochs)

