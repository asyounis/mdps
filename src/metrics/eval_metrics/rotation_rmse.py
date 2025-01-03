# Python Imports

# Package Imports
import torch

# Ali Package Import
from general_ml_framework.utils.config import *

# Project Imports
from evaluators.metrics.rotation_mse import RotationMSE


class RotationRMSE (RotationMSE):
    def __init__(self, name, configs):
        super(RotationRMSE, self).__init__(name, configs)


    def get_aggregated_result(self):

        # Need to stack them into 1 tensor
        # There should be only 1 dim so this should work
        all_values = torch.cat(self.all_values)

        # Take the sqrt of all the values to compute the RMSE for each sequence
        # we will then take the average RMSE value for all the sequences to get the final value
        sqrt_all_values = torch.sqrt(all_values)

        # Compute some aggregated results
        mean = torch.mean(sqrt_all_values)
        std = torch.std(sqrt_all_values)

        # Pack
        aggregated_results = dict()
        aggregated_results["mean"] = mean
        aggregated_results["std"] = std

        return aggregated_results

