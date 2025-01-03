# Python Imports

# Module Imports
import torch

# General ML Framework Imports
from general_ml_framework.utils.config import *

# Project Imports
from metrics.metrics_base import MetricsBase

class PassThroughMetric(MetricsBase):
    def __init__(self, name, configs):
        super(PassThroughMetric, self).__init__(name, configs)

    def add_values(self, model_output, data):

        # Extract the relevant things
        loss = model_output["loss"]

        # Append save the value for later
        self.all_values.append(loss.unsqueeze(0))

    def get_aggregated_result(self):

        # Need to stack them into 1 tensor
        # There should be only 1 dim so this should work
        all_values = torch.cat(self.all_values)

        # Compute some aggregated results
        mean = torch.mean(all_values)
        std = torch.std(all_values)

        # Pack
        aggregated_results = dict()
        aggregated_results["name"] = self.name
        aggregated_results["mean"] = mean
        aggregated_results["std"] = std

        # Need to return a list since some metrics return more than 1 set of values
        # But in this case we only have 1 case
        return [aggregated_results]



