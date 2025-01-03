# Python Imports

# Module Imports
import torch

# General ML Framework Imports
from general_ml_framework.utils.config import *

# Project Imports
from utils.general import nll_loss_xyr
from utils.memory import move_data_to_device
from metrics.metrics_base import MetricsBase


class DimensionRegularizer(MetricsBase):
    def __init__(self, name, configs):
        super(DimensionRegularizer, self).__init__(name, configs)

        # get the different keys we need to compute this metric
        self.variable_to_use = get_mandatory_config("variable_to_use", configs, "configs")

        # Get the parameters for how we are to extract the particles
        self.dims_to_use = get_mandatory_config("dims_to_use", configs, "configs")
        assert(isinstance(self.dims_to_use, list))
        assert(len(self.dims_to_use) > 0)

        # Get the regularizer type
        self.regularizer_type = get_mandatory_config("regularizer_type", configs, "configs")
        assert(self.regularizer_type in ["L2"])

    def add_values(self, model_output, data):

        # Extract the variable we will be regularizing
        assert(self.variable_to_use in model_output)
        variable_to_regularize = model_output[self.variable_to_use]

        # stack it so it is [Batch, Seq, # Parts, Dim]
        variable_to_regularize = torch.stack(variable_to_regularize)
        variable_to_regularize = torch.permute(variable_to_regularize, [1, 0, 2, 3])

        # Extract the dims we are going to regularize
        extracted_dims = []
        for dim in self.dims_to_use:
            extracted_dims.append(variable_to_regularize[..., dim].unsqueeze(-1))
        extracted_dims = torch.cat(extracted_dims, dim=-1)


        # Compute the regularization
        if(self.regularizer_type == "L2"):

            # Compute the L2 for each batch, time-step and particle
            squared_distance = extracted_dims**2

            # Aggregate the dim (This is the sum in the L2 equation)
            squared_distance = torch.sum(squared_distance, dim=-1) 

            # Aggregate the particles (Note the mean instead of sum)
            squared_distance = torch.mean(squared_distance, dim=-1) 

            # Compute the mean over the sequence
            value = torch.mean(squared_distance, dim=-1)

        # Append save the value for later
        self.all_values.append(value)


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
