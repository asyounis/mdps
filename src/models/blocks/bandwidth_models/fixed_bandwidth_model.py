# Python Imports
import time

# Package Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

# Ali Package Import
from general_ml_framework.utils.config import *


class FixedBandwith(nn.Module):
    def __init__(self, configs):
        super(FixedBandwith, self).__init__()

        # Get the parameters that we need
        starting_bandwidths = get_mandatory_config("starting_bandwidths", configs, "configs")
        min_bandwidths = get_mandatory_config("min_bandwidths", configs, "configs")

        # convert to a tensor
        self.min_bandwidths = torch.FloatTensor(min_bandwidths)

        # Create the parameter that we will learn
        log_bands = torch.FloatTensor(np.log(starting_bandwidths))
        self.log_bandwidths = nn.parameter.Parameter(log_bands)

    def forward(self, particles):

        if(particles is None):
            device = self.log_bandwidths.device
            batch_size = 1

        else:

            # Make sure the state dimensions match the dimensions of the particles we have
            assert(particles.shape[-1] == self.log_bandwidths.shape[-1])

            # Get some info 
            device = particles.device
            batch_size = particles.shape[0]

        # Return the exponential to make sure the bandwidth
        return_bands = torch.exp(self.log_bandwidths)

        # Move the min band to the correct device
        if(self.min_bandwidths.device != device):
            self.min_bandwidths = self.min_bandwidths.to(device)

        # Apply the min band
        return_bands = return_bands + self.min_bandwidths

        # Tile the log_bandwidths (aka copy the log_bandwidths so we have 1 bandwidth per batch)
        return_bands = torch.tile(return_bands, (batch_size, 1))

        return return_bands


