# Python Imports
import time

# Package Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

# Ali Package Import
from general_ml_framework.models.base_model import BaseModel
from general_ml_framework.utils.config import *



class GaussianDynamicsModel(nn.Module):
    def __init__(self, configs):
        super(GaussianDynamicsModel, self).__init__()

        # Get the parameters that we need
        self.noise_scales = get_mandatory_config("noise_scales", configs, "configs")

        # Convert the noise scales to a tensor
        self.noise_scales = torch.FloatTensor(self.noise_scales)

        # Get the particle dimension types and check that they are valid
        self.particle_dimension_types = get_mandatory_config("particle_dimension_types", configs, "configs")
        for pdt in self.particle_dimension_types:
            assert((pdt == "RealNumbers") or (pdt == "Angles"))

    def forward(self, particles, actions):

        # Get some info
        device = particles.device
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]

        # Make sure its on the correct device
        if(self.noise_scales.device != device):
            self.noise_scales = self.noise_scales.to(device)

        # The new particles are basically the old ones with Gaussian noise added
        new_particles = particles.clone()

        # Add the actions 
        if(actions is not None):
            new_particles = new_particles + actions.unsqueeze(1)

        # Add in the noise
        noise = torch.randn(new_particles.shape, device=device)
        noise = noise * self.noise_scales.unsqueeze(0).unsqueeze(1)
        new_particles = new_particles + noise

        # keep angle dims between 0 and 2pi
        for p_dim in range(new_particles.shape[-1]):
            if(self.particle_dimension_types[p_dim] == "Angles"):
                new_particles[..., p_dim] = new_particles[..., p_dim] % (2*np.pi)

        return new_particles