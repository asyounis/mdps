# Python Imports
import time

# Package Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision
import numpy as np

# Ali Package Import
from general_ml_framework.utils.config import *

# Project Imports
from utils.general import sample_xyr

class RetrievalForGaussianPFWeightModel(nn.Module):
    def __init__(self, configs):
        super(RetrievalForGaussianPFWeightModel, self).__init__()

        # Get the parameters to use for local map extraction
        particle_dims_to_use_for_local_map_extraction = get_mandatory_config("particle_dims_to_use_for_local_map_extraction", configs, "configs")
        self.particle_dim_translate_x = get_mandatory_config("translate_x", particle_dims_to_use_for_local_map_extraction, "particle_dims_to_use_for_local_map_extraction")
        self.particle_dim_translate_y = get_mandatory_config("translate_y", particle_dims_to_use_for_local_map_extraction, "particle_dims_to_use_for_local_map_extraction")
        self.particle_dim_rotation = get_mandatory_config("rotation", particle_dims_to_use_for_local_map_extraction, "particle_dims_to_use_for_local_map_extraction")

        # Extract the std
        obs_std = get_mandatory_config("obs_std", configs, "configs")
        self.obs_std_squared = obs_std**2


    def forward(self, input_dict):

        # Unpack
        particles = input_dict["particles"]
        encoded_global_map = input_dict["encoded_global_map"]
        encoded_observation = input_dict["encoded_observations"]
        unnormalized_resampled_particle_log_weights = input_dict["unnormalized_resampled_particle_log_weights"]

        # Get the map encodings
        map_encodings, _ = sample_xyr(encoded_global_map, particles[..., 0:2].unsqueeze(2).unsqueeze(2), particles[..., 2].unsqueeze(2).unsqueeze(2))
        map_encodings = map_encodings.squeeze(-1).squeeze(-1)
        map_encodings = torch.permute(map_encodings, [0, 2, 1])

        # Get the encoded observation
        encoded_observation = encoded_observation

        # Compute the unnormed particle weights as described by eqn.2 in https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9635972&tag=1
        # Note we dont take the sqrt when computing the equclidean distance
        unnormalized_particle_weights = torch.sum((map_encodings - encoded_observation.unsqueeze(1))**2, dim=-1)
        unnormalized_particle_weights = -unnormalized_particle_weights / (2*self.obs_std_squared)
        unnormalized_particle_weights = torch.exp(unnormalized_particle_weights)

        # Do this to get good gradients + its good math
        if(unnormalized_resampled_particle_log_weights is not None):
            unnormalized_particle_weights = unnormalized_particle_weights * torch.exp(unnormalized_resampled_particle_log_weights)    

        # Normalize the weights
        new_particle_weights = unnormalized_particle_weights / torch.sum(unnormalized_particle_weights, dim=1, keepdim=True)

        return new_particle_weights

