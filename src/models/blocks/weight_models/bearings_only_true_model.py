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

# Project Imports
from models.blocks.observation_encoder_models.encoded_observation import EncodedObservation



class BeaingsOnlyTrueWeightModel(nn.Module):
    def __init__(self, configs):
        super(BeaingsOnlyTrueWeightModel, self).__init__()


        self.concentration = 50.0
        self.uniform_mix = 0.85

        self.sensor_location = torch.FloatTensor([-5, 0])

    def forward(self, input_dict):

        # Unpack
        particles = input_dict["particles"]
        encoded_observation = input_dict["encoded_observations"]
        unnormalized_resampled_particle_log_weights = input_dict["unnormalized_resampled_particle_log_weights"]


        # Get some info
        device = particles.device
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]

        # Put back into a bearing_obs
        bearing_obs = torch.atan2(encoded_observation[..., 0], encoded_observation[..., 1])

        # Move to the correct device
        self.sensor_location = self.sensor_location.to(device)


        # Get the true bearing between the car and the sensor
        diff = particles[..., 0:2] - self.sensor_location.unsqueeze(0).unsqueeze(0)
        true_bearings = torch.arctan2(diff[..., 1], diff[..., 0])

        # Compute the von mises
        dist = D.VonMises(particles[..., -1], self.concentration)
        log_probs = dist.log_prob(true_bearings)

        # Add in the uniform
        probs = torch.exp(log_probs)
        probs = probs * self.uniform_mix
        probs = probs + (1-self.uniform_mix)*(1/(2.0*np.pi))
        unnormalized_particle_weights = probs

        # Do this to get good gradients + its good math
        if(unnormalized_resampled_particle_log_weights is not None):
            unnormalized_particle_weights = unnormalized_particle_weights * torch.exp(unnormalized_resampled_particle_log_weights)

        # Normalize the weights
        new_particle_weights = torch.nn.functional.normalize(unnormalized_particle_weights, p=1.0, eps=1e-8, dim=1)

        return new_particle_weights




