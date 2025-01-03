# Python Imports
import time

# Package Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ali Package Import
from general_ml_framework.models.base_model import BaseModel
from general_ml_framework.utils.config import *

# Project Imports
from kde.nms_mode_finder import NMSModeFinder
from kde.kde import KDE


class ParticleBasedModels(BaseModel):
    def __init__(self, model_configs, model_architecture_configs):
        super(ParticleBasedModels, self).__init__()

        # Extract the name of the model so we can get its model architecture params
        main_model_name = model_configs["main_model_name"]
        main_model_arch_configs = model_architecture_configs[main_model_name]

        # Get the parameters for mode finding
        # If not provided then no mode finding will happen
        self.nms_mode_finding_configs = get_optional_config_with_default("nms_mode_finding_configs", main_model_arch_configs, "main_model_arch_configs", default_value=None)


    def _get_model_config(self, model_name, internal_model_names, model_architecture_configs):
        internal_model_name = internal_model_names[model_name]
        configs = model_architecture_configs.get(internal_model_name, None)
        return configs


    def _compute_single_point_predictions_max_prob(self, all_outputs, kde_distribution_types, particles_key, weights_key, bandwidths_key):

        # Unpack
        all_particles = all_outputs[particles_key]
        all_particle_weights = all_outputs[weights_key]
        all_particle_bandwidths = all_outputs[bandwidths_key]

        # All the single point predictions
        all_single_point_predictions = []

        # Need to do this 1 time-step at a time
        for i in range(len(all_particles)):

            # Get the ones for this specific sequence
            particles = all_particles[i]
            particle_weights = all_particle_weights[i]  
            bandwidths = all_particle_bandwidths[i]  

            # Get some info
            batch_size = particles.shape[0]
            particle_dim = particles.shape[-1]
            device = particles.device

            # Create a KDE so we can find the highest prob particles
            # We are not going to sample from this KDE so we ignore the resampling method parameters
            dist = KDE(kde_distribution_types, particles, particle_weights, bandwidths, particle_resampling_method="stratified")

            # # Get the log prob for all the particles
            # log_probs = dist.log_prob(particles)

            
            # Sample a bunch to get a dense estimate of the actual MAP
            samples = dist.sample((10000, ))

            # Get the log prob for all the particles
            log_probs = dist.log_prob(samples)

            # Select the particles
            selected_samples_indices = torch.argmax(log_probs, dim=-1).unsqueeze(-1)

            # Select the samples
            selected_samples = torch.zeros(size=(batch_size, 1, particle_dim), device=device)
            for d in range(particle_dim):
                selected_samples[...,d] = torch.gather(samples[..., d], 1, selected_samples_indices)

            # Squeeze out the dim we dont need
            selected_samples = selected_samples.squeeze(1)

            # Append!!
            all_single_point_predictions.append(selected_samples)

        # Convert to a tensor
        all_single_point_predictions = torch.stack(all_single_point_predictions, dim=1)

        return all_single_point_predictions


    def _compute_single_point_predictions_max_weight(self, all_outputs, all_encoded_observations, encoded_global_map, kde_distribution_types, particles_key, weights_key, bandwidths_key):

        # If we dont have this model then there is nothing to compute....
        if(self.single_point_prediction_weight_model is None):
            return None

        # Unpack
        all_particles = all_outputs[particles_key]
        all_particle_weights = all_outputs[weights_key]
        all_particle_bandwidths = all_outputs[bandwidths_key]

        # All the single point predictions
        all_single_point_predictions = []

        # Need to do this 1 time-step at a time
        for i in range(len(all_particles)):

            # Get the ones for this specific sequence
            particles = all_particles[i]
            particle_weights = all_particle_weights[i]  
            bandwidths = all_particle_bandwidths[i]  
            encoded_observation = all_encoded_observations.get_subsequence_index(i)

            # Get some info
            batch_size = particles.shape[0]
            particle_dim = particles.shape[-1]
            device = particles.device

            # Create a KDE so we can find the highest prob particles
            # We are not going to sample from this KDE so we ignore the resampling method parameters
            dist = KDE(kde_distribution_types, particles, particle_weights, bandwidths, particle_resampling_method="stratified")
            
            # Sample a bunch to get a dense estimate of the actual MAP
            samples = dist.sample((10000, ))

            # Compute new weights
            input_dict = dict()
            input_dict["particles"] = samples
            input_dict["encoded_global_map"] = encoded_global_map
            input_dict["encoded_observations"] = encoded_observation 
            input_dict["unnormalized_resampled_particle_log_weights"] = None
            weights = self.single_point_prediction_weight_model(input_dict)

            # Select the particles
            selected_samples_indices = torch.argmax(weights, dim=-1).unsqueeze(-1)

            # Select the samples
            selected_samples = torch.zeros(size=(batch_size, 1, particle_dim), device=device)
            for d in range(particle_dim):
                selected_samples[...,d] = torch.gather(samples[..., d], 1, selected_samples_indices)

            # Squeeze out the dim we dont need
            selected_samples = selected_samples.squeeze(1)

            # Append!!
            all_single_point_predictions.append(selected_samples)

        # Convert to a tensor
        all_single_point_predictions = torch.stack(all_single_point_predictions, dim=1)

        return all_single_point_predictions


    def _get_top_modes(self, all_outputs, kde_distribution_types, particles_key, weights_key, bandwidths_key):

        # Compute multiple mode output
        mode_finder = NMSModeFinder(all_outputs, kde_distribution_types, particles_key, weights_key, bandwidths_key, self.nms_mode_finding_configs)
        modes = mode_finder.get_modes()

        # Stack them!
        modes = torch.stack(modes, dim=1)

        return modes







    # def _compute_single_point_predictions(self, all_outputs, particles_key, weights_key):

    #     # Unpack
    #     all_particles = all_outputs[particles_key]
    #     all_particle_weights = all_outputs[weights_key]

    #     # All the mean particles that we compute
    #     all_mean_particles = []

    #     # Need to do this 1 time-step at a time
    #     for i in range(len(all_particles)):

    #         # Get the ones for this specific sequence
    #         particles = all_particles[i]
    #         particle_weights = all_particle_weights[i]  

    #         # Get some info
    #         device = particles.device
    #         batch_size = particles.shape[0]
    #         number_of_particles = particles.shape[1]

    #         # transform the particles
    #         transformed_particles = torch.zeros((batch_size, number_of_particles, 4), device=device)
    #         transformed_particles[..., 0] = particles[..., 0]
    #         transformed_particles[..., 1] = particles[..., 1]
    #         transformed_particles[..., 2] = torch.sin(particles[..., 2])
    #         transformed_particles[..., 3] = torch.cos(particles[..., 2])

    #         # Compute the mean particle
    #         transformed_mean_particle = torch.sum(transformed_particles * particle_weights.unsqueeze(-1), dim=1)

    #         # transform it back
    #         mean_particle = torch.zeros((batch_size, 3), device=device)
    #         mean_particle[..., 0] = transformed_mean_particle[..., 0]
    #         mean_particle[..., 1] = transformed_mean_particle[..., 1]
    #         mean_particle[..., 2] = torch.atan2(transformed_mean_particle[..., 2], transformed_mean_particle[..., 3])

    #         # Save the mean particle
    #         all_mean_particles.append(mean_particle)

    #     # Convert to a tensor
    #     all_mean_particles = torch.stack(all_mean_particles, dim=1)

    #     return all_mean_particles
