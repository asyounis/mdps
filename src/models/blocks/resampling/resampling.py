# Python Imports

# Package Imports
import torch
import torch.distributions as D
import numpy as np

# Ali Package Import
from general_ml_framework.utils.config import *

# Project Import 
from utils.resampling import *
from kde.kde import KDE
from utils.resampling import *




class BaseResampler:
    def __init__(self):
        pass

    def __call__(self, number_of_particles, particles, particle_weights, bandwidths, stage):
        return self.resample_particles(number_of_particles, particles, particle_weights, bandwidths, stage)

    def resample_particles(self, number_of_particles, particles, particle_weights, bandwidths, stage):
        raise NotImplemented




class KDEResampler(BaseResampler):
    def __init__(self, kde_distribution_types, particle_resampling_method):

        # Save for later
        self.kde_distribution_types = kde_distribution_types
        self.particle_resampling_method = particle_resampling_method


    def resample_particles(self, number_of_particles, particles, particle_weights, bandwidths, stage):

        # Create the KDE
        kde_dist = KDE(self.kde_distribution_types, particles, particle_weights, bandwidths, particle_resampling_method=self.particle_resampling_method)

        # Sample new particles
        new_particles = kde_dist.sample((number_of_particles, )).detach()

        # Compute the new weights
        log_probs = kde_dist.log_prob(new_particles)
        new_particle_log_weights = log_probs - log_probs.detach()

        return new_particles, new_particle_log_weights


class SoftResamplingResampler(BaseResampler):

    # The available resampling methods for this method
    AVAILABLE_RESAMPLING_METHODS = set(["multinomial", "residual", "stratified", None])


    def __init__(self, configs):

        # Get the uniform mixing parameter (range [0, 1]).
        # Value of 1 means fully uniform
        # Value of 0 means no uniform mixing
        self.uniform_mixing_parameter = get_mandatory_config("uniform_mixing_parameter", configs, "configs")
        assert((self.uniform_mixing_parameter >= 0.0) and (self.uniform_mixing_parameter <= 1.0))

        # Get what kind of resampling we should do after we mix the weights with a uniform distribution
        self.particle_resampling_method = get_mandatory_config("particle_selection_method", configs, "configs")
        assert(self.particle_resampling_method in self.AVAILABLE_RESAMPLING_METHODS)

        # Get if we should enable or disable the mixing
        self.enable_soft_resampling_for_stages = get_mandatory_config("enabled", configs, "configs")

    def resample_particles(self, number_of_particles, particles, particle_weights, bandwidths, stage):

        # Get some info
        batch_size = particles.shape[0]
        current_number_of_particles = particles.shape[1]
        particle_dim = particles.shape[2]
        device = particles.device

        # Get if we are enabled or disabled for this stage
        enabled = self.enable_soft_resampling_for_stages[stage]

        if(enabled):
            # If we are enabled then mix the weights with a uniform distribution
            resampling_weights = (1.0 - self.uniform_mixing_parameter) * particle_weights
            resampling_weights = resampling_weights + (self.uniform_mixing_parameter * (1.0 / float(current_number_of_particles)))

        else:
            # If we are disabled then do no mixing
            resampling_weights = particle_weights

        # Select based on the method
        if(self.particle_resampling_method == "multinomial"):
            selected_particle_indices = select_particles_multinomial_resampling_method(number_of_particles, resampling_weights)
        elif(self.particle_resampling_method == "residual"):
            selected_particle_indices = select_particles_residual_resampling_method(number_of_particles, resampling_weights)
        elif(self.particle_resampling_method == "stratified"):                
            selected_particle_indices = select_particles_stratified_resampling_method(number_of_particles, resampling_weights)

        # Draw the samples
        new_particles = torch.zeros(size=(batch_size, number_of_particles,particle_dim), device=device)
        for d in range(particle_dim):
            new_particles[...,d] = torch.gather(particles[..., d], 1, selected_particle_indices)

        # If we are enabled then we should compute the soft weights
        if(enabled):
            # Slice out the particle weights
            selected_weights = torch.gather(particle_weights, 1, selected_particle_indices)
            selected_mixed_weights = torch.gather(resampling_weights, 1, selected_particle_indices)

            # Figure out the eps we should add to the log weights to prevent underflow
            eps = 0.0
            if(self.uniform_mixing_parameter < 1e-8):
                eps = 1e-8
            if((1.0 / float(current_number_of_particles)) < 1e-8):
                eps = 1e-8

            # Compute the new log weights
            new_weights = selected_weights / selected_mixed_weights
            new_particle_log_weights = torch.log(new_weights + eps)

        else:

            # Not enabled then the weights are uniform 
            fill_value = 1.0 / float(number_of_particles)
            new_particle_weights = torch.full((batch_size, number_of_particles), fill_value=fill_value, device=device)
            new_particle_log_weights = torch.log(new_particle_weights).detach()


        return new_particles.detach(), new_particle_log_weights




class DiscreteResamplingResampler(BaseResampler):

    # The available resampling methods for this method
    AVAILABLE_RESAMPLING_METHODS = set(["multinomial", "residual", "stratified", None])

    def __init__(self, configs):

        # Get what kind of resampling we should do after we mix the weights with a uniform distribution
        self.particle_resampling_method = get_mandatory_config("particle_selection_method", configs, "configs")
        assert(self.particle_resampling_method in self.AVAILABLE_RESAMPLING_METHODS)

    def resample_particles(self, number_of_particles, particles, particle_weights, bandwidths, stage):

        # Get some info
        batch_size = particles.shape[0]
        current_number_of_particles = particles.shape[1]
        particle_dim = particles.shape[2]
        device = particles.device

        # If we are disabled then do no mixing
        resampling_weights = particle_weights

        # Select based on the method
        if(self.particle_resampling_method == "multinomial"):
            selected_particle_indices = select_particles_multinomial_resampling_method(number_of_particles, resampling_weights)
        elif(self.particle_resampling_method == "residual"):
            selected_particle_indices = select_particles_residual_resampling_method(number_of_particles, resampling_weights)
        elif(self.particle_resampling_method == "stratified"):                
            selected_particle_indices = select_particles_stratified_resampling_method(number_of_particles, resampling_weights)

        # Draw the samples
        new_particles = torch.zeros(size=(batch_size, number_of_particles,particle_dim), device=device)
        for d in range(particle_dim):
            new_particles[...,d] = torch.gather(particles[..., d], 1, selected_particle_indices)


        # Not enabled then the weights are uniform 
        fill_value = 1.0 / float(number_of_particles)
        new_particle_weights = torch.full((batch_size, number_of_particles), fill_value=fill_value, device=device)
        new_particle_log_weights = torch.log(new_particle_weights)


        return new_particles.detach(), new_particle_log_weights.detach()
