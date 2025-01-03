# Python Imports
import time

# Package Imports
import torch
import torch.distributions as D
import numpy as np

# Ali Package Import
from general_ml_framework.utils.config import *

# Project Imports
from kde.kde import KDE


class NMSModeFinder:
    def __init__(self, all_outputs, kde_distribution_types, particles_key, weights_key, bandwidths_key, configs):

        # Unpack the configs
        nms_thresholds = get_mandatory_config("nms_thresholds", configs, "configs")
        self.number_of_modes = get_mandatory_config("number_of_modes", configs, "configs")
        self.number_of_samples_to_draw = get_mandatory_config("number_of_samples_to_draw", configs, "configs")

        # If we should treat the posterior as a discrete distribution or as a continuous one
        self.treat_posterior_as_discrete = get_optional_config_with_default("treat_posterior_as_discrete", configs, "configs", default_value=False)

        # Save for later
        self.all_outputs = all_outputs
        self.kde_distribution_types = kde_distribution_types
        self.all_particles = all_outputs[particles_key]
        self.all_weights = all_outputs[weights_key]
        self.all_bandwidths = all_outputs[bandwidths_key]

        # Put into a tensor so we can operate on it
        self.nms_thresholds = torch.FloatTensor(nms_thresholds)



    def get_modes(self):

        # Get some info
        sequence_length = len(self.all_particles)

        all_modes = []
        for seq_idx in range(sequence_length):

            # Get the particles and stuff for this specific index
            particles = self.all_particles[seq_idx]
            weights = self.all_weights[seq_idx]
            bandwidths = self.all_bandwidths[seq_idx]

            # Get the modes
            modes = self._get_modes_helper(self.number_of_modes, particles, weights, bandwidths)

            # Put the modes into a tensor
            modes = torch.stack(modes, dim=1)

            # Keep track
            all_modes.append(modes)

        return all_modes

    def _get_modes_helper(self, number_of_modes, particles, weights, bandwidths):

        # IF there are no modes to be had then do nothing
        if(number_of_modes == 0):
            return []

        # Get some info
        batch_size = particles.shape[0]

        ########################################################################################################################################
        ## Select a Mode
        ########################################################################################################################################

        if(self.treat_posterior_as_discrete):

            # If we are discrete then just choose the highest weight particle
            
            # Select the particles
            selected_samples_indices = torch.argmax(weights, dim=-1).unsqueeze(-1)

            # Select the sample (aka the mode)
            selected_mode = torch.zeros(size=(batch_size, 1, particles.shape[-1]), device=particles.device)
            for d in range(particles.shape[-1]):
                selected_mode[...,d] = torch.gather(particles[..., d], 1, selected_samples_indices)

            # Squeeze out this dim since its always 1 and we dont need it
            selected_mode = selected_mode.squeeze(1)

        else:

            # If we are a continuous posterior then we should select the peak

            # Make the KDE that we will use to score the modes
            kde = KDE(self.kde_distribution_types, particles, weights, bandwidths, particle_resampling_method="multinomial")

            # generate samples to look for modes
            samples = kde.sample((self.number_of_samples_to_draw, ))

            # Score the samples
            log_probs = kde.log_prob(samples)

            # Select the particles
            selected_samples_indices = torch.argmax(log_probs, dim=-1).unsqueeze(-1)

            # Select the sample (aka the mode)
            selected_mode = torch.zeros(size=(batch_size, 1, particles.shape[-1]), device=particles.device)
            for d in range(particles.shape[-1]):
                selected_mode[...,d] = torch.gather(samples[..., d], 1, selected_samples_indices)

            # Squeeze out this dim since its always 1 and we dont need it
            selected_mode = selected_mode.squeeze(1)


        ########################################################################################################################################
        ## Suppress the weights
        ########################################################################################################################################

        # Make a copy of the weights since we will be editing them and we dont want to edit the original tensor
        weights_copy = weights.detach().clone()

        # Compute the dist between the selected mode and the particles so we can 
        # figure out which particles we need to suppress
        diffs = torch.zeros_like(particles)
        for d in range(particles.shape[-1]):

            # Compute the difference
            diff_local = particles[..., d] - selected_mode[..., d].unsqueeze(1) 

            # there may be additional processing to do
            if(self.kde_distribution_types[d] == "Normal"):
                diffs[..., d] = diff_local
            elif(self.kde_distribution_types[d] == "VonMises"):
                diffs[..., d] = torch.atan2(torch.sin(diff_local), torch.cos(diff_local), )
            else:
                assert(False) 

        # Do the ellipsoid test
        # See 
        #   - https://math.stackexchange.com/questions/76457/check-if-a-point-is-within-an-ellipse
        #   - https://stackoverflow.com/questions/17770555/how-to-check-if-a-point-is-inside-an-ellipsoid
        ellipsoid_equation_check_lhs = diffs / self.nms_thresholds.to(particles.device).unsqueeze(0).unsqueeze(1)
        ellipsoid_equation_check_lhs = ellipsoid_equation_check_lhs**2
        ellipsoid_equation_check_lhs = torch.sum(ellipsoid_equation_check_lhs, dim=-1)

        # Run the check
        inside_ellipsoid = ellipsoid_equation_check_lhs <= 1.0

        # Suppress the weights within the ellipsoid and normalize
        weights_copy[inside_ellipsoid] = 1e-8
        weights_copy = torch.nn.functional.normalize(weights_copy, p=1.0, dim=1, eps=1e-8)


        ########################################################################################################################################
        ## Recurse to get more modes and then combine with the modes we already have
        ########################################################################################################################################
    
        # Get more modes
        more_modes = self._get_modes_helper(number_of_modes-1, particles, weights_copy, bandwidths) 

        # Add in our mode (to the front since it is a higher prob mode)
        more_modes.insert(0, selected_mode)


        return more_modes