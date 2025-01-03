# Python Imports
import time

# Package Imports
import torch
import torch.distributions as D
import numpy as np

# Ali Package Import
from utils.resampling import *

# Project Imports
from kde.probability_distribution import ProbabilityDistribution

class KDE(ProbabilityDistribution):

    # The available resampling methods
    # These are the methods that this class implements
    AVAILABLE_RESAMPLING_METHODS = set(["multinomial", "residual", "stratified", None])

    def __init__(self, distribution_types, particles, particle_weights, bandwidths, validate_args=True, particle_resampling_method=None):
        super(KDE, self).__init__()

        # The distribution types for each of the dims
        self.distribution_types = distribution_types


        # The method that we will use to select particles when doing resampling.
        # The methods are:
        #       - "multinomial"
        #       - "residual"
        #       - "stratified"
        #       - None: This means that no sampling will be allowed
        #
        # Good Resources:
        #       - https://robotics.stackexchange.com/questions/479/particle-filters-how-to-do-resampling
        #       - https://github.com/mnicely/particle_filter/blob/master/docs/Dissertation_Nicely_111319.pdf
        self.particle_resampling_selection_method = particle_resampling_method
        assert((self.particle_resampling_selection_method in self.AVAILABLE_RESAMPLING_METHODS) or (self.particle_resampling_selection_method is None))


        # Get some info
        self.device = particles.device
        self.batch_size = particles.shape[0]

        # The particle weights can be thought as the mixture component weighs
        self.particles = particles
        self.weights = particle_weights
        self.bandwidths = bandwidths

        # Create the distributions for all the dims
        self.distributions = []
        for d, distribution_type in enumerate(self.distribution_types):

            # Get the Parameters for this dim
            dim_particles = self.particles[...,d]
            dim_bandwidths = self.bandwidths[...,d].unsqueeze(-1)

            # Create the distribution
            if(distribution_type == "Normal"):
                dist = D.Normal(loc=dim_particles, scale=dim_bandwidths, validate_args=validate_args)
            elif(distribution_type == "VonMises"):
                # Need to make sure that this is the concentration and not the "std"
                dim_bandwidths = 1.0 / (dim_bandwidths + 1e-8)
                dist = D.VonMises(loc=dim_particles, concentration=dim_bandwidths)

            else:
                print("Unknown distribution_type \"{}\"".format(distribution_type))
                assert(False)

            # Keep track of the dist
            self.distributions.append(dist)

    @staticmethod
    def create_from_configs(configs):
        distribution_types = configs["distribution_types"]
        particles = configs["particles"]
        weights = configs["weights"]
        bandwidths = configs["bandwidths"]
        return KDE(distribution_types, particles, weights, bandwidths)
        

    def get_configs(self):
        configs = dict()
        configs["distribution_types"] = self.distribution_types
        configs["particles"] = self.particles
        configs["weights"] = self.weights
        configs["bandwidths"] = self.bandwidths
        configs["cls"] = "KDE"
        return configs



    def get_device(self):
        return self.device

    def sample(self, shape):

        # Make sure we can actually resample
        if(self.particle_resampling_selection_method is None):
            print("Tried to draw samples without setting \"particle_resampling_selection_method\"")
            assert(False)


        # There are no gradients when doing resampling so 
        # we might as well stop the gradients now to save computation time
        with torch.no_grad():
            assert(len(shape) == 1)

            # Figure out how many samples we need
            total_samples = 1
            for s in shape:
                total_samples *= s

            # Create structure that will hold all the samples
            all_samples_shape = list()
            all_samples_shape.append(total_samples)
            all_samples_shape.append(self.batch_size)
            all_samples_shape.append(len(self.distributions))
            all_samples = torch.zeros(size=all_samples_shape, device=self.device)

            # Select the particles we want to use
            if(self.particle_resampling_selection_method == "multinomial"):
                selected_particle_indices = select_particles_multinomial_resampling_method(total_samples, self.weights)
            elif(self.particle_resampling_selection_method == "residual"):
                selected_particle_indices = select_particles_residual_resampling_method(total_samples, self.weights)
            elif(self.particle_resampling_selection_method == "stratified"):                
                selected_particle_indices = select_particles_stratified_resampling_method(total_samples, self.weights)
            
            # exit()
            # Sample from each dim 1 dim at a time
            for d in range(len(self.distributions)):

                # Select the particles
                selected_particles = torch.gather(self.particles[..., d], 1, selected_particle_indices)
            
                # Get the bandwidth for this dim
                bandwidth = self.bandwidths[...,d].unsqueeze(-1)

                # create the correct dist for this dim
                distribution_type = self.distribution_types[d]
                if(distribution_type == "Normal"):
                    dist = D.Normal(loc=selected_particles, scale=bandwidth, validate_args=False)

                elif(distribution_type == "VonMises"):
                    bandwidth = 1.0 / (bandwidth + 1e-8)
                    dist = D.VonMises(loc=selected_particles, concentration=bandwidth, validate_args=False)
                
                # Draw samples
                samples = dist.sample()

                # Add them to the all samples tensor for us to 
                # all_samples[...,d] = torch.permute(samples, (1, 0)).squeeze(-1)
                all_samples[...,d] = torch.permute(samples, (1, 0))

            return torch.permute(all_samples,(1, 0, 2)).detach()

    def log_prob(self, x, do_normalize_weights=True):
        
        # Right now we only support shapes of size 3 [batch size, samples, dims]
        assert(len(x.shape) == 3)

        # Need to convert x from [batch, shape , dims] to [shape, batch, dims]
        x = torch.permute(x, (1,0,2))

        # All the log probabilities
        all_log_probs = None 

        # Do 1 dim at a time
        for d, dist in enumerate(self.distributions):

            log_prob = dist.log_prob(x[..., d].unsqueeze(-1))
            
            if(all_log_probs is None):
                all_log_probs = log_prob
            else:
                all_log_probs = all_log_probs + log_prob

        # Log the weights
        log_weights = torch.log(self.weights.unsqueeze(0) + 1e-8)

        # Normalize the weights if we need to
        if(do_normalize_weights):
            log_weights = log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)

        # Finish off the computation
        all_log_probs = all_log_probs + log_weights
        all_log_probs = torch.logsumexp(all_log_probs, dim=-1)
        all_log_probs = torch.permute(all_log_probs, (1,0))

        return all_log_probs



