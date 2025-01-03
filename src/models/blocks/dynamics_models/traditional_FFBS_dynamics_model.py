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


class TraditionalFFBSDynamicsModel(nn.Module):
    def __init__(self, configs):
        super(TraditionalFFBSDynamicsModel, self).__init__()

        # Get the parameters that we need
        self.residual_scale_factor = get_mandatory_config("residual_scale_factor", configs, "configs")
        self.noise_dim = get_mandatory_config("noise_dim", configs, "configs")
        self.particles_mask_out_dims = get_mandatory_config("particles_mask_out_dims", configs, "configs")
        self.particle_distribution_types = get_mandatory_config("particle_distribution_types", configs, "configs")

        # Get the particle dimension types and check that they are valid
        self.particle_dimension_types = get_mandatory_config("particle_dimension_types", configs, "configs")
        for pdt in self.particle_dimension_types:
            assert((pdt == "RealNumbers") or (pdt == "Angles"))

        # See how many dims the transformed particles have
        particle_dims_to_use = [i for i in range(len(self.particle_dimension_types))]
        self.total_transformed_particle_dims = self._count_transformed_dims(particle_dims_to_use, self.particle_dimension_types)

        # get the parameters that we need
        latent_space = get_mandatory_config("latent_space", configs, "configs")
        number_of_layers = get_mandatory_config("number_of_layers", configs, "configs")
        non_linear_type = get_mandatory_config("non_linear_type", configs, "configs")

        # The input dims
        particle_dims_to_use = [i for i in range(len(self.particle_dimension_types)) if i not in self.particles_mask_out_dims]
        input_dim = self._count_transformed_dims(particle_dims_to_use, self.particle_dimension_types)
        self.total_particle_encoder_input_dims = input_dim

        # output_dim
        output_dim = self.total_transformed_particle_dims
        # output_dim += len(self.particle_dimension_types)

        # Create the NN layers
        self.layers = self._create_linear_FF_network(input_dim, output_dim, non_linear_type, latent_space, number_of_layers)

        self.residual_scale_factor = torch.FloatTensor(self.residual_scale_factor)


        # Create the parameter that we will learn
        starting_bandwidths = [0.5, 0.5, 0.5]
        log_bands = torch.FloatTensor(np.log(starting_bandwidths))
        self.log_bandwidths = nn.parameter.Parameter(log_bands)



    def forward(self, particles, actions):

        assert(actions is None)

        # Get some info
        device = particles.device
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]

        # Get the means and stds
        means, stds = self.get_means_and_stds(particles, actions)


        # Draw samples
        new_particles = torch.zeros_like(particles)
        for i in range(particles.shape[-1]):

            # Get the info for this particle dim
            dist_type = self.particle_distribution_types[i]

            # Create the final particle dim
            if(dist_type == "Normal"):
                dist = D.Normal(loc=means[..., i], scale=stds[..., i])
                new_particles[..., i] = dist.rsample()

            elif(dist_type == "VonMises"):

                # Approximate with a Guassian so need to bound the STD
                assert(torch.sum(stds[..., i] > 0.5) == 0)

                dist = D.Normal(loc=means[..., i], scale=stds[..., i])
                x = dist.rsample()

                # Make sure we stay in the [-pi, pi] range
                x = torch.atan2(torch.sin(x), torch.cos(x))

                new_particles[..., i] = x

            else:
                assert(False)



        return new_particles




    def evaluate_particles(self, particles, particles_next_time):

        # Get some info
        device = particles.device
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]

        # Get the means and stds
        means, stds = self.get_means_and_stds(particles, None)


        # Draw samples
        log_probs = 0

        for i in range(particles.shape[-1]):

            # Get the info for this particle dim
            dist_type = self.particle_distribution_types[i]

            # Create the final particle dim
            if(dist_type == "Normal"):
                dist = D.Normal(loc=means[..., i], scale=stds[..., i])
                p = particles_next_time[..., i]
                p = torch.permute(p, [1, 0])
                lp = dist.log_prob(p.unsqueeze(-1))
                lp = torch.permute(lp, [1, 2, 0])
                log_probs += lp



            elif(dist_type == "VonMises"):

                # Approximate with a Guassian so need to bound the STD
                assert(torch.sum(stds[..., i] > 0.5) == 0)


                dist = D.Normal(loc=0, scale=stds[0,0,i])

                p_t = particles[..., i]
                p_t_1 = particles_next_time[..., i]

                diff = p_t.unsqueeze(-1) - p_t_1.unsqueeze(1)
                diff = torch.atan2(torch.sin(diff), torch.cos(diff))

                lp = dist.log_prob(diff)
                log_probs += lp

            else:
                assert(False)



        return log_probs









    def _create_linear_FF_network(self, input_dim, output_dim, non_linear_type, latent_space, number_of_layers):

        # Need at least 2 layers, the input and output layers
        assert(number_of_layers >= 2)

        # Get the non linear object from the name
        non_linear_object = self._get_non_linear_object_from_string(non_linear_type)

        # All the layers that this model will have
        layers = nn.Sequential()

        # Create the input layer
        layers.append(nn.Linear(in_features=(input_dim),out_features=latent_space))
        layers.append(non_linear_object())

        # the middle layers are all the same fully connected layers
        for i in range(number_of_layers-2):
            layers.append(nn.Linear(in_features=latent_space,out_features=latent_space))
            layers.append(non_linear_object())
        
        # Final layer is the output space and so does not need a non-linearity
        layers.append(nn.Linear(in_features=latent_space, out_features=output_dim))

        return layers


    def _get_non_linear_object_from_string(self, non_linear_type):

        # Select the non_linear type object to use
        if(non_linear_type == "ReLU"):
            non_linear_object = nn.ReLU
        elif(non_linear_type == "PReLU"):
            non_linear_object = nn.PReLU    
        elif(non_linear_type == "Tanh"):
            non_linear_object = nn.Tanh    
        elif(non_linear_type == "Sigmoid"):
            non_linear_object = nn.Sigmoid    
        else:
            assert(False)

        return non_linear_object


    def _transform_particles(self, particles, mask, transformed_size):

        # Get some info
        device = particles.device
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]

        # Create the transformed particles
        transformed_particles = torch.zeros((batch_size, number_of_particles, transformed_size), device=device)

        # Go through the dims 1 by 1 and transform/fill in the transformed_particles
        transformed_particles_dim_idx = 0
        for particles_dim_idx in range(particles.shape[-1]):

            if((mask is not None) and (particles_dim_idx in mask)):
                continue

            # Get the type of particle dim
            pd_type = self.particle_dimension_types[particles_dim_idx]

            if(pd_type == "RealNumbers"):
                transformed_particles[..., transformed_particles_dim_idx] = particles[..., particles_dim_idx]
                transformed_particles_dim_idx += 1
            elif(pd_type == "Angles"):
                transformed_particles[..., transformed_particles_dim_idx] = torch.sin(particles[..., particles_dim_idx])
                transformed_particles_dim_idx += 1
                transformed_particles[..., transformed_particles_dim_idx] = torch.cos(particles[..., particles_dim_idx])
                transformed_particles_dim_idx += 1


        transformed_particles = transformed_particles.contiguous()
        
        return transformed_particles


    def _count_transformed_dims(self, particle_dimensions_to_use, particle_dimension_types):

        # Count how many transformed particle inputs we have
        total = 0
        for particle_dim_idx in particle_dimensions_to_use:

            # Get the type of particle dim
            pd_type = particle_dimension_types[particle_dim_idx]

            # See how many dims this will be once transformed
            if(pd_type == "RealNumbers"):
                total += 1
            elif(pd_type == "Angles"):
                total += 2

        return total











    def get_means_and_stds(self, particles, actions):

        # Get some info
        device = particles.device
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]

        # Transform the particles
        transformed_particles = self._transform_particles(particles, mask=None, transformed_size=self.total_transformed_particle_dims)
        transformed_particles_for_encoder = self._transform_particles(particles, mask=self.particles_mask_out_dims, transformed_size=self.total_particle_encoder_input_dims)

        # Create the input
        inputs = torch.reshape(transformed_particles_for_encoder, (-1, transformed_particles_for_encoder.shape[-1]))

        # Run the layers
        out = self.layers(inputs)

        # reshape back into the correct unflattened shape
        out = out.view(batch_size, number_of_particles, -1)

        # get the means and stds
        means = out[..., 0:transformed_particles.shape[-1]]
        # stds = out[..., transformed_particles.shape[-1]:]

        # Squeeze it between -1 and 1
        means = (torch.sigmoid(means) * 2.0) - 1.0

        # Scale for the residuals
        residuals = (means * self.residual_scale_factor.to(means.device))

        # We want to learn the residual!
        means = transformed_particles + residuals

        # Un-Transform the residual back into the particle dim
        new_means = torch.zeros_like(particles)
        counter = 0
        for i in range(particles.shape[-1]):

            # Get the info for this particle dim
            pd_type = self.particle_dimension_types[i]

            # Create the final particle dim
            if(pd_type == "RealNumbers"):
                new_means[..., i] = means[..., counter]
                counter += 1
            elif(pd_type == "Angles"):
                new_means[..., i] = torch.atan2(means[...,counter], means[...,counter+1])
                counter += 2
            else:
                assert(False)



        # # Scale the stds
        # new_stds = torch.zeros_like(stds)
        # for i in range(particles.shape[-1]):

        #     # Get the info for this particle dim
        #     pd_type = self.particle_dimension_types[i]

        #     # Create the final particle dim
        #     if(pd_type == "RealNumbers"):
        #         new_stds[..., i] = torch.sigmoid(stds[..., i]) * 2.0

        #     elif(pd_type == "Angles"):
        #         new_stds[..., i] = torch.sigmoid(stds[..., i]) * 0.5

        #     else:
        #         assert(False)




        # Scale the stds
        new_stds = torch.zeros_like(new_means)
        for i in range(particles.shape[-1]):

            # Get the info for this particle dim
            pd_type = self.particle_dimension_types[i]

            stds = torch.exp(self.log_bandwidths[i])

            # Create the final particle dim
            if(pd_type == "RealNumbers"):
                new_stds[..., i] = torch.sigmoid(stds) * 2.0

            elif(pd_type == "Angles"):
                new_stds[..., i] = torch.sigmoid(stds) * 0.5

            else:
                assert(False)


        # print(new_stds[0,0,:].cpu().detach().numpy())


        return new_means, new_stds