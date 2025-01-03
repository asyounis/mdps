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


class LearnedDynamicsResidual(nn.Module):
    def __init__(self, configs):
        super(LearnedDynamicsResidual, self).__init__()

        # Get the parameters that we need
        self.residual_scale_factor = get_mandatory_config("residual_scale_factor", configs, "configs")
        self.noise_dim = get_mandatory_config("noise_dim", configs, "configs")
        self.particles_mask_out_dims = get_mandatory_config("particles_mask_out_dims", configs, "configs")

        # If its a list then it needs to become a tensor
        if(isinstance(self.residual_scale_factor, list)):
            self.residual_scale_factor = torch.FloatTensor(self.residual_scale_factor)
            self.residual_scale_factor = self.residual_scale_factor.unsqueeze(0).unsqueeze(0)
        else:
            assert(False)


        # Get the particle dimension types and check that they are valid
        self.particle_dimension_types = get_mandatory_config("particle_dimension_types", configs, "configs")
        for pdt in self.particle_dimension_types:
            assert((pdt == "RealNumbers") or (pdt == "Angles"))

        # Create the particle encoder
        self.particle_encoder, particle_encoder_latent_space, self.total_particle_encoder_input_dims = self._create_particle_encoder(configs, self.particles_mask_out_dims, self.particle_dimension_types)

        # See how many dims the transformed particles have
        particle_dims_to_use = [i for i in range(len(self.particle_dimension_types))]
        self.total_transformed_particle_dims = self._count_transformed_dims(particle_dims_to_use, self.particle_dimension_types)


        # Create the action encoder
        self.action_encoder, action_encoder_latent_space = self._create_action_encoder(configs)

        # Create the residual network
        self.residual_network = self._create_residual_network(configs, particle_encoder_latent_space, action_encoder_latent_space, self.noise_dim, self.total_transformed_particle_dims)

    # @torch.compile(mode="reduce-overhead")
    def forward(self, particles, actions):

        # Get some info
        device = particles.device
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]

        # Transform the particles
        transformed_particles = self._transform_particles(particles, mask=None, transformed_size=self.total_transformed_particle_dims)
        transformed_particles_for_encoder = self._transform_particles(particles, mask=self.particles_mask_out_dims, transformed_size=self.total_particle_encoder_input_dims)

        # Teh final input into the residual net
        final_input = []

        # Encode the particles
        encoded_particles = self.particle_encoder(torch.reshape(transformed_particles_for_encoder, (-1, transformed_particles_for_encoder.shape[-1])))
        encoded_particles = torch.reshape(encoded_particles, (batch_size, number_of_particles, -1))
        final_input.append(encoded_particles)

        # Encode the actions
        if((self.action_encoder is not None) and (actions is not None)):
            encoded_actions = self.action_encoder(actions)
            encoded_actions = torch.tile(encoded_actions.unsqueeze(1), [1, number_of_particles, 1])
            final_input.append(encoded_actions)

        # Sample noise
        noise = torch.randn(size=(batch_size, number_of_particles, self.noise_dim), device=device)
        final_input.append(noise)

        # The final input is the encoding and noise
        final_input = torch.cat(final_input, dim=-1)
        final_input = final_input.view(-1, final_input.shape[-1])

        # Do the forward pass
        out = self.residual_network(final_input)

        # reshape back into the correct unflattened shape
        out = out.view(batch_size, number_of_particles, -1)

        # Squeeze it between -1 and 1
        out = (torch.sigmoid(out) * 2.0) - 1.0

        # We want to learn the residual!
        residuals = (out * self.residual_scale_factor.to(out.device))
        out = transformed_particles + residuals

        # Un-Transform the residual back into the particle dim
        new_particles = torch.zeros_like(particles)
        counter = 0
        for i in range(particles.shape[-1]):

            # Get the info for this particle dim
            pd_type = self.particle_dimension_types[i]

            # Create the final particle dim
            if(pd_type == "RealNumbers"):
                new_particles[..., i] = out[..., counter]
                counter += 1
            elif(pd_type == "Angles"):
                new_particles[..., i] = torch.atan2(out[...,counter], out[...,counter+1])
                counter += 2
            else:
                assert(False)



        # print("")
        # print("------------------------------------------------------------")


        # # indx = 0
        # # for i in range(particles[0, :, -1].shape[0]):
        #     # print("{:+06.3f}   {:+06.3f}    {:+06.3f}".format(particles[0, i, indx].item(), residuals[0, i, indx].item(), new_particles[0, i, indx].item()))

        # # print(particles[0, :, -1])
        # # print("")
        # # print(residuals[0, :, -1])
        # # print("")
        # # print(new_particles[0, :, -1])
        # print("------------------------------------------------------------")
        # print("")


        # exit()



        # Return the particles
        return new_particles


    def _create_particle_encoder(self, configs, particles_mask_out_dims, particle_dimension_types):

        # get the parameters that we need
        particle_encoder_latent_space = get_mandatory_config("particle_encoder_latent_space", configs, "configs")
        particle_encoder_number_of_layers = get_mandatory_config("particle_encoder_number_of_layers", configs, "configs")
        particle_encoder_non_linear_type = get_mandatory_config("particle_encoder_non_linear_type", configs, "configs")

        # The number of particle dims to use
        particle_dims_to_use = [i for i in range(len(particle_dimension_types)) if i not in particles_mask_out_dims]
        total_particle_encoder_input_dims = self._count_transformed_dims(particle_dims_to_use, particle_dimension_types)

        # Create the network
        net = self._create_linear_FF_network(total_particle_encoder_input_dims, particle_encoder_latent_space, particle_encoder_non_linear_type, particle_encoder_latent_space, particle_encoder_number_of_layers)

        return net, particle_encoder_latent_space, total_particle_encoder_input_dims


    def _create_action_encoder(self, configs):

        # get the parameters that we need
        action_encoder_latent_space = get_mandatory_config("action_encoder_latent_space", configs, "configs")
        action_encoder_number_of_layers = get_mandatory_config("action_encoder_number_of_layers", configs, "configs")
        action_encoder_non_linear_type = get_mandatory_config("action_encoder_non_linear_type", configs, "configs")

        # If we have 0 layers then we are basically turning off the action encoder
        if(action_encoder_number_of_layers == 0):
            return None, 0

        return self._create_linear_FF_network(3, action_encoder_latent_space, action_encoder_non_linear_type, action_encoder_latent_space, action_encoder_number_of_layers), action_encoder_latent_space


    def _create_residual_network(self, configs, particle_encoder_latent_space, action_encoder_latent_space, noise_dim, total_transformed_particle_dims):

        # get the parameters that we need
        dynamics_latent_space = get_mandatory_config("dynamics_latent_space", configs, "configs")
        dynamics_number_of_layers = get_mandatory_config("dynamics_number_of_layers", configs, "configs")
        dynamics_non_linear_type = get_mandatory_config("dynamics_non_linear_type", configs, "configs")

        # The output dim
        output_dim = total_transformed_particle_dims

        input_dim = particle_encoder_latent_space + action_encoder_latent_space + noise_dim
        return self._create_linear_FF_network(input_dim, output_dim, dynamics_non_linear_type, dynamics_latent_space, dynamics_number_of_layers)


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







