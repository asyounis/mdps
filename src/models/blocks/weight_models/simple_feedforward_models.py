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



class SimpleFeedForwardWeightModel(nn.Module):
    def __init__(self, configs):
        super(SimpleFeedForwardWeightModel, self).__init__()

        # get the parameters that we need
        encoded_observation_latent_space = get_mandatory_config("encoded_observation_latent_space", configs, "configs")
        additional_inputs_latent_space = get_mandatory_config("additional_inputs_latent_space", configs, "configs")
        self.min_weight = get_mandatory_config("min_weight", configs, "configs")
        self.max_weight = get_mandatory_config("max_weight", configs, "configs")

        # Get the particle dimension types and check that they are valid
        self.particle_dimension_types = get_mandatory_config("particle_dimension_types", configs, "configs")
        for i in range(len(self.particle_dimension_types.keys())):
            assert(i in self.particle_dimension_types)
            assert((self.particle_dimension_types[i] == "RealNumbers") or (self.particle_dimension_types[i] == "Angles"))

        # Extract the particle dimes to use
        self.particle_dimensions_to_use =  get_mandatory_config("particle_dimensions_to_use", configs, "configs")
        assert(len(self.particle_dimensions_to_use) > 0)

        # Create the particle encoder
        self.particle_encoder, particle_encoder_latent_space, self.total_particle_encoder_input_dims = self._create_particle_encoder(configs, self.particle_dimensions_to_use, self.particle_dimension_types)

        # Create the bulk of the weight model
        input_size = particle_encoder_latent_space + encoded_observation_latent_space + additional_inputs_latent_space
        self.weight_model = self._create_weight_model(configs, input_size)

        # Add a sigmoid so we can bound the weights
        self.sigmoid = nn.Sigmoid()

        # Flag if we have additional inputs to consider
        self.use_additional_inputs = (additional_inputs_latent_space > 0)



    def forward(self, input_dict):

        # Unpack
        particles = input_dict["particles"]
        encoded_observation = input_dict["encoded_observations"]
        unnormalized_resampled_particle_log_weights = input_dict["unnormalized_resampled_particle_log_weights"]
        additional_inputs = input_dict.get("additional_inputs", None)

        # Get some info
        device = particles.device
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]

        # The final inputs
        final_input = []

        # Get the transformed particles
        transformed_particles = self._extract_and_transform_particles(particles)

        # Encode the particles
        encoded_particles = self.particle_encoder(torch.reshape(transformed_particles, (batch_size*number_of_particles, -1)))
        encoded_particles = torch.reshape(encoded_particles, (batch_size, number_of_particles, -1))
        final_input.append(encoded_particles)

        # Need to tile the encoded_observation
        tiled_encoded_observation = torch.tile(encoded_observation.unsqueeze(1), (1, number_of_particles, 1))
        final_input.append(tiled_encoded_observation)

        # Add the additional_inputs if we have them and if we are supposed to
        if(self.use_additional_inputs and (additional_inputs is not None)):
            final_input.append(additional_inputs)

        # The final input is the encoding and noise
        final_input = torch.cat(final_input,dim=-1)
        final_input = final_input.view(-1, final_input.shape[-1])

        # Run the weight model
        unnormalized_particle_weights = self.weight_model(final_input)

        # Unflatten
        unnormalized_particle_weights = torch.reshape(unnormalized_particle_weights, (batch_size, number_of_particles))

        # Bound the weight
        unnormalized_particle_weights = self.sigmoid(unnormalized_particle_weights)        
        unnormalized_particle_weights = unnormalized_particle_weights * (self.max_weight - self.min_weight)
        unnormalized_particle_weights = unnormalized_particle_weights + self.min_weight

        # Do this to get good gradients + its good math
        if(unnormalized_resampled_particle_log_weights is not None):
            unnormalized_particle_weights = unnormalized_particle_weights * torch.exp(unnormalized_resampled_particle_log_weights)

        # Normalize the weights
        new_particle_weights = torch.nn.functional.normalize(unnormalized_particle_weights, p=1.0, eps=1e-8, dim=1)

        return new_particle_weights





    def _create_particle_encoder(self, configs, particle_dimensions_to_use, particle_dimension_types):

        # get the parameters that we need
        particle_encoder_latent_space = get_mandatory_config("particle_encoder_latent_space", configs, "configs")
        particle_encoder_number_of_layers = get_mandatory_config("particle_encoder_number_of_layers", configs, "configs")
        particle_encoder_non_linear_type = get_mandatory_config("particle_encoder_non_linear_type", configs, "configs")

        # Count how many transformed particle inputs we have
        total_particle_encoder_input_dims = 0
        for particle_dim_idx in particle_dimensions_to_use:

            # Get the type of particle dim
            pd_type = particle_dimension_types[particle_dim_idx]

            # See how many dims this will be once transformed
            if(pd_type == "RealNumbers"):
                total_particle_encoder_input_dims += 1
            elif(pd_type == "Angles"):
                total_particle_encoder_input_dims += 2

        return self._create_linear_FF_network(total_particle_encoder_input_dims, particle_encoder_latent_space, particle_encoder_non_linear_type, particle_encoder_latent_space, particle_encoder_number_of_layers), particle_encoder_latent_space, total_particle_encoder_input_dims


    def _create_weight_model(self, configs, input_size):

        # get the parameters that we need
        weight_predictor_latent_space = get_mandatory_config("weight_predictor_latent_space", configs, "configs")
        weight_predictor_number_of_layers = get_mandatory_config("weight_predictor_number_of_layers", configs, "configs")
        weight_predictor_non_linear_type = get_mandatory_config("weight_predictor_non_linear_type", configs, "configs")

        return self._create_linear_FF_network(input_size, 1, weight_predictor_non_linear_type, weight_predictor_latent_space, weight_predictor_number_of_layers, apply_activation_to_output=False)


    def _create_linear_FF_network(self, input_dim, output_dim, non_linear_type, latent_space, number_of_layers, apply_activation_to_output=True):

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

        # Do the final layer
        layers.append(nn.Linear(in_features=latent_space,out_features=output_dim))

        # For some outputs we dont want the final output we might want a linear output
        if(apply_activation_to_output):
            layers.append(non_linear_object())

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








    def _extract_and_transform_particles(self, particles):

        # Get some info
        device = particles.device
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]

        # Create the transformed particles
        transformed_particles = torch.zeros((batch_size, number_of_particles, self.total_particle_encoder_input_dims), device=device)

        # Go through the dims 1 by 1 and transform/fill in the transformed_particles
        transformed_particles_dim_idx = 0
        for particles_dim_idx in self.particle_dimensions_to_use:

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
