# Python Imports
import time

# Package Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

# Ali Package Import
from general_ml_framework.utils.config import *


class TwoStageResidualDynamicsModel(nn.Module):
    def __init__(self, configs):
        super(TwoStageResidualDynamicsModel, self).__init__()

        # Get the action encoder configs and create the encoder
        action_encoder_configs = get_mandatory_config("action_encoder_configs", configs, "configs")
        self.action_encoder, action_encoder_output_dims = self._create_action_encoder(action_encoder_configs)

        # Get the particle dimension types and check that they are valid
        self.particle_dimension_types = get_mandatory_config("particle_dimension_types", configs, "configs")
        for i in range(len(self.particle_dimension_types.keys())):
            assert(i in self.particle_dimension_types)
            assert((self.particle_dimension_types[i] == "RealNumbers") or (self.particle_dimension_types[i] == "Angles"))


        # Get the stage 1 and stage 2 configs
        stage_1_configs = get_mandatory_config("stage_1_configs", configs, "configs")
        stage_2_configs = get_mandatory_config("stage_2_configs", configs, "configs")

        # Make stage 1
        self.stage_1_stuff = self._create_stage_models(stage_1_configs, action_encoder_output_dims, self.particle_dimension_types)
        stage_1_particle_encoder_latent_space = self.stage_1_stuff["particle_encoder_latent_space"]
        self.stage_1_particle_encoder = self.stage_1_stuff["particle_encoder"]
        self.stage_1_residual_network = self.stage_1_stuff["residual_network"]

        # Make stage 
        self.stage_2_stuff = self._create_stage_models(stage_2_configs, action_encoder_output_dims, self.particle_dimension_types, stage_1_particle_encoder_latent_space)
        self.stage_2_particle_encoder = self.stage_2_stuff["particle_encoder"]
        self.stage_2_residual_network = self.stage_2_stuff["residual_network"]

        # Create the inverse mapping that we will need
        self.transformed_particle_to_particle_mapping = dict()
        self._create_transformed_particle_to_particle_mapping(self.stage_1_stuff["particle_dimensions_to_use"], 1, self.transformed_particle_to_particle_mapping)
        self._create_transformed_particle_to_particle_mapping(self.stage_2_stuff["particle_dimensions_to_use"], 2, self.transformed_particle_to_particle_mapping)

        # print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING ")
        # print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING ")
        # print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING ")
        # print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING ")
        # print("Dynamics model is hacked!! Check forward function")
        # print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING ")
        # print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING ")
        # print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING ")
        # print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING ")


    def forward(self, particles, actions):

        # Get some info
        device = particles.device
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]

        # Encode the actions if we have them and have an encoder to encode with
        if((self.action_encoder is not None) and (actions is not None)):
            encoded_actions = self.action_encoder(actions)
            encoded_actions = torch.tile(encoded_actions.unsqueeze(1), [1, number_of_particles, 1])
        else:
            encoded_actions = None

        # Compute the stage 1 outputs
        stage_1_output = self._run_stage(particles,self.stage_1_particle_encoder, self.stage_1_residual_network, self.stage_1_stuff, encoded_actions)

        # Re-encoder the stage 1 outputs so we can pass them into stage 2
        stage_1_output_encoded = self.stage_1_particle_encoder(torch.reshape(stage_1_output, (-1, stage_1_output.shape[-1])))
        stage_1_output_encoded = torch.reshape(stage_1_output_encoded, (batch_size, number_of_particles, -1))

        # stage_1_output_encoded = stage_1_output_encoded * 0.0

        # Compute the stage 2 outputs
        stage_2_output = self._run_stage(particles, self.stage_2_particle_encoder, self.stage_2_residual_network, self.stage_2_stuff, encoded_actions, stage_1_output_encoded)

        # Un-Transform the residual back into the particle dim
        new_particles = torch.zeros_like(particles)
        for i in range(particles.shape[-1]):

            # Get the info for this particle dim
            info = self.transformed_particle_to_particle_mapping[i]
            stage_1_or_stage_2, transformed_particles_dims_to_use, pd_type = info

            # Extract where the parts are coming from
            if(stage_1_or_stage_2 == 1):
                source = stage_1_output 
            elif(stage_1_or_stage_2 == 2):
                source = stage_2_output 

            # Create the final particle dim
            if(pd_type == "RealNumbers"):
                new_particles[..., i] = source[...,transformed_particles_dims_to_use[0]]
            elif(pd_type == "Angles"):
                new_particles[..., i] = torch.atan2(source[...,transformed_particles_dims_to_use[0]], source[...,transformed_particles_dims_to_use[1]])
            else:
                assert(False)

        # Return the particles
        return new_particles


    def _run_stage(self, particles, particle_encoder, residual_network, stage_stuff,  encoded_actions, additional_inputs=None):

        # Get some info
        device = particles.device
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]

        # Unpack the stuff we need
        residual_scale_factors = stage_stuff["residual_scale_factors"]
        particle_dimensions_to_use = stage_stuff["particle_dimensions_to_use"]
        total_transformed_dims = stage_stuff["total_transformed_input_dims"]
        noise_dim = stage_stuff["noise_dim"]
        particle_encoder_dims_to_use = stage_stuff["particle_encoder_dims_to_use"]
        total_particle_encoder_input_dims = stage_stuff["total_particle_encoder_input_dims"]



        # Move it to the correct device
        if(residual_scale_factors.device != device):
            residual_scale_factors = residual_scale_factors.to(device)

        # Get the transformed particles
        transformed_particles = self._extract_and_transform_particles(particles, particle_dimensions_to_use, total_transformed_dims)

        # Get the particles to use for the particle encoder
        transformed_particles_for_particle_encoder = self._extract_and_transform_particles(particles, particle_encoder_dims_to_use, total_particle_encoder_input_dims)

        # Encode the particles
        encoded_particles = particle_encoder(torch.reshape(transformed_particles_for_particle_encoder, (-1, transformed_particles_for_particle_encoder.shape[-1])))
        encoded_particles = torch.reshape(encoded_particles, (batch_size, number_of_particles, -1))

        # Make the residual network inputs
        residual_network_input = []
        residual_network_input.append(encoded_particles)
        if(encoded_actions is not None):
            residual_network_input.append(encoded_actions)
        if(additional_inputs is not None):
            residual_network_input.append(additional_inputs)
        noise = torch.normal(0.0, 1.0, size=(batch_size, number_of_particles, noise_dim)).to(device)
        residual_network_input.append(noise)
        residual_network_input = torch.cat(residual_network_input, dim=-1)

        # Run the residual network        
        residual_network_input = residual_network_input.view(-1, residual_network_input.shape[-1])
        residual_network_output = residual_network(residual_network_input)
        residual_network_output = residual_network_output.view(batch_size, number_of_particles, -1)

        # Scale the residuals
        residuals = (torch.sigmoid(residual_network_output) * 2.0) - 1.0
        residuals = residuals * residual_scale_factors

        # Compute the final output
        transformed_output = transformed_particles + residuals

        return transformed_output



    def _extract_and_transform_particles(self, particles, particle_dimensions_to_use, total_particle_encoder_input_dims):

        # Get some info
        device = particles.device
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]

        # Create the transformed particles
        transformed_particles = torch.zeros((batch_size, number_of_particles, total_particle_encoder_input_dims), device=device)

        # Go through the dims 1 by 1 and transform/fill in the transformed_particles
        transformed_particles_dim_idx = 0
        for particles_dim_idx in particle_dimensions_to_use:

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

        # Make sure we use all the dims
        assert(transformed_particles_dim_idx == total_particle_encoder_input_dims)

        # Future ops require the particles to be contiguous. 
        transformed_particles = transformed_particles.contiguous()
        
        return transformed_particles


    def _create_stage_models(self, configs, action_encoder_output_dims, particle_dimension_types, residual_additional_input_dims=None):

        # Get all the needed parameters for this stage
        particle_dimensions_to_use = get_mandatory_config("particle_dimensions_to_use", configs, "configs")
        particle_dimensions_residual_scale_factor = get_mandatory_config("particle_dimensions_residual_scale_factor", configs, "configs")
        particle_encoder_configs = get_mandatory_config("particle_encoder_configs", configs, "configs")
        residual_network_configs = get_mandatory_config("residual_network_configs", configs, "configs")
        particles_mask_out_dims = get_mandatory_config("particles_mask_out_dims", particle_encoder_configs, "particle_encoder_configs")

        # Create the particle encoder:
        particle_encoder_dims_to_use = [i for i in particle_dimensions_to_use if i not in particles_mask_out_dims]
        particle_encoder, particle_encoder_latent_space, total_particle_encoder_input_dims = self._create_particle_encoder(particle_encoder_configs, particle_encoder_dims_to_use, particle_dimension_types)

        # Count how many dims we will have
        total_transformed_input_dims = self._count_transformed_dims(particle_dimensions_to_use, particle_dimension_types)

        # The amount of noise to inject is just the size of the transformed particle encoder inputs
        noise_dim = total_transformed_input_dims

        # Figure out how many inputs the residual model has
        residual_input_dim = particle_encoder_latent_space + action_encoder_output_dims + noise_dim
        if(residual_additional_input_dims is not None):
            residual_input_dim += residual_additional_input_dims

        # Create the residual network        
        residual_network, residual_scale_factors = self._create_residual_network(residual_network_configs, residual_input_dim, particle_dimensions_to_use, particle_dimension_types, particle_dimensions_residual_scale_factor)

        # Pack into a return dict
        return_dict = dict()
        return_dict["particle_encoder"] = particle_encoder
        return_dict["particle_encoder_latent_space"] = particle_encoder_latent_space
        return_dict["residual_network"] = residual_network
        return_dict["residual_scale_factors"] = residual_scale_factors
        return_dict["particle_dimensions_to_use"] = particle_dimensions_to_use
        return_dict["total_transformed_input_dims"] = total_transformed_input_dims
        return_dict["noise_dim"] = noise_dim
        return_dict["particle_encoder_dims_to_use"] = particle_encoder_dims_to_use
        return_dict["total_particle_encoder_input_dims"] = total_particle_encoder_input_dims

        return return_dict


    def _create_particle_encoder(self, configs, particle_dimensions_to_use, particle_dimension_types):

        # Extract all the needed configs
        latent_space = get_mandatory_config("latent_space", configs, "configs")
        number_of_layers = get_mandatory_config("number_of_layers", configs, "configs")
        non_linear_type = get_mandatory_config("non_linear_type", configs, "action_encoder_configs")

        # Make sure we have particle dims to ingest
        assert(len(particle_dimensions_to_use) > 0)

        # Count how many dimes the transformed particles will be
        total_particle_encoder_input_dims = self._count_transformed_dims(particle_dimensions_to_use, particle_dimension_types)

        # Make the particle encoder
        return self._create_linear_FF_network(total_particle_encoder_input_dims, latent_space, non_linear_type, latent_space, number_of_layers), latent_space, total_particle_encoder_input_dims


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


    def _create_residual_network(self, configs, residual_input_dim, particle_dimensions_to_use, particle_dimension_types, particle_dimensions_residual_scale_factor):

        # Extract all the needed configs
        latent_space = get_mandatory_config("latent_space", configs, "configs")
        number_of_layers = get_mandatory_config("number_of_layers", configs, "configs")
        non_linear_type = get_mandatory_config("non_linear_type", configs, "action_encoder_configs")

        # Create residual scale factors that we will be using
        residual_scale_factors = []
        for i, particle_dim_idx in enumerate(particle_dimensions_to_use):

            # Get the info for this dims
            pd_type = particle_dimension_types[particle_dim_idx]
            residual_scale_factor = particle_dimensions_residual_scale_factor[i]

            # Create the scale factors
            if(pd_type == "RealNumbers"):
                residual_scale_factors.append(residual_scale_factor)
            elif(pd_type == "Angles"):
                residual_scale_factors.append(residual_scale_factor)
                residual_scale_factors.append(residual_scale_factor)

        # Compute the output size
        output_size = len(residual_scale_factors)

        # Convert to a tensor
        residual_scale_factors = torch.FloatTensor(residual_scale_factors)

        # Make the residual network
        return self._create_linear_FF_network(residual_input_dim, output_size, non_linear_type, latent_space, number_of_layers), residual_scale_factors


    def _create_action_encoder(self, configs):

        # get the parameters that we need
        action_input_number_of_dimensions = get_mandatory_config("action_input_number_of_dimensions", configs, "configs")
        latent_space = get_mandatory_config("latent_space", configs, "configs")
        number_of_layers = get_mandatory_config("number_of_layers", configs, "configs")
        non_linear_type = get_mandatory_config("non_linear_type", configs, "configs")

        # If we have 0 layers then we are basically turning off the action encoder
        if(number_of_layers == 0):
            return None, 0

        return self._create_linear_FF_network(action_input_number_of_dimensions, latent_space, non_linear_type, latent_space, number_of_layers), latent_space


    def _create_linear_FF_network(self, input_dim, output_dim, non_linear_type, latent_space, number_of_layers):

        # Need at least 2 layers, the input and output layers
        assert(number_of_layers >= 2)

        # Get the non linear object from the name
        non_linear_object = self._get_non_linear_object_from_string(non_linear_type)

        # All the layers that this model will have
        layers = nn.Sequential()
        
        # Create the input layer
        layers.append(nn.Linear(in_features=input_dim, out_features=latent_space))
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

    
    def _create_transformed_particle_to_particle_mapping(self, particles_to_use, stage_1_or_stage_2, transformed_particle_to_particle_mapping):

        # Make sure we only use 1 of the 2 stages
        assert((stage_1_or_stage_2 == 1) or (stage_1_or_stage_2 == 2))

        transformed_particles_dim_idx = 0
        for i, particles_dim_idx in enumerate(particles_to_use):

            # Make sure the dim is unique
            assert(particles_dim_idx not in transformed_particle_to_particle_mapping)

            # Get the type for this dimension
            pd_type = self.particle_dimension_types[particles_dim_idx]

            # Get which dims to use
            transformed_particles_dims_to_use = []
            if(pd_type == "RealNumbers"):
                transformed_particles_dims_to_use = [transformed_particles_dim_idx]
                transformed_particles_dim_idx += 1
            elif(pd_type == "Angles"):
                transformed_particles_dims_to_use = [transformed_particles_dim_idx, transformed_particles_dim_idx+1]
                transformed_particles_dim_idx += 2

            # Add it in
            transformed_particle_to_particle_mapping[particles_dim_idx] = (stage_1_or_stage_2, transformed_particles_dims_to_use, pd_type)
