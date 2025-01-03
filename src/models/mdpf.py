# Python Imports
import time

# Package Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

# Ali Package Import
from general_ml_framework.models.base_model import BaseModel
from general_ml_framework.utils.config import *

# Project Imports
from models.blocks.weight_models.create_models import create_weight_model
from models.blocks.dynamics_models.create_models import create_dynamics_model
from models.blocks.bandwidth_models.create_models import create_bandwidth_model
from models.blocks.observation_encoder_models.create_models import create_observation_encoder_model
from models.blocks.map_encoder_models.create_models import create_map_encoder_model
from models.base_models.particle_based_models import ParticleBasedModels
from kde.kde import KDE
from models.blocks.resampling.resampling import *


class InternalMDPF(nn.Module):

    # The available resampling methods
    # These are the methods that this class implements
    AVAILABLE_RESAMPLING_METHODS = set(["discrete_soft_resampling", "discrete_resampling"])

    def __init__(self, 
                dynamics_model, 
                output_weights_model, 
                output_bandwidth_model, 
                resampling_weights_model, 
                resampling_bandwidth_model, 
                number_of_initial_particles, 
                number_of_particles, 
                particle_dimension_parameters, 
                direction_mode, 
                particle_resampling_configs):
        super(InternalMDPF, self).__init__()

        # Make sure that the KDE and this classes resampling methods dont overlap
        union_of_sets = self.AVAILABLE_RESAMPLING_METHODS.union(KDE.AVAILABLE_RESAMPLING_METHODS)
        a = len(union_of_sets)
        b = len(self.AVAILABLE_RESAMPLING_METHODS)
        c = len(KDE.AVAILABLE_RESAMPLING_METHODS)
        assert(a == (b + c))

        # Save the models for later
        self.dynamics_model = dynamics_model
        self.output_weights_model = output_weights_model
        self.output_bandwidth_model = output_bandwidth_model
        self.resampling_weights_model = resampling_weights_model
        self.resampling_bandwidth_model = resampling_bandwidth_model

        # Make sure that if 1 is none then both are None
        assert(((self.resampling_weights_model is None) and (self.resampling_bandwidth_model is None)) or ((self.resampling_weights_model is not None) and (self.resampling_bandwidth_model is not None)))

        # Save the parameters
        self.number_of_initial_particles = number_of_initial_particles
        self.number_of_particles = number_of_particles
        self.direction_mode = direction_mode
        self.particle_dimension_parameters = particle_dimension_parameters
    
        # Check that the resampling method provided is valid
        self.particle_resampling_method = get_mandatory_config("resampling_method", particle_resampling_configs, "particle_resampling_configs")
        assert(self.particle_resampling_method in self.AVAILABLE_RESAMPLING_METHODS.union(KDE.AVAILABLE_RESAMPLING_METHODS))
    
        # Check the particle_dimension_parameters labels the keys
        # as 0, 1, 2, .... with no gaps
        all_keys = list(self.particle_dimension_parameters.keys())
        for i in range(len(all_keys)):
            assert(i == all_keys[i])

        # Create the kde_distribution_types
        self.kde_distribution_types = []
        for dim_idx in range(len(self.particle_dimension_parameters.keys())):

            # Get the parameters for this specific dim
            particle_dim_params = self.particle_dimension_parameters[dim_idx]

            # Get the kde_distribution_type for this paramater and save it
            kde_distribution_type = get_mandatory_config("kde_distribution_type", particle_dim_params, "particle_dim_params")
            self.kde_distribution_types.append(kde_distribution_type)


        # Create the particle resampling object
        if(self.particle_resampling_method in KDE.AVAILABLE_RESAMPLING_METHODS):
            self.particle_resampler = KDEResampler(self.kde_distribution_types, self.particle_resampling_method)
        elif(self.particle_resampling_method == "discrete_soft_resampling"):
            self.particle_resampler = SoftResamplingResampler(particle_resampling_configs)
        elif(self.particle_resampling_method == "discrete_resampling"):
            self.particle_resampler = DiscreteResamplingResampler(particle_resampling_configs)
        else:
            assert(False)


    def get_kde_distribution_types(self):
        return self.kde_distribution_types

    def forward(self, data, all_encoded_observations, encoded_global_map):

        # Unpack the data
        all_actions = data.get("actions", None)
        xy_position_global_frame_init = data.get("xy_position_global_frame_init", None)
        yaw_init = data.get("yaw_init", None)
        global_x_limits = data.get("global_x_limits", None)
        global_y_limits = data.get("global_y_limits", None)
        xy_position_global_frame = data.get("xy_position_global_frame", None)
        roll_pitch_yaw = data.get("roll_pitch_yaw", None)
        stage = data.get("stage", None)
        model_control_parameters = data.get("model_control_parameters", None)

        # If we have to truncate the gradients
        # If the truncation is less than 1 (aka invalid) then we effectively do not 
        # truncate so turn it off
        truncated_bptt_modulo = data.get("truncated_bptt_modulo", None)
        if((truncated_bptt_modulo is not None) and (truncated_bptt_modulo < 1)):
            truncated_bptt_modulo = None

        # get some information 
        device = all_encoded_observations.device
        batch_size = all_encoded_observations.shape[0]
        sequence_length = all_encoded_observations.shape[1]

        # Keep track of all the outputs
        all_outputs = dict()
        all_outputs["particles"] = []
        all_outputs["particle_weights"] = []
        all_outputs["bandwidths"] = []

        all_outputs["after_dynamics_particles"] = []
        all_outputs["after_dynamics_particle_weights"] = []
        all_outputs["after_dynamics_bandwidths"] = []

        # Run the sequence. skip the first step since that is out init step
        for step in range(0, sequence_length):

            # Get the seq idx for this iteration
            if(self.direction_mode == "forward"):
                seq_idx = step
            else:
                seq_idx = (sequence_length - step) - 1

            # Get the stuff for this sequence
            encoded_observations = all_encoded_observations.get_subsequence_index(seq_idx)
            
            if(step == 0):

                # Figure out how many particles to init with in case we have an override
                if((model_control_parameters is not None) and ("number_of_initial_particles" in model_control_parameters)):
                    number_of_particles_to_init_with = model_control_parameters["number_of_initial_particles"]
                else:
                    number_of_particles_to_init_with = self.number_of_initial_particles


                # Create the initial particle set
                if(self.direction_mode == "forward"):
                    new_particles, particle_weights = self._create_initial_particle_set(number_of_particles_to_init_with, xy_position_global_frame_init, yaw_init, global_x_limits, global_y_limits)
                else:
                    new_particles, particle_weights = self._create_initial_particle_set(number_of_particles_to_init_with, xy_position_global_frame[:, -1, :], roll_pitch_yaw[:, -1, -1], global_x_limits, global_y_limits)

                # Need this for later computation
                unnormalized_resampled_particle_log_weights = torch.log(particle_weights)

                # Get what the normalization eps should be
                if(unnormalized_resampled_particle_log_weights.dtype == torch.float32):
                    eps = 1e-8
                else:
                    eps = 1e-4

                # Compute the weights for after the dynamics
                after_dynamics_particle_weights = torch.exp(unnormalized_resampled_particle_log_weights)
                after_dynamics_particle_weights = torch.nn.functional.normalize(after_dynamics_particle_weights, p=1.0, eps=eps, dim=1)

                # Also need the bandwidth
                if(self.resampling_bandwidth_model is not None):
                    bandwidths = self.resampling_bandwidth_model(new_particles)
                else:
                    bandwidths = self.output_bandwidth_model(new_particles)

            else:


                # We may not actions to splice into but if we do get them
                if(all_actions is not None):

                    if(self.direction_mode == "forward"):
                        # In the forward mode we need the action from the previous state since that tells 
                        # us how to go from state_{t-1} to state_{t}.
                        actions = all_actions[:, seq_idx-1, ...]
                    else:
                        # In the backward mode we want to get the current action a_{t} because that tells us how to go from s_{t+1} to s_{t}. 
                        actions = all_actions[:, seq_idx, ...]                        
                else:
                    # No actions so set as None
                    actions = None


                # Figure out how many particles to resampled in case we have an override
                if((model_control_parameters is not None) and ("number_of_particles" in model_control_parameters)):
                    number_of_particles_to_resample = model_control_parameters["number_of_particles"]
                else:
                    number_of_particles_to_resample = self.number_of_particles

                # Resample the particles
                resampled_particles, unnormalized_resampled_particle_log_weights = self.particle_resampler(number_of_particles_to_resample, particles, particle_weights, bandwidths, stage)

                # Augment the particles
                new_particles = self.dynamics_model(resampled_particles, actions)


                # print(torch.min(new_particles[..., 3]).item(), torch.max(new_particles[..., 3]).item(),)

                # Get what the normalization eps should be
                if(unnormalized_resampled_particle_log_weights.dtype == torch.float32):
                    eps = 1e-8
                else:
                    eps = 1e-4

                # Compute the weights for after the dynamics
                after_dynamics_particle_weights = torch.exp(unnormalized_resampled_particle_log_weights)
                after_dynamics_particle_weights = torch.nn.functional.normalize(after_dynamics_particle_weights, p=1.0, eps=eps, dim=1)

            # Save the output that is right after the dynamics model
            # This is so we can use this MDPF object when implementing MDPS
            all_outputs["after_dynamics_particles"].append(new_particles)
            all_outputs["after_dynamics_particle_weights"].append(after_dynamics_particle_weights)
            all_outputs["after_dynamics_bandwidths"].append(bandwidths)

            # Compute output weights and bandwidths
            output_new_particle_weights = self._weight_particles(self.output_weights_model, new_particles, encoded_global_map, encoded_observations, unnormalized_resampled_particle_log_weights)
            output_new_bandwidths = self.output_bandwidth_model(new_particles)

            # If we are decoupling then we need to compute resampling weights and bandwidths otherwise they are the same
            if(self.resampling_weights_model is not None):
                resampling_new_particle_weights = self._weight_particles(self.resampling_weights_model, new_particles, encoded_global_map, encoded_observations, unnormalized_resampled_particle_log_weights)
                resampling_new_bandwidths = self.resampling_bandwidth_model(new_particles)
            else:
                resampling_new_particle_weights = output_new_particle_weights
                resampling_new_bandwidths = output_new_bandwidths

            # Save the output 
            all_outputs["particles"].append(new_particles)
            all_outputs["particle_weights"].append(output_new_particle_weights)
            all_outputs["bandwidths"].append(output_new_bandwidths)

            # Update the current particles and weights
            if(truncated_bptt_modulo is None):
                particles = new_particles
                particle_weights = resampling_new_particle_weights
                bandwidths = resampling_new_bandwidths
            else:
                if(((step % truncated_bptt_modulo) == 0) and (step != 0)):
                    particles = new_particles.detach()
                    particle_weights = resampling_new_particle_weights.detach()
                    bandwidths = resampling_new_bandwidths.detach()
                else:
                    particles = new_particles
                    particle_weights = resampling_new_particle_weights
                    bandwidths = resampling_new_bandwidths


        # If we went backwards in time then we need to reverse the sequences
        if(self.direction_mode == "backward"):
            for key in all_outputs.keys():
                all_outputs[key].reverse()

        return all_outputs


    def _create_initial_particle_set(self, number_of_particles_to_init_with, xy_position_global_frame_init, yaw_init, global_x_limits, global_y_limits):

        # Get some info
        batch_size = xy_position_global_frame_init.shape[0]
        device = xy_position_global_frame_init.device

        # Count how many dims we have
        number_of_particle_dims = len(self.particle_dimension_parameters.keys())

        # Create the particles
        particles = torch.zeros((batch_size, number_of_particles_to_init_with, number_of_particle_dims), device=device)

        # Init each of the particle dims
        for dim_idx in range(number_of_particle_dims):

            # Get the parameters for this specific dim
            particle_dim_params = self.particle_dimension_parameters[dim_idx]

            # Get the concentration type incase we need it for later
            kde_distribution_type = get_mandatory_config("kde_distribution_type", particle_dim_params, "particle_dim_params")
            assert((kde_distribution_type == "Normal") or (kde_distribution_type == "VonMises"))

            # Get the initialization parameters for this dim
            initialization_parameters = get_mandatory_config("initialization_parameters", particle_dim_params, "particle_dim_params")
            method = get_mandatory_config("method", initialization_parameters, "initialization_parameters")

            # If we initialize with the true state with a little bit of noise
            if(method == "true_state_with_small_noise"):

                # Get the parameters we need for this method
                noise_spread = get_mandatory_config("noise_spread", initialization_parameters, "initialization_parameters")
                
                # Get the True state parameters
                true_state_parameters = get_mandatory_config("true_state_parameters", initialization_parameters, "initialization_parameters")
                variable_name = get_mandatory_config("variable_name", true_state_parameters, "true_state_parameters")
                variable_dimenstion_index = get_mandatory_config("variable_dimenstion_index", true_state_parameters, "true_state_parameters")

                # Make sure the variable is valid
                assert((variable_name == "xy_position_global_frame_init") or (variable_name == "yaw_init"))

                # Extract the mean
                if(variable_name == "xy_position_global_frame_init"):
                    assert((variable_dimenstion_index == 0) or (variable_dimenstion_index == 1))
                    mean = xy_position_global_frame_init[..., variable_dimenstion_index]

                elif(variable_name == "yaw_init"):
                    assert(variable_dimenstion_index == 0)
                    mean = yaw_init

                # Make the distribution
                if(kde_distribution_type == "Normal"):
                    sampling_dist = D.Normal(loc=mean, scale=noise_spread)
                elif(kde_distribution_type == "VonMises"):
                    sampling_dist = D.VonMises(loc=mean, concentration=noise_spread)

                # Sample from the distribution 
                samples = sampling_dist.sample((number_of_particles_to_init_with,))

                # Need to permute the dims because the batch dim is the past dim when we samples here
                samples = torch.permute(samples, [1, 0])

                # Set the particles
                particles[..., dim_idx] = samples

            elif(method == "random"):
    
                # Get the parameters we need for this method
                min_value = get_mandatory_config("min_value", initialization_parameters, "initialization_parameters")
                max_value = get_mandatory_config("max_value", initialization_parameters, "initialization_parameters")

                # If one of them is a string then we are going to use the map to set the 
                # min and max values
                if((min_value is str) or (max_value is str)):

                    # Do some checks to make sure that the values are set correctly
                    assert(min_value is str)
                    assert(max_value is str)
                    assert(min_value == max_value)
                    assert((min_value == "global_x_limits") or (min_value == "global_y_limits"))

                    # If we are using the global x or y limits
                    if(min_value == "global_x_limits"):
                        assert(global_x_limits is not None)
                        min_value = global_x_limits[..., 0]
                        max_value = global_x_limits[..., 1]
                    elif(min_value == "global_y_limits"):
                        assert(global_y_limits is not None)
                        min_value = global_y_limits[..., 0]
                        max_value = global_y_limits[..., 1]


                # Draw the samples
                samples = torch.rand((batch_size, number_of_particles_to_init_with), device=device)

                # Scale the samples
                samples = samples * (max_value - min_value)
                samples = samples + min_value

                # Set the particles
                particles[..., dim_idx] = samples


            else:
                print("Unknown value for initialization \"method\": {}".format(method))
                assert(False)

        # Just make sure its contiguous
        particles = particles.contiguous()

        # The initial particle weights are uniform
        fill_value = 1.0 / float(number_of_particles_to_init_with)
        particle_weights = torch.full((batch_size, number_of_particles_to_init_with), fill_value, device=device)

        return particles, particle_weights


    def _weight_particles(self, weights_model, particles, encoded_global_map, encoded_observations, unnormalized_resampled_particle_log_weights):

        # Make the input dict
        input_dict = dict()
        input_dict["particles"] = particles
        input_dict["encoded_global_map"] = encoded_global_map
        input_dict["encoded_observations"] = encoded_observations
        input_dict["unnormalized_resampled_particle_log_weights"] = unnormalized_resampled_particle_log_weights



        # Compute the weights
        new_particle_weights = weights_model(input_dict)

        return new_particle_weights




class MDPF(ParticleBasedModels):
    def __init__(self, model_configs, model_architecture_configs):
        super(MDPF, self).__init__(model_configs, model_architecture_configs)

        # Extract the name of the model so we can get its model architecture params
        main_model_name = model_configs["main_model_name"]
        main_model_arch_configs = model_architecture_configs[main_model_name]

        # Get some configs
        self.number_of_initial_particles = get_mandatory_config("number_of_initial_particles", main_model_arch_configs, "main_model_arch_configs")
        self.number_of_particles = get_mandatory_config("number_of_particles", main_model_arch_configs, "main_model_arch_configs")
        self.decouple_output_and_resampling_distributions = get_mandatory_config("decouple_output_and_resampling_distributions", main_model_arch_configs, "main_model_arch_configs")
        self.direction_mode = get_mandatory_config("direction_mode", main_model_arch_configs, "main_model_arch_configs")
        self.particle_resampling_configs = get_mandatory_config("particle_resampling_configs", main_model_arch_configs, "main_model_arch_configs")
        particle_dimension_parameters = get_mandatory_config("particle_dimension_parameters", main_model_arch_configs, "main_model_arch_configs")

        # Get the internal model names
        internal_model_names = get_mandatory_config("internal_model_names", main_model_arch_configs, "main_model_arch_configs")

        # Create the internal models
        self.create_internal_models(internal_model_names, model_architecture_configs)

        # Make sure the direction mode is one of the correct directions
        assert((self.direction_mode == "forward") or (self.direction_mode == "backward"))
 
        # Create the internal MDPF
        self.internal_mdpf = InternalMDPF(
                                    self.dynamics_model,
                                    self.output_weights_model,
                                    self.output_bandwidth_model,
                                    self.resampling_weights_model,
                                    self.resampling_bandwidth_model,
                                    self.number_of_initial_particles,
                                    self.number_of_particles,
                                    particle_dimension_parameters,
                                    self.direction_mode,
                                    self.particle_resampling_configs)


    def create_internal_models(self, internal_model_names, model_architecture_configs):

        ####################################################################################
        ## Dynamics Models
        ####################################################################################
        self.dynamics_model = create_dynamics_model(self._get_model_config("dynamics_model", internal_model_names, model_architecture_configs))

        ####################################################################################
        ## Weight Models
        ####################################################################################
        self.output_weights_model = create_weight_model(self._get_model_config("weights_model", internal_model_names, model_architecture_configs))
        if(self.decouple_output_and_resampling_distributions):
            self.resampling_weights_model = create_weight_model(self._get_model_config("weights_model", internal_model_names, model_architecture_configs))
        else:
            self.resampling_weights_model = None


        ####################################################################################
        ## Bandwidth Models
        ####################################################################################
        self.output_bandwidth_model = create_bandwidth_model(self._get_model_config("bandwidth_model", internal_model_names, model_architecture_configs))
        if(self.decouple_output_and_resampling_distributions):

            # Check if we have the resampling model config, if not then use use the other bandwidth model config
            if("bandwidth_model_resampling" in internal_model_names):
                model_cfg = self._get_model_config("bandwidth_model_resampling", internal_model_names, model_architecture_configs)
            else:
                model_cfg = self._get_model_config("bandwidth_model", internal_model_names, model_architecture_configs)

            self.resampling_bandwidth_model = create_bandwidth_model(model_cfg)
        else:
            self.resampling_bandwidth_model = None

        ####################################################################################
        ## Observation Encoder models
        ####################################################################################
        self.observation_encoder_model = create_observation_encoder_model(self._get_model_config("observation_encoder_model", internal_model_names, model_architecture_configs))

        ####################################################################################
        ## Map Encoder models
        ####################################################################################
        self.map_encoder_model = create_map_encoder_model(self._get_model_config("map_encoder_model", internal_model_names, model_architecture_configs))

    def forward(self, data):

        # Unpack the data
        all_observations = data.get("observations", None)
        all_actions = data.get("actions", None)
        global_map = data.get("global_map", None)
        all_camera_data = data.get("camera_data", None)
        xy_position_global_frame_init = data.get("xy_position_global_frame_init", None)
        yaw_init = data.get("yaw_init", None)
        global_x_limits = data.get("global_x_limits", None)
        global_y_limits = data.get("global_y_limits", None)
        xy_position_global_frame = data.get("xy_position_global_frame", None)
        roll_pitch_yaw = data.get("roll_pitch_yaw", None)
        stage = data.get("stage", None)

        # If we have to truncate the gradients
        # If the truncation is less than 1 (aka invalid) then we effectively do not 
        # truncate so turn it off
        truncated_bptt_modulo = data.get("truncated_bptt_modulo", None)
        if((truncated_bptt_modulo is not None) and (truncated_bptt_modulo < 1)):
            truncated_bptt_modulo = None

        # We dont use truncation so set it to be off
        # assert(truncated_bptt_modulo == None)

        # get some information 
        device = all_observations.device
        batch_size = all_observations.shape[0]
        sequence_length = all_observations.shape[1]

        # with torch.autocast(device_type="cuda"):
            
        #     # Encode the global map
        #     if(self.map_encoder_model is not None):
        #         encoded_global_map = self.map_encoder_model(global_map)
        #     else:
        #         encoded_global_map = global_map

        #     # Encode the observations
        #     all_encoded_observations = self.observation_encoder_model(all_observations, all_camera_data)

        #     # Run the MDPF
        #     all_outputs = self.internal_mdpf(data, all_encoded_observations, encoded_global_map)


        
        # Encode the global map
        if(self.map_encoder_model is not None):
            encoded_global_map = self.map_encoder_model(global_map)
        else:
            encoded_global_map = global_map

        # Encode the observations
        all_encoded_observations = self.observation_encoder_model(all_observations, all_camera_data)

        # Run the MDPF
        all_outputs = self.internal_mdpf(data, all_encoded_observations, encoded_global_map)

        # We only do this if the stage is evaluation, otherwise we dont need the single point computations and so 
        # dont do them to save compute time
        if(stage == "evaluation"):

            # Compute the single point predictions using the max weight method
            # all_outputs["single_point_prediction_max_weight"] = self._compute_single_point_predictions_max_weight(all_outputs, all_encoded_observations, encoded_global_map, self.internal_mdpf.get_kde_distribution_types(), "particles", "particle_weights", "bandwidths")
            # all_outputs["forward_single_point_prediction_max_weight"] = self._compute_single_point_predictions_max_weight(all_outputs, all_encoded_observations, encoded_global_map, self.internal_mdpf.get_kde_distribution_types(), "forward_particles", "forward_particle_weights", "forward_bandwidths")
            # all_outputs["backward_single_point_prediction_max_weight"] = self._compute_single_point_predictions_max_weight(all_outputs, all_encoded_observations, encoded_global_map, self.backward_filter.get_kde_distribution_types(), "backward_particles", "backward_particle_weights", "backward_bandwidths")

            # Compute the top modes
            if(self.nms_mode_finding_configs is not None):

                all_outputs["top_modes"] = self._get_top_modes(all_outputs, self.internal_mdpf.get_kde_distribution_types(), "particles", "particle_weights", "bandwidths")

                # This is just the top mode so if we already have the modes then there is no need to compute this again, instead just get the top mode
                all_outputs["single_point_prediction"] = all_outputs["top_modes"][..., 0, :]

            else:
                # Compute the single point predictions for this sequence from the set of particles and weights
                all_outputs["single_point_prediction"] = self._compute_single_point_predictions_max_prob(all_outputs, self.internal_mdpf.get_kde_distribution_types(), "particles", "particle_weights", "bandwidths")
                all_outputs["after_dynamics_single_point_prediction"] = self._compute_single_point_predictions_max_prob(all_outputs, self.internal_mdpf.get_kde_distribution_types(), "after_dynamics_particles", "after_dynamics_particle_weights", "after_dynamics_bandwidths")


        return all_outputs


    def get_submodels(self):
        submodels = dict()
        submodels["dynamics_model"] = self.dynamics_model 
        submodels["output_weights_model"] = self.output_weights_model 
        submodels["output_bandwidth_model"] = self.output_bandwidth_model 

        if(self.decouple_output_and_resampling_distributions):
            submodels["resampling_weights_model"] = self.resampling_weights_model 
            submodels["resampling_bandwidth_model"] = self.resampling_bandwidth_model 


        if(self.observation_encoder_model is not None):        
            submodels["observation_encoder_model"] = self.observation_encoder_model

        if(self.map_encoder_model is not None):
            submodels["map_encoder_model"] = self.map_encoder_model

        return submodels


    def get_bandwidth_models(self):
        ''' 
            Get all the models that compute bandwidths for this model

            Parameters:
                None

            Returns:
                A dict of all the bandwidth models with their names
        '''

        bandwidth_models = dict()
        bandwidth_models["output_bandwidth_model"] = self.output_bandwidth_model 
        if(self.decouple_output_and_resampling_distributions):
            bandwidth_models["resampling_bandwidth_model"] = self.resampling_bandwidth_model 

        return bandwidth_models

