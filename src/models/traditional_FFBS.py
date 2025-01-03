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
from models.mdpf import InternalMDPF
from kde.kde import KDE

from models.blocks.voting import conv2d_fft_batchwise, TemplateSampler
from utils.general import rotmat2d, make_grid, sample_xyr



class TraditionalFFBS(ParticleBasedModels):
    def __init__(self, model_configs, model_architecture_configs):
        super(TraditionalFFBS, self).__init__(model_configs, model_architecture_configs)

        # Extract the name of the model so we can get its model architecture params
        main_model_name = model_configs["main_model_name"]
        main_model_arch_configs = model_architecture_configs[main_model_name]

        # Get some configs
        self.number_of_initial_particles = get_mandatory_config("number_of_initial_particles", main_model_arch_configs, "main_model_arch_configs")
        self.number_of_particles = get_mandatory_config("number_of_particles", main_model_arch_configs, "main_model_arch_configs")
        self.particle_resampling_configs = get_mandatory_config("particle_resampling_configs", main_model_arch_configs, "main_model_arch_configs")

        # The particle dims params for the forward and backward filters
        forward_particle_dimension_parameters = get_mandatory_config("forward_particle_dimension_parameters", main_model_arch_configs, "main_model_arch_configs")

        # Get the internal model names
        internal_model_names = get_mandatory_config("internal_model_names", main_model_arch_configs, "main_model_arch_configs")

        # Create the internal models
        self.create_internal_models(internal_model_names, model_architecture_configs)

        # The KDE distribution dim types
        self.kde_distribution_types = ["Normal", "Normal", "VonMises"]


        # Create the internal forward MDPF
        self.forward_filter = InternalMDPF(
                                    self.forward_dynamics_model,
                                    self.forward_output_weights_model,
                                    self.forward_output_bandwidth_model,
                                    self.forward_output_weights_model,
                                    self.forward_resampling_bandwidth_model,
                                    self.number_of_initial_particles,
                                    self.number_of_particles,
                                    forward_particle_dimension_parameters,
                                    "forward",
                                    self.particle_resampling_configs)

        # Needed for when we do dense search refinement
        self.template_sampler = None

    def create_internal_models(self, internal_model_names, model_architecture_configs):

        ####################################################################################
        ## Dynamics Models
        ####################################################################################
        self.forward_dynamics_model = create_dynamics_model(self._get_model_config("forward_dynamics_model", internal_model_names, model_architecture_configs))

        ####################################################################################
        ## Weight Models
        ####################################################################################
        self.forward_output_weights_model = create_weight_model(self._get_model_config("forward_weights_model", internal_model_names, model_architecture_configs))
        self.forward_resampling_weights_model = create_weight_model(self._get_model_config("forward_weights_model", internal_model_names, model_architecture_configs))

        ####################################################################################
        ## Bandwidth Models
        ####################################################################################
        self.forward_output_bandwidth_model = create_bandwidth_model(self._get_model_config("forward_bandwidth_model", internal_model_names, model_architecture_configs))
        self.output_bandwidth_model = create_bandwidth_model(self._get_model_config("output_bandwidth_model", internal_model_names, model_architecture_configs))
            
        ####################################################################################
        ## The final output single point prediction weight model. Note that 
        ## this is an optional model and so we may not always create this model
        ####################################################################################
        if("single_point_prediction_weight_model" in internal_model_names):
            self.single_point_prediction_weight_model = create_weight_model(self._get_model_config("single_point_prediction_weight_model", internal_model_names, model_architecture_configs))
        else:
            self.single_point_prediction_weight_model = None 



        # Check if we have the resampling model config, if not then use use the other bandwidth model config
        if("forward_bandwidth_model_resampling" in internal_model_names):
            model_cfg = self._get_model_config("forward_bandwidth_model_resampling", internal_model_names, model_architecture_configs)
        else:
            model_cfg = self._get_model_config("forward_bandwidth_model", internal_model_names, model_architecture_configs)
        self.forward_resampling_bandwidth_model = create_bandwidth_model(model_cfg)

        ####################################################################################
        ## Observation Encoder models
        ####################################################################################
        self.observation_encoder_model = create_observation_encoder_model(self._get_model_config("observation_encoder_model", internal_model_names, model_architecture_configs))

        ####################################################################################
        ## Map Encoder models
        ####################################################################################
        self.map_encoder_model = create_map_encoder_model(self._get_model_config("map_encoder_model", internal_model_names, model_architecture_configs))


    def get_kde_distribution_types(self):
        return self.kde_distribution_types


    def get_submodels(self):
        submodels = dict()

        submodels["forward_dynamics_model"] = self.forward_dynamics_model 

        submodels["forward_output_weights_model"] = self.forward_output_weights_model
        submodels["forward_resampling_weights_model"] = self.forward_resampling_weights_model

        submodels["forward_output_bandwidth_model"] = self.forward_output_bandwidth_model
        submodels["forward_resampling_bandwidth_model"] = self.forward_resampling_bandwidth_model
        submodels["output_bandwidth_model"] = self.output_bandwidth_model


        if(self.observation_encoder_model is not None):        
            submodels["observation_encoder_model"] = self.observation_encoder_model

        if(self.map_encoder_model is not None):
            submodels["map_encoder_model"] = self.map_encoder_model

        if(self.single_point_prediction_weight_model is not None):
            submodels["single_point_prediction_weight_model"] = self.single_point_prediction_weight_model

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

        bandwidth_models["forward_output_bandwidth_model"] = self.forward_output_bandwidth_model
        bandwidth_models["forward_resampling_bandwidth_model"] = self.forward_resampling_bandwidth_model
        bandwidth_models["output_bandwidth_model"] = self.output_bandwidth_model

        return bandwidth_models


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
        stage = data.get("stage", None)
        xy_gt = data["xy_position_global_frame"]

        # If we have to truncate the gradients
        # If the truncation is less than 1 (aka invalid) then we effectively do not 
        # truncate so turn it off
        truncated_bptt_modulo = data.get("truncated_bptt_modulo", None)
        if((truncated_bptt_modulo is not None) and (truncated_bptt_modulo < 1)):
            truncated_bptt_modulo = None

        # get some information 
        device = all_observations.device
        batch_size = all_observations.shape[0]
        sequence_length = all_observations.shape[1]

        # Encode the global map
        if(self.map_encoder_model is not None):
            encoded_global_map = self.map_encoder_model(global_map)
        else:
            encoded_global_map = global_map

        # Encode the observations
        all_encoded_observations = self.observation_encoder_model(all_observations, all_camera_data)

        # Do forward and backward passes
        forward_outputs = self.forward_filter(data, all_encoded_observations, encoded_global_map)

        # get what we need
        forward_particles = forward_outputs["particles"]
        forward_particle_weights = forward_outputs["particle_weights"]

        # Compute the Backward Smoother
        backward_particle_weights = [None] * sequence_length
        backward_particle_weights[-1] = forward_particle_weights[-1]

        bandwidths = [None] * sequence_length

        for i in range(sequence_length-1, -1, -1):

            # Skip the last in the sequence
            if(i == sequence_length-1):
                continue

            # Get the stuff
            particles_t = forward_particles[i]
            particles_t_1 = forward_particles[i+1]
            forward_weights_t = forward_particle_weights[i]
            backward_weights_t_1 = backward_particle_weights[i+1]

            # Compute the log probs for all the particles
            log_probs = self.forward_dynamics_model.evaluate_particles(particles_t, particles_t_1)
            probs = torch.exp(log_probs)

            # Compute denominator
            denominator = probs * forward_weights_t.unsqueeze(-1)
            denominator = torch.sum(denominator, dim=1)
                
            # Compute inside the sum
            w_sum = probs / denominator.unsqueeze(1)    
            w_sum = w_sum * backward_weights_t_1.unsqueeze(1)
            w_sum = torch.sum(w_sum, dim=-1)

            # Compute the weight
            backward_weights_t = w_sum * forward_weights_t

            # Set it
            backward_particle_weights[i] = backward_weights_t
        
            # Compute bandwidth
            bandwidths[i] = self.output_bandwidth_model(particles_t)

        # Save the output 
        all_outputs = dict()
        all_outputs["particles"] = forward_particles
        all_outputs["particle_weights"] = backward_particle_weights
        all_outputs["bandwidths"] = bandwidths

        return all_outputs

