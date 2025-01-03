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



class MDPS(ParticleBasedModels):
    def __init__(self, model_configs, model_architecture_configs):
        super(MDPS, self).__init__(model_configs, model_architecture_configs)

        # Extract the name of the model so we can get its model architecture params
        main_model_name = model_configs["main_model_name"]
        main_model_arch_configs = model_architecture_configs[main_model_name]

        # Get some configs
        self.number_of_initial_particles = get_mandatory_config("number_of_initial_particles", main_model_arch_configs, "main_model_arch_configs")
        self.number_of_particles = get_mandatory_config("number_of_particles", main_model_arch_configs, "main_model_arch_configs")
        self.particle_resampling_configs = get_mandatory_config("particle_resampling_configs", main_model_arch_configs, "main_model_arch_configs")
        self.do_dense_search_refinement = get_mandatory_config("do_dense_search_refinement", main_model_arch_configs, "main_model_arch_configs")

        # The particle dims params for the forward and backward filters
        forward_particle_dimension_parameters = get_mandatory_config("forward_particle_dimension_parameters", main_model_arch_configs, "main_model_arch_configs")
        backward_particle_dimension_parameters = get_mandatory_config("backward_particle_dimension_parameters", main_model_arch_configs, "main_model_arch_configs")

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

        # Create the internal forward MDPF
        self.backward_filter = InternalMDPF(
                                    self.backward_dynamics_model,
                                    self.backward_output_weights_model,
                                    self.backward_output_bandwidth_model,
                                    self.backward_output_weights_model,
                                    self.backward_resampling_bandwidth_model,
                                    self.number_of_initial_particles,
                                    self.number_of_particles,
                                    backward_particle_dimension_parameters,
                                    "backward",
                                    self.particle_resampling_configs)

        # Needed for when we do dense search refinement
        self.template_sampler = None

    def create_internal_models(self, internal_model_names, model_architecture_configs):

        ####################################################################################
        ## Dynamics Models
        ####################################################################################
        self.forward_dynamics_model = create_dynamics_model(self._get_model_config("forward_dynamics_model", internal_model_names, model_architecture_configs))
        self.backward_dynamics_model = create_dynamics_model(self._get_model_config("backward_dynamics_model", internal_model_names, model_architecture_configs))

        ####################################################################################
        ## Weight Models
        ####################################################################################
        self.forward_output_weights_model = create_weight_model(self._get_model_config("forward_weights_model", internal_model_names, model_architecture_configs))
        self.backward_output_weights_model = create_weight_model(self._get_model_config("backward_weights_model", internal_model_names, model_architecture_configs))
        self.forward_resampling_weights_model = create_weight_model(self._get_model_config("forward_weights_model", internal_model_names, model_architecture_configs))
        self.backward_resampling_weights_model = create_weight_model(self._get_model_config("backward_weights_model", internal_model_names, model_architecture_configs))
        self.output_weights_model = create_weight_model(self._get_model_config("output_weights_model", internal_model_names, model_architecture_configs))

        ####################################################################################
        ## Bandwidth Models
        ####################################################################################
        self.forward_output_bandwidth_model = create_bandwidth_model(self._get_model_config("forward_bandwidth_model", internal_model_names, model_architecture_configs))
        self.backward_output_bandwidth_model = create_bandwidth_model(self._get_model_config("backward_bandwidth_model", internal_model_names, model_architecture_configs))
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

        # Check if we have the resampling model config, if not then use use the other bandwidth model config
        if("backward_bandwidth_model_resampling" in internal_model_names):
            model_cfg = self._get_model_config("backward_bandwidth_model_resampling", internal_model_names, model_architecture_configs)
        else:
            model_cfg = self._get_model_config("backward_bandwidth_model", internal_model_names, model_architecture_configs)
        self.backward_resampling_bandwidth_model = create_bandwidth_model(model_cfg)

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
        submodels["backward_dynamics_model"] = self.backward_dynamics_model 

        submodels["forward_output_weights_model"] = self.forward_output_weights_model
        submodels["backward_output_weights_model"] = self.backward_output_weights_model
        submodels["forward_resampling_weights_model"] = self.forward_resampling_weights_model
        submodels["backward_resampling_weights_model"] = self.backward_resampling_weights_model
        submodels["output_weights_model"] = self.output_weights_model

        submodels["forward_output_bandwidth_model"] = self.forward_output_bandwidth_model
        submodels["backward_output_bandwidth_model"] = self.backward_output_bandwidth_model
        submodels["forward_resampling_bandwidth_model"] = self.forward_resampling_bandwidth_model
        submodels["backward_resampling_bandwidth_model"] = self.backward_resampling_bandwidth_model
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
        bandwidth_models["backward_output_bandwidth_model"] = self.backward_output_bandwidth_model
        bandwidth_models["forward_resampling_bandwidth_model"] = self.forward_resampling_bandwidth_model
        bandwidth_models["backward_resampling_bandwidth_model"] = self.backward_resampling_bandwidth_model
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
        backward_outputs = self.backward_filter(data, all_encoded_observations, encoded_global_map)

        all_outputs = dict()

        # Pack in the forward outputs
        for key in forward_outputs.keys():
            all_outputs["forward_{}".format(key)] = forward_outputs[key]   

        # Pack in the backward outputs
        for key in backward_outputs.keys():
            all_outputs["backward_{}".format(key)] = backward_outputs[key]   

        # Compute the posterior distribution
        self._compute_posterior(all_outputs, forward_outputs, backward_outputs, encoded_global_map, all_encoded_observations)


        # We only do this if the stage is evaluation, otherwise we dont need the single point computations and so 
        # dont do them to save compute time
        if(stage == "evaluation"):

            # Compute the single point predictions using the max weight method
            # all_outputs["single_point_prediction_max_weight"] = self._compute_single_point_predictions_max_weight(all_outputs, all_encoded_observations, encoded_global_map, self.forward_filter.get_kde_distribution_types(), "particles", "particle_weights", "bandwidths")
            # all_outputs["forward_single_point_prediction_max_weight"] = self._compute_single_point_predictions_max_weight(all_outputs, all_encoded_observations, encoded_global_map, self.forward_filter.get_kde_distribution_types(), "forward_particles", "forward_particle_weights", "forward_bandwidths")
            # all_outputs["backward_single_point_prediction_max_weight"] = self._compute_single_point_predictions_max_weight(all_outputs, all_encoded_observations, encoded_global_map, self.backward_filter.get_kde_distribution_types(), "backward_particles", "backward_particle_weights", "backward_bandwidths")

            # Compute the top modes
            if(self.nms_mode_finding_configs is not None):
                all_outputs["top_modes"] = self._get_top_modes(all_outputs, self.forward_filter.get_kde_distribution_types(), "particles", "particle_weights", "bandwidths")
                all_outputs["forward_top_modes"] = self._get_top_modes(all_outputs, self.forward_filter.get_kde_distribution_types(), "forward_particles", "forward_particle_weights", "forward_bandwidths")
                all_outputs["backward_top_modes"] = self._get_top_modes(all_outputs, self.backward_filter.get_kde_distribution_types(), "backward_particles", "backward_particle_weights", "backward_bandwidths")

                # This is just the top mode so if we already have the modes then there is no need to compute this again, instead just get the top mode
                all_outputs["single_point_prediction"] = all_outputs["top_modes"][..., 0, :]
                all_outputs["forward_single_point_prediction"] = all_outputs["forward_top_modes"][..., 0, :]
                all_outputs["backward_single_point_prediction"] = all_outputs["backward_top_modes"][..., 0, :]

                # Do some refinement!!!!                
                if(self.do_dense_search_refinement):    
                    all_outputs["top_modes_refined"] = self._do_dense_search_refinement(all_outputs["top_modes"], encoded_global_map, all_encoded_observations)
                    all_outputs["forward_top_modes_refined"] = self._do_dense_search_refinement(all_outputs["forward_top_modes"], encoded_global_map, all_encoded_observations)
                    all_outputs["backward_top_modes_refined"] = self._do_dense_search_refinement(all_outputs["backward_top_modes"], encoded_global_map, all_encoded_observations)

                    # And again extract the top mode
                    all_outputs["single_point_prediction_refined"] = all_outputs["top_modes_refined"][..., 0, :]
                    all_outputs["forward_single_point_prediction_refined"] = all_outputs["forward_top_modes_refined"][..., 0, :]
                    all_outputs["backward_single_point_prediction_refined"] = all_outputs["backward_top_modes_refined"][..., 0, :]


            else:
                # Compute the single point predictions for this sequence from the set of particles and weights
                all_outputs["single_point_prediction"] = self._compute_single_point_predictions_max_prob(all_outputs, self.forward_filter.get_kde_distribution_types(), "particles", "particle_weights", "bandwidths")
                all_outputs["forward_single_point_prediction"] = self._compute_single_point_predictions_max_prob(all_outputs, self.forward_filter.get_kde_distribution_types(), "forward_particles", "forward_particle_weights", "forward_bandwidths")
                all_outputs["backward_single_point_prediction"] = self._compute_single_point_predictions_max_prob(all_outputs, self.backward_filter.get_kde_distribution_types(), "backward_particles", "backward_particle_weights", "backward_bandwidths")




        return all_outputs


    def _compute_posterior(self, all_outputs, forward_outputs, backward_outputs, encoded_global_map, all_encoded_observations):

        # Keep track of all the outputs
        all_outputs["particles"] = []
        all_outputs["particle_weights"] = []
        all_outputs["bandwidths"] = []

        for i in range(len(forward_outputs["particles"])):

            # Get the stuff for this sequence
            encoded_observations = all_encoded_observations.get_subsequence_index(i)

            # Unpack the forward outputs
            forward_particles = forward_outputs["after_dynamics_particles"][i]
            forward_particle_weights = forward_outputs["after_dynamics_particle_weights"][i]
            forward_bandwidths = forward_outputs["after_dynamics_bandwidths"][i]

            # Unpack the backward outputs
            backward_particles = backward_outputs["after_dynamics_particles"][i]
            backward_particle_weights = backward_outputs["after_dynamics_particle_weights"][i]
            backward_bandwidths = backward_outputs["after_dynamics_bandwidths"][i]

            # Create the forward and backward KDEs
            # We dont need to specify a resampling method because there is no resampling that needs to happen here
            forward_kde_dist = KDE(self.forward_filter.get_kde_distribution_types(), forward_particles, forward_particle_weights, forward_bandwidths, particle_resampling_method=None)
            backward_kde_dist = KDE(self.backward_filter.get_kde_distribution_types(), backward_particles, backward_particle_weights, backward_bandwidths, particle_resampling_method=None)

            # Create a unified particle set
            unified_particle_set = torch.cat([forward_particles, backward_particles], dim=1)

            # Compute the log prob of the unified particles against the 2 filter outputs
            forward_log_probs = forward_kde_dist.log_prob(unified_particle_set)
            backward_log_probs = backward_kde_dist.log_prob(unified_particle_set)

            # Compute the resampling weights
            unnormalized_resampled_particle_log_weights = torch.cat([forward_log_probs.unsqueeze(-1), backward_log_probs.unsqueeze(-1)], dim=-1)
            unnormalized_resampled_particle_log_weights = unnormalized_resampled_particle_log_weights + float(np.log(0.5))
            unnormalized_resampled_particle_log_weights = torch.logsumexp(unnormalized_resampled_particle_log_weights, dim=-1)
            # unnormalized_resampled_particle_log_weights = unnormalized_resampled_particle_log_weights - unnormalized_resampled_particle_log_weights.detach()
            unnormalized_resampled_particle_log_weights = -unnormalized_resampled_particle_log_weights.detach() # BUG FIX BUG FIX to match eqn. 20 in the paper

            # Create the additional inputs
            additional_inputs = torch.cat([forward_log_probs.unsqueeze(-1), backward_log_probs.unsqueeze(-1)], dim=-1)

            # Make the input dict
            input_dict = dict()
            input_dict["particles"] = unified_particle_set
            input_dict["encoded_observations"] = encoded_observations
            input_dict["unnormalized_resampled_particle_log_weights"] = unnormalized_resampled_particle_log_weights
            input_dict["additional_inputs"] = additional_inputs
            input_dict["encoded_global_map"] = encoded_global_map

            # Compute the weights
            unified_particle_weights = self.output_weights_model(input_dict)

            # Predict a bandwidth
            bandwidth = self.output_bandwidth_model(unified_particle_set)

            # Save the output 
            all_outputs["particles"].append(unified_particle_set)
            all_outputs["particle_weights"].append(unified_particle_weights)
            all_outputs["bandwidths"].append(bandwidth)



    def _do_dense_search_refinement(self, all_top_modes, encoded_global_map, all_encoded_observations):

        # Get some info
        device = all_encoded_observations.device
        batch_size = all_encoded_observations.shape[0]
        sequence_length = all_encoded_observations.shape[1]

        # Process step at a time
        all_refined_positions = []
        for seq_idx in range(sequence_length):

            # Get the stuff needed for this timestep
            encoded_observations = all_encoded_observations.get_subsequence_index(seq_idx)
            top_modes = all_top_modes[:, seq_idx, ...]

            # We only care about the x and y dims since we will do a dense search
            xy_top_modes = top_modes[..., 0:2]

            # for each of the modes refine the position
            refined_positions = []
            for mode_idx in range(top_modes.shape[1]):

                # Get the local map
                local_map, local_map_center = self._crop_local_map_from_global_map(encoded_global_map, xy_top_modes[:, mode_idx, :])

                # Compute the refined position
                refined_position = self._compute_refined_local_position(encoded_observations, local_map, local_map_center.float())
                refined_positions.append(refined_position)

            # Stack and append
            all_refined_positions.append(torch.stack(refined_positions, dim=1))

        # Final stack
        all_refined_positions = torch.stack(all_refined_positions, dim=1)

        return all_refined_positions

    def _crop_local_map_from_global_map(self, global_map, positions):

        # Get some info
        batch_size = positions.shape[0]
        device = positions.device

        # Cropped local maps
        local_maps = []
        local_map_centers = []

        # The size of the map to extract
        # @TODO: make this smaller
        # @TODO: make this a config parameter that gets loaded in 
        size_of_extracted_local_map_when_using_the_global_map = 256

        # Process each batch one at a time.
        # This is so not efficient but its fine for now until we do something more fancy and GPU friendly
        for b_idx in range(batch_size): 

            # Get the start and end x indices 
            s_x = int(positions[b_idx, 0].item()) - (size_of_extracted_local_map_when_using_the_global_map // 2)
            e_x = s_x + size_of_extracted_local_map_when_using_the_global_map

            # Get the start and end y indices 
            s_y = int(positions[b_idx, 1].item()) - (size_of_extracted_local_map_when_using_the_global_map // 2)
            e_y = s_y + size_of_extracted_local_map_when_using_the_global_map

            # Make sure its in range
            if(s_x < 0):
                e_x = e_x + (-s_x)
                s_x = 0

            if(e_x >= global_map.shape[3]):
                diff = e_x - global_map.shape[3]
                e_x -= diff
                s_x -= diff

            if(s_y < 0):
                e_y = e_y + (-s_y)
                s_y = 0

            if(e_y >= global_map.shape[2]):
                diff = e_y - global_map.shape[2]
                e_y -= diff
                s_y -= diff

            # Compute the center of the local map and save it
            c_x = (e_x + s_x) / 2.0
            c_y = (e_y + s_y) / 2.0
            local_map_center = torch.FloatTensor([c_x, c_y]).to(device)
            local_map_centers.append(local_map_center.unsqueeze(0))

            # Do the crop
            cropped_map = global_map[b_idx, :, s_y:e_y, s_x:e_x]
            local_maps.append(cropped_map.unsqueeze(0))

        # Make into a tensor again
        local_maps = torch.cat(local_maps, dim=0)
        local_map_centers = torch.cat(local_map_centers, dim=0)

        return local_maps, local_map_centers



    def _exhaustive_voting(self, f_bev, f_map, valid_bev, confidence_bev):
    

        # Create the template sampler
        if(self.template_sampler is None):

            # Get the number of matching rotations based on the stage we are currently in
            # num_of_matching_rotations = self.number_of_matching_rotations[stage]
            # @TODO: make this a config parameter that gets loaded in 
            num_of_matching_rotations = 32
            pixels_per_meter = 2

            # Create the sampler
            # This has no learned parameters
            self.template_sampler = TemplateSampler(self.observation_encoder_model.get_bev_grid_xz(), pixels_per_meter, num_of_matching_rotations)

            # Move it to the correct device
            self.template_sampler = self.template_sampler.to(f_bev.device)


        # Multiply by the confidence
        f_bev = f_bev * confidence_bev.unsqueeze(1)

        # Mask out the invalid pixels
        f_bev = f_bev.masked_fill(~valid_bev.unsqueeze(1), 0.0)

        # Get the sampling template
        templates = self.template_sampler(f_bev)

        # Compute the matching scores
        with torch.autocast("cuda", enabled=False):
            matching_scores = conv2d_fft_batchwise(f_map.float(), templates.float(), padding_mode="replicate")

        # Reweight the different rotations based on the number of valid pixels
        # in each template. Axis-aligned rotation have the maximum number of valid pixels.
        valid_templates = self.template_sampler(valid_bev.float()[None]) > (1 - 1e-4)
        num_valid = valid_templates.float().sum((-3, -2, -1))
        matching_scores = matching_scores / num_valid[..., None, None]

        return matching_scores


    def _compute_refined_local_position(self, encoded_observations, local_maps, log_probs_center_global_frame):

        # Unpack the obs
        bev_observations, bev_valid, bev_confidence = encoded_observations

        # Compute the matching scores
        matching_scores = self._exhaustive_voting(bev_observations, local_maps, bev_valid, bev_confidence)

        # Convert from [B, Rot, H, W] -> [B, H, W, Rot]
        # Do this because all the loss functions and rendering code expects this dim ordering
        # Annoying but I dont want to re-write all that rendering code and stuff so we live 
        # with this
        matching_scores = torch.movedim(matching_scores, 1, -1)

        # Compute the log probs for the locations
        log_probs = torch.nn.functional.log_softmax(matching_scores.flatten(-3), dim=-1).reshape(matching_scores.shape)

        # get some info
        batch_size  = log_probs.shape[0]
        H  = log_probs.shape[1]
        W  = log_probs.shape[2]
        R  = log_probs.shape[3]

        # We need to find the argmax of this so we need to flatten
        log_probs_flattened = log_probs.view([batch_size, -1])

        # Find the arg max
        _, max_idx = torch.max(log_probs_flattened, dim=-1)

        # Convert back from a max of 1D to a max of 3D
        WR = W * R
        y = torch.div(max_idx, WR, rounding_mode="floor")
        x = torch.div(max_idx % WR, R, rounding_mode="floor")
        r = max_idx % R

        # Need to convert from index into an angle
        r = r / float(R)
        r = r * 2.0 * np.pi

        # Stack into 1 tensor
        max_value = torch.stack([x, y, r], dim=-1)

        # Make the middle of the map the center
        max_value[:, 0] -= H // 2
        max_value[:, 1] -= W // 2

        # Add in the local map offset if we have it
        max_value[:, 0:2] += log_probs_center_global_frame

        return max_value
