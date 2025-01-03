# Python Imports
import time

# Package Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from tqdm import tqdm

# Ali Package Import
from general_ml_framework.models.base_model import BaseModel
from general_ml_framework.utils.config import *

# Project Imports
from utils.visualization import Colormap
from models.blocks.map_encoder_models.create_models import create_map_encoder_model
from models.blocks.observation_encoder_models.create_models import create_observation_encoder_model
from models.base_models.discrete_prob_volume_models import DiscreteProbVolumeModel 

class EmbeddingMapsAndImages(DiscreteProbVolumeModel):

    '''
        Implementation of this paper:
            - "You Are Here: Geolocation by Embedding Maps and Images"
            - https://arxiv.org/pdf/1911.08797.pdf
    '''
    def __init__(self, model_configs, model_architecture_configs):
        super(EmbeddingMapsAndImages, self).__init__(model_configs, model_architecture_configs)

        # Extract the name of the model so we can get its model architecture params
        main_model_name = model_configs["main_model_name"]
        main_model_arch_configs = model_architecture_configs[main_model_name]

        # Get the internal model names
        internal_model_names = get_mandatory_config("internal_model_names", main_model_arch_configs, "main_model_arch_configs")

        # Create the encoders
        self.observation_encoder_model = create_observation_encoder_model(self._get_model_config("observation_encoder_model", internal_model_names, model_architecture_configs))
        self.map_encoder_model = create_map_encoder_model(self._get_model_config("map_encoder_model", internal_model_names, model_architecture_configs))
    
        # Get the number of matching rotations to use
        self.number_of_matching_rotations = get_mandatory_config("number_of_matching_rotations", main_model_arch_configs, "main_model_arch_configs")

        # Get the size of the cropped local map and make sure its divisible by 2
        self.size_of_extracted_local_map_when_using_the_global_map = get_mandatory_config("size_of_extracted_local_map_when_using_the_global_map", main_model_arch_configs, "main_model_arch_configs")
        assert((self.size_of_extracted_local_map_when_using_the_global_map % 2) == 0)


    def get_submodels(self):
        submodels = dict()
        submodels["observation_encoder_model"] = self.observation_encoder_model
        submodels["map_encoder_model"] = self.map_encoder_model
        return submodels


    def forward(self, data):

        # Unpack the stage we are in since that will determine what kind of output we will be doing
        stage = data["stage"]

        # If we are training then we will train the encoders only 
        if((stage == "training") or (stage == "validation")):

            # Get the data we need and pass it in
            observations = data["observations"]
            local_maps = data["local_maps"]
            local_map_center_global_frame = data.get("local_map_center_global_frame", None)

            # Process
            log_probs, log_probs_center_global_frame = self._process_sequence_local(stage, observations, local_maps, local_map_center_global_frame)

            # Pack into the return dict and return it
            return_dict = dict()    
            return_dict["discrete_map_log_probs"] = log_probs
            return_dict["discrete_map_log_probs_center_global_frame"] = log_probs_center_global_frame
            return_dict["extracted_local_maps"] = local_maps
            return return_dict

        elif(stage == "evaluation"):
            return self._do_evaluation_forward(data)

        else:
            raise NotImplemented


    def _process_sequence_local(self, stage, observations, local_maps, local_map_center_global_frame, observation_already_encoded=False):

        # Get some info
        device = observations.device
        batch_size = observations.shape[0]
        sequence_length = observations.shape[1]

        # Sometimes its already processed so check if we need to process or not
        if(observation_already_encoded == False):
            
            # Flatten Observations so we can process them
            B, S, oC, oH, oW = observations.shape
            observations = torch.reshape(observations, (B*S, oC, oH, oW))

            # Encode the observations
            encoded_observations = self.observation_encoder_model(observations, None).get()

        else:

            # Its already processed so we just need to flatten it
            encoded_observations = torch.reshape(observations, (batch_size*sequence_length, -1))

        # Flatten local maps so we can process them
        B, S, lC, lH, lW = local_maps.shape
        local_maps = torch.reshape(local_maps, (B*S, lC, lH, lW))


        # # Encode the local maps
        # encoded_local_maps = []
        # for rot_idx in range(4):

        #     # Rotate the map
        #     if(rot_idx == 0):
        #         local_maps_tmp = local_maps
        #     else:
        #         local_maps_tmp = torch.rot90(local_maps, k=rot_idx, dims=[-2, -1])

        #     # Encode the local map
        #     encoded_local_map = self.map_encoder_model(local_maps_tmp)

        #     # Need to unrotate the encoding
        #     if(rot_idx == 0):
        #         encoded_local_map_unrotated = encoded_local_map
        #     else:
        #         encoded_local_map_unrotated = torch.rot90(encoded_local_map, k=4-rot_idx, dims=[-2, -1])

        #     # Save for later
        #     encoded_local_maps.append(encoded_local_map_unrotated)

        # Stack them
        # encoded_local_maps = torch.stack(encoded_local_maps, dim=1)


        # Encode the local maps
        encoded_local_maps = self.map_encoder_model(local_maps)

        # Unflatten everything
        all_encoded_local_maps = torch.reshape(encoded_local_maps, [batch_size, sequence_length, encoded_local_maps.shape[1], encoded_local_maps.shape[2], encoded_local_maps.shape[3], encoded_local_maps.shape[4]])
        all_encoded_observations = torch.reshape(encoded_observations, [batch_size, sequence_length, encoded_observations.shape[1]])

        
        all_log_probs = []
        for seq_idx in range(sequence_length):

            # Extract what we need for this sequence
            encoded_local_maps = all_encoded_local_maps[:, seq_idx]
            encoded_observations = all_encoded_observations[:, seq_idx]

            # Only interpolate if we need it
            if(self.number_of_matching_rotations[stage] != encoded_local_maps.shape[-1]):

                # Interpolate the rotation dim
                B, elmC, elmH, elmW, elmR = encoded_local_maps.shape
                encoded_local_maps = torch.permute(encoded_local_maps, [0, 2, 3, 1, 4])
                encoded_local_maps = torch.reshape(encoded_local_maps, [B*elmH*elmW, elmC, elmR])
                encoded_local_maps = torch.nn.functional.interpolate(encoded_local_maps, mode="linear", scale_factor=self.number_of_matching_rotations[stage] // 4)
                encoded_local_maps = torch.reshape(encoded_local_maps, [B, elmH, elmW, elmC, -1])
                encoded_local_maps = torch.permute(encoded_local_maps, [0, 3, 1, 2, 4])


            # Compute the matching scores
            matching_scores = encoded_local_maps * encoded_observations.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            matching_scores = torch.sum(matching_scores, dim=1)
            
            # Compute the log probs for the locations
            log_probs = torch.nn.functional.log_softmax(matching_scores.flatten(-3), dim=-1).reshape(matching_scores.shape)
            all_log_probs.append(log_probs)

        all_log_probs = torch.stack(all_log_probs, dim=1)

        return all_log_probs, local_map_center_global_frame


    def _do_evaluation_forward(self, data):

        # Unpack the data
        stage = data["stage"]
        all_observations = data["observations"]
        global_map = data.get("global_map", None)
        xy_position_global_frame_init = data.get("xy_position_global_frame_init", None)
        all_actions = data.get("actions", None)
        local_maps = data["local_maps"]
        local_map_center_global_frame = data.get("local_map_center_global_frame", None)

        # Get some model control parameters that allow us to change the behavior of the model
        model_control_parameters = data["model_control_parameters"]

        # Get some info
        device = all_observations.device
        batch_size = all_observations.shape[0]
        sequence_length = all_observations.shape[1]

        # Encode the observations
        B, S, oC, oH, oW = all_observations.shape
        all_observations = torch.reshape(all_observations, (B*S, oC, oH, oW))
        all_encoded_observations = self.observation_encoder_model(all_observations, None).get()
        all_encoded_observations = torch.reshape(all_encoded_observations, (B, S, -1))

        # Encode the global map
        encoded_global_map = self.map_encoder_model(global_map)

        # Figure out how we should move the map.
        map_moving_method = model_control_parameters.get("map_moving_method", None)
        assert(map_moving_method in ["map_centering", "actions", "use_gt"])
        if(map_moving_method == "map_centering"):

            # Process the sequences globally by recentering each local map as the MAP estimate of the previous time step 
            log_probs, log_probs_center_global_frame, all_local_maps = self._process_sequence_global(stage, all_encoded_observations, encoded_global_map, global_map, xy_position_global_frame_init, all_actions=None)
            extracted_local_maps = all_local_maps

        elif(map_moving_method == "actions"):

            # Process the sequences globally by recentering each local map as the MAP estimate of the previous time step 
            log_probs, log_probs_center_global_frame, all_local_maps =  self._process_sequence_global(stage, all_encoded_observations, encoded_global_map, global_map, xy_position_global_frame_init, all_actions=all_actions)
            extracted_local_maps = all_local_maps

        elif(map_moving_method == "use_gt"):

            # Use the GT data to center the local maps
            # This is a hack since it assumes you already have the true state or some estimate of it
            log_probs, log_probs_center_global_frame = self._process_sequence_local(stage, all_encoded_observations, local_maps, local_map_center_global_frame, observation_already_encoded=True)
            extracted_local_maps = local_maps



        # Pack into the return dict
        return_dict = dict()    
        return_dict["discrete_map_log_probs"] = log_probs
        return_dict["discrete_map_log_probs_center_global_frame"] = log_probs_center_global_frame
        return_dict["extracted_local_maps"] = extracted_local_maps


        # Return the modes
        return_dict["top_modes"] = self._get_modes(log_probs, log_probs_center_global_frame)

        # The single_point_prediction is actually just the top mode so instead of recomputing it, just slice out the top mode
        # return_dict["single_point_prediction"] = self._get_single_point_prediction(log_probs, log_probs_center_global_frame).float()
        return_dict["single_point_prediction"] = return_dict["top_modes"][:, :, 0, :]


        return return_dict


    def _process_sequence_global(self, stage, all_encoded_observations, encoded_global_map, global_map, initial_position, all_actions):

        # Get some info
        sequence_length = all_encoded_observations.shape[1]

        # All the log probability tensors
        all_log_probs = []

        # All the current positions of the local maps
        all_local_map_centers = []

        # Keep track of the local maps in case we need them
        all_local_maps = []

        local_map_center = initial_position
        for seq_idx in range(sequence_length):

            # Get the stuff for this step
            encoded_observations = all_encoded_observations[:, seq_idx, ...]
            
            # Get the actions if we have them
            if(all_actions is not None):
                actions = all_actions[:, seq_idx, ...]
            else:
                actions = None

            # Convert to an int so we are pixel aligned
            # This will help with the map cropping later
            local_map_center = local_map_center.int()

            # Get the local maps for this steps
            encoded_local_maps, local_map_center1 = self._crop_local_map_from_global_map(encoded_global_map, local_map_center)
            local_maps, local_map_center2 = self._crop_local_map_from_global_map(global_map, local_map_center)
            all_local_map_centers.append(local_map_center1.unsqueeze(1))
            all_local_maps.append(local_maps.unsqueeze(1))
            assert(torch.sum(local_map_center1 - local_map_center2) == 0)


            # Interpolate the rotation dim
            B, elmC, elmH, elmW, elmR = encoded_local_maps.shape
            encoded_local_maps = torch.permute(encoded_local_maps, [0, 2, 3, 1, 4])
            encoded_local_maps = torch.reshape(encoded_local_maps, [B*elmH*elmW, elmC, elmR])
            encoded_local_maps = torch.nn.functional.interpolate(encoded_local_maps, mode="linear", scale_factor=self.number_of_matching_rotations[stage]//4)
            encoded_local_maps = torch.reshape(encoded_local_maps, [B, elmH, elmW, elmC, -1])
            encoded_local_maps = torch.permute(encoded_local_maps, [0, 3, 1, 2, 4])

            # Compute the matching score
            matching_scores = encoded_local_maps * encoded_observations.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            matching_scores = torch.sum(matching_scores, dim=1)

            # Compute the log probs for the locations
            log_probs = torch.nn.functional.log_softmax(matching_scores.flatten(-3), dim=-1).reshape(matching_scores.shape)

            # Get some info about the log_probs
            H  = log_probs.shape[1]
            W  = log_probs.shape[2]

            # Need to compute the new centers to be a map centered around the current most likely state
            single_point_prediction_local_map = self._get_single_point_prediction(log_probs.unsqueeze(1), log_probs_center_global_frame=None)
            single_point_prediction_local_map = single_point_prediction_local_map.squeeze(1)
            single_point_prediction_local_map = single_point_prediction_local_map[:,0:2]

            # Update the map center using the
            local_map_center = single_point_prediction_local_map + local_map_center

            # Add in the actions if we have them to allow for propagation using actions
            if(actions is not None):
                local_map_center = local_map_center + actions[..., 0:2]

            # Save for later!
            all_log_probs.append(log_probs.unsqueeze(1))

        # Put into 1 tensor
        all_log_probs = torch.cat(all_log_probs, dim=1)
        all_local_map_centers = torch.cat(all_local_map_centers, dim=1)
        all_local_maps = torch.cat(all_local_maps, dim=1)

        return all_log_probs, all_local_map_centers, all_local_maps

