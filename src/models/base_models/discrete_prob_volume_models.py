# Python Imports
import time

# Package Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ali Package Import
from general_ml_framework.models.base_model import BaseModel
from general_ml_framework.utils.config import *


class DiscreteProbVolumeModel(BaseModel):
    def __init__(self, model_configs, model_architecture_configs):
        super(DiscreteProbVolumeModel, self).__init__()

        # Extract the name of the model so we can get its model architecture params
        main_model_name = model_configs["main_model_name"]
        main_model_arch_configs = model_architecture_configs[main_model_name]

        # Get the parameters for mode finding
        # If not provided then no mode finding will happen
        self.nms_mode_finding_configs = get_optional_config_with_default("nms_mode_finding_configs", main_model_arch_configs, "main_model_arch_configs", default_value=None)


    def _get_model_config(self, model_name, internal_model_names, model_architecture_configs):
        internal_model_name = internal_model_names[model_name]
        configs = model_architecture_configs.get(internal_model_name, None)
        return configs


    def _get_modes(self, log_probs, log_probs_center_global_frame):

        # Unpack the NMS configs
        nms_thresholds = get_mandatory_config("nms_thresholds", self.nms_mode_finding_configs, "nms_mode_finding_configs")
        number_of_modes = get_mandatory_config("number_of_modes", self.nms_mode_finding_configs, "nms_mode_finding_configs")

        # get some info
        device = log_probs.device
        batch_size  = log_probs.shape[0]
        sequence_length  = log_probs.shape[1]
        H = log_probs.shape[2]
        W = log_probs.shape[3]
        R = log_probs.shape[4]    

        # Convert to a tensor so we can actually use it
        nms_thresholds = torch.FloatTensor(nms_thresholds).to(device)

        # Create a grid that we can use over and over instead of creating it everytime
        h_indices = torch.arange(0, H, 1)
        w_indices = torch.arange(0, W, 1)
        r_indices = torch.arange(0, R, 1)
        grid_h, grid_w, grid_r = torch.meshgrid([h_indices, w_indices, r_indices], indexing="ij")
        grid = torch.stack([grid_h, grid_w, grid_r], dim=-1).to(device)
        
        # Convert the grid angle dim from indices to angles
        grid[..., 2] = (grid[..., 2] / R) * 2.0 * np.pi

        # Create the tiled version of it. This is what we will be reusing everytime
        tiled_grid = torch.tile(grid.unsqueeze(0), [batch_size, 1, 1, 1, 1])

        # Process 1 timestep at a time since we dont have enough VRAM
        all_modes = []
        for seq_idx in range(sequence_length):

            # Get the log prob, make a copy so we can edit it
            local_log_prob = log_probs[:, seq_idx].detach().clone()

            # Get the modes
            modes = self._get_modes_helper(number_of_modes, local_log_prob, nms_thresholds, tiled_grid)
                
            # Stack the modes into a tensor
            modes = torch.stack(modes, dim=1)

            # keep track
            all_modes.append(modes)


        # Convert to a tensor
        all_modes = torch.stack(all_modes, dim=1)

        # Convert to float
        all_modes = all_modes.float()

        # Need to convert from index into an angle
        all_modes[..., -1] = all_modes[..., -1] / float(R)
        all_modes[..., -1] = all_modes[..., -1] * 2.0 * np.pi

        # Make the middle of the map the center
        all_modes[..., 0] -= H // 2
        all_modes[..., 1] -= W // 2

        # Add in the local map offset if we have it
        if(log_probs_center_global_frame is not None):
            all_modes[...,  0:2] += log_probs_center_global_frame.unsqueeze(2)

        return all_modes


    def _get_modes_helper(self, number_of_modes, log_probs, nms_thresholds, tiled_grid):

        # print("_get_modes_helper", number_of_modes)

        # IF there are no modes to be had then do nothing
        if(number_of_modes == 0):
            return []

        # get some info
        device = log_probs.device
        batch_size  = log_probs.shape[0]
        H = log_probs.shape[1]
        W = log_probs.shape[2]
        R = log_probs.shape[3]

        ########################################################################################################################################
        ## Select a Mode
        ########################################################################################################################################

        # We need to find the argmax of this so we need to flatten
        log_probs_flattened = log_probs.view([batch_size, -1])

        # Find the arg max
        _, max_idx = torch.max(log_probs_flattened, dim=-1)

        # Convert back from a max of 1D to a max of 3D
        # Note that these are all indices
        # With y being the first index, x being the seconds index and r being the 3rd index
        # AKA [Batch, Sequence, y, x, r] (because its an "image")
        WR = W * R
        y = torch.div(max_idx, WR, rounding_mode="floor")
        x = torch.div(max_idx % WR, R, rounding_mode="floor")
        r = max_idx % R

        # Stack into 1 tensor in the order we want the mode to be in when we return it
        selected_mode = torch.stack([x, y, r], dim=-1)

        # Stack into 1 tensor in the order we want it to be in so we can do suppression
        # This differs from the order we want to return it since we need [H, W, R] instead of [W, H, R] when doing suppression
        selected_mode_in_index_order = torch.stack([y, x, r], dim=-1)


        ########################################################################################################################################
        ## Suppress the weights
        ########################################################################################################################################


        # Compute the dist between the selected mode and the particles so we can 
        # figure out which particles we need to suppress
        diffs = torch.zeros_like(tiled_grid)
        for d in range(tiled_grid.shape[-1]):

            # Compute the difference
            diff_local = tiled_grid[..., d] - selected_mode_in_index_order[..., d].unsqueeze(1).unsqueeze(1).unsqueeze(1)

            # there may be additional processing to do if we are in the angle dim
            if((d == 0) or (d == 1)):
                diffs[..., d] = diff_local
            else:
                diffs[..., d] = torch.atan2(torch.sin(diff_local), torch.cos(diff_local), )

        # Do the ellipsoid test
        # See 
        #   - https://math.stackexchange.com/questions/76457/check-if-a-point-is-within-an-ellipse
        #   - https://stackoverflow.com/questions/17770555/how-to-check-if-a-point-is-inside-an-ellipsoid
        ellipsoid_equation_check_lhs = diffs / nms_thresholds.to(device).unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        ellipsoid_equation_check_lhs = ellipsoid_equation_check_lhs**2
        ellipsoid_equation_check_lhs = torch.sum(ellipsoid_equation_check_lhs, dim=-1)

        # Run the check
        inside_ellipsoid = ellipsoid_equation_check_lhs <= 1.0

        # Suppress the values within the ellipsoid
        log_probs[inside_ellipsoid] = -np.inf

        ########################################################################################################################################
        ## Recurse to get more modes and then combine with the modes we already have
        ########################################################################################################################################
    
        # Get more modes
        more_modes = self._get_modes_helper(number_of_modes-1, log_probs, nms_thresholds, tiled_grid) 

        # Add in our mode (to the front since it is a higher prob mode)
        more_modes.insert(0, selected_mode)

        return more_modes


    def _get_single_point_prediction(self, log_probs, log_probs_center_global_frame):

        # get some info
        batch_size  = log_probs.shape[0]
        sequence_length  = log_probs.shape[1]
        H  = log_probs.shape[2]
        W  = log_probs.shape[3]
        R  = log_probs.shape[4]

        # We need to find the argmax of this so we need to flatten
        log_probs_flattened = log_probs.view([batch_size, sequence_length, -1])

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
        max_value[:, :, 0] -= H // 2
        max_value[:, :, 1] -= W // 2


        # Add in the local map offset if we have it
        if(log_probs_center_global_frame is not None):
            max_value[:, :, 0:2] += log_probs_center_global_frame

        return max_value


    def _crop_local_map_from_global_map(self, global_map, positions):

        # Get some info
        batch_size = positions.shape[0]
        device = positions.device

        # Cropped local maps
        local_maps = []
        local_map_centers = []

        # Process each batch one at a time.
        # This is so not efficient but its fine for now until we do something more fancy and GPU friendly
        for b_idx in range(batch_size): 

            # Get the start and end x indices 
            s_x = int(positions[b_idx, 0].item()) - (self.size_of_extracted_local_map_when_using_the_global_map // 2)
            e_x = s_x + self.size_of_extracted_local_map_when_using_the_global_map

            # Get the start and end y indices 
            s_y = int(positions[b_idx, 1].item()) - (self.size_of_extracted_local_map_when_using_the_global_map // 2)
            e_y = s_y + self.size_of_extracted_local_map_when_using_the_global_map

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



