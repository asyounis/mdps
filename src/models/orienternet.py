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

# Project Imports
from models.blocks.voting import conv2d_fft_batchwise, TemplateSampler
from utils.wrappers import Camera

from models.blocks.map_encoder_models.create_models import create_map_encoder_model
from models.blocks.observation_encoder_models.create_models import create_observation_encoder_model
from models.base_models.discrete_prob_volume_models import DiscreteProbVolumeModel 

from osm.raster import Canvas
from utils.general import rotmat2d, make_grid, sample_xyr


class OrienterNet(DiscreteProbVolumeModel):
    def __init__(self, model_configs, model_architecture_configs):
        super(OrienterNet, self).__init__(model_configs, model_architecture_configs)

        # Extract the name of the model so we can get its model architecture params
        main_model_name = model_configs["main_model_name"]
        main_model_arch_configs = model_architecture_configs[main_model_name]

        # Get some configs
        self.normalize_feature_maps_before_matching = get_mandatory_config("normalize_feature_maps_before_matching", main_model_arch_configs, "main_model_arch_configs")
        self.pixels_per_meter = get_mandatory_config("pixels_per_meter", main_model_arch_configs, "main_model_arch_configs")
        self.max_sequence_length_before_chopping = get_mandatory_config("max_sequence_length_before_chopping", main_model_arch_configs, "main_model_arch_configs")
        
        # Get the size of the cropped local map and make sure its divisible by 2
        self.size_of_extracted_local_map_when_using_the_global_map = get_mandatory_config("size_of_extracted_local_map_when_using_the_global_map", main_model_arch_configs, "main_model_arch_configs")
        assert((self.size_of_extracted_local_map_when_using_the_global_map % 2) == 0)

        # If we should use the local or global frame for the different functions
        self.use_local_or_global_frame = get_mandatory_config("use_local_or_global_frame", main_model_arch_configs, "main_model_arch_configs")
        for key in self.use_local_or_global_frame.keys():
            assert((key == "training")  or (key == "validation") or (key == "evaluation"))
            assert((self.use_local_or_global_frame[key] == "Local") or (self.use_local_or_global_frame[key] == "Global"))

        # If we should use the individual frames or the sequences
        self.use_sequence_or_individual_images = get_mandatory_config("use_sequence_or_individual_images", main_model_arch_configs, "main_model_arch_configs")
        for key in self.use_sequence_or_individual_images.keys():
            assert((key == "training") or (key == "validation") or (key == "evaluation"))
            assert((self.use_sequence_or_individual_images[key] == "Individual") or (self.use_sequence_or_individual_images[key] == "Sequence"))

        # Get the number of matching rotations to use
        self.number_of_matching_rotations = get_mandatory_config("number_of_matching_rotations", main_model_arch_configs, "main_model_arch_configs")

        # For each of the stages, make sure that the parameters are correct
        for key in self.use_local_or_global_frame.keys():
            local_or_global = self.use_local_or_global_frame[key]
            sequence_or_individual = self.use_sequence_or_individual_images[key]

            # If we are local then we must only look at each frame individually and cant use the sequence
            if(local_or_global == "Local"):
                assert(sequence_or_individual == "Individual")



        # Get the internal model names
        internal_model_names = get_mandatory_config("internal_model_names", main_model_arch_configs, "main_model_arch_configs")

        # Observation Encoder models
        self.observation_encoder_model = create_observation_encoder_model(self._get_model_config("observation_encoder_model", internal_model_names, model_architecture_configs))

        # Map Encoder models
        self.map_encoder_model = create_map_encoder_model(self._get_model_config("map_encoder_model", internal_model_names, model_architecture_configs))

        # We create this once we know what stage we are in since the number of rotations we need is determined by the stage we are in
        self.template_sampler = None

    def forward(self, data):

        # Unpack the data
        stage = data["stage"]
        observations = data["observations"]
        local_maps = data["local_maps"]
        global_map = data.get("global_map", None)
        camera_data = data["camera_data"]
        map_mask = data["map_mask"]
        xy_position_global_frame_init = data.get("xy_position_global_frame_init", None)
        local_map_center_global_frame = data.get("local_map_center_global_frame", None)
        all_actions = data.get("actions", None)


        # Create the template sampler
        if(self.template_sampler is None):

            # Get the number of matching rotations based on the stage we are currently in
            num_of_matching_rotations = self.number_of_matching_rotations[stage]

            # Create the sampler
            # This has no learned parameters
            self.template_sampler = TemplateSampler(self.observation_encoder_model.get_bev_grid_xz(), self.pixels_per_meter, num_of_matching_rotations)

            # Move it to the correct device
            self.template_sampler = self.template_sampler.to(observations.device)


        # Get some model control parameters that allow us to change the behavior of orienternet
        model_control_parameters = data["model_control_parameters"]

        # Get some info
        batch_size = observations.shape[0]
        sequence_length = observations.shape[1]


        # Make sure we have this
        assert(stage in self.use_sequence_or_individual_images)
        assert(stage in self.use_local_or_global_frame)

        # Get the parameters
        local_or_global = self.use_local_or_global_frame[stage]
        sequence_or_individual = self.use_sequence_or_individual_images[stage]


        if(local_or_global == "Local"):
            log_probs = self._process_sequence_local(observations, local_maps, camera_data, map_mask)
            log_probs_center_global_frame = local_map_center_global_frame
            extracted_local_maps = local_maps

        elif(local_or_global == "Global"):

            # If we are in global, figure out how we should move the map.
            map_moving_method = model_control_parameters.get("map_moving_method", None)
            assert(map_moving_method in ["map_centering", "actions", "use_gt"])

            if(map_moving_method == "map_centering"):

                # Process the sequences globally by recentering each local map as the MAP estimate of the previous time step 
                log_probs, log_probs_center_global_frame, extracted_local_maps = self._process_sequence_global(observations, global_map, camera_data, map_mask, xy_position_global_frame_init, all_actions=None)
            
            elif(map_moving_method == "actions"):

                # Process the sequences globally by recentering each local map as the MAP estimate of the previous time step 
                log_probs, log_probs_center_global_frame, extracted_local_maps = self._process_sequence_global(observations, global_map, camera_data, map_mask, xy_position_global_frame_init, all_actions=all_actions)

            elif(map_moving_method == "use_gt"):

                # Use the GT data to center the local maps
                # This is a hack since it assumes you already have the true state or some estimate of it
                log_probs = self._process_sequence_local(observations, local_maps, camera_data, map_mask)
                log_probs_center_global_frame = local_map_center_global_frame
                extracted_local_maps = local_maps



        #     if(sequence_or_individual == "Sequence"):
        #         # Not implemented yet 
        #         # assert(False)

        #         # This is a hack right now
        #         log_probs = self._process_sequence_local(observations, local_maps, camera_data, map_mask)
        #         log_probs_center_global_frame = local_map_center_global_frame
        #         extracted_local_maps = local_maps

        #         # # Need to get the canvas data for this
        #         # local_map_canvas_data = data.get("local_map_canvas_data")

        #         # # Need them to be in numpy
        #         # local_map_canvas_data = local_map_canvas_data.cpu().numpy()

        #         # # Break them into individual canvases
        #         # all_local_map_canvas = []
        #         # for b_idx in range(batch_size):
        #         #     sequence_local_map_canvas = []
        #         #     for s_idx in range(sequence_length):
        #         #         lmcd = local_map_canvas_data[b_idx, s_idx]
        #         #         canvas = Canvas.create_from_data(lmcd)
        #         #         sequence_local_map_canvas.append(canvas)
        #         #     all_local_map_canvas.append(sequence_local_map_canvas)

        #         # log_probs_pre = log_probs.clone()

        #         # for b_idx in range(batch_size):
        #         #     lps, _ = self.markov_filtering(log_probs[b_idx], all_local_map_canvas[b_idx], data["xy_position_world_frame"][b_idx], data["roll_pitch_yaw"][b_idx, ..., -1])
        #         #     lps = torch.stack(lps)
        #         #     log_probs[b_idx] = lps

        #     elif(sequence_or_individual == "Individual"):
        #         # No need to do any additional processing
        #         pass



        # Pack into the return dict
        return_dict = dict()    
        return_dict["discrete_map_log_probs"] = log_probs
        return_dict["discrete_map_log_probs_center_global_frame"] = log_probs_center_global_frame
        return_dict["extracted_local_maps"] = extracted_local_maps

        # We only do this if the stage is evaluation, otherwise we dont need the single point computations and so 
        # dont do them to save compute time
        if(stage == "evaluation"):

            # Return the modes
            return_dict["top_modes"] = self._get_modes(log_probs, log_probs_center_global_frame)

            # The single_point_prediction is actually just the top mode so instead of recomputing it, just slice out the top mode
            # return_dict["single_point_prediction"] = self._get_single_point_prediction(log_probs, log_probs_center_global_frame).float()
            return_dict["single_point_prediction"] = return_dict["top_modes"][:, :, 0, :]


        return return_dict

    def get_submodels(self):
        submodels = dict()
        submodels["map_encoder"] = self.map_encoder_model
        submodels["observation_encoder_model"] = self.observation_encoder_model
        return submodels

    def _process_sequence_global(self, all_observations, global_map, all_camera_data, map_mask, initial_position, all_actions):

        # Get some info
        sequence_length = all_observations.shape[1]

        # All the log probability tensors
        all_log_probs = []

        # All the current positions of the local maps
        all_local_map_centers = []

        # Keep track of the local maps in case we need them
        all_local_maps = []

        local_map_center = initial_position
        for seq_idx in range(sequence_length):

            # Get the stuff for this step
            observation = all_observations[:, seq_idx, ...]
            camera_data = all_camera_data[:, seq_idx, ...]
            
            # Get the actions if we have them
            if(all_actions is not None):
                actions = all_actions[:, seq_idx, ...]
            else:
                actions = None

            # Convert to an int so we are pixel aligned
            # This will help with the map cropping later
            local_map_center = local_map_center.int()

            # Get the local maps for this steps
            local_maps, local_map_center = self._crop_local_map_from_global_map(global_map, local_map_center)
            all_local_map_centers.append(local_map_center.unsqueeze(1))
            all_local_maps.append(local_maps.unsqueeze(1))

            # Process the BEV
            encoded_observation = self.observation_encoder_model(observation, camera_data)
            bev_observations, bev_valid, bev_confidence = encoded_observation.get()

            # Process the maps
            encoded_maps = self.map_encoder_model(local_maps)

            # Compute the matching scores
            matching_scores = self._exhaustive_voting(bev_observations, encoded_maps, bev_valid, bev_confidence)

            # Convert from [B, Rot, H, W] -> [B, H, W, Rot]
            # Do this because all the loss functions and rendering code expects this dim ordering
            # Annoying but I dont want to re-write all that rendering code and stuff so we live 
            # with this
            matching_scores = torch.movedim(matching_scores, 1, -1)

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

    def _process_sequence_local(self, observations, local_maps, camera_data, map_mask):


        # Get some info
        sequence_length = observations.shape[1]

        # If the sequence is too long then we need to process it in chunks
        if(sequence_length <= self.max_sequence_length_before_chopping):

            # No need to chop so just process it all at once
            log_probs = self._process_sequence_helper(observations, local_maps, camera_data, map_mask)

        else:

            # The sequence is too long so we need to chop for memory purposes

            s = 0
            e = s + self.max_sequence_length_before_chopping

            all_log_probs = []
            while (s < sequence_length):

                # Make sure we dont go over the end of the sequence
                e = min(e, sequence_length)

                # Chop the sequence
                chopped_observations = observations[:, s:e, ...]
                chopped_local_maps = local_maps[:, s:e, ...]
                chopped_camera_data = camera_data[:, s:e, ...]
                chopped_map_mask = map_mask[:, s:e, ...]

                # Process the chopped sequence
                chopped_log_probs = self._process_sequence_helper(chopped_observations, chopped_local_maps, chopped_camera_data, chopped_map_mask)
                all_log_probs.append(chopped_log_probs)

                # Move the pointers
                s += self.max_sequence_length_before_chopping
                e = s + self.max_sequence_length_before_chopping
                
            # Restack into a single tensor
            log_probs = torch.cat(all_log_probs, dim=1)

        return log_probs

    def _process_sequence_helper(self, observations, local_maps, camera_data, map_mask):

        # get some information 
        device = observations.device
        batch_size = observations.shape[0]
        sequence_length = observations.shape[1]

        # Flatten Everything needed for the BEV encoder so we can batch process them
        obs_C, obs_H, obs_W = observations.shape[2:]  
        observations_flattened = torch.reshape(observations, (batch_size*sequence_length, obs_C, obs_H, obs_W))
        cameras_flattened = torch.reshape(camera_data, (batch_size*sequence_length, -1))

        # Flatten Everything for the Map encoder
        lm_C, lm_H, lm_W = local_maps.shape[2:]  
        local_maps_flattened = torch.reshape(local_maps, (batch_size*sequence_length, lm_C, lm_H, lm_W))

        # Process the BEV
        encoded_observation = self.observation_encoder_model(observations_flattened, cameras_flattened)
        bev_observations, bev_valid, bev_confidence = encoded_observation.get()

        # Process the map
        encoded_maps = self.map_encoder_model(local_maps_flattened)

        # Compute the matching scores
        matching_scores = self._exhaustive_voting(bev_observations, encoded_maps, bev_valid, bev_confidence)

        # Reshape it back into batch and sequence dims
        ms_C, ms_H, ms_W = matching_scores.shape[1:]      
        matching_scores = torch.reshape(matching_scores, (batch_size, sequence_length, ms_C, ms_H, ms_W))

        # Convert from [B, S, Rot, H, W] -> [B, S, H, W, Rot]
        # Do this because all the loss functions and rendering code expects this dim ordering
        # Annoying but I dont want to re-write all that rendering code and stuff so we live 
        # with this
        matching_scores = torch.movedim(matching_scores, 2, -1)

        # CHEEEATINg!!! This is so cheating
        # If we have a map mask then mask out the other probabilities
        # if(map_mask is not None):
            # matching_scores.masked_fill_(~map_mask.unsqueeze(-1), -np.inf)
            
        # Compute the log probs for the locations
        log_probs = torch.nn.functional.log_softmax(matching_scores.flatten(-3), dim=-1).reshape(matching_scores.shape)

        return log_probs

    def _exhaustive_voting(self, f_bev, f_map, valid_bev, confidence_bev):
    
        # Sometimes we want to normalize first
        if self.normalize_feature_maps_before_matching:
            f_bev = torch.nn.functional.normalize(f_bev, dim=1)
            f_map = torch.nn.functional.normalize(f_map, dim=1)

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
















    # def propagate_belief(self, Δ_xy, Δ_yaw, canvas_target, canvas_source, belief, num_rotations=None):
    #     # We allow a different sampling resolution in the target frame
    #     if num_rotations is None:
    #         num_rotations = belief.shape[-1]

    #     angles = torch.arange(0, 360, 360 / num_rotations, device=Δ_xy.device, dtype=Δ_xy.dtype)
    #     uv_grid = make_grid(canvas_target.w, canvas_target.h, device=Δ_xy.device)
    #     xy_grid = canvas_target.to_xy(uv_grid.to(Δ_xy))


    #     Δ_xy_world = torch.einsum("nij,j->ni", rotmat2d(-angles), Δ_xy)
    #     xy_grid_prev = xy_grid[..., None, :] + Δ_xy_world[..., None, None, :, :]
    #     uv_grid_prev = canvas_source.to_uv(xy_grid_prev).to(Δ_xy)

    #     angles_prev = angles + Δ_yaw
    #     angles_grid_prev = angles_prev.tile((canvas_target.h, canvas_target.w, 1))

    #     prior, valid = sample_xyr( belief[None, None], uv_grid_prev.to(belief)[None], angles_grid_prev.to(belief)[None], nearest_for_inf=True)
            


    #     return prior, valid


    # def markov_filtering(self, observations, canvas, xys, yaws, idxs=None):

    #     def log_softmax_spatial(x, dims=3):
    #         return torch.nn.functional.log_softmax(x.flatten(-dims), dim=-1).reshape(x.shape)


    #     xys = xys.float()
    #     yaws = yaws.float()

    #     if idxs is None:
    #         idxs = range(len(observations))



    #     belief = None
    #     beliefs = []
    #     for i in idxs:


    #         obs = observations[i]

    #         if belief is None:
    #             belief = obs
    #         else:

    #             # if(i > 0):
    #                 # belief = observations[i-1][...]

    #             # Original
    #             Δ_xy = rotmat2d(yaws[i]) @ (xys[i - 1] - xys[i])
    #             Δ_yaw = yaws[i - 1] - yaws[i]

    #             # Mine
    #             # Δ_xy = (xys[i - 1] - xys[i])
    #             # Δ_yaw = yaws[i - 1] - yaws[i]

    #             prior, valid = self.propagate_belief(Δ_xy, Δ_yaw, canvas[i], canvas[i - 1], belief)
    #             prior = prior[0, 0].masked_fill_(~valid[0], -np.inf)
    #             belief = prior + obs
    #             belief = log_softmax_spatial(belief)
    #         beliefs.append(belief)
    #     uvt_seq = torch.stack([self.argmax_xyr(p) for p in beliefs])



    #     return beliefs, uvt_seq




    # def argmax_xyr(self, scores):
    #     indices = scores.flatten(-3).max(-1).indices
    #     width, num_rotations = scores.shape[-2:]
    #     wr = width * num_rotations
    #     y = torch.div(indices, wr, rounding_mode="floor")
    #     x = torch.div(indices % wr, num_rotations, rounding_mode="floor")
    #     angle_index = indices % num_rotations
        
    #     # Convert to an angle in radians
    #     angle = angle_index / num_rotations
    #     angle = angle * 2.0 * np.pi


    #     xyr = torch.stack((x, y, angle), -1)
    #     return xyr


