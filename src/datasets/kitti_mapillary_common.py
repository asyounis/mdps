
# Python Imports
import json
from collections import defaultdict

# Package Imports
import torch
import torchvision
import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation

# General ML Framework Imports
from general_ml_framework.utils.config import *

# Project imports
from utils.image_utils import rectify_image, resize_image, pad_image
from utils.wrappers import Camera
from utils.geometry import decompose_rotmat
from utils.geo import BoundaryBox
from osm.tiling import TileManager

class KittiMapillaryCommon(torch.utils.data.Dataset):
    def __init__(self, dataset_configs, dataset_type):

        # Needed for later
        self.dataset_type = dataset_type

        # Get the mandatory configs
        self.pixels_per_meter = get_mandatory_config("pixels_per_meter", dataset_configs, "dataset_configs")
        self.local_map_crop_size_meters = get_mandatory_config("local_map_crop_size_meters", dataset_configs, "dataset_configs")
        self.max_xy_noise = get_mandatory_config("max_xy_noise", dataset_configs, "dataset_configs")
        self.return_sequence = get_mandatory_config("return_sequence", dataset_configs, "dataset_configs")
        self.action_noise_xy = float(get_mandatory_config("action_noise_xy", dataset_configs, "dataset_configs"))
        self.action_noise_yaw_degrees = float(get_mandatory_config("action_noise_yaw_degrees", dataset_configs, "dataset_configs"))

        # If the global map center should be random or such that the the sequence is kinda centered around in the global map
        use_random_global_map_center = get_mandatory_config("use_random_global_map_center", dataset_configs, "dataset_configs")
        self.use_random_global_map_center = use_random_global_map_center[dataset_type]

        # Get the cropping!!
        global_map_crop_size_meters = get_mandatory_config("global_map_crop_size_meters", dataset_configs, "dataset_configs")
        self.global_map_crop_size_meters = global_map_crop_size_meters[dataset_type]

    def __getitem__(self, idx):

        # Return a sequence if we are in sequence mode otherwise just return the one image
        if(self.return_sequence):
            return_dict = self._get_item_sequence(idx)
        else:
            return_dict = self._get_item_image(idx)
       
        return return_dict


    def get_collate_function(self):
        return KittiMapillaryCommon.custom_collate_function

    @staticmethod
    def custom_collate_function(batch_list):

        # Get the keys of the things we want to batch
        keys = batch_list[0].keys()

        # create the dictionary to return
        return_dict = dict()
        return_dict = {k: [] for k in keys}

        # Pack into a large thing
        for i in range(len(batch_list)):
            for k in return_dict.keys():
                return_dict[k].append(batch_list[i][k])

        # Stack the correct keys
        for k in return_dict.keys():
            if(torch.is_tensor(return_dict[k][0])):
                return_dict[k] = torch.stack(return_dict[k], dim=0)    

            elif(isinstance(return_dict[k][0], Camera)):
                return_dict[k] = torch.stack(return_dict[k], dim=0)     

            elif(isinstance(return_dict[k][0], Camera)):
                return_dict[k] = torch.stack(return_dict[k], dim=0)     


            elif(k == "map_classes_order"):
                # We just want one of these
                return_dict[k] = return_dict[k][0]

            elif(k == "image_filepath"):
                return_dict[k] = return_dict[k][0]

            elif(k == "pixels_per_meter"):
                return_dict[k] = return_dict[k][0]

            else:
                print("Unknown datatype in collate function", type(return_dict[k][0]), "for key", k)
                assert(False)

        return return_dict



    def _get_item_sequence(self, idx):

         # Get the specific sequence
        sequence_indices = self.all_sequences[idx]

        # All the data for this sequence
        all_sequence_data = []

        # For this sequence get all the info for each index in the sequence
        for image_idx in sequence_indices:
                
            # Get the data for this image
            data = self._get_item_image(image_idx)
            all_sequence_data.append(data)

        # Convert from a list of dicts to dict of lists
        data = {k: [dic[k] for dic in all_sequence_data] for k in all_sequence_data[0]}

        # Get the map class order so we can re-apply it
        # We dont want a list of lists for this, just a single list with the map orders
        # so we extract the first entry (since all of the entries are the same)
        # and we will just add that to the map after aggregating all the lists of lists into tensors
        # map_classes_order = data["map_classes_order"][0]
        # data.pop("map_classes_order")

        # Stack all the data into a single tensor
        for key in data.keys():
            if(torch.is_tensor(data[key][0])):
                data[key] = torch.vstack(data[key])

        # Add this back in (see comment above)
        # data["map_classes_order"] = map_classes_order

        # Get the map center
        xy_position_world_frame = data["xy_position_world_frame"]
        xy_position_world_frame_min = torch.min(xy_position_world_frame, dim=0)[0]
        xy_position_world_frame_max = torch.max(xy_position_world_frame, dim=0)[0]
        xy_position_world_frame_center = (xy_position_world_frame_max + xy_position_world_frame_min) / 2.0
        xy_position_world_frame_range = (xy_position_world_frame_max - xy_position_world_frame_min) 
        xy_position_world_frame_range = torch.max(xy_position_world_frame_range)
        remaining_global_map_crop_size_meters = self.global_map_crop_size_meters - xy_position_world_frame_range

        # return xy_position_world_frame_range

        # if(remaining_global_map_crop_size_meters.item() < 0):
            
        #     # print(self.global_map_crop_size_meters, xy_position_world_frame_range.item())
        #     # print(xy_position_world_frame_range)

        #     return 1

        #     # print(xy_position_world_frame)
        #     # print("\n\n")
        #     # print(xy_position_world_frame_min)
        #     # print(xy_position_world_frame_max)

        #     # exit()

        # return 0

        # print(remaining_global_map_crop_size_meters)
        # Make sure that we dont have some kind of negative condition
        assert(remaining_global_map_crop_size_meters.item() >= 0)

        # Create the map center, this might be random or just the center of the 
        # sequence
        if(self.use_random_global_map_center):
            # Draw a uniform value in this range for the x and y
            xy_offset = torch.rand((2,)) * remaining_global_map_crop_size_meters
            xy_offset = xy_offset - (remaining_global_map_crop_size_meters / 2)
            map_center = xy_offset + xy_position_world_frame_center
        else:
            map_center = xy_position_world_frame_center

        # Create the bounding box of the map that we will extract from the tile
        half_global_map_crop_size_meters = int(self.global_map_crop_size_meters // 2.0)
        map_bounding_box = BoundaryBox(map_center - half_global_map_crop_size_meters, map_center + half_global_map_crop_size_meters)

        # Get the scene name so we can load the canvas
        # image_info = self.all_images_list[sequence_indices[0]]
        # (scene_name, _, _) = image_info

        # Get the tile manager
        tile_manager = self._get_tile_manager(sequence_indices)

        # Get the global map
        global_map = tile_manager.query(map_bounding_box)

        # Get the rasterized image of the global map
        # Dims are: [C, H, W] 
        global_map_rasterized_image = global_map.raster
        global_map_rasterized_image = self._convert_object_to_tensor(global_map_rasterized_image)



        # if((global_map_rasterized_image.shape[1] !=  int(self.global_map_crop_size_meters * self.pixels_per_meter)) or (global_map_rasterized_image.shape[2] !=  int(self.global_map_crop_size_meters * self.pixels_per_meter))):

        #     print(self.global_map_crop_size_meters * self.pixels_per_meter)
        #     print(global_map_rasterized_image.shape[1])
        #     print(global_map_rasterized_image.shape[2])
        #     print(map_center)
        #     print(half_global_map_crop_size_meters)
        #     print("idx", idx)
        
        # print("Done")
        # exit()

        # Make sure that they are the right size
        assert(global_map_rasterized_image.shape[1] ==  int(self.global_map_crop_size_meters * self.pixels_per_meter))
        assert(global_map_rasterized_image.shape[2] ==  int(self.global_map_crop_size_meters * self.pixels_per_meter))

        # Need to convert to long since these are classes
        global_map_rasterized_image = global_map_rasterized_image.long()

        # Get the coordinates in the global frame
        xy_position_global_frame = global_map.to_uv(xy_position_world_frame.numpy())
        xy_position_global_frame = self._convert_object_to_tensor(xy_position_global_frame)

        # Get the initial position in the world frame
        xy_position_world_frame_init = data["xy_position_world_frame_init"][0]
        xy_position_global_frame_init = global_map.to_uv(xy_position_world_frame_init.numpy())
        xy_position_global_frame_init = self._convert_object_to_tensor(xy_position_global_frame_init)

        # Get the local map center in the global frame
        local_map_center_global_frame = []
        for xy in data["xy_position_world_frame_init"]:
            tmp = global_map.to_uv(xy.numpy())
            tmp = self._convert_object_to_tensor(tmp)
            local_map_center_global_frame.append(tmp)
        local_map_center_global_frame = torch.stack(local_map_center_global_frame)

        # Convert to degrees
        # Note: this is commented out because I moved the degrees to radian conversion into "_get_item_image()".
        # Soooooo basically we are already in radians and always work in radians
        # data["roll_pitch_yaw"] = torch.deg2rad(data["roll_pitch_yaw"])

        # Get the initial yaw
        roll_pitch_yaw_init = data["roll_pitch_yaw"][0]
        yaw_init = roll_pitch_yaw_init[-1]

        # Get the actions to use
        actions_world_frame, actions_global_frame  = self._get_actions(xy_position_world_frame, global_map, data["roll_pitch_yaw"][..., -1])

        # Compute the driving direction 
        driving_direction = self._get_driving_direction(xy_position_world_frame)

        # Create the ground truth masks (this is dense)
        ground_truth_mask = torch.full(size=(len(sequence_indices),), fill_value=True)

        # Add in the global things
        data["global_map"] = global_map_rasterized_image
        data["xy_position_global_frame"] = xy_position_global_frame
        data["xy_position_global_frame_init"] = xy_position_global_frame_init
        data["yaw_init"] = yaw_init
        data["actions"] = actions_global_frame
        data["ground_truth_mask"] = ground_truth_mask
        data["driving_direction"] = driving_direction

        # Set the limits to be the map size in pixels
        data["global_x_limits"] = torch.FloatTensor([0, self.global_map_crop_size_meters*self.pixels_per_meter])
        data["global_y_limits"] = torch.FloatTensor([0, self.global_map_crop_size_meters*self.pixels_per_meter])

        data["local_map_center_global_frame"] = local_map_center_global_frame

        data["pixels_per_meter"] = self.pixels_per_meter

        return data

    def _get_actions(self, xy_position_world_frame, global_map, yaws):

        ##################################################
        # Compute the XY actions
        ##################################################

        # Add Noise
        xy_action_noise = (torch.rand(xy_position_world_frame.shape) * 2.0) - 1.0
        xy_action_noise = xy_action_noise * self.action_noise_xy
        xy_position_world_frame_perturbed = xy_position_world_frame + xy_action_noise

        # Create the global frame version
        xy_position_global_frame_perturbed = global_map.to_uv(xy_position_world_frame_perturbed.numpy())
        xy_position_global_frame_perturbed = self._convert_object_to_tensor(xy_position_global_frame_perturbed)


        # create the XY action
        xy_actions_world_frame = xy_position_world_frame_perturbed[1:, ...] - xy_position_world_frame_perturbed[:-1, ...]
        xy_actions_global_frame = xy_position_global_frame_perturbed[1:, ...] - xy_position_global_frame_perturbed[:-1, ...]

        ##################################################
        # Compute the Yaw actions
        ##################################################

        # Create the yaw actions
        yaw_actions = yaws[1:] - yaws[:-1]

        # Add noise
        yaws_action_noise = (torch.rand(yaw_actions.shape) * 2.0) - 1.0
        yaws_action_noise = yaws_action_noise * np.deg2rad(self.action_noise_yaw_degrees)
        yaw_actions = yaw_actions + yaws_action_noise


        # Stack them all into 1 action set
        actions_world_frame = torch.zeros((yaw_actions.shape[0], 3))
        actions_world_frame[..., :2] = xy_actions_world_frame
        actions_world_frame[..., 2] = yaw_actions


        # Stack them all into 1 action set
        actions_global_frame = torch.zeros((yaw_actions.shape[0], 3))
        actions_global_frame[..., :2] = xy_actions_global_frame
        actions_global_frame[..., 2] = yaw_actions

        # Add zeros to the end to make the actions the right size
        zeros = torch.zeros(actions_world_frame.shape[1]).unsqueeze(0)
        actions_world_frame = torch.cat([actions_world_frame, zeros], dim=0)
        actions_global_frame = torch.cat([actions_global_frame, zeros], dim=0)

        return actions_world_frame, actions_global_frame

    def _get_driving_direction(self, xy_position_world_frame):

        # this is the x-y deltas that we will compute the driving direction using trig
        # Note since we are computing deltas, we will have 1 less delta than we have positions
        # For the last delta just set it to be the second to last one. This isnt ideal but 
        # its better than setting it to zero since how fast does the car turn anyways
        deltas = torch.zeros_like(xy_position_world_frame)
        deltas[:-1] = xy_position_world_frame[1:] - xy_position_world_frame[:-1]

        # Compute the driving directions 
        driving_direction = torch.atan2(deltas[:, 1], deltas[:, 0])

        return driving_direction


    def _convert_object_to_tensor(self, obj):

        if(obj is None):
            return None

        if(torch.is_tensor(obj)):
            return obj

        if(isinstance(obj, Camera)):
            return obj

        # If we can then we should convert to pytorch
        if(np.issubdtype(obj.dtype, np.integer) or np.issubdtype(obj.dtype, np.floating) or np.issubdtype(obj.dtype, bool)):
            return torch.from_numpy(obj)


        assert(False)

    def _create_map_mask(self, canvas):
        map_mask = np.zeros(canvas.raster.shape[-2:], bool)
        radius = self.max_xy_noise
        mask_pad = 1
        mask_min, mask_max = np.round(canvas.to_uv(canvas.bbox.center)+ np.array([[-1], [1]]) * (radius + mask_pad) * canvas.ppm).astype(int)
        map_mask[mask_min[1] : mask_max[1], mask_min[0] : mask_max[0]] = True
        return map_mask




    def _get_tile_manager(self, sequence_indices):
        raise NotImplemented