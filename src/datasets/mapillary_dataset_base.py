
# Python Imports
import json
from collections import defaultdict

# Package Imports
import torch
import torchvision
import numpy as np
import cv2
from tqdm import tqdm

# General ML Framework Imports
from general_ml_framework.utils.config import *

# Project imports
from utils.image_utils import rectify_image, resize_image, pad_image
from utils.wrappers import Camera
from utils.geometry import decompose_rotmat
from utils.geo import BoundaryBox
from osm.tiling import TileManager
from datasets.kitti_mapillary_common import KittiMapillaryCommon

class MapillaryDatasetBase(KittiMapillaryCommon):
    def __init__(self, dataset_configs, dataset_type):
        super(MapillaryDatasetBase, self).__init__(dataset_configs, dataset_type)

        # Get the mandatory configs
        self.dataset_directory = get_mandatory_config("dataset_directory", dataset_configs, "dataset_configs")
        tile_filename = get_mandatory_config("tile_filename", dataset_configs, "dataset_configs")

        # scenes = ['paris', 'avignon']
        # scenes = ["sanfrancisco_soma"]
        # ['sanfrancisco_soma', 'nantes', 'paris', 'sanfrancisco_hayes', 'vilnius', 'avignon', 'amsterdam', 'montrouge', 'berlin', 'helsinki', 'milan', 'toulouse', 'lemans']
        scenes = ['sanfrancisco_soma', 'nantes', 'paris', 'sanfrancisco_hayes', 'avignon', 'amsterdam', 'montrouge', 'berlin', 'helsinki', 'milan', 'toulouse', 'lemans']

        # Load the tile managers
        self.tile_managers =  self._load_tile_managers(scenes, tile_filename)

        # Create the image augmentations if we are in training mode
        self.augmentations = []
        if(dataset_type == "training"):
            self.augmentations.append(torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.4, hue=(0.5/3.14)))
        self.augmentations = torchvision.transforms.Compose(self.augmentations)

        # self._get_min_map_size_for_sequences()

        # We can only handle sequences right now
        # Need to fix the way the splits are done for the image list
        # assert(self.return_sequence)

    def __len__(self):

        if(self.return_sequence):
            size = len(self.all_sequences)
        else:
            size = len(self.all_images_list)

        return size 

  
    def _get_item_image(self, idx):

        # Get the image info
        image_info = self.all_images_list[idx]
        (scene_name, sequence_name, view_name) = image_info

        #################################################################
        ## Get the image 
        #################################################################

        # Get the camera info and create a camera object for this frame 
        all_camera_infos = self.consolidated_data["cameras"] 
        camera_info = all_camera_infos[scene_name][sequence_name]
        camera_info = camera_info[self.consolidated_data["camera_id"][idx]] 
        camera = Camera.from_dict(camera_info)

        # Get the angular data for the image
        if "roll_pitch_yaw" in self.consolidated_data:
            roll, pitch, yaw = self.consolidated_data["roll_pitch_yaw"][idx]
        else:
            roll, pitch, yaw = decompose_rotmat(self.consolidated_data["R_c2w"][idx])

        # Create the image file path and load the image.
        image_filepath = "{}/{}/images/{}.jpg".format(self.dataset_directory, scene_name, view_name)
        image, valid, camera, roll, pitch = self._load_and_process_image(image_filepath, camera, roll, pitch)

        # These should be 0
        assert(roll == 0)
        assert(pitch == 0)

        # Convert Yaw to radians
        yaw = yaw * np.pi / 180.0

        #################################################################
        ## Get the Map
        #################################################################

        # Get the GPS position
        latlon_gps_gt = self.consolidated_data["gps_position"][idx][:2].clone()
        gps_xy_position_world_frame_gt = self.tile_managers[scene_name].projection.project(latlon_gps_gt.numpy())

        # Get the XY from the camera to world translation vector
        t_c2w_xy_position_world_frame_gt = self.consolidated_data["t_c2w"][idx][:2].clone().double().numpy()

        # If we should init from the GPS or from the camera to world translation vector
        if self.init_from_gps:
            # xy_position_world_frame = gps_xy_position_world_frame_gt.copy()
            assert(False)
        else:
            xy_position_world_frame = t_c2w_xy_position_world_frame_gt.copy()

        # Add some noise to the position in both directions
        noise = np.random.uniform(low=-1, high=1, size=2) * self.max_xy_noise
        xy_position_world_frame_init = xy_position_world_frame + noise

        # Create the bounding box of the map that we will extract from the tile
        map_bounding_box = BoundaryBox(xy_position_world_frame_init - self.local_map_crop_size_meters, xy_position_world_frame_init + self.local_map_crop_size_meters)

        # Get the local world map around the x-y position
        local_map = self.tile_managers[scene_name].query(map_bounding_box)

        # Get the XY position in the local local map coordinate frame
        t_c2w_xy_position_local_map_frame_gt = local_map.to_uv(t_c2w_xy_position_world_frame_gt)
        gps_xy_position_local_map_frame_gt = local_map.to_uv(gps_xy_position_world_frame_gt)
        xy_position_local_frame_init = local_map.to_uv(xy_position_world_frame_init)

        # Compute the GPS accuracy (which really isnt a compitation)
        gps_xy_position_local_map_frame_gt_accuracy = torch.tensor(min(self.gps_accuracy, self.local_map_crop_size_meters))

        # # get the local world map
        # uv_init = canvas.to_uv(bbox_tile.center) 

        # Create the map mask that will be used somehow.  No clue wtf this does
        map_mask = self._create_map_mask(local_map)

        # Get the rasterized image of the local map
        # Dims are: [C, H, W] 
        local_map_rasterized_image = local_map.raster
        local_map_rasterized_image = self._convert_object_to_tensor(local_map_rasterized_image)

        # Need to convert to long since these are classes
        local_map_rasterized_image = local_map_rasterized_image.long()

        # Pack into the return dict
        return_dict = dict()

        return_dict["xy_position_world_frame"] = self._convert_object_to_tensor(xy_position_world_frame).unsqueeze(0)
        return_dict["observations"] = self._convert_object_to_tensor(image).unsqueeze(0)
        return_dict["local_maps"] = self._convert_object_to_tensor(local_map_rasterized_image).unsqueeze(0)
        return_dict["map_mask"] = self._convert_object_to_tensor(map_mask).unsqueeze(0)
        return_dict["t_c2w_xy_position_local_map_frame_gt"] = self._convert_object_to_tensor(t_c2w_xy_position_local_map_frame_gt).unsqueeze(0).float()
        return_dict["gps_xy_position_local_map_frame_gt"] = self._convert_object_to_tensor(gps_xy_position_local_map_frame_gt).unsqueeze(0).float()
        return_dict["gps_xy_position_local_map_frame_gt_accuracies"] = self._convert_object_to_tensor(gps_xy_position_local_map_frame_gt_accuracy).unsqueeze(0)
        return_dict["camera_data"] = camera._data.unsqueeze(0)
        return_dict["roll_pitch_yaw"] = torch.FloatTensor([roll, pitch, yaw]).unsqueeze(0)
        return_dict["xy_position_local_frame_init"] = self._convert_object_to_tensor(xy_position_local_frame_init).unsqueeze(0).float()
        return_dict["xy_position_world_frame_init"] = self._convert_object_to_tensor(xy_position_world_frame_init).unsqueeze(0).float()

        return_dict["local_map_canvas_data"] = self._convert_object_to_tensor(local_map.get_data()).unsqueeze(0).float()

        # return_dict["map_classes_order"] = ["areas", "ways", "nodes"]
        return_dict["image_filepath"] = image_filepath

        return return_dict

    def _load_tile_managers(self, scenes, tile_filename):

        # Each scene will have its own tile manager
        tile_managers = dict()

        for scene in scenes:

            # Create the tile file path
            tile_filepath = "{}/{}/{}".format(self.dataset_directory, scene, tile_filename)

            # Load the tile data
            tile_managers[scene] = TileManager.load(tile_filepath)

            # Make sure the pixels per meter match
            if(tile_managers[scene].ppm != self.pixels_per_meter):
                print("Mismatch Pixels per meter for the tile manager")
                print("Got {} but expected {}".format(tile_managers[scene].ppm, self.pixels_per_meter))
                assert(False)

            # @todo Do this check
            # groups = self.tile_managers[scene].groups
            # if self.cfg.num_classes:  # check consistency
            #     if set(groups.keys()) != set(self.cfg.num_classes.keys()):
            #         raise ValueError(
            #             f"Inconsistent groups: {groups.keys()} {self.cfg.num_classes.keys()}"
            #         )
            #     for k in groups:
            #         if len(groups[k]) != self.cfg.num_classes[k]:
            #             raise ValueError(
            #                 f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
            #             )


        return tile_managers

    def _load_and_process_image(self, image_filepath, camera, roll, pitch):

        # Load the image
        image = cv2.imread(image_filepath, cv2.IMREAD_COLOR)

        # We need to convert from BGR to RGB because opencv is BGR for some stupid reasons
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to pytorch
        image = torch.from_numpy(image)

        # Move the channels to the correct dim
        image = torch.permute(image, [2, 0, 1])

        # Convert to float and convert from [0, 255] to [0, 1]
        image = image.float()
        image = image / 255.0            

        # Rectify the image so that it has 0 roll and 0 pitch
        image, valid = rectify_image(image, camera, roll, pitch)
        roll = 0.0
        pitch = 0.0

        # Resize the image
        image, _, camera, valid = resize_image(image, self.output_image_size, fn=max, camera=camera, valid=valid)

        # pad such that both edges are of the given size
        image, valid, camera = pad_image(image, self.output_image_size, camera, valid)

        # Apply augmentations
        image = self.augmentations(image)

        return image, valid, camera, roll, pitch

    def _get_tile_manager(self, sequence_indices):
        
        # Get the scene name so we can load the canvas
        image_info = self.all_images_list[sequence_indices[0]]
        (scene_name, _, _) = image_info

        return self.tile_managers[scene_name]
