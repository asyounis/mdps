
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
from datasets.kitti_mapillary_common import KittiMapillaryCommon

class KittiDataset(KittiMapillaryCommon):
    def __init__(self, dataset_configs, dataset_type):
        super(KittiDataset, self).__init__(dataset_configs, dataset_type)

        # Constants that we need to set
        self.DUMMY_SCENE_NAME = "kitti_dummy_scene"
        self.CAMERA_INDEX = 2


        # Get the mandatory configs
        self.dataset_directory = get_mandatory_config("dataset_directory", dataset_configs, "dataset_configs")
        splits_filenames = get_mandatory_config("splits_filenames", dataset_configs, "dataset_configs")
        tile_filename = get_mandatory_config("tile_filename", dataset_configs, "dataset_configs")
        self.gps_accuracy = get_mandatory_config("gps_accuracy", dataset_configs, "dataset_configs")
        self.output_image_size = get_mandatory_config("output_image_size", dataset_configs, "dataset_configs")
        self.target_focal_length = get_mandatory_config("target_focal_length", dataset_configs, "dataset_configs")
        # self.pad_to_multiple = get_mandatory_config("pad_to_multiple", dataset_configs, "dataset_configs")
        self.cache_dir = get_mandatory_config("cache_dir", dataset_configs, "dataset_configs")

        # Get the sequence length for this data
        max_sequence_lengths = get_mandatory_config("max_sequence_lengths", dataset_configs, "dataset_configs")
        self.max_sequence_length = max_sequence_lengths[self.dataset_type]

        # Make sure the cache directory exists
        ensure_directory_exists(self.cache_dir)

        # Get the split filename for this dataset
        splits_filename = splits_filenames[self.dataset_type]
        splits_filepath = "{}/{}".format(self.dataset_directory, splits_filename)

        # Load the tile manager
        tile_filepath = "{}/tiles.pkl".format(self.dataset_directory)
        self.tile_manager = TileManager.load(tile_filepath)

        # Make sure the cache directory exists
        ensure_directory_exists(self.cache_dir)

        # Create the cache filepath
        cache_filepath = "{}/kitti_cache_{}.pt".format(self.cache_dir, dataset_type)

        # Check if it exists and if it does, load the cache
        if(os.path.exists(cache_filepath)):
            print("Loading from cachfile: {}".format(cache_filepath))

            # Load the cache
            save_dict = torch.load(cache_filepath)
            self.consolidated_data = save_dict["consolidated_data"]
            self.all_sequences = save_dict["all_sequences"]

        else:

            # Load the dataset
            self.consolidated_data = self._load_and_process_dataset(splits_filepath)

            # Create the sequence data
            self.all_sequences = self._create_sequences(self.consolidated_data)

            # Save to the cache
            save_dict = dict()
            save_dict["consolidated_data"] = self.consolidated_data
            save_dict["all_sequences"] = self.all_sequences
            torch.save(save_dict, cache_filepath)


        # Create the image augmentations if we are in training mode
        self.augmentations = []
        if(dataset_type == "training"):
            self.augmentations.append(torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.4, hue=(0.5/3.14)))
        self.augmentations = torchvision.transforms.Compose(self.augmentations)

    def __len__(self):

        if(self.return_sequence):
            size = len(self.all_sequences)
        else:
            size = len(self.consolidated_data["sequence_index"])

        return size 


    def _get_item_image(self, idx):

        # Unpack
        date = self.consolidated_data["date"][idx]
        sequence_name = self.consolidated_data["sequence_name"][idx]
        sequence_index = self.consolidated_data["sequence_index"][idx]


        #################################################################
        ## Get the image 
        #################################################################

        # Get the camera info and create a camera object for this frame 
        all_camera_infos = self.consolidated_data["cameras"] 
        camera_info = all_camera_infos[self.DUMMY_SCENE_NAME][sequence_name]
        camera_info = camera_info[self.CAMERA_INDEX] 
        camera = Camera.from_dict(camera_info)

        # Get the angular data for the image
        roll, pitch, yaw = self.consolidated_data["roll_pitch_yaw"][idx]

        # Create the image file path and load the image.
        image_filepath = "{}/{}/{}/image_02/data//{:010d}.png".format(self.dataset_directory, date, sequence_name, sequence_index)
        image, valid, camera, roll, pitch, image_orig = self._load_and_process_image(image_filepath, camera, roll, pitch)


        # These should be 0
        assert(roll == 0)
        assert(pitch == 0)

        # Convert Yaw to radians
        yaw = yaw * np.pi / 180.0

        #################################################################
        ## Get the Map
        #################################################################

        # Get the XY from the camera to world translation vector
        t_c2w_xy_position_world_frame_gt = self.consolidated_data["t_c2w"][idx][:2].clone().double().numpy()

        # If we should init from the camera to world translation vector
        xy_position_world_frame = t_c2w_xy_position_world_frame_gt.copy()

        # Add some noise to the position in both directions
        noise = np.random.uniform(low=-1, high=1, size=2) * self.max_xy_noise
        xy_position_world_frame_init = xy_position_world_frame + noise

        # Create the bounding box of the map that we will extract from the tile
        map_bounding_box = BoundaryBox(xy_position_world_frame_init - self.local_map_crop_size_meters, xy_position_world_frame_init + self.local_map_crop_size_meters)

        # Get the local world map around the x-y position
        local_map = self.tile_manager.query(map_bounding_box)

        # Get the XY position in the local local map coordinate frame
        t_c2w_xy_position_local_map_frame_gt = local_map.to_uv(t_c2w_xy_position_world_frame_gt)
        xy_position_local_frame_init = local_map.to_uv(xy_position_world_frame_init)

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
        return_dict["camera_data"] = camera._data.unsqueeze(0)
        return_dict["roll_pitch_yaw"] = torch.FloatTensor([roll, pitch, yaw]).unsqueeze(0)
        return_dict["xy_position_local_frame_init"] = self._convert_object_to_tensor(xy_position_local_frame_init).unsqueeze(0).float()
        return_dict["xy_position_world_frame_init"] = self._convert_object_to_tensor(xy_position_world_frame_init).unsqueeze(0).float()

        return_dict["local_map_canvas_data"] = self._convert_object_to_tensor(local_map.get_data()).unsqueeze(0).float()

        # return_dict["observations_orig"] = self._convert_object_to_tensor(image_orig).unsqueeze(0)

        # return_dict["map_classes_order"] = ["areas", "ways", "nodes"]

        return return_dict

  



    def _load_and_process_dataset(self, splits_filepath):

        # Get the names and the shifts for this split file
        #    Names are (date, sequence_name, image ID)
        #    Shifts are optional and are 3D vectors (x, y, theta)        
        names, shifts = self._parse_split_file(splits_filepath)

        # Get the dates present in this names file (as a set of dates to avoid duplicates)
        dates = {d for d, _, _ in names}

        # For each of the dates, load the calibration files that was used
        calibrations = dict()
        for date in dates:
            calibration_directory = "{}/{}".format(self.dataset_directory, date)
            calibrations[date] = self._get_calibration(calibration_directory)

        # Make sure the pixels per meter match
        if(self.tile_manager.ppm != self.pixels_per_meter):
            print("Mismatch Pixels per meter for the tile manager")
            print("Got {} but expected {}".format(self.tile_manager.ppm, self.pixels_per_meter))
            assert(False)

        # The consolidated data
        consolidated_data = dict()
        consolidated_data["t_c2w"] = []
        consolidated_data["roll_pitch_yaw"] = []
        consolidated_data["camera_id"] = []
        consolidated_data["date"] = []
        consolidated_data["sequence_name"] = []
        consolidated_data["sequence_index"] = []

        # Get all the data for all the frames 
        for name in names:  

            # Unpack the name
            date = name[0]
            sequence_name = name[1]
            img_filename = name[2]

            # get the sequence index from from the img filename
            sequence_index = int(img_filename.replace(".png", ""))

            # Load the data for this specific frame
            frame_data = self._get_frame_data(date, sequence_name, img_filename, calibrations)

            # Save the information
            consolidated_data["date"].append(date)
            consolidated_data["sequence_name"].append(sequence_name)
            consolidated_data["sequence_index"].append(sequence_index)
            consolidated_data["t_c2w"].append(torch.from_numpy(frame_data["t_c2w"]))
            consolidated_data["roll_pitch_yaw"].append(torch.from_numpy(frame_data["roll_pitch_yaw"]))

            # All have the same camera ID
            consolidated_data["camera_id"].append(self.CAMERA_INDEX)


        # Create a mapping of sequence name to dates that we will need for later
        sequence_name_to_date = dict()
        for i in range(len(consolidated_data["sequence_name"])):
    
            # Unpack and set
            sequence_name = consolidated_data["sequence_name"][i]
            date = consolidated_data["date"][i]
            sequence_name_to_date[sequence_name] = date



        # For each sequence create the camera 
        cameras_dict = dict()
        for sequence_name in sequence_name_to_date.keys():
            date = sequence_name_to_date[sequence_name]
            cameras_dict[sequence_name] = {self.CAMERA_INDEX: calibrations[date][0]}
        consolidated_data["cameras"] = {self.DUMMY_SCENE_NAME:cameras_dict}



        return consolidated_data

    def _create_sequences(self, consolidated_data):

        # Collect all the indices of all the frames into which sequence they are a part of
        sequence_name_to_index = dict()
        for i in range(len(consolidated_data["sequence_name"])):

            # Unpack the sequence name
            sequence_name = consolidated_data["sequence_name"][i]

            # Make sure it exists
            if sequence_name not in sequence_name_to_index:
                sequence_name_to_index[sequence_name] = []

            # Add i
            sequence_name_to_index[sequence_name].append(i)


        # Sort all the sequences by their frame number
        for sequence_name in sequence_name_to_index.keys():
            sequence_indices = sequence_name_to_index[sequence_name]
            sequence_indices = sorted(sequence_indices, key=lambda i: consolidated_data["sequence_index"][i])
            sequence_name_to_index[sequence_name] = sequence_indices

        # Remove any sequences that are too short
        for sequence_name in list(sequence_name_to_index.keys()):
            sequence_indices = sequence_name_to_index[sequence_name]
            if(len(sequence_indices) < self.max_sequence_length):
                sequence_name_to_index.pop(sequence_name)

        # Chop up the sequences
        sequences = []
        for sequence_name in list(sequence_name_to_index.keys()):
            sequence_indices = sequence_name_to_index[sequence_name]

            # Break up into smaller sequences
            s = 0
            e = self.max_sequence_length
            while e < len(sequence_indices):

                # slice out the sequence indices and save them
                sliced_indices = sequence_indices[s:e]
                sequences.append(sliced_indices)

                # Move the sequence head and tail
                s += self.max_sequence_length
                e = s + self.max_sequence_length

        return sequences

    def _get_frame_data(self, date, drive_sequence, img_filename, calibrations):

        # Get the calibrations for this date
        _, R_cam_gps, t_cam_gps = calibrations[date]

        # Get the path to the GPS file
        gps_path = "{}/{}/{}/oxts/data/{}".format(self.dataset_directory, date, drive_sequence, img_filename.replace("png", "txt"))

        # Transform the GPS pose to the camera pose
        _, R_world_gps, t_world_gps = self._parse_gps_file(gps_path, self.tile_manager.projection)
        R_world_cam = R_world_gps @ R_cam_gps.T
        t_world_cam = t_world_gps - R_world_gps @ R_cam_gps.T @ t_cam_gps

        # Some voodoo to extract correct Euler angles from R_world_cam
        R_cv_xyz = Rotation.from_euler("YX", [-90, 90], degrees=True).as_matrix()
        R_world_cam_xyz = R_world_cam @ R_cv_xyz
        y, p, r = Rotation.from_matrix(R_world_cam_xyz).as_euler("ZYX", degrees=True)
        roll, pitch, yaw = r, -p, 90 - y
        roll_pitch_yaw = np.array([-roll, -pitch, yaw], np.float32)  # for some reason


        # Return the information
        return {
            "t_c2w": t_world_cam.astype(np.float32),
            "roll_pitch_yaw": roll_pitch_yaw,
            "index": int(img_filename.split(".")[0]),
        }

    def _parse_split_file(self, path):

        # Open the file and get all the lines
        with open(path, "r") as file:
            lines = file.read()

        # the names and the splits for this file
        names = []
        shifts = []

        # Go through line by line
        for line in lines.split("\n"):

            # Make sure the line is valid
            if not line:
                continue


            # Split the file by spaces into the file name and the shifts that we should apply
            # Not all files will have shifts
            name, *shift = line.split()

            # Split the name into its components and save that
            name_split = tuple(name.split("/"))
            names.append(name_split)

            # If we have shifts then we should record them and save
            if len(shift) > 0:
                assert len(shift) == 3
                shifts.append(np.array(shift, float))


        # If we have the shifts then we should stack them otherwise we set them to None
        # to indicate that we dont have any shifts
        if(len(shifts) == 0):
            shifts = None
        else:
            shifts = np.stack(shifts)
        
        return names, shifts

    def _get_calibration(self, calib_dir, cam_index=2):
        calib_path = "{}/{}".format(calib_dir, "calib_cam_to_cam.txt")
        calib_cam = self._parse_calibration_file(calib_path)
        P = calib_cam[f"P_rect_{cam_index:02}"]
        K = P[:3, :3]
        size = np.array(calib_cam[f"S_rect_{cam_index:02}"], float).astype(int)
        camera = {
            "model": "PINHOLE",
            "width": size[0],
            "height": size[1],
            "params": K[[0, 1, 0, 1], [0, 1, 2, 2]],
        }

        t_cam_cam0 = P[:3, 3] / K[[0, 1, 2], [0, 1, 2]]
        R_rect_cam0 = calib_cam["R_rect_00"]

        calib_gps_velo = self._parse_calibration_file("{}/{}".format(calib_dir, "calib_imu_to_velo.txt"))
        calib_velo_cam0 = self._parse_calibration_file("{}/{}".format(calib_dir, "calib_velo_to_cam.txt"))
        R_cam0_gps = calib_velo_cam0["R"] @ calib_gps_velo["R"]
        t_cam0_gps = calib_velo_cam0["R"] @ calib_gps_velo["T"] + calib_velo_cam0["T"]
        R_cam_gps = R_rect_cam0 @ R_cam0_gps
        t_cam_gps = t_cam_cam0 + R_rect_cam0 @ t_cam0_gps


        R_cam_gps = R_cam_gps.astype(np.float32)
        t_cam_gps = t_cam_gps.astype(np.float32)
        camera["params"] = camera["params"].astype(np.float32)

        return camera, R_cam_gps, t_cam_gps

    def _parse_calibration_file(self, path):
        calib = {}
        with open(path, "r") as fid:
            for line in fid.read().split("\n"):
                if not line:
                    continue
                key, *data = line.split(" ")
                key = key.rstrip(":")
                if key.startswith("R"):
                    data = np.array(data, float).reshape(3, 3)
                elif key.startswith("T"):
                    data = np.array(data, float).reshape(3)
                elif key.startswith("P"):
                    data = np.array(data, float).reshape(3, 4)
                calib[key] = data


        return calib

    def _parse_gps_file(self, path, projection):
        with open(path, "r") as fid:
            lat, lon, _, roll, pitch, yaw, *_ = map(float, fid.read().split())
        latlon = np.array([lat, lon])
        R_world_gps = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
        t_world_gps = None if projection is None else np.r_[projection.project(latlon), 0]
        return latlon, R_world_gps, t_world_gps

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

        # Keep the orig image but resize it so they are all the same size
        image_orig = image.clone()  
        image_orig, _= resize_image(image_orig, (1248, 384))

        # Rectify the image so that it has 0 roll and 0 pitch
        image, valid = rectify_image(image, camera, roll, pitch)
        roll = 0.0
        pitch = 0.0

        # resize to a canonical focal length
        factor = self.target_focal_length / camera.f.numpy()
        size = (np.array(image.shape[-2:][::-1]) * factor).astype(int)
        image, _, camera, valid = resize_image(image, size, camera=camera, valid=valid)

        # # If we should pad the image then pad the image
        # if(self.pad_to_multiple > 0):
        #     stride = self.pad_to_multiple
        #     size_out = (np.ceil((size / stride)) * stride).astype(int)

        image, valid, cam = pad_image(image, self.output_image_size, camera, valid, crop_and_center=True)

        # Apply augmentations
        image = self.augmentations(image)

        return image, valid, camera, roll, pitch, image_orig



    def _get_tile_manager(self, sequence_indices):
        
        # There is only 1 tile manager for this dataset
        return self.tile_manager
