
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
from datasets.mapillary_dataset_base import MapillaryDatasetBase

class MapillaryDataset(MapillaryDatasetBase):
    def __init__(self, dataset_configs, dataset_type):
        super(MapillaryDataset, self).__init__(dataset_configs, dataset_type)

        # Get the mandatory configs
        self.dataset_directory = get_mandatory_config("dataset_directory", dataset_configs, "dataset_configs")
        splits_filename = get_mandatory_config("splits_filename", dataset_configs, "dataset_configs")
        dump_filename = get_mandatory_config("dump_filename", dataset_configs, "dataset_configs")
        tile_filename = get_mandatory_config("tile_filename", dataset_configs, "dataset_configs")
        self.output_image_size = get_mandatory_config("output_image_size", dataset_configs, "dataset_configs")
        self.init_from_gps = get_mandatory_config("init_from_gps", dataset_configs, "dataset_configs")
        self.gps_accuracy = get_mandatory_config("gps_accuracy", dataset_configs, "dataset_configs")
        self.global_map_crop_size_meters = get_mandatory_config("global_map_crop_size_meters", dataset_configs, "dataset_configs")
        self.cache_dir = get_mandatory_config("cache_dir", dataset_configs, "dataset_configs")

        # Get the sequence length for this data
        max_sequence_lengths = get_mandatory_config("max_sequence_lengths", dataset_configs, "dataset_configs")
        self.max_sequence_length = max_sequence_lengths[self.dataset_type]

        # Make sure the cache directory exists
        ensure_directory_exists(self.cache_dir)

        # Get the dataset split
        dataset_split = self._get_dataset_split(splits_filename, dataset_type)

        # Get the scenes in the dataset split
        scenes = list(dataset_split.keys())

        # scenes = ['paris', 'avignon']
        # scenes = ["sanfrancisco_soma"]
        # ['sanfrancisco_soma', 'nantes', 'paris', 'sanfrancisco_hayes', 'vilnius', 'avignon', 'amsterdam', 'montrouge', 'berlin', 'helsinki', 'milan', 'toulouse', 'lemans']
        scenes = ['sanfrancisco_soma', 'nantes', 'paris', 'sanfrancisco_hayes', 'avignon', 'amsterdam', 'montrouge', 'berlin', 'helsinki', 'milan', 'toulouse', 'lemans']

        # Make sure the cache directory exists
        ensure_directory_exists(self.cache_dir)

        # Create the cache filepath
        cache_filepath = "{}/mapillary_cache_{}.pt".format(self.cache_dir, dataset_type)

        # Check if it exists and if it does, load the cache
        if(os.path.exists(cache_filepath)):
            print("Loading from cachfile: {}", cache_filepath)

            # Load the cache
            save_dict = torch.load(cache_filepath)
            self.all_images_list = save_dict["all_images_list"]
            self.consolidated_data = save_dict["consolidated_data"]
            self.all_sequences = save_dict["all_sequences"]

        else:
            print("No cache file found {}", cache_filepath)

            # Get the camera and view info
            all_camera_info, all_view_info = self._load_camera_and_view_info(scenes, dump_filename)

            # Get the list of all the images
            self.all_images_list = self._get_image_list(all_view_info)

            # Consolidate the data into a single dict of lists
            self.consolidated_data = self._consolidate_into_data_dict(all_view_info, all_camera_info, self.all_images_list)

            # Keep only the images we have for this split
            self.all_images_list, self.consolidated_data = self._select_out_data_split(dataset_split, self.all_images_list, self.consolidated_data)

            # Convert to tensors
            self.consolidated_data = self._convert_to_numpy(self.consolidated_data)

            # Create the 
            self.all_sequences = self._create_sequences(self.consolidated_data, self.all_images_list, self.max_sequence_length)

            # Need to make sure that we actually have some sequences
            assert(len(self.all_sequences) > 0)

            # Save to the cache
            save_dict = dict()
            save_dict["all_images_list"] = self.all_images_list
            save_dict["consolidated_data"] = self.consolidated_data
            save_dict["all_sequences"] = self.all_sequences
            torch.save(save_dict, cache_filepath)


    def _load_json_file(self, filepath):

        # Load the json
        with open(filepath, "r") as file:
            data = json.load(file)

        return data

    def _get_dataset_split(self, splits_filename, dataset_type):
        # Load the splits file name
        splits_filepath = "{}/{}".format(self.dataset_directory, splits_filename)
        dataset_splits = self._load_json_file(splits_filepath)

        # Extract the split we want
        if(dataset_type == "training"):
            dataset_split = dataset_splits["train"]
        elif(dataset_type == "validation"):
            dataset_split = dataset_splits["val"]
        elif(dataset_type == "evaluation"):
            dataset_split = dataset_splits["val"]
            dataset_split = dataset_splits["train"]
        else:
            assert(False)

        return dataset_split

    def _load_camera_and_view_info(self, scenes, dump_filename):

        # These will hold the info for all the scenes
        all_camera_info = dict()
        all_view_info = dict()

        for scene in scenes:

            # Get the dump file data
            dump_filepath = "{}/{}/{}".format(self.dataset_directory, scene, dump_filename)
            dump_data = self._load_json_file(dump_filepath)

            # Make the camera and view info dicts for this scene
            camera_info = dict()
            view_info = dict()

            for sequence_name in dump_data.keys():
                per_seq = dump_data[sequence_name]

                # get the raw camera and view info 
                raw_camera_info = per_seq["cameras"]
                raw_view_info = per_seq["views"]

                # print("\n\n\n")
                # print(sequence_name) 
                # print(raw_camera_info)

                # Process the camera info
                for camera in raw_camera_info.values():

                    # Make sure its numpy
                    camera["params"] = np.array(camera["params"], np.float32)

                # Process the view info
                for view_name in raw_view_info.keys():

                    # Get the view so we can operate on it
                    view = raw_view_info[view_name]
                    
                    # Make sure its numpy
                    view["R_c2w"] = np.array(view["R_c2w"], np.float32)
                    view["roll_pitch_yaw"] = np.array(view["roll_pitch_yaw"], np.float32)

                    if("observations" in view):
                        view["observations"] = np.array(view["observations"])

                    # remove the "chunk_id" since we dont need it
                    if("chunk_id" in view):
                        view.pop("chunk_id")

                    raw_view_info[view_name] = view


                # Add to the dict
                camera_info[sequence_name] = raw_camera_info
                view_info[sequence_name] = raw_view_info

            # with open('tmp.yaml', 'w') as file:
            #     yaml.dump(camera_info, file)
            # exit()

            # Keep track of this
            all_camera_info[scene] = camera_info
            all_view_info[scene] = view_info

        return all_camera_info, all_view_info

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

    def _get_image_list(self, all_view_info):

        image_list = []

        unique = set()

        for scene in all_view_info.keys():
            for sequence_name in all_view_info[scene].keys():
                for view_name in all_view_info[scene][sequence_name].keys():
                    image_list.append((scene, sequence_name, view_name))

                    name = "{}{}{}".format(scene, sequence_name, view_name)
                    assert(name not in unique)
                    unique.add(name)

        return image_list

    def _consolidate_into_data_dict(self, all_view_info, all_camera_info, all_images_list):

        # We dont need these so get rid of them
        # keys_to_exclude = ["compass_angle","compass_accuracy", "gps_accuracy", "chunk_key", "panorama_offset"]

        # Keys to include
        keys_to_include = ["camera_id", "latlong", "t_c2w", "R_c2w", "roll_pitch_yaw", "capture_time", "gps_position", "index"]

        # All the consolidated data 
        consolidated_data = {k: [] for k in keys_to_include}

        # Get all the info for each of the frames 
        for image_info in all_images_list:
            (scene, sequence_name, view_name) = image_info

            # Get each of the keys we want to include
            for k in consolidated_data.keys():
                consolidated_data[k].append(all_view_info[scene][sequence_name][view_name].get(k, None))

            # Make sure all the camera IDs match correctly with the camera info
            camera_id = consolidated_data["camera_id"][-1]
            camera_info = all_camera_info[scene][sequence_name]
            assert(camera_id in camera_info)

        # Add in the camera data just so we have it.  We dont actually need it
        consolidated_data["cameras"] = all_camera_info

        return consolidated_data

    def _convert_to_numpy(self, consolidated_data):

        # Keys to include
        keys_to_include = ["camera_id", "latlong", "t_c2w", "R_c2w", "roll_pitch_yaw", "capture_time", "gps_position", "index"]

        # Convert to tensor
        for k in keys_to_include:
            
            # convert to numpy
            v = np.array(consolidated_data[k])

            # If we can then we should convert to pytorch
            if(np.issubdtype(v.dtype, np.integer) or np.issubdtype(v.dtype, np.floating)):
                v = torch.from_numpy(v)

            # Update the data
            consolidated_data[k] = v

        return consolidated_data

    def _select_out_data_split(self, dataset_split, all_images_list, consolidated_data):

        # Get the scenes in the dataset split
        scenes = list(dataset_split.keys())

        # Convert the dataset split to be a set for fast lookup
        dataset_split_set = {k: set(dataset_split[k]) for k in dataset_split.keys()}

        # Keep only data that is in this dataset split
        indices_to_remove = []
        for i, (scene_name, sequence_name, view_name) in enumerate(all_images_list):

            # If we dont even have the scene then def we want to get rid of it
            if(scene_name not in scenes):
                indices_to_remove.append(i)
                continue
        
            # See if this image is in the split for this one
            view_name_base = int(view_name.rsplit("_", 1)[0])
            if(view_name_base not in dataset_split_set[scene_name]):
                indices_to_remove.append(i)

        # Deleted the indices from all the data structs.
        # Note we reverse the list so we can pop off the list without having to do a bunch
        # re-index calculations
        indices_to_remove.reverse()
        for i in indices_to_remove:
            all_images_list.pop(i)

            for k in ["camera_id", "latlong", "t_c2w", "R_c2w", "roll_pitch_yaw", "capture_time", "gps_position", "index"]:
                consolidated_data[k].pop(i)

        return all_images_list, consolidated_data

    def _create_sequences(self, consolidated_data, all_images_list, max_sequence_length):

        # For each of the images, get which sequence it is in 
        sequence_to_image_indices = defaultdict(list)
        for i, (scene_name, sequence_name, view_name) in enumerate(all_images_list):
            name = "{}_{}".format(scene_name, sequence_name)
            sequence_to_image_indices[name].append(i)

        # Get the sorting information. This is the info we will use to sort the sequence
        # indices.  If the capture time isnt present then use the index (not ideal)
        sorting_information = consolidated_data.get("capture_time", consolidated_data.get("index"))

        # Sort the indices so that they are in order
        for sequence_name in sequence_to_image_indices.keys():
            sequence_to_image_indices[sequence_name] = sorted(sequence_to_image_indices[sequence_name], key=lambda i: sorting_information[i].item())


        all_sequences = []

        for sequence_name in sequence_to_image_indices.keys():

            # Get the indices
            indices = sequence_to_image_indices[sequence_name]

            # Get the time for each of the indices
            times = []
            for idx in indices:
                times.append(sorting_information[idx].item())

            # Get all the positions for the indices
            positions = []
            for idx in indices: 

                # Get the image info
                (scene_name, sequence_name, view_name) = all_images_list[idx]

                # Get the GPS position
                latlon_gps_gt = self.consolidated_data["gps_position"][idx][:2].clone()
                gps_xy_position_world_frame_gt = self.tile_managers[scene_name].projection.project(latlon_gps_gt.numpy())

                # Get the XY from the camera to world translation vector
                t_c2w_xy_position_world_frame_gt = self.consolidated_data["t_c2w"][idx][:2].clone().double().numpy()

                # If we should init from the GPS or from the camera to world translation vector
                if self.init_from_gps:
                    # position = gps_xy_position_world_frame_gt.copy()
                    assert(False)
                else:
                    position = t_c2w_xy_position_world_frame_gt.copy()

                positions.append(position)

            # Make into numpy arrays 
            times = np.asarray(times)
            positions = np.vstack(positions)

            # Make into sequences that dont "Jump" in time or space
            current_sequence = []
            for i, idx in enumerate(indices):

                # If its the first item in the current sequence then add it
                if(len(current_sequence) == 0):
                    current_sequence.append(idx)
                    continue

                # Get some information
                current_time = times[i]
                current_pos = positions[i]
                last_time = times[i-1]
                last_pos = positions[i-1]

                # Compute the distance traveled (in meters)
                position_diff = (current_pos - last_pos) ** 2
                position_diff = np.sqrt(np.sum(position_diff))

                # Compute the time diff (in milliseconds)
                time_diff = current_time - last_time
                assert(time_diff >= 0)

                # Check if we should make a new sequence or not
                if((time_diff <= 30000) and (position_diff <= 100)):
                    current_sequence.append(idx)
                else:
                    all_sequences.append(current_sequence)
                    current_sequence = [idx]

            # Need to add the last sequence 
            all_sequences.append(current_sequence)


        count = 0
        for sequence_name in sequence_to_image_indices.keys():
            indices = sequence_to_image_indices[sequence_name]
            count += len(indices)

        print("count", count)
        print("all_sequences", len(all_sequences))
        print("all_images_list", len(self.all_images_list))


        # Process each sequence into the final sequences
        final_sequences = []
        for indices in all_sequences:

            print(len(indices))

            # If the sequence is too short then dont use it, just skip it
            if(len(indices) < max_sequence_length):
                continue

            # Chunk the sequence into smaller sub-sequences
            s = 0
            e = s + max_sequence_length
            while(e < len(indices)):
                final_sequences.append(indices[s:e])
                s += max_sequence_length
                e = s + max_sequence_length

        print("final_sequences", len(final_sequences))

        exit()


        return final_sequences

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

    def _get_min_map_size_for_sequences(self):

        largest_map_size = 0

        # sorting_information = self.consolidated_data.get("capture_time", None)
        # print(sorting_information.shape)
        # print(len(self.all_images_list))
        # exit()

        large_map_counter = 0

        for seq_idx in tqdm(range(len(self.all_sequences)), leave=False, desc="Computing Largest Map Size"):

            # We want all the positions in the sequence so we can figure out how big to make the global map
            positions = []

            # seq_idx = 30093

            for seq_img_idx in self.all_sequences[seq_idx]:

                image_info = self.all_images_list[seq_img_idx]
                (scene_name, sequence_name, view_name) = image_info

                # Get the GPS position
                latlon_gps_gt = self.consolidated_data["gps_position"][seq_img_idx][:2].clone()
                gps_xy_position_world_frame_gt = self.tile_managers[scene_name].projection.project(latlon_gps_gt.numpy())

                # Get the XY from the camera to world translation vector
                t_c2w_xy_position_world_frame_gt = self.consolidated_data["t_c2w"][seq_img_idx][:2].clone().double().numpy()

                # If we should init from the GPS or from the camera to world translation vector
                if self.init_from_gps:
                    # xy_position_world_frame_init = gps_xy_position_world_frame_gt.copy()
                    assert(False)
                else:
                    xy_position_world_frame_init = t_c2w_xy_position_world_frame_gt.copy()

                # Convert from numpy to pytorch  and keep track of it
                positions.append(torch.from_numpy(xy_position_world_frame_init))


                # print("\n")
                # print(scene_name, sequence_name, view_name)
                # print(xy_position_world_frame_init)


            # exit()

            # Stack them into a tensor to so we can operate on it
            positions = torch.stack(positions)

            # Get the min and max
            min_x = torch.min(positions[..., 0]).item()
            max_x = torch.max(positions[..., 0]).item()
            min_y = torch.min(positions[..., 1]).item()
            max_y = torch.max(positions[..., 1]).item()

            # Compute the range in the x and y directions
            x_range = max_x - min_x
            y_range = max_y - min_y

            # The map size
            map_size = max(x_range, y_range)

            # if(map_size > 1000):
            #     large_map_counter += 1
            #     print(large_map_counter, seq_idx, map_size)
            #     print(positions.numpy())
            #     print(scene_name)
            #     print(sequence_name)
            #     print(seq_idx)
            #     print("\n")
            #     exit()
                # exit()

            # The overall largest map size
            largest_map_size = max(largest_map_size, map_size)

        print("largest_map_size", largest_map_size)
        print("len(self.all_sequences)", len(self.all_sequences))
        exit()








