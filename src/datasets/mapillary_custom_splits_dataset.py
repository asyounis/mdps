
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

class MapillaryCustomSplitsDataset(MapillaryDatasetBase):
    def __init__(self, dataset_configs, dataset_type):
        super(MapillaryCustomSplitsDataset, self).__init__(dataset_configs, dataset_type)

        # Get the mandatory configs
        self.dataset_directory = get_mandatory_config("dataset_directory", dataset_configs, "dataset_configs")
        splits_filename = get_mandatory_config("splits_filename", dataset_configs, "dataset_configs")
        self.output_image_size = get_mandatory_config("output_image_size", dataset_configs, "dataset_configs")
        self.init_from_gps = get_mandatory_config("init_from_gps", dataset_configs, "dataset_configs")
        self.gps_accuracy = get_mandatory_config("gps_accuracy", dataset_configs, "dataset_configs")
        
        # The name of the length that we will be using
        length_name = get_mandatory_config("length_name", dataset_configs, "dataset_configs")
        assert((length_name == "long") or (length_name == "short"))

        # scenes = ['paris', 'avignon']
        # scenes = ["sanfrancisco_soma"]
        # ['sanfrancisco_soma', 'nantes', 'paris', 'sanfrancisco_hayes', 'vilnius', 'avignon', 'amsterdam', 'montrouge', 'berlin', 'helsinki', 'milan', 'toulouse', 'lemans']
        scenes = ['sanfrancisco_soma', 'nantes', 'paris', 'sanfrancisco_hayes', 'avignon', 'amsterdam', 'montrouge', 'berlin', 'helsinki', 'milan', 'toulouse', 'lemans']

        # Unpack the splits data
        save_dict = torch.load(splits_filename)
        self.all_images_list = save_dict["all_images_list"]
        self.consolidated_data = save_dict["consolidated_data"]
        sequence_splits = save_dict["sequence_splits"]

        # Create the name of the split to use
        split_name = "{}_{}_sequences".format(length_name, dataset_type)

        # Get the specific one for our split
        self.all_sequences = sequence_splits[split_name]

        # get all the indices
        all_indices = []
        for sequence in self.all_sequences:
            for idx in sequence:
                all_indices.append(idx)


        # Create a map from the current index to the new index
        current_index_to_new_index_map = dict()
        for i in range(len(all_indices)):
            current_index_to_new_index_map[all_indices[i]] = i

        # Re-index the images
        new_all_images_list = []    
        for idx in all_indices:
            new_all_images_list.append(self.all_images_list[idx])
        self.all_images_list = new_all_images_list

        # Re-index the consolidated data
        for key in ['camera_id', 'latlong', 't_c2w', 'R_c2w', 'roll_pitch_yaw', 'capture_time', 'gps_position', 'index']:
            new_list = []
            for idx in all_indices:
                new_list.append(self.consolidated_data[key][idx])
            self.consolidated_data[key] = new_list

        # Re-index the sequences
        for sequence_idx in range(len(self.all_sequences)):
            for i in range(len(self.all_sequences[sequence_idx])):
                self.all_sequences[sequence_idx][i] = current_index_to_new_index_map[self.all_sequences[sequence_idx][i]]
