# Python Imports
import copy

# Package Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Ali Package Import
from general_ml_framework.models.base_model import BaseModel
from general_ml_framework.utils.config import *

# Project Imports
from models.blocks.map_encoder_models.large_map_encoder import LargeMapEncoder

class RetievalMapEncoder(LargeMapEncoder):
    def __init__(self, configs):

        # Compute the output latent dim for the upper model
        self.number_of_rotation_outputs = get_mandatory_config("number_of_rotation_outputs", configs, "configs")
        self.output_embedding_dim = get_mandatory_config("output_embedding_dim", configs, "configs")

        # Make a copy of the dict so we can edit it for the parent class
        configs_copy = copy.deepcopy(configs)
        configs_copy["output_latent_dim"] = self.number_of_rotation_outputs * self.output_embedding_dim 

        # Finally call the parent constructor
        super(RetievalMapEncoder, self).__init__(configs_copy)

    def forward(self, image, dummy_input=None):

        # We need the batch and sequence dims to be squashed together
        assert(len(image.shape) == 4)

        # Get the feature maps
        decoder_feature_maps = self.get_all_map_features(image)

        # Get the final feature map
        final_feature_map = decoder_feature_maps[-1]

        # Process it one more time
        final_feature_map = self.final_layer(final_feature_map)

        # Convert the output
        B, _, H, W = final_feature_map.shape
        final_feature_map = torch.reshape(final_feature_map, [B, self.output_embedding_dim, self.number_of_rotation_outputs, H, W])

        # Convert from [B, C, R, H, W] -> [B, C, H, W, R]
        final_feature_map = torch.permute(final_feature_map, [0, 1, 3, 4, 2])

        # Normalize and scale by 32
        final_feature_map = torch.nn.functional.normalize(final_feature_map, dim=1)
        final_feature_map = final_feature_map * 32.0

        return final_feature_map

