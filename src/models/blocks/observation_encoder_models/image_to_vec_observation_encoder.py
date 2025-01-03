# Python Imports
import copy

# Package Imports
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

# Ali Package Import
from general_ml_framework.models.base_model import BaseModel
from general_ml_framework.utils.config import *

# Project Imports
from models.blocks.observation_encoder_models.encoded_observation import EncodedObservation

class ImageToVecObservationEncoder(nn.Module):
    def __init__(self, configs):
        super(ImageToVecObservationEncoder, self).__init__()

        # Extract some parameters
        self.observation_size = get_mandatory_config("observation_size", configs, "configs")
        self.output_encoding_dim = get_mandatory_config("output_encoding_dim", configs, "configs")

        # Create the encoder
        self.observation_encoder = self._create_observation_encoder_model()

        # We must preprocess the images to have the correct mean and std as the resnet expects 
        self.resnet_image_preprocessor = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    def forward(self, observations, camera_data, dummy_input=None):

        # Check to see if we need to flatten 
        need_to_flatten = (len(observations.shape) == 5)
        if(need_to_flatten):

            # Get some info
            bs, sl, obs_C, H, W = observations.shape

            # Flatten
            observations = torch.reshape(observations, (bs*sl, obs_C, H, W))

        # Pre-process the obs
        observations = self.resnet_image_preprocessor(observations)

        # Run the model
        features = self.observation_encoder(observations)

        # Unflatten if needed
        if(need_to_flatten):
            features = torch.reshape(features, (bs, sl, -1))

        # Normalize and scale by 32
        features = torch.nn.functional.normalize(features, dim=-1)
        features = features * 32.0


        return EncodedObservation(features)



    def _create_observation_encoder_model(self):

        # Get the backbone.  This is with the adaptive pooling and last FC layer removed
        backbone = torchvision.models.resnet50(weights="DEFAULT")
        backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))

        # Compute the size of the flattened output
        # resnet_flattened_output_size = 512 * ((self.observation_size // 32) **2)      
        resnet_flattened_output_size = 512 
        resnet_flattened_output_size*= self.observation_size[0] // 32
        resnet_flattened_output_size*= self.observation_size[1] // 32

        # Create the network
        # See https://github.com/ZhouMengjie/Image-Map-Embeddings/blob/master/models/nets/resnet_nets.py
        layers = nn.Sequential()
        layers.append(backbone)
        layers.append(nn.Conv2d(2048, 512, kernel_size=1, padding=0, stride=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.Conv2d(512, 512, kernel_size=1, padding=0, stride=1))
        layers.append(nn.Flatten())
        layers.append(nn.BatchNorm1d(resnet_flattened_output_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=resnet_flattened_output_size, out_features=1024))
        layers.append(nn.BatchNorm1d(1024))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=1024, out_features=self.output_encoding_dim))

        return layers
