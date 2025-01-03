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
from models.blocks.observation_encoder_models.bev.bev_projection import PolarProjectionDepth, CartesianProjection
from models.blocks.observation_encoder_models.encoded_observation import EncodedObservation
from utils.wrappers import Camera

class BEVFeatureEncoder(nn.Module):
    def __init__(self, input_image_size, backbone_name):
        super(BEVFeatureEncoder, self).__init__()

        # The image stats
        self.IMAGE_MEAN = [0.485, 0.456, 0.406]
        self.IMAGE_STD = [0.229, 0.224, 0.225]
        self.register_buffer("image_mean_internal", torch.tensor(self.IMAGE_MEAN), persistent=False)
        self.register_buffer("image_std_internal", torch.tensor(self.IMAGE_STD), persistent=False)

        # Save this for later so we can check against the actual image we get
        self.input_image_size = input_image_size

        # Create the encoder 
        self.encoder, self.feature_channels, _, self.scale = self._get_encoder_backbone_blocks(input_image_size, backbone_name)

    def get_feature_channels(self):
        return self.feature_channels

    def get_scale(self):
        return self.scale


    #@torch.compile(mode="reduce-overhead")
    def forward(self, image):

        # Check the image size
        assert(image.shape[-1] == self.input_image_size[0])
        assert(image.shape[-2] == self.input_image_size[1])

        # Get some info
        device = image.device
        batch_size = image.shape[0]

        # Normalize the image
        image = image - self.image_mean_internal.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        image = image / self.image_std_internal.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # Run the encoder
        features = self.encoder(image)

        return features

    def _get_encoder_backbone_blocks(self, input_image_size, backbone_name):

        # The model we want to use as the backbone
        # We want the pre-trained version of this and we want to not dilate
        kw = {}
        kw["replace_stride_with_dilation"] = [False, False, False]

        if(backbone_name == "resnet18"):
            backbone = torchvision.models.resnet18(weights="DEFAULT", **kw)
        elif(backbone_name == "resnet50"):
            backbone = torchvision.models.resnet50(weights="DEFAULT", **kw)
        elif(backbone_name == "resnet101"):
            backbone = torchvision.models.resnet101(weights="DEFAULT", **kw)
        else:
            print("Unknown backbone name \"{}\"".format(backbone_name))
            assert(False)


        # Use the feature extractor functionality in torchvision to extract specific layers we wanted
        layers = ["relu", "layer1", "layer2", "layer3", "layer4"]
        encoder = create_feature_extractor(backbone, return_nodes=layers)

        # Create a test image so that we can see the output shapes for all the feature maps
        representative_image = torch.zeros((1, 3, input_image_size[0], input_image_size[1]))

        # Get the feature map sizes
        representative_features = list(encoder(representative_image).values())

        # Get information about the features
        feature_channels = [x.shape[1] for x in representative_features]
        layer_strides = [np.asarray(input_image_size) / f.shape[-1] for f in representative_features]
        scale = layer_strides[0]

        return encoder, feature_channels, layer_strides, scale

class DecoderBlock(nn.Module):
    def __init__(self, feature_map_in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # The layers that process it
        self.layers = nn.Sequential()
        self.layers.append(nn.Conv2d(feature_map_in_channels, out_channels, kernel_size=3, padding=1, padding_mode="replicate", bias=False))
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.ReLU(inplace=True))

    #@torch.compile(mode="reduce-overhead")
    def forward(self, feature_map, skip_layer_feature_map):

       # Upsample it
        _, _, hp, wp = feature_map.shape
        _, _, hs, ws = skip_layer_feature_map.shape
        scale = 2 ** np.round(np.log2(np.array([hs / hp, ws / wp])))
        upsampled = nn.functional.interpolate(feature_map, scale_factor=scale.tolist(), mode="bilinear", align_corners=False)

        # If the shape of the input map `skip` is not a multiple of 2,
        # it will not match the shape of the upsampled map `upsampled`.
        # If the downsampling uses ceil_mode=False, we nedd to crop `skip`.
        # If it uses ceil_mode=True (not supported here), we should pad it.
        _, _, hu, wu = upsampled.shape
        _, _, hs, ws = skip_layer_feature_map.shape
        if (hu <= hs) and (wu <= ws):
            skip_layer_feature_map = skip_layer_feature_map[:, :, :hu, :wu]
        elif (hu >= hs) and (wu >= ws):
            skip_layer_feature_map = nn.functional.pad(skip_layer_feature_map, [0, wu - ws, 0, hu - hs])
        else:
            print("Inconsistent skip_layer_feature_map vs upsampled shapes: {(hs, ws)}, {(hu, wu)}")
            assert(False)

        return self.layers(skip_layer_feature_map) + upsampled

class BEVFeatureDecoder(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(BEVFeatureDecoder, self).__init__()

        # Create The first layer
        self.first_layer = nn.Conv2d(in_channels_list[-1], out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        # Create the specific decoder blocks
        self.decoder_blocks = []
        for c in in_channels_list[::-1][1:]:
            self.decoder_blocks.append(DecoderBlock(c, out_channels))
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

        # The final layer before the output
        self.final_layer = nn.Sequential()
        self.final_layer.append(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False))
        self.final_layer.append(nn.BatchNorm2d(out_channels),)
        self.final_layer.append(nn.ReLU(inplace=True))
            
    #@torch.compile(mode="reduce-overhead")
    def forward(self, encoder_feature_maps):
        feats = None
        for idx, x in enumerate(reversed(encoder_feature_maps.values())):
            if feats is None:
                feats = self.first_layer(x)
            else:
                feats = self.decoder_blocks[idx - 1](feats, x)
        out = self.final_layer(feats)
        return out


    def get_all_feature_maps(self, encoder_feature_maps):
        feats = []
        for idx, x in enumerate(reversed(encoder_feature_maps.values())):
            if len(feats) == 0:
                feats.append(self.first_layer(x))
            else:
                feats.append(self.decoder_blocks[idx - 1](feats[-1], x))
        return feats



class BEVNet(nn.Module):
    def __init__(self, configs):
        super(BEVNet, self).__init__()

        # Get some parameters
        number_of_blocks = get_mandatory_config("number_of_blocks", configs, "configs")
        latent_dim = get_mandatory_config("latent_dim", configs, "configs")
        input_dim = get_mandatory_config("input_dim", configs, "configs")
        self.output_dim = get_mandatory_config("output_dim", configs, "configs")

        # Create the blocks
        self.blocks = nn.Sequential()
        for i in range(number_of_blocks):

            # Make sure we have the correct input dim for this block
            if(i == 0):
                dim = input_dim
            else:
                dim = latent_dim

            # Create the block
            expansion = torchvision.models.resnet.Bottleneck.expansion
            layer = torchvision.models.resnet.Bottleneck(dim, latent_dim // expansion, norm_layer=nn.BatchNorm2d)
            self.blocks.append(layer)

        # Create the final layer
        self.output_layer = nn.Conv2d(latent_dim, self.output_dim, kernel_size=1, padding=0, bias=True)

        # We also predict confidence
        self.confidence_predictor = nn.Conv2d(latent_dim, 1, kernel_size=1, padding=0, bias=True)

    #@torch.compile(mode="reduce-overhead")
    def forward(self, x):
        
        # Get the output
        features = self.blocks(x)
        output = self.output_layer(features)

        # Predict the confidence
        confidence = self.confidence_predictor(features).squeeze(1)
        confidence = torch.sigmoid(confidence)

        return output, confidence


class BEVEncoder(nn.Module):
    def __init__(self, configs):
        super(BEVEncoder, self).__init__()

        # Get some parameters
        input_image_size = get_mandatory_config("input_image_size", configs, "configs")
        self.output_latent_dim = get_mandatory_config("output_latent_dim", configs, "configs")
        z_max = get_mandatory_config("z_max", configs, "configs")
        x_max = get_mandatory_config("x_max", configs, "configs")
        self.pixels_per_meter = get_mandatory_config("pixels_per_meter", configs, "configs")
        scale_range = get_mandatory_config("scale_range", configs, "configs")
        number_of_scales_bins = get_mandatory_config("number_of_scales_bins", configs, "configs")
        bev_net_configs = get_mandatory_config("bev_net_configs", configs, "configs")
        backbone_name = get_mandatory_config("backbone_name", configs, "configs")

        # Create the encoder
        self.encoder = BEVFeatureEncoder(input_image_size, backbone_name)

        # Create the decoder 
        self.decoder = BEVFeatureDecoder(self.encoder.get_feature_channels(), self.output_latent_dim)

        # The scales classifier that will do the linear combination for the different scales
        self.scale_classifier = torch.nn.Linear(self.output_latent_dim, number_of_scales_bins)

        # Create the Polar coordinate projection
        self.projection_polar = PolarProjectionDepth(z_max, self.pixels_per_meter, scale_range, z_min=None)

        # The projection to the birds eye view
        self.projection_bev = CartesianProjection(z_max, x_max, self.pixels_per_meter, z_min=None)

        # Create the BEV Network
        self.bev_net = BEVNet(bev_net_configs)

    #@torch.compile(mode="reduce-overhead")
    def forward(self, image, camera_data, dummy_input=None):

        # Convert the camera data into the Camera Object 
        camera = Camera(camera_data)

        # Check to see if we need to flatten 
        need_to_flatten = (len(image.shape) == 5)
        if(need_to_flatten):

            # Get some info
            bs, sl, obs_C, obs_H, obs_W = image.shape

            # Flatten
            image = torch.reshape(image, (bs*sl, obs_C, obs_H, obs_W))
            camera = torch.reshape(camera, (bs*sl, ))

        # Get some info
        device = image.device

        # Run the encoder
        feature_maps = self.encoder(image)

        # Run the decoder
        feature_map = self.decoder(feature_maps)

        # Scale the camera
        camera = camera.scale(1 / self.encoder.get_scale())
        camera = camera.to(device, non_blocking=True)

        # Compute the scale weights
        scale_weights = self.scale_classifier(feature_map.moveaxis(1, -1))

        # Project into polar coordinates
        polar_feature_map = self.projection_polar(feature_map, scale_weights, camera)

        # Map to the BEV.
        with torch.autocast("cuda", enabled=False):
            bev_feature_map, bev_valid, _ = self.projection_bev(polar_feature_map.float(), None, camera.float())

        # Do some final processing
        bev_feature_map, confidence = self.bev_net(bev_feature_map)

        # Unflatten if we need to
        if(need_to_flatten):
            bev_feature_map = torch.reshape(bev_feature_map, (bs, sl, bev_feature_map.shape[1], bev_feature_map.shape[2], bev_feature_map.shape[3]))
            bev_valid = torch.reshape(bev_valid, (bs, sl, bev_valid.shape[1], bev_valid.shape[2]))
            confidence = torch.reshape(confidence, (bs, sl, confidence.shape[1], confidence.shape[2]))

        return EncodedObservation((bev_feature_map, bev_valid, confidence))


    def get_bev_grid_xz(self):
        return self.projection_bev.grid_xz

    def get_pixels_per_meter(self):
        return self.pixels_per_meter

