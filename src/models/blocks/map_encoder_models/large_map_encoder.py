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

class MapFeatureEncoder(nn.Module):
    def __init__(self, number_of_blocks, input_dim, backbone_name):
        super(MapFeatureEncoder, self).__init__()

        # Create the encoder 
        self.encoder_blocks, self.encoder_output_layers_out_channels = self._get_encoder_backbone_blocks(number_of_blocks, input_dim, backbone_name)

        # Need to update the padding mode of the encoder blocks to be "replicate" instead of zeros
        def update_padding_fn(module):
            if isinstance(module, nn.Conv2d):
                module.padding_mode = "replicate"
        self.encoder_blocks.apply(update_padding_fn)


    def get_output_layers_out_channels(self):
        return self.encoder_output_layers_out_channels

    #@torch.compile(mode="reduce-overhead")
    def forward(self, image):

        # Get some info
        device = image.device
        batch_size = image.shape[0]

        # Run the encoder
        output_feature_maps = []
        current_feature_map = image
        for encoder_block in self.encoder_blocks:
            current_feature_map = encoder_block(current_feature_map)
            output_feature_maps.append(current_feature_map)

        return output_feature_maps




    def _get_encoder_backbone_blocks(self, number_of_blocks, input_dim, backbone_name):

        #     # The model we want to use as the backbone
        #     vgg19 = torchvision.models.vgg19(weights=None)
        #     pytorch_total_params = sum(p.numel() for p in vgg19.parameters() if p.requires_grad)
        #     print("vgg19: {:,}".format(pytorch_total_params))


        # All the blocks of the encoder
        encoder_blocks = []
        encoder_output_layers_out_channels = []

        if("resnet" in backbone_name):

            # Create the backbone object
            if(backbone_name == "resnet50"):
                backbone = torchvision.models.resnet50(weights=None)
            elif(backbone_name == "resnet101"):
                backbone = torchvision.models.resnet101(weights=None)
            elif(backbone_name == "resnet18"):
                backbone = torchvision.models.resnet18(weights=None)
            else:
                print("Unknown backbone name \"{}\"".format(backbone_name))
                assert(False)

            # The different backbones have different output features
            # We could probably automate this with code but right now
            # its quick and easy to just hard code these since the networks
            # dont really change anyways
            if((backbone_name == "resnet50") or (backbone_name == "resnet50")):
                encoder_output_layers_out_channels.append(64)
                encoder_output_layers_out_channels.append(256)
                encoder_output_layers_out_channels.append(512)
                encoder_output_layers_out_channels.append(1024)
                encoder_output_layers_out_channels.append(2048)
            elif(backbone_name == "resnet18"):
                encoder_output_layers_out_channels.append(64)
                encoder_output_layers_out_channels.append(64)
                encoder_output_layers_out_channels.append(128)
                encoder_output_layers_out_channels.append(256)
                encoder_output_layers_out_channels.append(512)


            # Encoder 0
            current_block = nn.Sequential()
            args = {k: getattr(backbone.conv1, k) for k in backbone.conv1.__constants__}
            args.pop("output_padding")
            layer = torch.nn.Conv2d(**{**args, "in_channels": input_dim})
            current_block.append(layer)
            current_block.append(backbone.bn1)
            current_block.append(backbone.relu)
            encoder_blocks.append(current_block)

            # Encoder 1
            current_block = nn.Sequential()
            current_block.append(backbone.maxpool)
            current_block.append(backbone.layer1)
            encoder_blocks.append(current_block)

            # Encoder 2
            encoder_blocks.append(backbone.layer2)

            # Encoder 3
            encoder_blocks.append(backbone.layer3)

            # Encoder 4
            encoder_blocks.append(backbone.layer4)

            # Make the encoder a module list so pytorch knows about their params
            encoder_blocks = nn.ModuleList(encoder_blocks)


        elif("vgg" in backbone_name):

            # Create the backbone object
            if(backbone_name == "vgg19"):
                backbone = torchvision.models.vgg19(weights=None)
            elif(backbone_name == "vgg16"):
                backbone = torchvision.models.vgg16(weights=None)
            else:
                print("Unknown backbone name \"{}\"".format(backbone_name))
                assert(False)

            # The number of channels in the layer of the last block (prior to the downsampling)
            last_layer_out_channels = None

            # The current block we are on
            current_block = nn.Sequential()

            # Break VGG into several blocks and keep track of the output channels so we can construct
            for i, layer in enumerate(backbone.features):

                # If it is a Conv layer then we just need to update the last layers outputs
                if isinstance(layer, nn.Conv2d):              

                    # If this is the first layer then the input dim may not be the same for VGG
                    # So we have to change it to make it match the dim we expected
                    # here we only change the input channels to be input dim, leaving everything else the same
                    if((i == 0) and (input_dim != layer.in_channels)):
                        args = {k: getattr(layer, k) for k in layer.__constants__}
                        args.pop("output_padding")
                        layer = torch.nn.Conv2d(**{**args, "in_channels": input_dim})

                    # Keep track of the out channels for this layer in case it is the last conv layer of the block
                    last_layer_out_channels = layer.out_channels

                elif isinstance(layer, nn.MaxPool2d):

                    # Make sure that the last lat
                    assert(last_layer_out_channels is not None)

                    # We just finished a block so we should output the final layer
                    encoder_output_layers_out_channels.append(last_layer_out_channels)

                    # start a new block
                    encoder_blocks.append(current_block)
                    current_block = nn.Sequential()

                    # If we have all the downsampling blocks we wanted so we should exit since 
                    # we have all the blocks we need
                    if((number_of_blocks + 1) == len(encoder_blocks)):    
                        break

                # Add the current layer to the current block
                current_block.append(layer)

            # Make the encoder a module list so pytorch knows about their params
            encoder_blocks = nn.ModuleList(encoder_blocks)


        else:
            print("Unknown backbone name \"{}\"".format(backbone_name))
            assert(False)

        return encoder_blocks, encoder_output_layers_out_channels


class DecoderBlock(nn.Module):
    def __init__(self, feature_map_in_channels, skip_layer_feature_map_in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # The up-sampling layer
        self.upsample_layer = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # The layers that process it
        self.layers = nn.Sequential()
        self.layers.append(nn.Conv2d(feature_map_in_channels + skip_layer_feature_map_in_channels, out_channels, kernel_size=3, padding=1, padding_mode="replicate", bias=False))
        # self.layers.append(nn.Conv2d(feature_map_in_channels + skip_layer_feature_map_in_channels, out_channels, kernel_size=3, padding=1, padding_mode="zeros", bias=False))
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.ReLU(inplace=True))

    #@torch.compile(mode="reduce-overhead")
    def forward(self, feature_map, skip_layer_feature_map):

        # Upsample it
        upsampled = self.upsample_layer(feature_map)

        # @todo is this needed? Seems hacky
        # # If the shape of the input map `skip` is not a multiple of 2,
        # # it will not match the shape of the upsampled map `upsampled`.
        # # If the downsampling uses ceil_mode=False, we nedd to crop `skip`.
        # # If it uses ceil_mode=True (not supported here), we should pad it.
        # _, _, hu, wu = upsampled.shape
        # _, _, hs, ws = skip.shape
        # assert (hu <= hs) and (wu <= ws), "Using ceil_mode=True in pooling?"
        # # assert (hu == hs) and (wu == ws), 'Careful about padding'
        # skip = skip[:, :, :hu, :wu]


        # Concat the up-sampled with the skip connection
        x = torch.cat([upsampled, skip_layer_feature_map], dim=1)

        # Process
        x = self.layers(x)

        return x

class MapFeatureDecoder(nn.Module):
    def __init__(self, feature_encoder_out_channels,decoder_latent_dims):
        super(MapFeatureDecoder, self).__init__()

        # Create the decoder blocks
        self.decoder_blocks = self._create_decoder_blocks(feature_encoder_out_channels, decoder_latent_dims)

    def _create_decoder_blocks(self, feature_encoder_out_channels, decoder_latent_dims):

        # Reverse the output dims since the decoder will go from smallest to larger
        feature_encoder_out_channels = copy.deepcopy(feature_encoder_out_channels)
        feature_encoder_out_channels.reverse()

        # The current output dim
        current_output_latent_dim = feature_encoder_out_channels[0]

        # Create the blocks one at a time
        decoder_blocks = []
        for i in range(len(decoder_latent_dims)):

            # Get the current latent dim for this decoder block (aka the dim of the output)
            decoder_latent_dim = decoder_latent_dims[i]

            # The skip output we will be using, this accounts for the initial input
            skip_dim = feature_encoder_out_channels[i + 1]

            # Create the decoder
            decoder_blocks.append(DecoderBlock(current_output_latent_dim, skip_dim, decoder_latent_dim))

            # The new output latent dim is the output of this block
            current_output_latent_dim = decoder_latent_dim


        # Make the encoder a module list so pytorch knows about their params
        decoder_blocks = nn.ModuleList(decoder_blocks)

        return decoder_blocks

    #@torch.compile(mode="reduce-overhead")
    def forward(self, encoder_feature_maps):

        # Create a copy of the encoder feature maps but backwards.  
        # Do this so we can make a reversed version of the list without modifying the original list 
        # There must be a better way of doing this
        reversed_encoder_feature_maps = [x for x in reversed(encoder_feature_maps)]

        # Decode!
        current_output = reversed_encoder_feature_maps[0]
        decoded_outputs = [current_output]
        for i, block in enumerate(self.decoder_blocks):

            # Get the skip featured maps
            skip_encoder_feature_map = reversed_encoder_feature_maps[i+1]

            # Run the decoder and save it
            output = block(current_output, skip_encoder_feature_map)
            decoded_outputs.append(output)

            # Update the output
            current_output = output


        return decoded_outputs


class LargeMapEncoder(nn.Module):
    def __init__(self, configs):
        super(LargeMapEncoder, self).__init__()

        # Get some parameters
        number_of_classes = get_mandatory_config("number_of_classes", configs, "configs")
        embedding_dim = get_mandatory_config("embedding_dim", configs, "configs")
        encoder_number_of_blocks = get_mandatory_config("encoder_number_of_blocks", configs, "configs")
        self.decoder_latent_dims = get_mandatory_config("decoder_latent_dims", configs, "configs")
        self.output_latent_dim = get_mandatory_config("output_latent_dim", configs, "configs")
        backbone_name = get_mandatory_config("backbone_name", configs, "configs")

        # Make sure that the number of encoder blocks matches up with the number of decoder layers
        # Since each encoder output will be used by some decoder
        assert(encoder_number_of_blocks == len(self.decoder_latent_dims))

        # Create the embeddings that we will use
        self.embeddings, self.category_names = self._create_embeddings(number_of_classes, embedding_dim)

        # create the feature encoder
        map_feature_encoder_input_dim = len(number_of_classes) * embedding_dim
        self.feature_encoder = MapFeatureEncoder(encoder_number_of_blocks, map_feature_encoder_input_dim, backbone_name)

        # Create the feature decoder
        self.feature_decoder = MapFeatureDecoder(self.feature_encoder.get_output_layers_out_channels(), self.decoder_latent_dims) 

        # The final layer to convert to the right output
        self.final_layer = nn.Conv2d(self.decoder_latent_dims[-1], self.output_latent_dim, kernel_size=1, padding=0, bias=True)

    def get_output_dim(self):
        return self.output_latent_dim

    def forward(self, image, dummy_input=None):

        # Get the feature maps
        decoder_feature_maps = self.get_all_map_features(image)

        # Get the final feature map
        final_feature_map = decoder_feature_maps[-1]

        # Process it one more time
        final_feature_map = self.final_layer(final_feature_map)

        return final_feature_map

    def get_all_map_features(self, image):

        # Embed
        computed_embedding = self.compute_embedding(image)

        # Get the encoder features
        encoder_feature_maps = self.feature_encoder(computed_embedding)

        # Run the decoder
        decoder_feature_maps = self.feature_decoder(encoder_feature_maps)

        # the first decider map is internal only, not one that was specified to be output
        # based on the input parameter "decoder_latent_dims""
        return decoder_feature_maps[1:]

    def compute_embedding(self, image, dummy_input=None):

        # Embed
        computed_embedding = []
        for i, k in enumerate(self.category_names):
            ev = self.embeddings[k](image[:, i, :, :])
            computed_embedding.append(ev)

        # Make into 1 embedding tensor
        computed_embedding = torch.cat(computed_embedding, dim=-1)

        # Reshape it since we are going to process with an image neural net which needs the channels to be in the
        # first dim
        computed_embedding = torch.permute(computed_embedding, (0, 3, 1, 2))        

        return computed_embedding



    def _create_embeddings(self, number_of_classes, embedding_dim):

        # All the embeddings
        embeddings = nn.ModuleDict()

        # The categories that we have
        category_names = []

        # Each class should have its own embedding
        for category_name in number_of_classes.keys():

            # Keep track of the order of this for later
            category_names.append(category_name)

            # The number of classes in that category
            num_classes = number_of_classes[category_name]

            # Create the embedding
            embeddings[category_name] = nn.Embedding(num_classes + 1, embedding_dim)


        return embeddings, category_names