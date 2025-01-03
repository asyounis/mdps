# Python Imports
import time

# Package Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision
import numpy as np

# Ali Package Import
from general_ml_framework.utils.config import *


from models.blocks.voting import conv2d_fft_batchwise, TemplateSampler

class GenericMapBasedWeightModel(nn.Module):
    def __init__(self, configs):
        super(GenericMapBasedWeightModel, self).__init__()

        # Get the parameters to use for local map extraction
        particle_dims_to_use_for_local_map_extraction = get_mandatory_config("particle_dims_to_use_for_local_map_extraction", configs, "configs")
        self.particle_dim_translate_x = get_mandatory_config("translate_x", particle_dims_to_use_for_local_map_extraction, "particle_dims_to_use_for_local_map_extraction")
        self.particle_dim_translate_y = get_mandatory_config("translate_y", particle_dims_to_use_for_local_map_extraction, "particle_dims_to_use_for_local_map_extraction")
        self.particle_dim_rotation = get_mandatory_config("rotation", particle_dims_to_use_for_local_map_extraction, "particle_dims_to_use_for_local_map_extraction")


    def forward(self, particles):
        raise NotImplemented


    def _compute_local_global_map_matching(self, local_maps, encoded_observations):

        # Unpack
        bev_observations, bev_valid, bev_confidence = encoded_observations

        # Get some info
        device = bev_observations.device
        batch_size = bev_observations.shape[0]
        bev_observations_height = bev_observations.shape[2]
        bev_observations_width = bev_observations.shape[3]

        # Multiply BEV by the confidence
        bev_observations = bev_observations * bev_confidence.unsqueeze(1)

        # Mask out the invalid pixels in the BEV 
        bev_observations = bev_observations.masked_fill(~bev_valid.unsqueeze(1), 0.0)

        # Normalize the vectors so that they are unit vectors and hopefully dont blow up during training
        # bev_observations = torch.nn.functional.normalize(bev_observations, dim=1)
        # local_maps = torch.nn.functional.normalize(local_maps, dim=2)

        # Compute the matching score by looking at the alignment (aka dot product)
        bev_observations_flattened = bev_observations.view(batch_size, bev_observations.shape[1], -1)
        local_maps_flattened = local_maps.view(batch_size, local_maps.shape[1], local_maps.shape[2], -1)    
        matching_score = bev_observations_flattened.unsqueeze(1) * local_maps_flattened
        matching_score_shape = matching_score.shape
        matching_score = torch.sum(matching_score, dim=2)

        # Sum all the different dot products
        matching_score = torch.sum(matching_score, dim=-1)

        # Normalize using the number of pixels being compared
        matching_score = matching_score / float(matching_score_shape[-1])        
        
        # Normalize the matching score using the number of valid
        bev_valid_flattened = bev_valid.view(batch_size, -1)
        matching_score = matching_score / torch.sum(bev_valid_flattened, dim=-1).unsqueeze(-1)

        return matching_score

    # @torch.compile(mode="reduce-overhead")
    def _extract_local_maps(self, particles, encoded_global_map, bev_observation_size):
        
        # Extract some info
        device = particles.device
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]

        # Flatten the particles down so we can 
        particles = torch.reshape(particles, (batch_size*number_of_particles, -1))

        # The batch size of the flattened version of the particles
        batch_size_flattened = particles.shape[0]

        # Pre-compute sin and cos
        # thetas = -particles[...,-1] - (0.5 * np.pi)
        # thetas = -particles[...,self.particle_dim_rotation] - (0.5 * np.pi)
        thetas = particles[...,self.particle_dim_rotation]
        sin_thetas = torch.sin(thetas)
        cos_thetas = torch.cos(thetas)


        # Some other information we need
        map_width = encoded_global_map.shape[2]
        map_height = encoded_global_map.shape[3]

        # Some useful things to have for later
        one = torch.ones((batch_size_flattened, ), device=device)
        zero = torch.zeros((batch_size_flattened, ), device=device)

        ##########################################################################################################################################
        ### Create the Affine Matrix
        ##########################################################################################################################################

        # ---------------------------------------------------------------------
        # Step 1: Move the map such that the map is centered around the particle
        # ---------------------------------------------------------------------

        # Compute the translation
        translate_x = particles[..., self.particle_dim_translate_x]
        translate_y = particles[..., self.particle_dim_translate_y]

        # Scale it since the grid sampler wants it from [-1, 1] not [map_width, map_height]
        translate_x = (translate_x * (2.0 / float(map_width))) - 1.0
        translate_y = (translate_y * (2.0 / float(map_height))) - 1.0

        # create the matrix
        translation_mat_1 = torch.stack([one, zero, translate_x, zero, one, translate_y, zero, zero, one], dim=-1)
        translation_mat_1 = torch.reshape(translation_mat_1, (-1, 3, 3))

        # ---------------------------------------------------------------------
        # Step 2: Rotate the map such that that the orientation matches that of the state
        # ---------------------------------------------------------------------
        # rotation_matrix = torch.stack([cos_thetas, sin_thetas, zero, -sin_thetas, cos_thetas, zero, zero, zero, one], dim=-1)
        rotation_matrix = torch.stack([cos_thetas, -sin_thetas, zero, sin_thetas, cos_thetas, zero, zero, zero, one], dim=-1)
        rotation_matrix = torch.reshape(rotation_matrix, (-1, 3, 3))

        # ---------------------------------------------------------------------
        # Step 3: scale down the map
        # ---------------------------------------------------------------------
        # scale_x = torch.full((batch_size_flattened, ), fill_value=(float(bev_observation_size[0]) / float(map_width)), device=device) # Broken but trained on originally and does something reasonable
        # scale_y = torch.full((batch_size_flattened, ), fill_value=(float(bev_observation_size[1]) / float(map_height)), device=device) # Broken but trained on originally and does something reasonable
        scale_x = torch.full((batch_size_flattened, ), fill_value=(float(bev_observation_size[1]) / float(map_width)), device=device)
        scale_y = torch.full((batch_size_flattened, ), fill_value=(float(bev_observation_size[0]) / float(map_height)), device=device)
        scale_mat = torch.stack([scale_x, zero, zero, zero, scale_y, zero, zero, zero, one], dim =-1)
        scale_mat = torch.reshape(scale_mat, (-1, 3, 3))

        # ---------------------------------------------------------------------
        # Step 4: translate the local map such that the state defines the bottom mid-point instead of the center
        # ---------------------------------------------------------------------
        translate_y = -torch.ones((batch_size_flattened ,), device=particles.device)
        translation_mat_2 = torch.stack([one, zero, zero, zero, one, translate_y, zero, zero, one], dim =-1)
        translation_mat_2 = torch.reshape(translation_mat_2, (-1, 3, 3))

        # ---------------------------------------------------------------------
        # Step 5: Combine all the matrices into a final affine matrix
        # ---------------------------------------------------------------------
        final_affine_matrix = translation_mat_1
        final_affine_matrix = torch.bmm(final_affine_matrix, rotation_matrix)
        final_affine_matrix = torch.bmm(final_affine_matrix, scale_mat)
        final_affine_matrix = torch.bmm(final_affine_matrix, translation_mat_2)
        final_affine_matrix = final_affine_matrix[:, :2, :]

        ########################################################################################################################################
        # Sample the local map using the affine matrix
        ########################################################################################################################################
        
        # Create the affine grid
        affine_sampling_grid = torch.nn.functional.affine_grid(final_affine_matrix, (batch_size_flattened, 1, bev_observation_size[0], bev_observation_size[1]) , align_corners=False)

        # Reshape the grid to have a correct particle dim
        affine_sampling_grid = affine_sampling_grid.view((batch_size, number_of_particles, affine_sampling_grid.shape[1], affine_sampling_grid.shape[2], affine_sampling_grid.shape[3]))

        # Add zeros to the end so we can do that "use the 5D sampler instead of 4D" to get particles hack
        zeros = torch.zeros(affine_sampling_grid.shape[:-1], device=device)
        zeros = zeros.unsqueeze(-1)
        affine_sampling_grid = torch.cat([affine_sampling_grid, zeros], dim=-1)
        
        # Get the local maps
        local_maps = torch.nn.functional.grid_sample(encoded_global_map.unsqueeze(2), affine_sampling_grid, align_corners=False)
        local_maps = torch.permute(local_maps, [0, 2, 1, 3, 4])

        return local_maps





class MapMatchingWeightModel(GenericMapBasedWeightModel):
    def __init__(self, configs):
        super(MapMatchingWeightModel, self).__init__(configs)

    def forward(self, input_dict):

        # Unpack
        particles = input_dict["particles"]
        encoded_global_map = input_dict["encoded_global_map"]
        encoded_observation = input_dict["encoded_observations"]
        unnormalized_resampled_particle_log_weights = input_dict["unnormalized_resampled_particle_log_weights"]

        # Unpack
        bev_observations, bev_valid, bev_confidence = encoded_observation

        # Extract the local maps and encode them
        bev_observation_size = bev_observations.shape[2:]
        local_maps = self._extract_local_maps(particles, encoded_global_map, bev_observation_size)

        # if(torch.sum(torch.isnan(particles)) > 0):
        #     print("particles")
        #     print(torch.sum(particles))
        #     assert(False)

        # if(torch.sum(torch.isnan(encoded_global_map)) > 0):
        #     print("encoded_global_map")
        #     print(torch.sum(encoded_global_map))
        #     assert(False)

        # if(torch.sum(torch.isnan(local_maps)) > 0):
        #     print("local_maps")
        #     print(torch.sum(local_maps))
        #     assert(False)





        # Compute the matching score
        matching_score = self._compute_local_global_map_matching(local_maps, encoded_observation)


        # if(torch.sum(torch.isnan(matching_score)) > 0):
        #     print("matching_score")
        #     print(torch.sum(matching_score, dim=1, keepdim=True))
        #     assert(False)

        # The unnormalized weights are just the matching costs
        unnormalized_particle_log_weights = matching_score

        # Do this to get good gradients + its good math
        if(unnormalized_resampled_particle_log_weights is not None):
            unnormalized_particle_log_weights = unnormalized_particle_log_weights + unnormalized_resampled_particle_log_weights    

        # add in the min weight
        unnormalized_particle_log_weights = unnormalized_particle_log_weights

        # Normalize the weights
        # new_particle_weights = unnormalized_particle_log_weights / torch.sum(unnormalized_particle_log_weights, dim=1, keepdim=True)
        new_particle_log_weights = unnormalized_particle_log_weights - torch.logsumexp(unnormalized_particle_log_weights, dim=1, keepdim=True)
        new_particle_weights = torch.exp(new_particle_log_weights)


        # if(torch.sum(torch.isnan(unnormalized_particle_log_weights)) > 0):
        #     print("unnormalized_particle_log_weights")
        #     print(torch.sum(unnormalized_particle_log_weights, dim=1, keepdim=True))
        #     assert(False)


        # if(torch.sum(torch.isnan(unnormalized_resampled_particle_log_weights)) > 0):
        #     print("unnormalized_resampled_particle_log_weights")
        #     print(torch.sum(unnormalized_resampled_particle_log_weights, dim=1, keepdim=True))
        #     assert(False)


        # if(torch.sum(torch.isnan(new_particle_log_weights)) > 0):
        #     print("new_particle_log_weights")
        #     print(torch.sum(new_particle_log_weights, dim=1, keepdim=True))
        #     assert(False)


        # if(torch.sum(torch.isnan(new_particle_weights)) > 0):
        #     print("new_particle_weights")
        #     print(torch.sum(new_particle_weights, dim=1, keepdim=True))
        #     assert(False)


        return new_particle_weights





class MapMatchingWithAdditionalInputsWeightModel(GenericMapBasedWeightModel):
    def __init__(self, configs):
        super(MapMatchingWithAdditionalInputsWeightModel, self).__init__(configs)

        # get the parameters that we need
        input_dim = get_mandatory_config("input_dim", configs, "configs")
        latent_space = get_mandatory_config("latent_space", configs, "configs")
        number_of_layers = get_mandatory_config("number_of_layers", configs, "configs")
        non_linear_type = get_mandatory_config("non_linear_type", configs, "configs")
        self.min_weight = get_mandatory_config("min_weight", configs, "configs")
        self.max_weight = get_mandatory_config("max_weight", configs, "configs")

        # Construct the network
        self.network = self._create_linear_FF_network(input_dim, 1, non_linear_type, latent_space, number_of_layers)

        # Add a sigmoid so we can bound the weights
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_dict):

        # Unpack
        particles = input_dict["particles"]
        encoded_global_map = input_dict["encoded_global_map"]
        encoded_observation = input_dict["encoded_observations"]
        unnormalized_resampled_particle_log_weights = input_dict["unnormalized_resampled_particle_log_weights"]
        additional_inputs = input_dict["additional_inputs"]

        # Unpack
        bev_observations, bev_valid, bev_confidence = encoded_observation

        # Extract the local maps and encode them
        bev_observation_size = bev_observations.shape[2:]
        local_maps = self._extract_local_maps(particles, encoded_global_map, bev_observation_size)

        # Compute the matching score
        matching_score = self._compute_local_global_map_matching(local_maps, encoded_observation)

        # Put everything into a single tensor
        unified_input = torch.cat([matching_score.unsqueeze(-1), additional_inputs], dim=-1)

        # Flatten
        B, N, C = unified_input.shape
        unified_input = torch.reshape(unified_input, (B*N, C))

        # Compute NN
        unnormalized_particle_weights = self.network(unified_input)

        # Unflatten
        unnormalized_particle_weights = torch.reshape(unnormalized_particle_weights, (B, N))


        # Bound the weight
        unnormalized_particle_weights = torch.sigmoid(unnormalized_particle_weights)        
        unnormalized_particle_weights = unnormalized_particle_weights * (self.max_weight - self.min_weight)
        unnormalized_particle_weights = unnormalized_particle_weights + self.min_weight
        
        # Do this to get good gradients + its good math
        if(unnormalized_resampled_particle_log_weights is not None):
            unnormalized_particle_weights = unnormalized_particle_weights * torch.exp(unnormalized_resampled_particle_log_weights)

        # Normalize the weights
        new_particle_weights = torch.nn.functional.normalize(unnormalized_particle_weights, p=1.0, eps=1e-8, dim=1)

        # print(new_particle_weights.shape)
        # print(new_particle_weights[0])

        return new_particle_weights


    def _create_linear_FF_network(self, input_dim, output_dim, non_linear_type, latent_space, number_of_layers):

        # Need at least 2 layers, the input and output layers
        assert(number_of_layers >= 2)

        # Get the non linear object from the name
        non_linear_object = self._get_non_linear_object_from_string(non_linear_type)

        # All the layers that this model will have
        layers = nn.Sequential()

        # Create the input layer
        layers.append(nn.Linear(in_features=(input_dim),out_features=latent_space))
        layers.append(non_linear_object())

        # the middle layers are all the same fully connected layers
        for i in range(number_of_layers-2):
            layers.append(nn.Linear(in_features=latent_space,out_features=latent_space))
            layers.append(non_linear_object())
        
        # Final layer is the output space and so does not need a non-linearity
        layers.append(nn.Linear(in_features=latent_space, out_features=output_dim))

        return layers


    def _get_non_linear_object_from_string(self, non_linear_type):

        # Select the non_linear type object to use
        if(non_linear_type == "ReLU"):
            non_linear_object = nn.ReLU
        elif(non_linear_type == "PReLU"):
            non_linear_object = nn.PReLU    
        elif(non_linear_type == "Tanh"):
            non_linear_object = nn.Tanh    
        elif(non_linear_type == "Sigmoid"):
            non_linear_object = nn.Sigmoid    
        else:
            assert(False)

        return non_linear_object













class MapMatchingWeightModelForOrienternetModels(GenericMapBasedWeightModel):
    def __init__(self, configs):
        super(MapMatchingWeightModelForOrienternetModels, self).__init__(configs)

    def forward(self, input_dict):

        # This is not grad safe so do not do any gradients
        with torch.no_grad():

            # Unpack
            particles = input_dict["particles"]
            encoded_global_map = input_dict["encoded_global_map"]
            encoded_observation = input_dict["encoded_observations"]
            unnormalized_resampled_particle_log_weights = input_dict["unnormalized_resampled_particle_log_weights"]
            observation_model = input_dict["observation_model"]

            num_of_matching_rotations = 1
            ts = TemplateSampler(observation_model.get_bev_grid_xz(), 2, num_of_matching_rotations, optimize=False)
            ts = ts.to(particles.device)

            # Unpack
            bev_observations, bev_valid, bev_confidence = encoded_observation

            # Extract the local maps and encode them
            # bev_observation_size = bev_observations.shape[2:]
            bev_observation_size = [129, 129]
            local_maps = self._extract_local_maps(particles, encoded_global_map, bev_observation_size)


            # Compute the matching score
            matching_score = self._compute_local_global_map_matching(local_maps, encoded_observation, ts)

            print(matching_score)

            # The unnormalized weights are just the matching costs
            unnormalized_particle_log_weights = matching_score

            # Do this to get good gradients + its good math
            if(unnormalized_resampled_particle_log_weights is not None):
                unnormalized_particle_log_weights = unnormalized_particle_log_weights + unnormalized_resampled_particle_log_weights    

            # Normalize via softmax
            # unnormalized_particle_log_weights = torch.nn.functional.log_softmax(unnormalized_particle_log_weights, dim=-1)

            # Convert to weights
            # new_particle_weights = torch.exp(unnormalized_particle_log_weights)


            print(unnormalized_particle_log_weights)

            print("")
            print(torch.max(unnormalized_particle_log_weights))
            print("")

            new_particle_weights = torch.nn.functional.softmax(unnormalized_particle_log_weights, dim=-1)



            print(new_particle_weights)
            exit()

        return new_particle_weights





    # def _compute_local_global_map_matching(self, local_maps, encoded_observations):

    #     # Unpack
    #     bev_observations, bev_valid, bev_confidence = encoded_observations

    #     # Get some info
    #     device = bev_observations.device
    #     batch_size = bev_observations.shape[0]
    #     bev_observations_height = bev_observations.shape[2]
    #     bev_observations_width = bev_observations.shape[3]

    #     # Multiply BEV by the confidence
    #     bev_observations = bev_observations * bev_confidence.unsqueeze(1)

    #     # Mask out the invalid pixels in the BEV 
    #     bev_observations = bev_observations.masked_fill(~bev_valid.unsqueeze(1), 0.0)

    #     # Normalize the vectors so that they are unit vectors and hopefully dont blow up during training
    #     # bev_observations = torch.nn.functional.normalize(bev_observations, dim=1)
    #     # local_maps = torch.nn.functional.normalize(local_maps, dim=2)

    #     # Compute the matching score by looking at the alignment (aka dot product)
    #     bev_observations_flattened = bev_observations.view(batch_size, bev_observations.shape[1], -1)
    #     local_maps_flattened = local_maps.view(batch_size, local_maps.shape[1], local_maps.shape[2], -1)    


    #     print(bev_observations[0, :])
    #     exit()



    #     matching_score = bev_observations_flattened.unsqueeze(1) * local_maps_flattened
    #     matching_score_shape = matching_score.shape
    #     matching_score = torch.sum(matching_score, dim=2)

    #     # Sum all the different dot products
    #     matching_score = torch.sum(matching_score, dim=-1)

    #     # Normalize using the number of pixels being compared
    #     # matching_score = matching_score / float(matching_score_shape[-1])        
        
    #     # Normalize the matching score using the number of valid
    #     bev_valid_flattened = bev_valid.view(batch_size, -1)
    #     matching_score = matching_score / torch.sum(bev_valid_flattened, dim=-1).unsqueeze(-1)

    #     return matching_score










    def _compute_local_global_map_matching(self, local_maps, encoded_observations, ts):

        # Unpack
        bev_observations, bev_valid, bev_confidence = encoded_observations

        # Get some info
        device = bev_observations.device
        batch_size = bev_observations.shape[0]
        bev_observations_height = bev_observations.shape[2]
        bev_observations_width = bev_observations.shape[3]

        # Multiply BEV by the confidence
        bev_observations = bev_observations * bev_confidence.unsqueeze(1)

        # Mask out the invalid pixels in the BEV 
        bev_observations = bev_observations.masked_fill(~bev_valid.unsqueeze(1), 0.0)

        bev_valid_flattened = bev_valid.view(batch_size, -1)

        bev_observations =  ts(bev_observations).squeeze(1)


        # Compute the matching score by looking at the alignment (aka dot product)
        bev_observations_flattened = bev_observations.view(batch_size, bev_observations.shape[1], -1)
        local_maps_flattened = local_maps.view(batch_size, local_maps.shape[1], local_maps.shape[2], -1)    
        matching_score = bev_observations_flattened.unsqueeze(1) * local_maps_flattened
        # matching_score = matching_score.masked_fill(~bev_valid_flattened.unsqueeze(1).unsqueeze(1), 0.0)



        matching_score_shape = matching_score.shape
        matching_score = torch.sum(matching_score, dim=2)

        # Sum all the different dot products
        matching_score = torch.sum(matching_score, dim=-1)

        # Normalize using the number of pixels being compared
        # matching_score = matching_score / float(matching_score_shape[-1])        
        
        # Normalize the matching score using the number of valid
        bev_valid_flattened = bev_valid.view(batch_size, -1)
        matching_score = matching_score / torch.sum(bev_valid_flattened, dim=-1).unsqueeze(-1)

        return matching_score





