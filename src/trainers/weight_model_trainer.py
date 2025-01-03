

# Python Imports

# Module Imports
import torch

# General ML Framework Imports
from general_ml_framework.training.base_trainer import BaseTrainer
from general_ml_framework.utils.config import *
from general_ml_framework.training.data_plotter import DataPlotter


# Project Imports
from utils.memory import move_data_to_device, get_device_from_dict
from utils.general import nll_loss_xyr
from metrics.create_metrics import create_metric
from models.mdpf import MDPF
from models.mdps import MDPS
from kde.kde import KDE



import matplotlib.pyplot as plt



class WeightModelTrainer(BaseTrainer):
    def __init__(self, experiment_name, experiment_configs, save_dir, logger, device, model,training_dataset, validation_dataset):
        super(WeightModelTrainer, self).__init__(experiment_name, experiment_configs, save_dir, logger, device, model, training_dataset, validation_dataset)

        # Get the configs for the particle cloud generator
        particle_cloud_configs = get_mandatory_config("particle_cloud_configs", self.training_configs, "training_configs")
        self.number_of_particles_in_cloud = get_mandatory_config("number_of_particles_in_cloud", particle_cloud_configs, "particle_cloud_configs")
        self.cloud_center_distribution_configs = get_mandatory_config("cloud_center_distribution_configs", particle_cloud_configs, "particle_cloud_configs")
        self.cloud_generating_distribution_configs = get_mandatory_config("cloud_generating_distribution_configs", particle_cloud_configs, "particle_cloud_configs")

        # Get the loss function configs
        loss_configs = get_mandatory_config("loss_configs", self.training_configs, "training_configs")
        self.distribution_types = get_mandatory_config("distribution_types", loss_configs, "loss_configs")
        manual_bandwidth = get_mandatory_config("manual_bandwidth", loss_configs, "loss_configs")
        self.manual_bandwidth = torch.FloatTensor(manual_bandwidth)



    def do_forward_pass(self, data):

        # Mode the data to the correct device
        data = move_data_to_device(data, self.device)

        # Unpack the data 
        ground_truth_mask = data["ground_truth_mask"]
        observations = data["observations"]
        xy_gt = data["xy_position_global_frame"]
        yaw_gt = data["roll_pitch_yaw"][..., -1]

        # Mask out
        observations = self._get_maked_data(observations, ground_truth_mask)
        xy_gt = self._get_maked_data(xy_gt, ground_truth_mask)
        yaw_gt = self._get_maked_data(yaw_gt, ground_truth_mask)

        # Get some info
        batch_size = observations.shape[0]
        sequence_length = observations.shape[1]

        # Flatten the batch and sequence dimensions
        observations = torch.reshape(observations, (batch_size*sequence_length, -1))
        xy_gt = torch.reshape(xy_gt, (batch_size*sequence_length, -1))
        yaw_gt = torch.reshape(yaw_gt, (batch_size*sequence_length, -1))

        # Get some new info
        flattened_batch_size = observations.shape[0]

        # Create a unified gt
        unified_gt = torch.zeros((flattened_batch_size, 3), device=self.device)
        unified_gt[:, 0:2] = xy_gt
        unified_gt[:, 2] = yaw_gt.squeeze(-1)

        # Extract the models we need
        base_model = self._get_base_model(self.model)
        observation_encoder_model = base_model.observation_encoder_model
        weights_model = base_model.output_weights_model

        # Encode the observations and tile them so that the have a particles dim
        encoded_observations = observation_encoder_model(observations, None).get()

        # Create the particle clouds
        particles = self._create_particle_cloud(unified_gt)

        # Compute the weights        
        input_dict = dict()
        input_dict["particles"] = particles
        input_dict["encoded_global_map"] = None
        input_dict["encoded_observations"] = encoded_observations
        input_dict["unnormalized_resampled_particle_log_weights"] = None
        particle_weights = weights_model(input_dict)

        # Make the bandwidth
        manual_bandwidth_copy = self.manual_bandwidth.clone()
        manual_bandwidth_copy = manual_bandwidth_copy.to(self.device)
        manual_bandwidth_copy = torch.tile(manual_bandwidth_copy.unsqueeze(0), [flattened_batch_size, 1])

        # Create the loss KDE
        loss_kde = KDE(self.distribution_types, particles, particle_weights, manual_bandwidth_copy)

        # Compute the NLL loss
        nll = loss_kde.log_prob(unified_gt.unsqueeze(1), do_normalize_weights=False)
        nll = -nll
        loss = torch.mean(nll)

        return loss, flattened_batch_size


    def _get_base_model(self, model):

        # Get the base model
        if(isinstance(model, torch.nn.DataParallel)):
            base_model = model.module
        else:
            base_model = model

        return base_model


    def _get_maked_data(self, d, mask):

        # Make sure they have the same shape
        assert(d.shape[0] == mask.shape[0])
        assert(d.shape[1] == mask.shape[1])

        # Filter out the data that has a negative mask
        data = []
        for s in range(d.shape[1]):
            if(mask[0, s] == True):
                assert(torch.all(mask[:, s]))
                data.append(d[:, s])

        # Stack back into a tensor
        data = torch.stack(data, dim=1)

        return data





    def _create_kde_from_configs(self, configs, particles):


        # Information needed to construct the KDE
        kde_distribution_types = []
        bandwidths = []        

        # Get all the parameters for this KDE
        dims = configs["dims"]
        for d in dims:

            # Get the configs for this specific dim
            dim_configs = dims[d]
            distribution_type = get_mandatory_config("distribution_type", dim_configs, "dim_configs")
            bandwidth = get_mandatory_config("bandwidth", dim_configs, "dim_configs")

            # Save so we can mek the KDE
            kde_distribution_types.append(distribution_type)
            bandwidths.append(bandwidth)


        # Put into a tensor and add batch dim
        bandwidths = torch.FloatTensor(bandwidths).to(self.device)

        # Create the weights
        weights = torch.ones((particles.shape[0], particles.shape[1]), device=self.device)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)


        # Add batch dim
        bandwidths = torch.tile(bandwidths.unsqueeze(0), [particles.shape[0], 1])

        # Construct the KDE
        kde = KDE(kde_distribution_types, particles, weights, bandwidths)

        return kde


    def _create_particle_cloud(self, unified_gt):

        # Get some new info
        flattened_batch_size = unified_gt.shape[0]

        # Get the cloud centers
        cloud_center_kde = self._create_kde_from_configs(self.cloud_center_distribution_configs, unified_gt.unsqueeze(1))
        cloud_centers = cloud_center_kde.sample((1, ))

        # Generate the cloud
        cloud_generator_kde = self._create_kde_from_configs(self.cloud_generating_distribution_configs, cloud_centers)
        clouds = cloud_generator_kde.sample((self.number_of_particles_in_cloud, ))

        return clouds