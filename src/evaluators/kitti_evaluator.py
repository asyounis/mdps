

# Python Imports

# Module Imports
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw

# General ML Framework Imports
from general_ml_framework.utils.config import *
from general_ml_framework.utils.rendering import *


# Project Imports
from utils.memory import move_data_to_device
from metrics.create_metrics import create_metric
from kde.kde import KDE
from models.mdps import MDPS
from models.mdpf import MDPF
from utils.visualization import Colormap
from evaluators.common_evaluator import CommonEvaluator



class KittiEvaluator(CommonEvaluator):
    def __init__(self, experiment_name, experiment_configs, save_dir, logger, device, model, evaluation_dataset, metric_classes):
        super(KittiEvaluator, self).__init__(experiment_name, experiment_configs, save_dir, logger, device, model, evaluation_dataset, metric_classes)

        # How dense to sample the space to render an accurate KDE density
        self.KDE_RENDERING_DENSITY = 100


    def render_model_output_data(self, save_dir, render_number, data, model_output, rendering_configs):

        # Make a save directory for this sequence
        sequence_save_dir = "{}/{:03d}".format(save_dir, render_number)
        ensure_directory_exists(sequence_save_dir)

        # Move everything to the CPU
        data = move_data_to_device(data, "cpu")
        model_output = move_data_to_device(model_output, "cpu")

        # We dont want the batch dim for this since we only have 1 batch
        data = self._remove_batch_dim(data)
        model_output = self._remove_batch_dim(model_output)

        # Extract the data
        global_map = data["global_map"]
        all_observations = data["observations"]
        all_gt_xy = data["xy_position_global_frame"]
        all_gt_yaw = data["roll_pitch_yaw"][..., 2]

        # Extract the model output
        all_particles = model_output.get("particles", None)
        all_particle_weights = model_output.get("particle_weights", None)
        all_bandwidths = model_output.get("bandwidths", None)

        # Construct the map x and y limits
        global_map_x_lim = [0, global_map.shape[2]]
        global_map_y_lim = [0, global_map.shape[1]]

        # Get some information
        sequence_length = all_observations.shape[0]

        # Convert the global map into something renderable
        global_map_img = Colormap.apply(global_map)

        for seq_idx in tqdm(range(sequence_length), leave=False, desc="Rendering a Sequence"):

            # Get the data for just this sequence index
            observation = self._get_data_for_seq_index(all_observations, seq_idx)
            gt_xy = self._get_data_for_seq_index(all_gt_xy, seq_idx)
            gt_yaw = self._get_data_for_seq_index(all_gt_yaw, seq_idx)
            particles = self._get_data_for_seq_index(all_particles, seq_idx)
            particle_weights = self._get_data_for_seq_index(all_particle_weights, seq_idx)
            bandwidths = self._get_data_for_seq_index(all_bandwidths, seq_idx)

            # Create the plotter
            rows = 1
            cols = 2
            fig, axes = plt.subplots(nrows=rows, ncols=cols, squeeze=False, figsize=(8*cols, 8*rows))

            # Render the global map
            axes[0,0].imshow(global_map_img, origin="lower", zorder=-100)

            # Render the true state      
            self._render_state_arrow(axes[0,0], gt_xy[0], gt_xy[1], gt_yaw, color="red", yaw_offset=-np.pi/2)           
            # self._render_state_arrow(axes[0,0], particles[..., 0], particles[..., 1], particles[..., 2], color="black", scale=50, yaw_offset=-np.pi/2)           

            # Render the KDE posterior if we can.  This will do nothing if we cant
            self._render_kde_posterior(axes[0,0], particles, particle_weights, bandwidths, global_map_x_lim, global_map_y_lim, rendering_configs=rendering_configs)

            self._render_particles(axes[0,0], particles)

            # Render the observation
            axes[0,1].imshow(torch.permute(observation, [1, 2, 0]).numpy())

            # Make layout pretty!
            fig.tight_layout()

            # Save into the save dir
            plt.savefig("{}/{:03d}.png".format(sequence_save_dir, seq_idx))

            # Close the figure when we are done to stop matplotlub from complaining
            plt.close('all')


    def _render_kde_posterior(self, ax, particles, particle_weights, bandwidths, x_lim, y_lim, rendering_configs, number_of_samples=100000):

        # Get the configs we need for this
        local_rendering_configs = rendering_configs["xy_kde_posterior_rendering"]
        do_render = local_rendering_configs["do_render"]
        xy_kde_posterior_particle_dims_to_use = local_rendering_configs["xy_kde_posterior_particle_dims_to_use"] 
        xy_kde_posterior_distribution_types = local_rendering_configs["xy_kde_posterior_distribution_types"] 


        # Make sure we have all the data we need. If we dont then do nothing
        if((particles is None) or (particle_weights is None) or (bandwidths is None)):
            return 

        # Check if we should even render the KDE
        if(do_render == False):
            return

        # Extract the particle and bandwidth dims we want to use since we 
        # may not want to use the whole particle state for the loss (aka unsupervised latent dims)
        extracted_particles = [particles[..., pdim].unsqueeze(-1) for pdim in xy_kde_posterior_particle_dims_to_use]
        extracted_bandwidths = [bandwidths[..., pdim].unsqueeze(-1) for pdim in xy_kde_posterior_particle_dims_to_use]
        extracted_particles = torch.cat(extracted_particles, dim=-1)
        extracted_bandwidths = torch.cat(extracted_bandwidths, dim=-1)

        # Construct the KDE
        kde = KDE(xy_kde_posterior_distribution_types, extracted_particles.unsqueeze(0), particle_weights.unsqueeze(0), extracted_bandwidths.unsqueeze(0), particle_resampling_method="multinomial")

        # Create the sampling mesh grid
        x = torch.linspace(x_lim[0], x_lim[1], x_lim[1] - x_lim[0])
        y = torch.linspace(y_lim[0], y_lim[1], y_lim[1] - y_lim[0])
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        probe_points = torch.cat([torch.reshape(grid_x, (-1,)).unsqueeze(-1), torch.reshape(grid_y, (-1,)).unsqueeze(-1)], dim=-1)

        # Compute the probs of each of the points
        probe_points = probe_points.unsqueeze(0).to(particles.device)
        log_probs = kde.log_prob(probe_points)
        log_probs = log_probs.squeeze(0)
        probs = torch.exp(log_probs)

        # Reshape it back into an image
        probs = probs.reshape(grid_x.shape)

        # Make the grid!
        ax.contourf(grid_x.cpu().numpy(), grid_y.cpu().numpy(), probs.cpu().numpy(), locator=ticker.MaxNLocator(prune = 'lower', nbins=10), cmap="Reds")


        # # Sample so we can create a histogram
        # samples = kde.sample((number_of_samples, ))
        # samples = samples.squeeze(0)
        # samples = samples.numpy()

        # # Create a 2d Histogram
        # x_samples = samples[..., 0]
        # y_samples = samples[..., 1]
        # H, xedges, yedges = np.histogram2d(x_samples, y_samples, bins=1000, range=[x_lim, y_lim])
        # # ax.imshow(H.T, cmap="Blues", interpolation='none', norm=matplotlib.colors.LogNorm(), extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], origin="lower")
        # # ax.imshow(H.T, cmap="plasma", interpolation='none', norm=matplotlib.colors.LogNorm(), extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], origin="lower")


        # ax.imshow(H.T, cmap="plasma", interpolation='none', norm=matplotlib.colors.LogNorm(), extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], origin="lower")

