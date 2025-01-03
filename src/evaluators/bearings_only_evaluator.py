

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
from evaluators.common_evaluator import CommonEvaluator




class BearingsOnlyEvaluator(CommonEvaluator):
    def __init__(self, experiment_name, experiment_configs, save_dir, logger, device, model, evaluation_dataset, metric_classes):
        super(BearingsOnlyEvaluator, self).__init__(experiment_name, experiment_configs, save_dir, logger, device, model, evaluation_dataset, metric_classes)

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
        all_observations = data["observations"]
        all_gt_xy = data["xy_position_global_frame"]
        all_gt_yaw = data["roll_pitch_yaw"][..., 2]
        all_sensor_locations = data["all_sensor_locations"]
        global_x_limits = data["global_x_limits"]
        global_y_limits = data["global_y_limits"]

        # Construct the map x and y limits
        global_map_x_lim = [global_x_limits[0].item(), global_x_limits[1].item()]
        global_map_y_lim = [global_y_limits[0].item(), global_y_limits[1].item()]

        # Get some information
        sequence_length = all_observations.shape[0]

        # Render the Sequence
        for seq_idx in tqdm(range(sequence_length), leave=False, desc="Rendering Sequence Images"):

            # Get the data for just this sequence index
            observation = self._get_data_for_seq_index(all_observations, seq_idx)
            gt_xy = self._get_data_for_seq_index(all_gt_xy, seq_idx)
            gt_yaw = self._get_data_for_seq_index(all_gt_yaw, seq_idx)


            # Render based on the model
            if(isinstance(self.base_model, MDPF)):
                rows = 2
                cols = 1

            elif(isinstance(self.base_model, MDPS)):
                rows = 2
                cols = 3

            # Create the plotter
            fig, axes = plt.subplots(nrows=rows, ncols=cols, squeeze=False, figsize=(4.0*cols, 4.0*rows))

            axes[0,0].set_title("{:03d}".format(seq_idx))

            for c in range(cols):

                # Render the true state      
                self._render_state_arrow(axes[0,c], gt_xy[0], gt_xy[1], gt_yaw, color="red", yaw_offset=0)           

                # Render the sensors and the observations
                self._render_observations(axes[0,c], all_sensor_locations, observation, color="green")


                # Set the limits for the rendering
                axes[0, c].set_xlim(xmin=global_map_x_lim[0], xmax=global_map_x_lim[1])
                axes[0, c].set_ylim(ymin=global_map_y_lim[0], ymax=global_map_y_lim[1])

            # Render based on the model
            if(isinstance(self.base_model, MDPF)):

                # Extract the needed info
                particles = self._get_data_for_seq_index(model_output.get("particles", None), seq_idx)
                particle_weights = self._get_data_for_seq_index(model_output.get("particle_weights", None), seq_idx)
                bandwidths = self._get_data_for_seq_index(model_output.get("bandwidths", None), seq_idx)

                # Render the mean state
                self._render_mean_particle_arrow(axes[0,0], particles, particle_weights, color="yellow")

                # Render the particles
                self._render_particles(axes[0, 0], particles, particle_weights, color="black")

                # Render the KDE posterior if we can.  This will do nothing if we cant
                image = self._render_kde_posterior_xy(axes[0,0], self.KDE_RENDERING_DENSITY, particles, particle_weights, bandwidths, global_map_x_lim, global_map_y_lim, cmap=sns.cubehelix_palette(start=-0.2, rot=0.0, dark=0.05, light=1.0, reverse=False, as_cmap=True), rendering_configs=rendering_configs)

                # Render the KDE posterior for the angle if we can.  This will do nothing if we cant
                self._render_kde_posterior_angle(axes[1, 0], particles, particle_weights, bandwidths, gt_yaw, rendering_configs=rendering_configs)

            


            elif(isinstance(self.base_model, MDPS)):

                # Things to render
                rendering_configs = []
                rendering_configs.append(("mdps", "particles", "particle_weights", "bandwidths"))
                rendering_configs.append(("mdpf_forward", "forward_particles", "forward_particle_weights", "forward_bandwidths"))
                rendering_configs.append(("mdpf_backward", "backward_particles", "backward_particle_weights", "backward_bandwidths"))

                for c in range(cols):

                    # get the parameters for this rendering
                    model_name = rendering_configs[c][0]
                    particles_key = rendering_configs[c][1]
                    particle_weights_key = rendering_configs[c][2]
                    bandwidths_key = rendering_configs[c][3]

                    # Extract the needed info
                    particles = self._get_data_for_seq_index(model_output.get(particles_key, None), seq_idx)
                    particle_weights = self._get_data_for_seq_index(model_output.get(particle_weights_key, None), seq_idx)
                    bandwidths = self._get_data_for_seq_index(model_output.get(bandwidths_key, None), seq_idx)

                    # Render the mean state
                    self._render_mean_particle_arrow(axes[0,c], particles, particle_weights, color="yellow")

                    # Render the particles
                    self._render_particles(axes[0, c], particles, particle_weights, color="black")

                    # Render the KDE posterior if we can.  This will do nothing if we cant
                    image = self._render_kde_posterior_xy(axes[0, c], self.KDE_RENDERING_DENSITY, particles, particle_weights, bandwidths, global_map_x_lim, global_map_y_lim, cmap=sns.cubehelix_palette(start=-0.2, rot=0.0, dark=0.05, light=1.0, reverse=False, as_cmap=True), rendering_configs=rendering_configs)

                     # Render the KDE posterior for the angle if we can.  This will do nothing if we cant
                    self._render_kde_posterior_angle(axes[1, c], particles, particle_weights, bandwidths, gt_yaw, rendering_configs=rendering_configs)


                    # Set the title for the axis
                    axes[0, c].set_title(model_name)


            # Make layout pretty!
            fig.tight_layout()

            # Save into the save dir
            plt.savefig("{}/{:03d}.png".format(sequence_save_dir, seq_idx))

            # Close the figure when we are done to stop matplotlub from complaining
            plt.close('all')

        # Generate the GIFS
        generate_gif_from_image_directory(sequence_save_dir, "{}/{:03d}.gif".format(save_dir, render_number), img_file_type="png", logger=self.logger)

    def _render_observations(self, ax, all_sensor_locations, observation, color):

        # Can only handle 1 sensor right now
        assert(all_sensor_locations.shape[0] == 2)

        # Move things to numpy if they are not
        all_sensor_locations = self._move_to_numpy(all_sensor_locations)

        # Draw the sensor
        ax.add_patch(plt.Circle((all_sensor_locations[0], all_sensor_locations[1]), 0.5, color=color))


        # The sensor observation
        sensor_obs_y = observation[0].item() * 100.0
        sensor_obs_x = observation[1].item() * 100.0
        x_values = [all_sensor_locations[0], sensor_obs_x + all_sensor_locations[0]]
        y_values = [all_sensor_locations[1], sensor_obs_y + all_sensor_locations[1]]
        ax.plot(x_values, y_values, color=color)

    def _render_kde_posterior_xy(self, ax, density, particles, particle_weights, bandwidths, x_lim, y_lim, cmap, rendering_configs):

        # Get the configs we need for this
        local_rendering_configs = rendering_configs["xy_kde_posterior_rendering"]
        do_render = local_rendering_configs["do_render"]
        xy_kde_posterior_particle_dims_to_use = local_rendering_configs["xy_kde_posterior_particle_dims_to_use"] 
        xy_kde_posterior_distribution_types = local_rendering_configs["xy_kde_posterior_distribution_types"] 

        # Check if we should even render the KDE
        if(do_render == False):
            return False

        # Make sure we have all the data we need. If we dont then do nothing
        if((particles is None) or (particle_weights is None) or (bandwidths is None)):
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
        x = torch.linspace(x_lim[0], x_lim[1], density)
        y = torch.linspace(y_lim[0], y_lim[1], density)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        probe_points = torch.cat([torch.reshape(grid_x, (-1,)).unsqueeze(-1), torch.reshape(grid_y, (-1,)).unsqueeze(-1)], dim=-1)

        # Compute the probs of each of the points
        probe_points = probe_points.unsqueeze(0).to(particles.device)
        log_probs = kde.log_prob(probe_points)
        log_probs = log_probs.squeeze(0)
        probs = torch.exp(log_probs)

        # Reshape it back into an image
        probs = probs.reshape(grid_x.shape)

        ax.contourf(grid_x.cpu().numpy(), grid_y.cpu().numpy(), probs.cpu().numpy(), locator=ticker.MaxNLocator(prune = 'lower', nbins=10), zorder=-10, cmap="Blues")

        # return x
        # print(grid_x.shape) 
        # print(grid_y.shape)
        # print(probs.shape)

        # exit()







        # # Scale the probabilties for nice rendering
        # probs_min = torch.min(probs)
        # probs_max = torch.max(probs)
        # probs -= probs_min
        # probs /= (probs_max - probs_min)


        # # Convert to numpy for rendering
        # probs = probs.cpu().numpy()

        # # Need to rotate it 
        # probs = np.transpose(probs, axes=[1,0])

        # # Make rgb
        # norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        # m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        # rgb = m.to_rgba(probs)

        # # Redner the image
        # ax.imshow(rgb, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], interpolation='bilinear', origin="lower")


        # return rgb


        # Render!
        # ax.imshow(probs, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], interpolation='bilinear', origin="lower", cmap=sns.cubehelix_palette(start=-0.2, rot=0.0, dark=0.05, light=1.0, reverse=False, as_cmap=True))



