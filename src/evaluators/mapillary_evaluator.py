

# Python Imports

# Module Imports
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import seaborn as sns


# General ML Framework Imports
from general_ml_framework.utils.config import *

# Project Imports
from utils.memory import move_data_to_device
from utils.visualization import Colormap
from kde.kde import KDE
from evaluators.common_evaluator import CommonEvaluator


class MapillaryEvaluator(CommonEvaluator):
    def __init__(self, experiment_name, experiment_configs, save_dir, logger, device, model, evaluation_dataset, metric_classes):
        super(MapillaryEvaluator, self).__init__(experiment_name, experiment_configs, save_dir, logger, device, model, evaluation_dataset, metric_classes)



    def render_model_output_data(self, save_dir, render_number, data, model_output, rendering_configs):

        if(render_number < 1):
            return

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
        all_local_map_center_global_frame = data["local_map_center_global_frame"]



        # Extract the model output
        # all_particles = model_output.get("particles", None)
        # all_particle_weights = model_output.get("particle_weights", None)
        # all_bandwidths = model_output.get("bandwidths", None)

        all_particles = model_output.get("forward_particles", None)
        all_particle_weights = model_output.get("forward_particle_weights", None)
        all_bandwidths = model_output.get("forward_bandwidths", None)




        all_discrete_map_log_probs = model_output.get("discrete_map_log_probs", None)
        all_single_point_prediction = model_output.get("single_point_prediction", None)
        all_extracted_local_maps = model_output.get("extracted_local_maps", None)
        all_discrete_map_log_probs_center_global_frame = model_output.get("discrete_map_log_probs_center_global_frame", None)

        # Construct the map x and y limits
        global_map_x_lim = [0, global_map.shape[2]]
        global_map_y_lim = [0, global_map.shape[1]]

        # Get some information
        sequence_length = all_observations.shape[0]

        # Convert the global map into something render-able
        global_map_img = Colormap.apply(global_map)

        for seq_idx in tqdm(range(sequence_length), desc="Rendering Sequence", leave=False):

            # Get the data for just this sequence index
            observation = self._get_data_for_seq_index(all_observations, seq_idx)
            gt_xy = self._get_data_for_seq_index(all_gt_xy, seq_idx)
            gt_yaw = self._get_data_for_seq_index(all_gt_yaw, seq_idx)
            local_map_center_global_frame = self._get_data_for_seq_index(all_local_map_center_global_frame, seq_idx)
            particles = self._get_data_for_seq_index(all_particles, seq_idx)
            particle_weights = self._get_data_for_seq_index(all_particle_weights, seq_idx)
            bandwidths = self._get_data_for_seq_index(all_bandwidths, seq_idx)
            discrete_map_log_probs = self._get_data_for_seq_index(all_discrete_map_log_probs, seq_idx)
            single_point_prediction = self._get_data_for_seq_index(all_single_point_prediction, seq_idx)
            extracted_local_maps = self._get_data_for_seq_index(all_extracted_local_maps, seq_idx)
            discrete_map_log_probs_center_global_frame = self._get_data_for_seq_index(all_discrete_map_log_probs_center_global_frame, seq_idx)

            # Create the plotter
            rows = 2
            cols = 3
            fig, axes = plt.subplots(nrows=rows, ncols=cols, squeeze=False, figsize=(8*cols, 8*rows))

            # Render the global map
            axes[0,0].imshow(global_map_img)
            axes[1,1].imshow(global_map_img)


            # Render the observation
            axes[0,1].imshow(torch.permute(observation, [1, 2, 0]).numpy())

            # Get the KDE XY posterior info
            # We do this once so we can reuse computation when plotting multiple times
            kde_xy_posterior_info = self._generate_kde_xy_posterior_info(2500, particles, particle_weights, bandwidths, global_map_x_lim, global_map_y_lim, cmap=sns.cubehelix_palette(start=-0.2, rot=0.0, dark=0.05, light=1.0, reverse=False, as_cmap=True), rendering_configs=rendering_configs)

            # Render the posteriors
            self._render_kde_xy_posterior(axes[0,0], kde_xy_posterior_info)
            self._render_kde_xy_posterior(axes[1,1], kde_xy_posterior_info)


            # Render the true state      
            self._render_state_arrow(axes[0,0], gt_xy[0], gt_xy[1], -gt_yaw, color="red", yaw_offset=np.pi/2)
            self._render_state_arrow(axes[1,1], gt_xy[0], gt_xy[1], -gt_yaw, color="red", yaw_offset=np.pi/2)

            # Render the MAP posterior estimate
            self._render_state_arrow(axes[0,0], single_point_prediction[0], single_point_prediction[1], -single_point_prediction[2], yaw_offset=np.pi/2, color="green")
            self._render_state_arrow(axes[1,1], single_point_prediction[0], single_point_prediction[1], -single_point_prediction[2], yaw_offset=np.pi/2, color="green")


            axes[1,1].set_xlim([gt_xy[0]-128, gt_xy[0]+128])
            axes[1,1].set_ylim([gt_xy[1]+128, gt_xy[1]-128])

            self._render_particles(axes[0,0], particles, particle_weights, color="black")
            self._render_particles(axes[1,1], particles, particle_weights, color="black")

            # # Render the KDE posterior if we can.  This will do nothing if we cant
            # self._render_kde_posterior(axes[0,0], 2500, particles, particle_weights, bandwidths, global_map_x_lim, global_map_y_lim, cmap=sns.cubehelix_palette(start=-0.2, rot=0.0, dark=0.05, light=1.0, reverse=False, as_cmap=True), rendering_configs=rendering_configs)
            
            # self._render_kde_posterior(axes[1,1], 512, particles, particle_weights, bandwidths, [gt_xy[0]-128, gt_xy[0]+128], [gt_xy[1]-128, gt_xy[1]+128], cmap=sns.cubehelix_palette(start=-0.2, rot=0.0, dark=0.05, light=1.0, reverse=False, as_cmap=True), rendering_configs=rendering_configs)
         
            # self._render_state_arrow(axes[1,1], gt_xy[0], gt_xy[1], -gt_yaw, color="red", yaw_offset=np.pi/2)
            # self._render_state_arrow(axes[1,1], single_point_prediction[0], single_point_prediction[1], -single_point_prediction[2], color="green", yaw_offset=np.pi/2)

            # axes[1,1].set_xlim([gt_xy[0]-128, gt_xy[0]+128])
            # axes[1,1].set_ylim([gt_xy[1]+128, gt_xy[1]-128])








            # # Render the discrete prob map if we can, does nothing if we cant
            # self._render_discrete_log_probs(axes[0,0], global_map_img, discrete_map_log_probs, extracted_local_maps, discrete_map_log_probs_center_global_frame, global_map_x_lim, global_map_y_lim, data, seq_idx)



            # all_single_point_prediction_post = model_output.get("single_point_prediction_post", None)
            # single_point_prediction_post = self._get_data_for_seq_index(all_single_point_prediction_post, seq_idx)
            
            # axes[1,0].imshow(global_map_img)
            # self._render_discrete_log_probs(axes[1,0], global_map_img, discrete_map_log_probs, extracted_local_maps, discrete_map_log_probs_center_global_frame, global_map_x_lim, global_map_y_lim, data, seq_idx)
            # axes[1,0].set_xlim([gt_xy[0]-128, gt_xy[0]+128])
            # axes[1,0].set_ylim([gt_xy[1]+128, gt_xy[1]-128])

            # self._render_state_arrow(axes[1,0], gt_xy[0], gt_xy[1], -gt_yaw, color="red", yaw_offset=np.pi/2)
            # self._render_state_arrow(axes[1,0], single_point_prediction_post[0], single_point_prediction_post[1], -single_point_prediction_post[2], color="green", yaw_offset=np.pi/2)



            # self._render_state_arrow(axes[1,1], gt_xy[0], gt_xy[1], -gt_yaw, color="red", yaw_offset=np.pi/2)
            # self._render_state_arrow(axes[1,1], single_point_prediction[0], single_point_prediction[1], -single_point_prediction[2], color="green")
            # all_log_probs_pre = model_output.get("log_probs_pre", None)
            # log_probs_pre = self._get_data_for_seq_index(all_log_probs_pre, seq_idx)
            # axes[1,1].imshow(global_map_img)
            # self._render_discrete_log_probs(axes[1,1], global_map_img, log_probs_pre, extracted_local_maps, discrete_map_log_probs_center_global_frame, global_map_x_lim, global_map_y_lim, data, seq_idx)
            # axes[1,1].set_xlim([gt_xy[0]-128, gt_xy[0]+128])
            # axes[1,1].set_ylim([gt_xy[1]+128, gt_xy[1]-128])




            # if(seq_idx > 0):

            #     # Get the data for just this sequence index
            #     observation = self._get_data_for_seq_index(all_observations, seq_idx-1)
            #     gt_xy = self._get_data_for_seq_index(all_gt_xy, seq_idx-1)
            #     gt_yaw = self._get_data_for_seq_index(all_gt_yaw, seq_idx-1)
            #     local_map_center_global_frame = self._get_data_for_seq_index(all_local_map_center_global_frame, seq_idx-1)
            #     particles = self._get_data_for_seq_index(all_particles, seq_idx-1)
            #     particle_weights = self._get_data_for_seq_index(all_particle_weights, seq_idx-1)
            #     bandwidths = self._get_data_for_seq_index(all_bandwidths, seq_idx-1)
            #     discrete_map_log_probs = self._get_data_for_seq_index(all_discrete_map_log_probs, seq_idx-1)
            #     single_point_prediction = self._get_data_for_seq_index(all_single_point_prediction, seq_idx-1)
            #     extracted_local_maps = self._get_data_for_seq_index(all_extracted_local_maps, seq_idx-1)
            #     discrete_map_log_probs_center_global_frame = self._get_data_for_seq_index(all_discrete_map_log_probs_center_global_frame, seq_idx-1)

            #     all_log_probs_pre = model_output.get("log_probs_pre", None)
            #     log_probs_pre = self._get_data_for_seq_index(all_log_probs_pre, seq_idx-1)


            #     self._render_state_arrow(axes[1,2], gt_xy[0], gt_xy[1], -gt_yaw, color="red", yaw_offset=np.pi/2)
            #     self._render_state_arrow(axes[1,2], single_point_prediction[0], single_point_prediction[1], -single_point_prediction[2], color="green")
            #     axes[1,2].imshow(global_map_img)
            #     self._render_discrete_log_probs(axes[1,2], global_map_img, log_probs_pre, extracted_local_maps, discrete_map_log_probs_center_global_frame, global_map_x_lim, global_map_y_lim, data, seq_idx)
            #     axes[1,2].set_xlim([gt_xy[0]-128, gt_xy[0]+128])
            #     axes[1,2].set_ylim([gt_xy[1]+128, gt_xy[1]-128])
























            # Make layout pretty!
            fig.tight_layout()

            # Save into the save dir
            plt.savefig("{}/{:03d}.png".format(sequence_save_dir, seq_idx))

            # Close the figure when we are done to stop matplotlub from complaining
            plt.close('all')



    # def _render_kde_posterior(self, ax, particles, particle_weights, bandwidths, x_lim, y_lim, rendering_configs, number_of_samples=100000):

    #     # Get the configs we need for this
    #     local_rendering_configs = rendering_configs["xy_kde_posterior_rendering"]
    #     do_render = local_rendering_configs["do_render"]
    #     xy_kde_posterior_particle_dims_to_use = local_rendering_configs["xy_kde_posterior_particle_dims_to_use"] 
    #     xy_kde_posterior_distribution_types = local_rendering_configs["xy_kde_posterior_distribution_types"] 

    #     # Make sure we have all the data we need. If we dont then do nothing
    #     if((particles is None) or (particle_weights is None) or (bandwidths is None)):
    #         return 

    #     # Check if we should even render the KDE
    #     if(do_render == False):
    #         return 

    #     # Extract the particle and bandwidth dims we want to use since we 
    #     # may not want to use the whole particle state for the loss (aka unsupervised latent dims)
    #     extracted_particles = [particles[..., pdim].unsqueeze(-1) for pdim in xy_kde_posterior_particle_dims_to_use]
    #     extracted_bandwidths = [bandwidths[..., pdim].unsqueeze(-1) for pdim in xy_kde_posterior_particle_dims_to_use]
    #     extracted_particles = torch.cat(extracted_particles, dim=-1)
    #     extracted_bandwidths = torch.cat(extracted_bandwidths, dim=-1)

    #     # Construct the KDE
    #     kde = KDE(xy_kde_posterior_distribution_types, extracted_particles.unsqueeze(0), particle_weights.unsqueeze(0), extracted_bandwidths.unsqueeze(0), particle_resampling_method="multinomial")

    #     # Sample so we can create a histogram
    #     samples = kde.sample((number_of_samples, ))
    #     samples = samples.squeeze(0)
    #     samples = samples.numpy()

    #     # Create a 2d Histogram
    #     x_samples = samples[..., 0]
    #     y_samples = samples[..., 1]
    #     H, xedges, yedges = np.histogram2d(x_samples, y_samples, bins=1000, range=[x_lim, y_lim])
    #     # ax.imshow(H.T, cmap="Blues", interpolation='none', norm=matplotlib.colors.LogNorm(), extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], origin="lower")
    #     # ax.imshow(H.T, cmap="plasma", interpolation='none', norm=matplotlib.colors.LogNorm(), extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], origin="lower")

    #     # ax.imshow(H.T, cmap="plasma", interpolation='none', norm=matplotlib.colors.LogNorm(), extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]])
    #     ax.imshow(H, cmap="plasma", interpolation='none', norm=matplotlib.colors.LogNorm(), extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]])


    def _render_discrete_log_probs(self, ax, global_map_img, discrete_map_log_probs, extracted_local_maps, local_map_center_global_frame, x_lim, y_lim, data, seq_idx):

        # Make sure we have all the data we need. If we dont then do nothing
        if(discrete_map_log_probs is None):
            return 

        # Marginalize out the rotation dimension so we can render just the x y
        discrete_map_probs = torch.exp(discrete_map_log_probs)
        discrete_map_probs = torch.sum(discrete_map_probs, dim=-1)

        # Get the image size to render
        size_x = x_lim[1] - x_lim[0]
        size_y = y_lim[1] - y_lim[0]

        # Copy over into the image
        s_x = int(local_map_center_global_frame[0].item()  - (discrete_map_probs.shape[1]// 2))
        e_x = int(s_x + discrete_map_probs.shape[0])
        s_y = int(local_map_center_global_frame[1].item()  - (discrete_map_probs.shape[0]// 2))
        e_y = int(s_y + discrete_map_probs.shape[1])

        # Outside of bounds so dont render
        if((e_x < 0) or (e_y <= 0) or (s_x >= size_x) or (s_y >= size_y)):
            return 

        source_start_x = 0
        if(s_x < 0):
            source_start_x = -s_x
            s_x = 0

        source_start_y = 0
        if(s_y < 0):
            source_start_y = -s_y
            s_y = 0


        source_end_x = discrete_map_probs.shape[0]
        if(e_x > size_x):
            source_end_x = discrete_map_probs.shape[0] - e_x
            e_x = size_x

        source_end_y = discrete_map_probs.shape[1]
        if(e_y > size_y):
            source_end_y = discrete_map_probs.shape[1] - e_y
            e_y = size_y

        # Outside of bounds so dont render
        if((source_end_x < 0) or (source_end_y <= 0) or (source_start_x >= discrete_map_probs.shape[0]) or (source_start_y >= discrete_map_probs.shape[1])):
            return 

        # local_map_img = Colormap.apply(extracted_local_maps)
        # img = np.zeros((size_x, size_y, 4))
        # img[s_y:e_y, s_x:e_x, :3] = local_map_img[source_start_y:source_end_y, source_start_x:source_end_x]
        # img[s_y:e_y, s_x:e_x, 3] = 0.8

        # # ax.imshow(img, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], origin="lower")
        # ax.imshow(img, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]])


        def likelihood_overlay(prob, map_viz=None, p_rgb=0.2, p_alpha=1 / 15, thresh=None, cmap="jet"):
            prob = prob / prob.max()
            cmap = plt.get_cmap(cmap)
            rgb = cmap(prob**p_rgb)
            alpha = prob[..., None] ** p_alpha
            if thresh is not None:
                alpha[prob <= thresh] = 0
            if map_viz is not None:
                faded = map_viz + (1 - map_viz) * 0.5
                rgb = rgb[..., :3] * alpha + faded * (1 - alpha)
                rgb = np.clip(rgb, 0, 1)
            else:
                rgb[..., -1] = alpha.squeeze(-1)
            return rgb



        # img = np.zeros((size_x, size_y))
        # img[s_y:e_y, s_x:e_x] = discrete_map_probs.numpy()[source_start_y:source_end_y, source_start_x:source_end_x]
        # ax.imshow(img, cmap="plasma", interpolation='none', norm=matplotlib.colors.LogNorm())
        


        img = np.zeros((size_x, size_y, 3))
        img[...] = global_map_img[...]
        img[s_y:e_y, s_x:e_x] = likelihood_overlay(discrete_map_probs.numpy()[source_start_y:source_end_y, source_start_x:source_end_x], img[s_y:e_y, s_x:e_x])
        ax.imshow(img)
        


        rect = matplotlib.patches.Rectangle(xy=(s_x, s_y),width=(e_x-s_x), height=(e_y-s_y), edgecolor="red", fill=False, linewidth=5)
        ax.add_patch(rect)






    def _generate_kde_xy_posterior_info(self, density, particles, particle_weights, bandwidths, x_lim, y_lim, cmap, rendering_configs):

        density = 2500

        # Get the configs we need for this
        local_rendering_configs = rendering_configs["xy_kde_posterior_rendering"]
        do_render = local_rendering_configs["do_render"]
        xy_kde_posterior_particle_dims_to_use = local_rendering_configs["xy_kde_posterior_particle_dims_to_use"] 
        xy_kde_posterior_distribution_types = local_rendering_configs["xy_kde_posterior_distribution_types"] 

        # Check if we should even render the KDE
        if(do_render == False):
            return None

        # Make sure we have all the data we need. If we dont then do nothing
        if((particles is None) or (particle_weights is None) or (bandwidths is None)):
            return None


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

        # Move to the CPU
        grid_x = grid_x.cpu()
        grid_y = grid_y.cpu()
        probs = probs.cpu()

        # Convert to Numpy
        grid_x = grid_x.numpy()
        grid_y = grid_y.numpy()
        probs = probs.numpy()


        return (grid_x, grid_y, probs)

    def _render_kde_xy_posterior(self, ax, kde_posterior_info):

        # If we have no info then render nothing
        if(kde_posterior_info is None):
            return

        # Unpack
        grid_x = kde_posterior_info[0]
        grid_y = kde_posterior_info[1]
        probs = kde_posterior_info[2]

        # render it!
        ax.contourf(grid_x, grid_y, probs, locator=ticker.MaxNLocator(prune='lower',nbins=50), cmap="Reds")
        # ax.contourf(grid_x.cpu().numpy(), grid_y.cpu().numpy(), probs.cpu().numpy(), cmap="Reds")
