import sys
sys.path.append('../../../src')

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Python Imports
import zipfile
import shutil
from multiprocessing import Pool

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

def _remove_batch_dim( data):

    # If its a string then we just do nothing
    if(isinstance(data, str)):
        return data

    # If its a tensor then remove the batch dim but make sure 
    # that the batch dim does indeed only have a size of 1
    elif(torch.is_tensor(data)):
        assert(data.shape[0] == 1)
        return data.squeeze(0)
    
    # If it is a dict then remove the batch dim for each entry in the dict
    elif(isinstance(data, dict)):
        new_data = dict()
        for k in data.keys():
            new_data[k] = _remove_batch_dim(data[k])
        return new_data

    # If it is a list then remove the batch dim for each value in the list
    elif(isinstance(data, list)):            
        return [_remove_batch_dim(v) for v in data]

    # If it is an int then there is nothing to do
    elif(isinstance(data, int)):            
        return data

    # If it is an long then there is nothing to do
    elif(isinstance(data, long)):            
        return data

    # If it is an float then there is nothing to do
    elif(isinstance(data, float)):            
        return data

    # This usually means its numpy
    elif(data is not None):
        assert(data.shape[0] == 1)
        return data.squeeze(0)

    # If the data is anything else then just return it
    # It isnt something that we can remove the batch dim from
    else:
        return data


def _get_data_for_seq_index(x, seq_idx):

    if(x is None):
        return None

    return x[seq_idx]


def _move_to_numpy(x):
    if(torch.is_tensor(x)):
        return x.detach().cpu().numpy()

    return x


def _render_state_arrow(ax, x, y, yaw, color, yaw_offset=0,label=None, big_or_small="small"):

    # Move things to numpy if they are not
    x = _move_to_numpy(x)
    y = _move_to_numpy(y)
    yaw = _move_to_numpy(yaw)

    # Apply a yaw offset sometimes
    yaw = yaw + yaw_offset

    assert(big_or_small in ["big", "small"])
    if(big_or_small == "big"):
        # width = 0.1
        # scale = 1.0
        width = 0.015
        scale = 10.0
    else:
        width = None
        scale = None

    # Render the arrow
    # ax.quiver(x, y, np.cos(yaw), np.sin(yaw), color=color, scale=scale, label=label)
    ax.quiver(x, y, np.cos(yaw), np.sin(yaw), color=color, label=label, units="width", width=width, scale=scale)

    # Add in the circle
    # ax.add_patch(plt.Circle((x, y), 0.1, color=color))




def _render_discrete_log_probs(ax, global_map_img, discrete_map_log_probs, extracted_local_maps, local_map_center_global_frame, x_lim, y_lim):

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

    rect = matplotlib.patches.Rectangle(xy=(s_x, s_y),width=(e_x-s_x), height=(e_y-s_y), edgecolor="red", fill=False, linewidth=2)
    ax.add_patch(rect)






def _generate_kde_xy_posterior_info(density, particles, particle_weights, bandwidths, x_lim, y_lim, rendering_configs, device="cpu"):

    density = 2500

    # Get the configs we need for this
    xy_kde_posterior_particle_dims_to_use = rendering_configs["xy_kde_posterior_particle_dims_to_use"] 
    xy_kde_posterior_distribution_types = rendering_configs["xy_kde_posterior_distribution_types"] 

    # Make sure we have all the data we need. If we dont then do nothing
    if((particles is None) or (particle_weights is None) or (bandwidths is None)):
        return None

    # Move to the correct device
    particles = particles.to(device)
    particle_weights = particle_weights.to(device)
    bandwidths = bandwidths.to(device)

    # Extract the particle and bandwidth dims we want to use since we 
    # may not want to use the whole particle state for the loss (aka unsupervised latent dims)
    extracted_particles = [particles[..., pdim].unsqueeze(-1) for pdim in xy_kde_posterior_particle_dims_to_use]
    extracted_bandwidths = [bandwidths[..., pdim].unsqueeze(-1) for pdim in xy_kde_posterior_particle_dims_to_use]
    extracted_particles = torch.cat(extracted_particles, dim=-1)
    extracted_bandwidths = torch.cat(extracted_bandwidths, dim=-1)


    # Construct the KDE
    kde = KDE(xy_kde_posterior_distribution_types, extracted_particles.unsqueeze(0), particle_weights.unsqueeze(0), extracted_bandwidths.unsqueeze(0), particle_resampling_method="multinomial")


    # Get the log probs
    if(device == "cpu"):

        # Create the sampling mesh grid
        x = torch.linspace(x_lim[0], x_lim[1], density)
        y = torch.linspace(y_lim[0], y_lim[1], density)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        probe_points = torch.cat([torch.reshape(grid_x, (-1,)).unsqueeze(-1), torch.reshape(grid_y, (-1,)).unsqueeze(-1)], dim=-1)

        # get the prob points
        probe_points = probe_points.unsqueeze(0).to(particles.device)


        log_probs = kde.log_prob(probe_points)
    else:

        # Create the sampling mesh grid
        x = torch.linspace(x_lim[0], x_lim[1], density, device=particles.device)
        y = torch.linspace(y_lim[0], y_lim[1], density, device=particles.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        probe_points = torch.cat([torch.reshape(grid_x, (-1,)).unsqueeze(-1), torch.reshape(grid_y, (-1,)).unsqueeze(-1)], dim=-1)

        probe_points = probe_points.unsqueeze(0)

        # max_number_to_probe_at_a_time = 1000000
        max_number_to_probe_at_a_time = 500000

        if(probe_points.shape[1] <= max_number_to_probe_at_a_time):
            log_probs = kde.log_prob(probe_points)
        else:

            # Number of cuts
            log_probs = []
            s = 0
            e = s + max_number_to_probe_at_a_time
            while (s < probe_points.shape[1]):
                e = min(e, probe_points.shape[1])
                cut = probe_points[:, s:e,:]
                lp = kde.log_prob(cut)
                log_probs.append(lp)
                s += max_number_to_probe_at_a_time
                e = s + max_number_to_probe_at_a_time

            log_probs = torch.cat(log_probs, dim=1)







    # Convert to probs
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


def _render_kde_xy_posterior(ax, kde_posterior_info):

    # If we have no info then render nothing
    if(kde_posterior_info is None):
        return

    # Unpack
    grid_x = kde_posterior_info[0]
    grid_y = kde_posterior_info[1]
    probs = kde_posterior_info[2]

    # render it!
    ax.contourf(grid_x, grid_y, probs, locator=ticker.MaxNLocator(prune='lower',nbins=50), cmap="Reds")
    # ax.contourf(grid_x, grid_y, probs, cmap="Reds")



def render_sequence(save_file):

    # Make a temporary directory to store things in
    TEMP_DIR = "./temp/"
    if(not os.path.exists(TEMP_DIR)):
        os.makedirs(TEMP_DIR)

    # Make an output directory
    OUTPUT_DIR = "./output/"
    if(not os.path.exists(OUTPUT_DIR)):
        os.makedirs(OUTPUT_DIR)


    # Unzip the file
    with zipfile.ZipFile(save_file, mode="r") as archive:
        for member in archive.namelist():
            
            # get the file
            filename = os.path.basename(member)

            # copy file (taken from zipfile's extract)
            source = archive.open(member)
            target = open(os.path.join(TEMP_DIR, filename), "wb")
            with source, target:
                shutil.copyfileobj(source, target)

    # Get the file to load
    file_to_load = save_file.split("/")[-1]
    file_to_load = file_to_load.replace(".zip", "")
    file_to_load = "{}/{}".format(TEMP_DIR, file_to_load)

    # Load the save file
    saved_dict = torch.load(file_to_load, map_location="cpu")

    # Delete the file we just loaded because it takes a lot of space
    os.remove(file_to_load)

    # Create the save directory
    save_dir = file_to_load.split("/")[-1]
    save_dir = save_dir.split(".")[0]
    save_dir = "{}/{}".format(OUTPUT_DIR, save_dir)
    if(not os.path.exists(save_dir)):
        os.makedirs(save_dir)


    # Unpack it into input data and model output
    data = saved_dict["data"]
    model_output = saved_dict["outputs"]

    # We dont want the batch dim for this since we only have 1 batch
    data = _remove_batch_dim(data)
    model_output = _remove_batch_dim(model_output)

    # Extract the data
    global_map = data["global_map"]
    all_observations = data["observations"]
    all_gt_xy = data["xy_position_global_frame"]
    all_gt_yaw = data["roll_pitch_yaw"][..., 2]
    all_local_map_center_global_frame = data["local_map_center_global_frame"]

    # Extract the model output
    all_discrete_map_log_probs = model_output.get("discrete_map_log_probs", None)
    all_extracted_local_maps = model_output.get("extracted_local_maps", None)
    all_discrete_map_log_probs_center_global_frame = model_output.get("discrete_map_log_probs_center_global_frame", None)


    all_particles = model_output.get("particles", None)
    all_particle_weights = model_output.get("particle_weights", None)
    all_bandwidths = model_output.get("bandwidths", None)
    all_single_point_prediction = model_output.get("single_point_prediction", None)
    all_single_point_prediction_max_weight = model_output.get("single_point_prediction_max_weight", None)
    all_modes = model_output.get("modes", None)

    # all_particles = model_output.get("forward_particles", None)
    # all_particle_weights = model_output.get("forward_particle_weights", None)
    # all_bandwidths = model_output.get("forward_bandwidths", None)
    # all_single_point_prediction = model_output.get("forward_single_point_prediction", None)

    # all_particles = model_output.get("backward_particles", None)
    # all_particle_weights = model_output.get("backward_particle_weights", None)
    # all_bandwidths = model_output.get("backward_bandwidths", None)
    # all_single_point_prediction = model_output.get("backward_single_point_prediction", None)

    # Construct the map x and y limits
    global_map_x_lim = [0, global_map.shape[2]]
    global_map_y_lim = [0, global_map.shape[1]]

    # Get some information
    sequence_length = all_observations.shape[0]

    # Convert the global map into something render-able
    global_map_img = Colormap.apply(global_map)

    for seq_idx in tqdm(range(sequence_length), desc="Rendering Sequence", leave=False):

        # if((seq_idx > 2) and (seq_idx < (sequence_length - 2))):
        #     continue


        # Get the data for just this sequence index
        observation = _get_data_for_seq_index(all_observations, seq_idx)
        gt_xy = _get_data_for_seq_index(all_gt_xy, seq_idx)
        gt_yaw = _get_data_for_seq_index(all_gt_yaw, seq_idx)
        local_map_center_global_frame = _get_data_for_seq_index(all_local_map_center_global_frame, seq_idx)
        particles = _get_data_for_seq_index(all_particles, seq_idx)
        particle_weights = _get_data_for_seq_index(all_particle_weights, seq_idx)
        bandwidths = _get_data_for_seq_index(all_bandwidths, seq_idx)
        discrete_map_log_probs = _get_data_for_seq_index(all_discrete_map_log_probs, seq_idx)
        extracted_local_maps = _get_data_for_seq_index(all_extracted_local_maps, seq_idx)
        discrete_map_log_probs_center_global_frame = _get_data_for_seq_index(all_discrete_map_log_probs_center_global_frame, seq_idx)

        single_point_prediction = _get_data_for_seq_index(all_single_point_prediction, seq_idx)
        single_point_prediction_max_weight = _get_data_for_seq_index(all_single_point_prediction_max_weight, seq_idx)
        modes = _get_data_for_seq_index(all_modes, seq_idx)


        # Create the plotter
        rows = 2
        cols = 2
        fig, axes = plt.subplots(nrows=rows, ncols=cols, squeeze=False, figsize=(8*cols, 8*rows))

        # Render the global map
        axes[0,0].imshow(global_map_img)
        axes[1,0].imshow(global_map_img)

        # Render the observation
        axes[0,1].imshow(torch.permute(observation, [1, 2, 0]).numpy())

        # # Get the KDE XY posterior info
        # We do this once so we can reuse computation when plotting multiple times
        rendering_configs = {"xy_kde_posterior_particle_dims_to_use": [0, 1], "xy_kde_posterior_distribution_types": ["Normal", "Normal"]}
        kde_xy_posterior_info = _generate_kde_xy_posterior_info(2500, particles, particle_weights, bandwidths, global_map_x_lim, global_map_y_lim, rendering_configs=rendering_configs, device="cuda")

        # # # Render the posteriors
        _render_kde_xy_posterior(axes[0,0], kde_xy_posterior_info)
        # _render_kde_xy_posterior(axes[1,0], kde_xy_posterior_info)

        # # Render the true state arrows
        _render_state_arrow(axes[0, 0], gt_xy[0], gt_xy[1], -gt_yaw, color="red", yaw_offset=np.pi/2)
        _render_state_arrow(axes[1, 0], gt_xy[0], gt_xy[1], -gt_yaw, color="red", yaw_offset=np.pi/2)

        # # Render the estimated state arrow
        _render_state_arrow(axes[0, 0], single_point_prediction[0], single_point_prediction[1], -single_point_prediction[2], yaw_offset=np.pi/2, color="green")
        _render_state_arrow(axes[1, 0], single_point_prediction[0], single_point_prediction[1], -single_point_prediction[2], yaw_offset=np.pi/2, color="green")


        # Plot the modes
        if(modes is not None):
            _render_state_arrow(axes[0, 0], modes[0, 0], modes[0, 1], -modes[0, 2], yaw_offset=np.pi/2, color="blue")
            _render_state_arrow(axes[1, 0], modes[0, 0], modes[0, 1], -modes[0, 2], yaw_offset=np.pi/2, color="blue")
            _render_state_arrow(axes[0, 0], modes[1, 0], modes[1, 1], -modes[1, 2], yaw_offset=np.pi/2, color="lime")
            _render_state_arrow(axes[1, 0], modes[1, 0], modes[1, 1], -modes[1, 2], yaw_offset=np.pi/2, color="lime")



        # if(single_point_prediction_max_weight is not None):
        #     _render_state_arrow(axes[0, 0], single_point_prediction_max_weight[0], single_point_prediction_max_weight[1], -single_point_prediction_max_weight[2], yaw_offset=np.pi/2, color="blue")
        #     _render_state_arrow(axes[1, 0], single_point_prediction_max_weight[0], single_point_prediction_max_weight[1], -single_point_prediction_max_weight[2], yaw_offset=np.pi/2, color="blue")

        # Render the discrete posterior
        # _render_discrete_log_probs(axes[row_number,i], global_map_img, discrete_map_log_probs, extracted_local_maps, discrete_map_log_probs_center_global_frame, global_map_x_lim, global_map_y_lim)

        # Zoom in so we dont have the whole map to render
        zoom_size = 256 
        zoom_size_half = zoom_size // 2
        axes[1, 0].set_xlim([gt_xy[0]-zoom_size_half, gt_xy[0]+zoom_size_half])
        axes[1, 0].set_ylim([gt_xy[1]+zoom_size_half, gt_xy[1]-zoom_size_half])

        # Plot the particles
        axes[0, 0].scatter(particles[..., 0].numpy(), particles[..., 1].numpy(), color="black", s=1)
        axes[1, 0].scatter(particles[..., 0].numpy(), particles[..., 1].numpy(), color="black", s=1)


        # Make layout pretty!
        fig.tight_layout()

        # Save into the save dir
        plt.savefig("{}/{:03d}.png".format(save_dir, seq_idx))

        # Close the figure when we are done to stop matplotlub from complaining
        plt.close('all')






def main():

    # All the prefixes that face forward
    # [1, 5, 6, 7, 11, 12, 21, 26, 27, 28, 46, 47, 49, 50, 52, 53, 54, 59, 60, 62, 63, 64, 69, 74, 79, 85, 88, 90, 91, 96, 97, 106, 107, 114, 120, 121, 124, 125, 126, 127, 130, 131, 132, 133, 136, 137, 140, 149, 150, 158, 159, 163, 164, 165, 166, 167, 172, 173, 176, 178, 179, 180, 181, 182, 189, 191, 193, 194, 195, 196, 198, 200, 201, 202, 203, 204, 205, 206, 207, 208, 212, 220, 222, 225, 232, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 260, 263, 265, 266, 267, 268, 272, 273, 274, 275, 276, 278, 279, 280, 281, 283, 284, 287, 288, 289, 290, 291, 292, 293, 294, 296, 304, 311, 312, 314, 315, 320, 322, 323, 324, 325, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 343, 348, 349, 350, 351, 353, 354, 357, 363, 364, 365, 370, 373, 377, 378, 379, 380, 382, 383, 384, 385, 386, 388, 389, 401, 409, 410, 416, 417, 419, 420, 421, 427, 428, 430, 444, 445, 456, 463, 464, 469, 470, 471, 472, 475, 479, 483, 484, 488, 492, 493, 498, 499, 500, 501, 502, 507, 508, 509, 510, 512, 518, 522, 525, 526, 528, 534, 538, 540, 541, 542, 547, 550, 553, 554, 558, 564, 565, 572, 575, 580, 587, 594, 595, 596, 597, 599, 600, 601, 602, 609, 613, 614, 618, 619, 625, 626, 634, 635, 640, 641, 646, 647, 648, 649, 651, 655, 656, 657, 661, 662, 663, 664, 665, 666, 667, 668, 669, 674, 680, 681, 682, 685, 686, 691, 692, 695, 696, 697, 698, 699, 700, 701, 705, 706, 711, 712, 713, 714, 727, 729, 730, 734, 735, 736, 739, 740, 742, 743, 744, 745, 746, 750, 751, 758, 759, 760, 761, 764, 765, 766, 767, 768, 769, 771, 772, 773, 780, 781, 782, 783, 784, 785, 786, 787, 788, 792, 793, 801, 802, 805, 811, 815, 816, 817, 822, 823, 832, 836, 837, 838, 842, 843, 844, 845, 849, 850, 857, 858, 859, 860, 861, 862, 864, 865, 866, 868, 873, 874, 875, 882, 886, 887, 888, 889, 890, 907, 908, 911, 912, 913, 914, 916, 922, 923, 924, 925, 929, 930]
    
    # The save files
    save_files_idxes = []
    save_files_idxes.append(0)
    # save_files_idxes.append(5)
    # save_files_idxes.append(6)
    # save_files_idxes.append(7)

    

    # Add the prefix
    save_files = ["../../../experiments/mapillary/gaussian_dynamics_pf/saves/001_evaluation_filters/run_0000/qualitative/model_input_output/seq_{:06d}.ptp.zip".format(sv) for sv in save_files_idxes]


    # with Pool(4) as p:
    #     p.map(render_sequence, save_files)


    for save_file in save_files:
        render_sequence(save_file)






if __name__ == '__main__':
    main()