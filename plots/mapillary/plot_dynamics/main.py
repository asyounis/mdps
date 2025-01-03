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
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
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


# def _render_state_arrow(ax, x, y, yaw, color, yaw_offset=0,label=None, big_or_small="small"):

#     # Move things to numpy if they are not
#     x = _move_to_numpy(x)
#     y = _move_to_numpy(y)
#     yaw = _move_to_numpy(yaw)

#     # Apply a yaw offset sometimes
#     yaw = yaw + yaw_offset

#     assert(big_or_small in ["big", "small"])
#     if(big_or_small == "big"):
#         # width = 0.1
#         # scale = 1.0
#         width = 0.015
#         scale = 10.0
#     else:
#         width = None
#         scale = None

#     # Render the arrow
#     # ax.quiver(x, y, np.cos(yaw), np.sin(yaw), color=color, scale=scale, label=label)
#     ax.quiver(x, y, np.cos(yaw), np.sin(yaw), color=color, label=label, units="width", width=width, scale=scale)

#     # Add in the circle
#     # ax.add_patch(plt.Circle((x, y), 0.1, color=color))




def _render_state_arrow(ax, x, y, yaw, color, yaw_offset=0,label=None, width=0.015, scale=10.0):

    # Move things to numpy if they are not
    x = _move_to_numpy(x)
    y = _move_to_numpy(y)
    yaw = _move_to_numpy(yaw)

    # Apply a yaw offset sometimes
    yaw = yaw + yaw_offset

    # Render the arrow
    # ax.quiver(x, y, np.cos(yaw), np.sin(yaw), color=color, scale=scale, label=label)
    ax.quiver(x, y, np.cos(yaw), np.sin(yaw), color=color, label=label, units="width", width=width, scale=scale)


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
    dynamics_samples = saved_dict["samples"]

    # We dont want the batch dim for this since we only have 1 batch
    data = _remove_batch_dim(data)

    # Extract the data
    global_map = data["global_map"]
    all_gt_xy = data["xy_position_global_frame"]
    all_gt_yaw = data["roll_pitch_yaw"][..., 2]
    all_local_map_center_global_frame = data["local_map_center_global_frame"]
    all_actions = data["actions"]

    # Construct the map x and y limits
    global_map_x_lim = [0, global_map.shape[2]]
    global_map_y_lim = [0, global_map.shape[1]]

    # Get some information
    sequence_length = all_gt_xy.shape[0]

    # Convert the global map into something render-able
    global_map_img = Colormap.apply(global_map, return_grayscale=True)

    # Convert to numpy for rendering
    dynamics_samples = dynamics_samples.numpy()
    all_gt_xy = all_gt_xy.numpy()

    # Create the plotter
    rows = 1
    cols = 8
    fig, axes = plt.subplots(nrows=rows, ncols=cols, squeeze=False, figsize=(2.8*cols, 3*rows))


    for r in range(rows):
        for c in range(cols):

            # Get the index to use and make sure its in range
            index = r*cols + c
            if(index >= (dynamics_samples.shape[0] - 1)):
                continue

            # Render the global map
            axes[r,c].imshow(global_map_img)


            # Zoom in so we dont have the whole map to render
            # zoom_size = 64
            zoom_size = 80
            # zoom_size = 96
            # zoom_size = 128
            zoom_size_half = zoom_size // 2
            x_lim = [all_gt_xy[index,0]-zoom_size_half, all_gt_xy[index,0]+zoom_size_half]
            # y_lim = [all_gt_xy[index,1]+zoom_size_half, all_gt_xy[index,1]-zoom_size_half]
            y_lim = [all_gt_xy[index,1]-zoom_size_half, all_gt_xy[index,1]+zoom_size_half]
            axes[r, c].set_xlim(x_lim)
            axes[r, c].set_ylim(y_lim)

            # Draw the particle cloud
            # axes[r, c].hist2d(dynamics_samples[index, :, 0], dynamics_samples[index, :, 1], range=[x_lim, [all_gt_xy[index,1]-zoom_size_half, all_gt_xy[index,1]+zoom_size_half]], bins=100, cmin=1)


            # cmap = sns.dark_palette("#69d", reverse=True, as_cmap=True)
            # cmap = sns.color_palette("rocket_r", as_cmap=True)
            # cmap = sns.color_palette("Blues", as_cmap=True)
            # cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
            cmap = sns.color_palette("mako_r", as_cmap=True)
            # cmap = sns.color_palette("crest", as_cmap=True)

            cmap = sns.color_palette("YlOrBr", as_cmap=True)


            axes[r, c].hist2d(dynamics_samples[index, :, 0], dynamics_samples[index, :, 1], range=[x_lim, [all_gt_xy[index,1]-zoom_size_half, all_gt_xy[index,1]+zoom_size_half]], bins=100, cmin=1, cmap=cmap)

            # # Draw the action
            # x1 = all_gt_xy[index,0]
            # y1 = all_gt_xy[index,1]
            # x2 = all_gt_xy[index,0] + all_actions[index, 0]
            # y2 = all_gt_xy[index,1] + all_actions[index, 1]
            # axes[r,c].plot([x1.item(), x2.item()], [y1.item(), y2.item()])

            # Render the true state
            _render_state_arrow(axes[r, c], all_gt_xy[index,0], all_gt_xy[index,1], all_gt_yaw[index], yaw_offset=-np.pi/2, color="black", width=0.02, scale=8.0)
            _render_state_arrow(axes[r, c], all_gt_xy[index+1,0], all_gt_xy[index+1,1], all_gt_yaw[index+1], yaw_offset=-np.pi/2, color=sns.color_palette("bright")[0], width=0.02, scale=8.0)



    # Turn of the x and y axes labels
    for r in range(rows):
        for c in range(cols):
            axes[r,c].set_xticks([])
            axes[r,c].set_yticks([])


    def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
        p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.95*height, width=0.5*height )
        return p


    # Create the legend
    arrow_1 = plt.arrow(0, 0, 0.5, 0.6, color="black")
    arrow_2 = plt.arrow(0, 0, 0.5, 0.6, color=sns.color_palette("bright")[0])
    handles = [arrow_1, arrow_2]
    labels = ["True State at Time t", "True State at Time t+1"]
    legend_properties = {'weight':'bold', "size":20}
    lgnd = fig.legend(handles, labels, loc='upper left', ncol=8, handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow)}, prop=legend_properties, facecolor='white', framealpha=1, edgecolor="black")

    # Adjust whitespace
    # fig.tight_layout(rect=(0,0,1,0.93))

    # Make layout pretty!
    fig.tight_layout()

    # Save into the save dir
    plt.savefig("{}/mapillary_dynamics_plot.png".format(save_dir))
    plt.savefig("{}/mapillary_dynamics_plot.pdf".format(save_dir))

    plt.show()

    # Close the figure when we are done to stop matplotlub from complaining
    plt.close('all')



def main():

    # All the prefixes that face forward
    # [1, 5, 6, 7, 11, 12, 21, 26, 27, 28, 46, 47, 49, 50, 52, 53, 54, 59, 60, 62, 63, 64, 69, 74, 79, 85, 88, 90, 91, 96, 97, 106, 107, 114, 120, 121, 124, 125, 126, 127, 130, 131, 132, 133, 136, 137, 140, 149, 150, 158, 159, 163, 164, 165, 166, 167, 172, 173, 176, 178, 179, 180, 181, 182, 189, 191, 193, 194, 195, 196, 198, 200, 201, 202, 203, 204, 205, 206, 207, 208, 212, 220, 222, 225, 232, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 260, 263, 265, 266, 267, 268, 272, 273, 274, 275, 276, 278, 279, 280, 281, 283, 284, 287, 288, 289, 290, 291, 292, 293, 294, 296, 304, 311, 312, 314, 315, 320, 322, 323, 324, 325, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 343, 348, 349, 350, 351, 353, 354, 357, 363, 364, 365, 370, 373, 377, 378, 379, 380, 382, 383, 384, 385, 386, 388, 389, 401, 409, 410, 416, 417, 419, 420, 421, 427, 428, 430, 444, 445, 456, 463, 464, 469, 470, 471, 472, 475, 479, 483, 484, 488, 492, 493, 498, 499, 500, 501, 502, 507, 508, 509, 510, 512, 518, 522, 525, 526, 528, 534, 538, 540, 541, 542, 547, 550, 553, 554, 558, 564, 565, 572, 575, 580, 587, 594, 595, 596, 597, 599, 600, 601, 602, 609, 613, 614, 618, 619, 625, 626, 634, 635, 640, 641, 646, 647, 648, 649, 651, 655, 656, 657, 661, 662, 663, 664, 665, 666, 667, 668, 669, 674, 680, 681, 682, 685, 686, 691, 692, 695, 696, 697, 698, 699, 700, 701, 705, 706, 711, 712, 713, 714, 727, 729, 730, 734, 735, 736, 739, 740, 742, 743, 744, 745, 746, 750, 751, 758, 759, 760, 761, 764, 765, 766, 767, 768, 769, 771, 772, 773, 780, 781, 782, 783, 784, 785, 786, 787, 788, 792, 793, 801, 802, 805, 811, 815, 816, 817, 822, 823, 832, 836, 837, 838, 842, 843, 844, 845, 849, 850, 857, 858, 859, 860, 861, 862, 864, 865, 866, 868, 873, 874, 875, 882, 886, 887, 888, 889, 890, 907, 908, 911, 912, 913, 914, 916, 922, 923, 924, 925, 929, 930]
    
    # The save files
    save_files_idxes = []
    save_files_idxes.append(1)
    save_files_idxes.append(5)
    save_files_idxes.append(6)
    save_files_idxes.append(7)

    

    # Add the prefix
    # save_files = ["../../../experiments/mapillary/mdps_strat_random_init/saves/004_evaluation_all/run_0000/qualitative/dynamics_model_propagation_input_output/seq_{:06d}.ptp.zip".format(sv) for sv in save_files_idxes]
    save_files = ["../../../experiments/mapillary/mdps_strat_random_init_fixed_map_extraction/saves/004_evaluation_all/run_0000/qualitative/dynamics_model_propagation_input_output/seq_{:06d}.ptp.zip".format(sv) for sv in save_files_idxes]


    render_sequence(save_files[0])
    exit()

    with Pool(4) as p:
        p.map(render_sequence, save_files)








if __name__ == '__main__':
    main()