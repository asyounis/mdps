import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.lines as mLines
import matplotlib.pyplot as plt
from matplotlib import rc,rcParams
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import os
import torch
import yaml
import sys




rc('font', weight='bold')

def load_data_for_experiment_save_file(base_dir, experiment, experiment_name, evaluation_directory, recall_metric_name_base, number_of_runs=5):

    # The data for this experiment
    data_for_each_run = dict()

    # If we have all the data then say we loaded it all
    has_all_data = True

    for i in range(number_of_runs):
        results_filepath = "{}/{}/saves/{}/run_{:04d}/quantitative/metrics.ptp".format(base_dir, experiment, evaluation_directory, i)

        # Check if the file exists, if it doesnt then skip this file
        if(os.path.isfile(results_filepath) == False):
            print("Could not find datafile: ")
            print("\t\t" + results_filepath)
            print("")

            # We could not load something
            has_all_data = False

            # return None
            continue

        # Load the metric data from the file
        metric_data = torch.load(results_filepath, map_location=torch.device('cpu'))


        # Get the position recall metric data
        recall_metric_data = []
        for md in metric_data:

            # recall_metric_name_base, md["name"][0:len(recall_metric_name_base)])

            if(recall_metric_name_base == md["name"][0:len(recall_metric_name_base)]):
                recall_metric_data.append(md)

        # sort based on the threshold values
        recall_metric_data.sort(key=lambda x: x["threshold"])

        # Pack the data into a dict
        recall_data_dict = dict()
        recall_data_dict["display_thresholds"] = [d["threshold_scaled"] for d in recall_metric_data]
        recall_data_dict["values"] = [d["value"] for d in recall_metric_data]

        # Pack it
        all_data_for_this_run = dict()
        all_data_for_this_run["recall"] = recall_data_dict

        # Save that dict in the run
        data_for_each_run[i] = all_data_for_this_run

    # Save the experiment name to the dict
    data_for_each_run["experiment_name"] = experiment_name 

    # Say if we loaded everything
    data_for_each_run["loaded_everything"] = has_all_data

    return data_for_each_run



def compute_stats_for_run_data(experiment_data, metric_label, number_of_runs=5):

    # Get the data for all the runs and put them into a np array
    all_run_data = [experiment_data[i][metric_label]["values"] for i in range(number_of_runs)]
    all_run_data = np.asarray(all_run_data)

    # Sort to help compute metrics later on
    all_run_data.sort(axis=0)

    # Calculate the median
    median = np.median(all_run_data, axis=0)

    # Calculate the IQR
    q75, q25 = np.percentile(all_run_data, [75 ,25], axis=0)
    iqr = q75 - q25

    # also extract the x values while we are here
    x_values = experiment_data[0][metric_label]["display_thresholds"]

    return median, x_values


def main():
	
    linestyle_tuple_list = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot'),  # Same as '-.'
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 2))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
    ]

    linestyle_tuple = dict()
    for i in range(len(linestyle_tuple_list)):
        linestyle_tuple[linestyle_tuple_list[i][0]] = linestyle_tuple_list[i][1]


    # get the config file and load it
    with open(sys.argv[1]) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)

    # Get the save dir
    save_dir = configs["save_dir"]

    # Get the base dir
    base_dir = configs["base_dir"]

    # Get the base name (no extensions) to save the plot as
    plot_save_basename = configs["plot_save_basename"]

    # Get the number of runs
    number_of_runs = configs["number_of_runs"]

    # Get the number of plots to generate
    rows = configs["number_of_rows"]
    cols = configs["number_of_cols"]

    # Get the x labels
    x_labels = configs["x_labels"]

    # Get the y labels
    x_ranges = configs["x_ranges"]

    # Get the color_mapping
    color_mapping = configs["color_mapping"]

    # Get the linestyle_mapping
    linestyle_mapping = configs["linestyle_mapping"]

    # Get the legend configs
    legend_configs = configs["legend"]


    # Get the experiment configs
    experiment_configs = []
    for experiment in configs["experiments"]:
        exp_cfg = experiment[list(experiment.keys())[0]]
        experiment_configs.append(exp_cfg)


    # Get all the experiment data
    all_experiment_data = []
    for exp_cfg in experiment_configs:

        data_source = exp_cfg["data_source"]
        if(data_source == "save_file"):

            # Load the experiment data
            experiment_data = load_data_for_experiment_save_file(base_dir, exp_cfg["dir_name"], exp_cfg["display_name"], exp_cfg["evaluation_dir_name"], exp_cfg["recall_metric_name_base"], number_of_runs=number_of_runs)

        elif(data_source == "provided_in_config"):

            experiment_data = dict()
            experiment_data["experiment_name"] = exp_cfg["display_name"] 
            experiment_data["loaded_everything"] = True

            experiment_data[0] = dict()
            experiment_data[0]["recall"] = dict()
            experiment_data[0]["recall"]["display_thresholds"] = exp_cfg["display_thresholds"]
            experiment_data[0]["recall"]["values"] = exp_cfg["values"]


        # Get the config information for the data
        experiment_data["row_pos"] = exp_cfg["row_pos"]
        experiment_data["col_pos"] = exp_cfg["col_pos"]
        experiment_data["color"] = exp_cfg["color"]
        experiment_data["line_style"] = exp_cfg["line_style"]

        # Save the data
        all_experiment_data.append(experiment_data)



    # # Plot!!!!!
    # rows = 1
    # cols = 3
    # width_ratios = [1, 1, 0.3]
    # col_size = sum([i*5 for i in width_ratios])
    # fig, axes = plt.subplots(rows, cols, sharex=False, sharey=True, figsize=(col_size, 3*rows), squeeze=False, gridspec_kw={"width_ratios":width_ratios})


    # # Plot!!!!!
    # rows = 1
    # cols = 2
    # width_ratios = [1, 1]
    # col_size = sum([i*8 for i in width_ratios])
    # fig, axes = plt.subplots(rows, cols, sharex=False, sharey=True, figsize=(col_size, 5*rows), squeeze=False, gridspec_kw={"width_ratios":width_ratios})

    # Plot!!!!!
    fig, axes = plt.subplots(rows, cols, sharex=False, sharey=True, figsize=(10*cols, 4*rows), squeeze=False)

    for i, experiment_data in enumerate(all_experiment_data):

        # Get which color index we should use
        color = experiment_data["color"]
        color_index = color_mapping[color]["index"]
        color_alpha = color_mapping[color]["alpha"]

        # Line style
        line_style = experiment_data["line_style"]
        line_style = linestyle_mapping[line_style]
        line_style = linestyle_tuple[line_style]

        # Select the color to use for this experiment
        color = sns.color_palette("tab10")[color_index]

        # Get the experiment name
        experiment_name = experiment_data["experiment_name"]


        # Get the experiment row and col pos
        row_pos = experiment_data["row_pos"]
        col_pos = experiment_data["col_pos"]

        # If we did not load everything then skip
        loaded_everything = experiment_data["loaded_everything"]
        if(loaded_everything == False):
            print("Skipping experiment \"{}\" because of missing data".format(experiment_name))
            continue

        # Get some stats that we want to plot for this and plot them
        y, x_values_position = compute_stats_for_run_data(experiment_data, "recall", number_of_runs=number_of_runs)
        axes[row_pos, col_pos].plot(x_values_position, y*100.0, label=experiment_name, color=color, linestyle=line_style, linewidth=3, alpha=color_alpha)


    # # Add the axis labels
    # axes[0, 0].set_xlabel("Position Error [m]", weight="bold", fontsize=16)
    # axes[0, 1].set_xlabel("Angle Error [°]", weight="bold", fontsize=16)
    
    # Set the x labels and ranges
    for r in range(rows):
        for c in range(cols):
            if((r in x_labels) and (c in x_labels[r])):
                axes[r, c].set_xlabel(x_labels[r][c], weight="bold", fontsize=16)

            if((r in x_ranges) and (c in x_ranges[r])):
                axes[r, c].set_xlim(x_ranges[r][c])

    # Set the Y axis labels
    for r in range(rows): 
        axes[r, 0].set_ylabel("Recall [%]", weight="bold", fontsize=16)


    # Set the recall y axis to be [0, 100] because its a percentage
    for r in range(rows):
        for c in range(cols):
            axes[r, c].set_ylim([0, 100])

    # # Create and add the legend
    # all_handles, all_labels = axes[0,0].get_legend_handles_labels()
    # handles = list()
    # labels = list()
    # for i, experiment_data in enumerate(all_experiment_data):

    #     # Get if we should include it in the Legend
    #     if(experiment_data["include_in_legend"] == False):
    #         continue

    #     # Include it
    #     handles.append(all_handles[i])
    #     labels.append(all_labels[i])


    handles = list()
    labels = list()
    for legend_item_idx in legend_configs.keys():

        # Get the values for this legend item
        legend_item_cfg = legend_configs[legend_item_idx]
        name = legend_item_cfg["name"]
        color = legend_item_cfg["color"]
        color_index = color_mapping[color]["index"]
        color_alpha = color_mapping[color]["alpha"]

        # Get the color
        color = sns.color_palette("tab10")[color_index]
        l = mLines.Line2D([0, 1], [0, 0], color=color, linewidth=5, alpha=color_alpha)

        # Add to the legend lists
        handles.append(l)
        labels.append(name)



    # Add the Legend
    lgnd = fig.legend(handles, labels, loc='upper center', ncol=8, fontsize=14)



    # Adjust whitespace
    fig.tight_layout(rect=(0,0,1,0.90))




    # Add legends
    # handles, labels = axes[0,0].get_legend_handles_labels()
    # axes[0, 2].legend(handles, labels, loc="upper left")
    # axes[0, 2].set_axis_off()

    # Make sure the labels are inside the image
    # fig.tight_layout()

    # Save
    plt.savefig("{}/{}.png".format(save_dir, plot_save_basename))
    plt.savefig("{}/{}.pdf".format(save_dir, plot_save_basename))


    # Plot
    plt.show()


if __name__ == '__main__':
	main()