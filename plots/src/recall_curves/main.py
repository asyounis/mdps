import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

def load_data_for_experiment(base_dir, experiment, experiment_name, evaluation_directory, position_recall_metric_name_base, angle_recall_metric_name_base, number_of_runs=5):

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
        position_recall_metric_data = []
        for md in metric_data:
            if(position_recall_metric_name_base in md["name"]):
                position_recall_metric_data.append(md)

        # sort based on the threshold values
        position_recall_metric_data.sort(key=lambda x: x["threshold"])

        # Pack the data into a dict
        position_recall_data_dict = dict()
        position_recall_data_dict["display_thresholds"] = [d["threshold_scaled"] for d in position_recall_metric_data]
        position_recall_data_dict["values"] = [d["value"] for d in position_recall_metric_data]

        # Get the position recall metric data
        angle_recall_metric_data = []
        for md in metric_data:
            if(angle_recall_metric_name_base in md["name"]):
                angle_recall_metric_data.append(md)

        # sort based on the threshold values
        angle_recall_metric_data.sort(key=lambda x: x["threshold"])

        # Pack the data into a dict
        angle_recall_data_dict = dict()
        angle_recall_data_dict["display_thresholds"] = [d["threshold_scaled"] for d in angle_recall_metric_data]
        angle_recall_data_dict["values"] = [d["value"] for d in angle_recall_metric_data]

        # Pack it
        all_data_for_this_run = dict()
        all_data_for_this_run["position_recall"] = position_recall_data_dict
        all_data_for_this_run["angle_recall"] = angle_recall_data_dict

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

    # Convert from [0, 1] to percentage (aka [0, 100])
    median = median * 100.0
    iqr = iqr * 100.0

    return median, iqr, x_values


def main():
	
    # get the config file and load it
    with open(sys.argv[1]) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)

    # Get the save dir
    save_dir = configs["save_dir"]

    # Get the base dir
    base_dir = configs["base_dir"]

    # Get the number of runs
    number_of_runs = configs["number_of_runs"]

    # Get the experiment configs
    experiment_configs = []
    for experiment in configs["experiments"]:
        exp_cfg = experiment[list(experiment.keys())[0]]
        experiment_configs.append(exp_cfg)


    # Get all the experiment data
    all_experiment_data = []
    for exp_cfg in experiment_configs:

        # Load the experiment data
        experiment_data = load_data_for_experiment(base_dir, exp_cfg["dir_name"], exp_cfg["display_name"], exp_cfg["evaluation_dir_name"], exp_cfg["position_recall_metric_name_base"], exp_cfg["angle_recall_metric_name_base"], number_of_runs=number_of_runs)
        # experiment_data = load_data_for_experiment(exp_cfg["dir_name"], exp_cfg["evaluation_dir_name"], exp_cfg["display_name"], exp_cfg["position_recall_metric_name_base"], exp_cfg["angle_recall_metric_name_base"], number_of_runs=number_of_runs)

        # Save the data
        all_experiment_data.append(experiment_data)


    # # Plot!!!!!
    # rows = 1
    # cols = 3
    # width_ratios = [1, 1, 0.3]
    # col_size = sum([i*5 for i in width_ratios])
    # fig, axes = plt.subplots(rows, cols, sharex=False, sharey=True, figsize=(col_size, 3*rows), squeeze=False, gridspec_kw={"width_ratios":width_ratios})


    # Plot!!!!!
    rows = 1
    cols = 2
    width_ratios = [1, 1]
    col_size = sum([i*8 for i in width_ratios])
    fig, axes = plt.subplots(rows, cols, sharex=False, sharey=True, figsize=(col_size, 5*rows), squeeze=False, gridspec_kw={"width_ratios":width_ratios})


    for i, experiment_data in enumerate(all_experiment_data):

        # Select the color to use for this experiment
        color = sns.color_palette("tab10")[i]

        # Get the experiment name
        experiment_name = experiment_data["experiment_name"]

        # If we did not load everything then skip
        loaded_everything = experiment_data["loaded_everything"]
        if(loaded_everything == False):
            print("Skipping experiment \"{}\" because of missing data".format(experiment_name))
            continue

        # Get some stats that we want to plot for this and plot them
        median, iqr, x_values_position = compute_stats_for_run_data(experiment_data, "position_recall", number_of_runs=number_of_runs)
        axes[0, 0].fill_between(x_values_position, median-iqr, median+iqr, color=color, alpha=0.75)
        axes[0, 0].plot(x_values_position, median, label=experiment_name, color=color, linewidth=3)

        # # Get some stats that we want to plot for this and plot them
        median, iqr, x_values_angle = compute_stats_for_run_data(experiment_data, "angle_recall", number_of_runs=number_of_runs)
        axes[0, 1].fill_between(x_values_angle, median-iqr, median+iqr, color=color, alpha=0.75)
        axes[0, 1].plot(x_values_angle, median, color=color, linewidth=3)


    # Add the axis labels
    axes[0, 0].set_xlabel("Position Error [m]", weight="bold", fontsize=16)
    axes[0, 1].set_xlabel("Angle Error [°]", weight="bold", fontsize=16)
    axes[0, 0].set_ylabel("Recall [%]", weight="bold", fontsize=16)

    # Set the recall y axis to be [0, 100] because its a perentage
    axes[0, 0].set_ylim([0, 100])
    axes[0, 0].set_ylim([0, 100])


    # Add the Legend
    handles, labels = axes[0,0].get_legend_handles_labels()
    lgnd = fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=10)

    # Adjust whitespace
    # fig.subplots_adjust(wspace=0.01, hspace=0.03)
    fig.tight_layout(rect=(0,0,1,0.94))




    # Add legends
    # handles, labels = axes[0,0].get_legend_handles_labels()
    # axes[0, 2].legend(handles, labels, loc="upper left")
    # axes[0, 2].set_axis_off()

    # Make sure the labels are inside the image
    # fig.tight_layout()

    # Save
    plt.savefig("{}/recall_curves_plot.png".format(save_dir))
    plt.savefig("{}/recall_curves_plot.pdf".format(save_dir))


    # Plot
    plt.show()


if __name__ == '__main__':
	main()