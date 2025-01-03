import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import matplotlib.pyplot as plt
from matplotlib import rc,rcParams
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import os
import torch
import pandas as pd
import csv
import sys
import yaml 

rc('font', weight='bold')

def load_data_for_experiment(base_dir, experiment, metric_name, evaluation_directory, number_of_runs=5):

    # The data for this experiment
    data = []

    for i in range(number_of_runs):
        results_filepath = "{}/{}/saves/{}/run_{:04d}/quantitative/metrics.csv".format(base_dir, experiment, evaluation_directory, i)

        # Check if the file exists, if it doesnt then skip this file
        if(os.path.isfile(results_filepath) == False):
            print("Could not load: ")
            print("\t{}".format(results_filepath))
            continue

        with open(results_filepath, mode ='r')as file:
            csv_data = csv.reader(file)
            lines = list(csv_data)

        # Skip the first line which is the header
        lines = lines[1:]

        # Pack them into a dict
        metrics = dict()
        for line in lines:
            metrics[line[0]] = line[1]


        print(metrics.keys())

        # Get the specific metric we care about
        assert(metric_name in metrics)
        data.append(float(metrics[metric_name]))


    return data


def load_data_onto_dataframe(base_dir,number_of_runs,  experiment_configs):

    # Pack all the data into a data struct that can be bassed into a pandas data frame
    all_experiment_data = []
    for experiment_config in experiment_configs:

        # Upack
        experiment = experiment_config["dir_name"]
        experiment_name = experiment_config["display_name"]
        evaluation_directory = experiment_config["evaluation_dir_name"]
        metric_name = experiment_config["metric_name"]

        # Load the experiment data
        experiment_data = load_data_for_experiment(base_dir, experiment, metric_name, evaluation_directory, number_of_runs=number_of_runs)
        experiment_data = [(d, experiment_name) for d in experiment_data]

        # Save the data
        all_experiment_data.extend(experiment_data)

    # Pack into a data frame for plotting
    df = pd.DataFrame(all_experiment_data, columns = ['value','experiment_name'])

    return df

def make_plot(configs):

    # Get the save file name
    save_file_base_name = configs["save_file_base_name"]

    # Get the save dir and make sure it exists
    save_dir = configs["save_dir"]
    if(not os.path.exists(save_dir)):
        os.makedirs(save_dir)

    # Get the base dir
    base_dir = configs["base_dir"]

    # Get the number of runs we need
    number_of_runs = configs["number_of_runs"]



    # Unpack the experiment configs
    experiment_configs = []
    for experiment in configs["experiments"]:
        exp_cfg = experiment[list(experiment.keys())[0]]
        # experiment_configs.append((exp_cfg["dir_name"], exp_cfg["display_name"], exp_cfg["evaluation_dir_name"], exp_cfg["metric_name"]))
        experiment_configs.append(exp_cfg)


    # Load into a data frame
    df = load_data_onto_dataframe(base_dir,number_of_runs, experiment_configs)

    # Create the experiment order
    experiment_order = [exp_cfg["display_name"] for exp_cfg in experiment_configs]


    # Plot!!!!!
    rows = 1
    cols = 1
    width = max(2*len(experiment_configs), 8)
    fig, axes = plt.subplots(rows, cols, sharex=True, figsize=(width, 4), squeeze=False)


    # Plot.  
    #   Note: set "whis" to a large number to prevent outliers from being rendered as diamonds, keep them in the min and max whiskers of the box plot
    # sns.boxplot(data=df, x="experiment_name", y="value", ax=axes[0, 0], palette=sns.color_palette(as_cmap=True), medianprops={"color": "red"}, orient="v", whis=1000000, order=experiment_order)
    # sns.boxplot(data=df, x="experiment_name", y="value", ax=axes[0, 0], color=sns.color_palette(as_cmap=True)[0], medianprops={"color": "red"}, orient="v", whis=1000000, order=experiment_order)
    sns.boxplot(data=df, x="experiment_name", y="value", ax=axes[0, 0], color=sns.color_palette(as_cmap=True)[0], medianprops={"color": "red"}, orient="v", whis=1000000, order=experiment_order)



    # Display the insets
    for i, experiment_config in enumerate(experiment_configs):

        # Upack
        use_inset = experiment_config["use_inset"]
        experiment_name = experiment_config["display_name"]

        # If no need for inset then do nothing
        if(use_inset == False):
            continue

        # Create the inset
        inset_x_edge_margins = 0.025
        inset_x_size = (1 / len(experiment_configs)) - (inset_x_edge_margins*2)
        inset_x_start = (i / len(experiment_configs)) + ((1 / len(experiment_configs)) / 2)
        inset_x_start -= inset_x_edge_margins
        inset_ax = axes[0, 0].inset_axes([inset_x_start,0.6,inset_x_size,0.35])

        # Get the data for this inset and plot it
        inset_df = df[df["experiment_name"] == experiment_name]
        sns.boxplot(data=inset_df, x="experiment_name", y="value", ax=inset_ax, color=sns.color_palette(as_cmap=True)[0], medianprops={"color": "red"}, orient="v", whis=1000000)

        # Style the inset
        inset_ax.patch.set_edgecolor('black')  
        inset_ax.patch.set_linewidth(1.5)  
        inset_ax.legend().remove()
        inset_ax.set(xlabel=None)
        inset_ax.set(ylabel=None)
        # inset_ax.yaxis.set_label_position("right")
        # inset_ax.yaxis.tick_right()

        # Label the inset
        labels = [experiment_name]
        x = np.arange(len(labels))
        inset_ax.set_xticks(x, labels, rotation=0, fontsize=9, ha="center",va="top", weight="bold")
        



    # Set the title
    # axes[0, 0].set_title(configs["title"], fontsize=20, fontweight="bold")

    # Set the x and y labels (or turn them off maybe )
    axes[0, 0].set_ylabel(configs["y_axis_label"], fontsize=16, fontweight="bold")

    # axes[0, 0].set_xlabel("Experiment Name", fontsize=16, fontweight="bold")
    axes[0, 0].set(xlabel=None)

    # Set the font size
    axes[0, 0].tick_params(axis='both', which='major', labelsize=14)



    # Set the y limits for this plot
    if("y_axis_limits" in configs):
        axes[0, 0].set_ylim(configs["y_axis_limits"])

    # Make sure the labels are inside the image
    fig.tight_layout()

    # Save
    plt.savefig("{}/{}.png".format(save_dir, save_file_base_name))
    plt.savefig("{}/{}.pdf".format(save_dir, save_file_base_name))

    # Plot
    plt.show()

def main():
	


    # get the config file and load it
    with open(sys.argv[1]) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    

    # Get the configs for all the plots to make
    plot_configs = configs["plot_configs"]

    # Plot them 1 by one
    for plot_name in plot_configs.keys():
        make_plot(plot_configs[plot_name])


if __name__ == '__main__':
	main()