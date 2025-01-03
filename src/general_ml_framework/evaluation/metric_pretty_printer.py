
# Python Imports
import csv

# Package Imports
import torch
from prettytable import PrettyTable


# Ali Package Import

# Project Imports


class MetricPrettyPrinter:
    def __init__(self, logger, save_dir):

        # Save for later
        self.logger = logger
        self.save_dir = save_dir



    def print_metrics(self, metrics):

        # get all the metric data
        all_metric_data = []
        for metric_name in metrics.keys():
            all_metric_data.extend(metrics[metric_name].get_aggregated_result())


        # Get a set of all unique metric sub-names
        unique_metric_labels = set()
        for metric_data in all_metric_data:
            for metric_labels in metric_data.keys():
                unique_metric_labels.add(metric_labels)

        # Skip the name one
        unique_metric_labels.remove("name")

        # Convert to a list and sort it
        unique_metric_labels = list(unique_metric_labels)
        unique_metric_labels.sort()

        # make a dict mapping the subname to where it is in the row
        subname_index_mapping = {unique_metric_labels[i]:i+1 for i in range(len(unique_metric_labels))}

        # Make the name come first
        subname_index_mapping["name"] = 0

        # Figure out the row length
        # This is the number of metric subnames + 1 (for the name of the metric)
        row_length = len(unique_metric_labels) + 1


        # Put them into rows
        rows = []
        for metric_data in all_metric_data:

            # Make the blank row
            row = [None for i in range(row_length)]

            # Add the metric name
            row[0] = metric_data["name"]

            # Add the metrics aggregated values
            for metric_label in metric_data.keys():

                # Get the index to put the data
                subname_idx = subname_index_mapping[metric_label]

                # Get the value
                value = metric_data[metric_label]

                # Put it depending on what datatype it is
                if(torch.is_tensor(value)):
                    row[subname_idx] = "{:0.4f}".format(value.item())
                elif(isinstance(value, str)):
                    row[subname_idx] = "{}".format(value)
                elif(isinstance(value, float)):
                    row[subname_idx] = "{:0.4f}".format(value)
                elif(isinstance(value, int) or isinstance(value, long)):
                    row[subname_idx] = "{:d}".format(value)

            # For all the rows that are left blank, fill them in with a filler
            for i in range(len(row)):
                if(row[i] is None):
                    row[i] = "--"

            # Add the row to all the rows
            rows.append(row)

        # Print the data
        self._print_table(rows, unique_metric_labels)
        self._print_csv(rows, unique_metric_labels)

    def _print_table(self, rows, unique_metric_labels):

        # Create the table
        table = PrettyTable()
        table.field_names = ["metric"] + unique_metric_labels

        # Add all the rows
        for row in rows:
            table.add_row(row)

        # Log it
        self.logger.log("\n\n")
        self.logger.log("==================================================================")
        self.logger.log("Quantitative Results:")
        self.logger.log("==================================================================")
        self.logger.log(str(table))
        self.logger.log("\n\n")


    def _print_csv(self, rows, unique_metric_labels):

        # Create the header
        header = ["metric"] + unique_metric_labels

        # Save to the file 
        csv_output_file = "{}/metrics.csv".format(self.save_dir)
        with open(csv_output_file, 'w') as f:

            # Create the CSV write
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write all the rows
            writer.writerows(rows)

