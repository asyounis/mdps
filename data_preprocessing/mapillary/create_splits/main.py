
# Python Imports
import json
from collections import defaultdict
import random
import os


# Package Imports
import yaml
import numpy as np
import torch
from prettytable import PrettyTable




def load_yaml_file(file_path):

    # Read and parse the config file
    with open(file_path) as file:

        # Load the whole file into a dictionary and return
        return yaml.load(file, Loader=yaml.FullLoader)


def load_json_file(file_path):

    # Load the json
    with open(file_path, "r") as file:
        data = json.load(file)

    return data


def load_camera_and_view_info(dataset_directory, scenes, dump_filename):

    # These will hold the info for all the scenes
    all_camera_info = dict()
    all_view_info = dict()

    for scene in scenes:

        # Get the dump file data
        dump_filepath = "{}/{}/{}".format(dataset_directory, scene, dump_filename)
        dump_data = load_json_file(dump_filepath)

        # Make the camera and view info dicts for this scene
        camera_info = dict()
        view_info = dict()

        for sequence_name in dump_data.keys():
            per_seq = dump_data[sequence_name]

            # get the raw camera and view info 
            raw_camera_info = per_seq["cameras"]
            raw_view_info = per_seq["views"]

            # Process the camera info
            for camera in raw_camera_info.values():

                # Make sure its numpy
                camera["params"] = np.array(camera["params"], np.float32)

            # Process the view info
            for view_name in raw_view_info.keys():

                # Get the view so we can operate on it
                view = raw_view_info[view_name]
                
                # Make sure its numpy
                view["R_c2w"] = np.array(view["R_c2w"], np.float32)
                view["roll_pitch_yaw"] = np.array(view["roll_pitch_yaw"], np.float32)

                if("observations" in view):
                    view["observations"] = np.array(view["observations"])

                # remove the "chunk_id" since we dont need it
                if("chunk_id" in view):
                    view.pop("chunk_id")

                raw_view_info[view_name] = view

            # Add to the dict
            camera_info[sequence_name] = raw_camera_info
            view_info[sequence_name] = raw_view_info

        # Keep track of this
        all_camera_info[scene] = camera_info
        all_view_info[scene] = view_info

    return all_camera_info, all_view_info


def get_image_list(all_view_info):

    image_list = []

    unique = set()

    for scene in all_view_info.keys():
        for sequence_name in all_view_info[scene].keys():
            for view_name in all_view_info[scene][sequence_name].keys():
                image_list.append((scene, sequence_name, view_name))

                name = "{}{}{}".format(scene, sequence_name, view_name)
                assert(name not in unique)
                unique.add(name)

    return image_list

def consolidate_into_data_dict(all_view_info, all_camera_info, all_images_list):

    # We dont need these so get rid of them
    # keys_to_exclude = ["compass_angle","compass_accuracy", "gps_accuracy", "chunk_key", "panorama_offset"]

    # Keys to include
    keys_to_include = ["camera_id", "latlong", "t_c2w", "R_c2w", "roll_pitch_yaw", "capture_time", "gps_position", "index"]

    # All the consolidated data 
    consolidated_data = {k: [] for k in keys_to_include}

    # Get all the info for each of the frames 
    for image_info in all_images_list:
        (scene, sequence_name, view_name) = image_info

        # Get each of the keys we want to include
        for k in consolidated_data.keys():
            consolidated_data[k].append(all_view_info[scene][sequence_name][view_name].get(k, None))

        # Make sure all the camera IDs match correctly with the camera info
        camera_id = consolidated_data["camera_id"][-1]
        camera_info = all_camera_info[scene][sequence_name]
        assert(camera_id in camera_info)

    # Add in the camera data just so we have it.  We dont actually need it
    consolidated_data["cameras"] = all_camera_info

    return consolidated_data



def convert_to_numpy(consolidated_data):

    # Keys to include
    keys_to_include = ["camera_id", "latlong", "t_c2w", "R_c2w", "roll_pitch_yaw", "capture_time", "gps_position", "index"]

    # Convert to tensor
    for k in keys_to_include:
        
        # convert to numpy
        v = np.array(consolidated_data[k])

        # If we can then we should convert to pytorch
        if(np.issubdtype(v.dtype, np.integer) or np.issubdtype(v.dtype, np.floating)):
            v = torch.from_numpy(v)

        # Update the data
        consolidated_data[k] = v

    return consolidated_data




def create_sequences(consolidated_data, all_images_list, sequence_length=None):

    # For each of the images, get which sequence it is in 
    sequence_to_image_indices = defaultdict(list)
    for i, (scene_name, sequence_name, view_name) in enumerate(all_images_list):
        name = "{}_{}".format(scene_name, sequence_name)
        sequence_to_image_indices[name].append(i)

    # Get the sorting information. This is the info we will use to sort the sequence
    # indices.  If the capture time isnt present then use the index (not ideal)
    sorting_information = consolidated_data.get("capture_time", consolidated_data.get("index"))

    # Sort the indices so that they are in order
    for sequence_name in sequence_to_image_indices.keys():
        sequence_to_image_indices[sequence_name] = sorted(sequence_to_image_indices[sequence_name], key=lambda i: sorting_information[i].item())


    all_sequences = []

    for sequence_name in sequence_to_image_indices.keys():

        # Get the indices
        indices = sequence_to_image_indices[sequence_name]

        # Get the time for each of the indices
        times = []
        for idx in indices:
            times.append(sorting_information[idx].item())

        # Get all the positions for the indices
        positions = []
        for idx in indices: 

            # Get the image info
            (scene_name, sequence_name, view_name) = all_images_list[idx]

            # Get the XY from the camera to world translation vector
            t_c2w_xy_position_world_frame_gt = consolidated_data["t_c2w"][idx][:2].clone().double().numpy()

            # If we should init from the GPS or from the camera to world translation vector
            # if self.init_from_gps:
            if(False):
                # position = gps_xy_position_world_frame_gt.copy()
                assert(False)
            else:
                position = t_c2w_xy_position_world_frame_gt.copy()

            positions.append(position)

        # Make into numpy arrays 
        times = np.asarray(times)
        positions = np.vstack(positions)

        # Make into sequences that dont "Jump" in time or space
        current_sequence = []
        for i, idx in enumerate(indices):

            # If its the first item in the current sequence then add it
            if(len(current_sequence) == 0):
                current_sequence.append(idx)
                continue

            # Get some information
            current_time = times[i]
            current_pos = positions[i]
            last_time = times[i-1]
            last_pos = positions[i-1]

            # Compute the distance traveled (in meters)
            position_diff = (current_pos - last_pos) ** 2
            position_diff = np.sqrt(np.sum(position_diff))

            # Compute the time diff (in milliseconds)
            time_diff = current_time - last_time
            assert(time_diff >= 0)

            # Check if we should make a new sequence or not
            if((time_diff <= 30000) and (position_diff <= 100)):
                current_sequence.append(idx)
            else:
                all_sequences.append(current_sequence)
                current_sequence = [idx]

        # Need to add the last sequence 
        all_sequences.append(current_sequence)


    count = 0
    for sequence_name in sequence_to_image_indices.keys():
        indices = sequence_to_image_indices[sequence_name]
        count += len(indices)

    # If no sequence length is specified then return all the sequences
    if(sequence_length == None):
        return all_sequences

    # Process each sequence into the final sequences
    final_sequences = []
    for indices in all_sequences:

        # If the sequence is too short then dont use it, just skip it
        if(len(indices) < sequence_length):
            continue

        # Chunk the sequence into smaller sub-sequences
        s = 0
        e = s + sequence_length
        while(e < len(indices)):
            final_sequences.append(indices[s:e])
            s += sequence_length
            e = s + sequence_length

    return final_sequences




def cut_sequences_into_correct_length(sequences, desired_length):

    cut_sequences = []

    for sequence in sequences:
        assert(len(sequence) >= desired_length)

        s = 0
        e = s + desired_length
        while(e <= len(sequence)):
            cut_sequences.append(sequence[s:e])
            s += desired_length
            e = s + desired_length
    return cut_sequences

def split_sequences(all_sequences, split_configs):

    # get the info for the split configs
    short_sequence_length = split_configs["short_sequence_length"]
    long_sequence_length = split_configs["long_sequence_length"]
    long_sequence_splits = split_configs["long_sequence_splits"]
    short_sequence_splits = split_configs["short_sequence_splits"]


    # Make sure things are correct
    assert(long_sequence_length >= short_sequence_length)
    assert(sum(long_sequence_splits) == 1)
    assert(sum(short_sequence_splits) == 1)
    assert(len(long_sequence_splits) == 3)
    assert(len(short_sequence_splits) == 2)


    # Randomize the sequence order
    random.shuffle(all_sequences)



    # Get all the sequences that are eligible for long and short sequence lengths
    long_sequences = [s for s in all_sequences if(len(s) >= long_sequence_length)]
    short_sequences = [s for s in all_sequences if((len(s) < long_sequence_length) and (len(s) >= short_sequence_length))]

    # For each of the long sequences, truncate the remainder and add those to the short sequences
    long_sequences_tmp = []
    for ls in long_sequences:

        # Compute where to truncate the sequence
        truncate_start = int(len(ls) // long_sequence_length) * long_sequence_length
            
        # Keep only the part of the long sequence that we will definitely use
        long_sequences_tmp.append(ls[0:truncate_start])

        # Get the rest of it as a short sequence
        ss = ls[truncate_start:]

        # If its too short then toss it
        if(len(ss) < short_sequence_length):
            continue

        # Otherwise save it for later
        short_sequences.append(ss)

    # Not a temp anymore
    long_sequences = long_sequences_tmp

    # Cut the sequences into their correct lengths
    cut_long_sequences = cut_sequences_into_correct_length(long_sequences, long_sequence_length)
    cut_short_sequences = cut_sequences_into_correct_length(short_sequences, short_sequence_length)

    for s in cut_long_sequences:
        assert(len(s) == long_sequence_length)

    for s in cut_short_sequences:
        assert(len(s) == short_sequence_length)


    # Split the long sequences into their desired splits
    split_location_a = int(len(cut_long_sequences) * long_sequence_splits[0])
    split_location_b = int(len(cut_long_sequences) * long_sequence_splits[1]) + split_location_a
    long_training_sequences = cut_long_sequences[:split_location_a]
    long_validation_sequences = cut_long_sequences[split_location_a:split_location_b]
    long_evaluation_sequences = cut_long_sequences[split_location_b:]

    # Split the short sequences into their desired splits
    split_location = int(len(cut_short_sequences) * short_sequence_splits[0])
    short_training_sequences = cut_short_sequences[:split_location]
    short_validation_sequences = cut_short_sequences[split_location:]

    # Pack into the splits
    splits_dict = dict()
    splits_dict["long_training_sequences"] = long_training_sequences
    splits_dict["long_validation_sequences"] = long_validation_sequences
    splits_dict["long_evaluation_sequences"] = long_evaluation_sequences
    splits_dict["short_training_sequences"] = short_training_sequences
    splits_dict["short_validation_sequences"] = short_validation_sequences



    # Calculate how many frames we kept from the whole dataset
    total_number_of_selected_frames = 0
    total_number_of_selected_frames += sum([len(s) for s in long_training_sequences])
    total_number_of_selected_frames += sum([len(s) for s in long_validation_sequences])
    total_number_of_selected_frames += sum([len(s) for s in long_evaluation_sequences])
    total_number_of_selected_frames += sum([len(s) for s in short_training_sequences])
    total_number_of_selected_frames += sum([len(s) for s in short_validation_sequences])

    # Calculate the total number of frames in the dataset
    total_number_of_frames = sum([len(s) for s in all_sequences])

    # # Print some stats
    table = PrettyTable()
    table.field_names = ["Name", "# Seqs", "Seq/Total Same", "# Frames", "Frame/Total Same", "Frame/Total Used", "Frame/Total All"]

    # Some totals we want to keep track of
    stats_total_number_of_frames = 0

    # Print some stats
    for name in ["short_training_sequences", "short_validation_sequences", "long_training_sequences", "long_validation_sequences", "long_evaluation_sequences"]:

        # Get some stats
        sequences = splits_dict[name]
        number_of_frames = sum([len(s) for s in sequences])
        number_of_sequences = len(sequences)

        if("long" in name):
            sequences_with_correct_length = cut_long_sequences
        else:
            sequences_with_correct_length = cut_short_sequences

        number_of_frames_with_correct_length = sum([len(s) for s in sequences_with_correct_length])


        row_info = []
        row_info.append(name)
        # row_info.append(splits[i])
        row_info.append(len(sequences))        
        row_info.append("{:0.3f}".format(number_of_sequences / len(sequences_with_correct_length)))

        row_info.append(number_of_frames)
        row_info.append("{:0.3f}".format(number_of_frames / number_of_frames_with_correct_length))
        row_info.append("{:0.3f}".format(number_of_frames / total_number_of_selected_frames))
        row_info.append("{:0.3f}".format(number_of_frames / total_number_of_frames))

        table.add_row(row_info)

        stats_total_number_of_frames += number_of_frames 

    print(table)

    print("")
    number_of_frames_being_thrown_away = total_number_of_frames - stats_total_number_of_frames
    percent_number_of_frames_being_thrown_away = number_of_frames_being_thrown_away / total_number_of_frames
    print("total_number_of_frames:", total_number_of_frames)
    print("total_number_of_frames used:", stats_total_number_of_frames)
    print("number_of_frames_being_thrown_away:", number_of_frames_being_thrown_away)
    print("ratio number_of_frames_being_thrown_away:", percent_number_of_frames_being_thrown_away)


    return splits_dict



def main():

    # The yaml file to load
    DATASET_CONFIG_FILE = "./mapillary.yaml"

    # Load the dataset configs
    configs = load_yaml_file(DATASET_CONFIG_FILE)

    # Get some info we need
    dataset_directory = configs["dataset_directory"]
    splits_filename = configs["splits_filename"]
    dump_filename = configs["dump_filename"]
    output_splits_save_filename = configs["output_splits_save_filename"]
    split_configs = configs["split_configs"]

    # the scenes we want to use
    # We are ignoring "vilnius" for some reason. I think there was something wrong with that one
    scenes = ['sanfrancisco_soma', 'nantes', 'paris', 'sanfrancisco_hayes', 'avignon', 'amsterdam', 'montrouge', 'berlin', 'helsinki', 'milan', 'toulouse', 'lemans']

    # Get the camera and view info
    all_camera_info, all_view_info = load_camera_and_view_info(dataset_directory, scenes, dump_filename)

    # Get the list of all the images
    all_images_list = get_image_list(all_view_info)

    # Consolidate the data into a single dict of lists
    consolidated_data = consolidate_into_data_dict(all_view_info, all_camera_info, all_images_list)

    # Convert to tensors
    consolidated_data = convert_to_numpy(consolidated_data)

    # Create the 
    all_sequences = create_sequences(consolidated_data, all_images_list, sequence_length=None)


    # torch.save(all_sequences, "all_sequences.pt")
    # exit()
    # all_sequences = torch.load("all_sequences.pt")

    # Compute the sequence splits
    sequence_splits = split_sequences(all_sequences, split_configs)

    # Make sure the sequences are well formatted
    for key in sequence_splits:
        all_sequences = sequence_splits[key]
        seen = set()
        for sequence in all_sequences:
            for idx in sequence:
                assert(idx not in seen)
                seen.add(idx)


    # Make sure the save directory exists
    if(not os.path.exists(output_splits_save_filename)):
        os.makedirs(output_splits_save_filename)

    # Save all the data and splits
    save_dict = dict()
    save_dict["consolidated_data"] = consolidated_data
    save_dict["all_images_list"] = all_images_list
    save_dict["sequence_splits"] = sequence_splits
    torch.save(save_dict, output_splits_save_filename+"/splits_data.pt")

    print("Saved to")
    print("\t {}".format(output_splits_save_filename+"/splits_data.pt"))





if __name__ == '__main__':
    main()