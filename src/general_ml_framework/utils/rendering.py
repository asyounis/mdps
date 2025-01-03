
# Python Imports
import os
import subprocess


# Module imports 
import yaml
import cv2



def generate_gif_from_image_directory(img_dir, output_filepath, img_file_type="png", frame_rate=3, logger=None):

    # Make sure the inputs are valid
    assert(isinstance(img_dir, str))
    assert(isinstance(output_filepath, str))
    assert(isinstance(img_file_type, str))
    assert(isinstance(frame_rate, int))

    # Get absolute paths
    img_dir = os.path.abspath(img_dir)
    output_filepath = os.path.abspath(output_filepath)

    # Get all the files
    all_files = []
    for file in os.listdir(img_dir):
        if file.endswith(".{}".format(img_file_type)):
            all_files.append(os.path.join(img_dir, file))

    # Get the size of a representative image by loading the image and getting its width and height
    img = cv2.imread(all_files[0])
    H, W = img.shape[0:2]

    # Build the ffmpeg command
    ffmpeg_command = []
    ffmpeg_command.append("ffmpeg")
    ffmpeg_command.append("-ss")
    ffmpeg_command.append("0")
    ffmpeg_command.append("-framerate")
    ffmpeg_command.append("{:d}".format(frame_rate))
    ffmpeg_command.append("-pattern_type")
    ffmpeg_command.append("glob")
    ffmpeg_command.append("-i")
    ffmpeg_command.append("'{}/*.{}'".format(img_dir, img_file_type))
    ffmpeg_command.append("-y")
    ffmpeg_command.append("-vf")
    ffmpeg_command.append('"fps={:d},scale={}:{}:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"'.format(frame_rate, W, H))
    ffmpeg_command.append(output_filepath)

    # Run the command
    p = subprocess.Popen(" ".join(ffmpeg_command), shell=True, stdout=subprocess.PIPE,  stderr=subprocess.PIPE)
    
    # Wait (forever) for the process to finish
    p.wait(timeout=None)

    # Check if something went wrong
    if(p.returncode != 0):

        # Something went wrong so get the output
        out, err = p.communicate()

        # Make them into normal strings
        out = out.decode()
        err = err.decode()

        # Log the output but dont crash 
        if(logger is not None):
            logger.log_error("Issue creating GIF for data in directory: \"{}\"".format(img_dir))
            logger.log_error("ffmepg output:")
            logger.log_error("---------------------------------------------------------------------------------------------------------")
            logger.log_error(out)
            logger.log_error(err)
            logger.log_error("---------------------------------------------------------------------------------------------------------")

        else:
            print("")
            print("ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR")
            print("Issue creating GIF for data in directory: \"{}\"".format(img_dir))
            print("ffmepg output:")
            print("---------------------------------------------------------------------------------------------------------")
            print(out)
            print(err)
            print("---------------------------------------------------------------------------------------------------------")
            print("ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR")
            print("")