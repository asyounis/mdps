
# Python Imports
import os
import zipfile

# Module imports 
import yaml

def print_dict_pretty(data_dict):
    print(yaml.dump(data_dict, allow_unicode=True, default_flow_style=False))


def compress_file(file_path, delete_file_after_compression=False, compression_level=9):

    # Make sure the compression level is correct
    assert((compression_level >= 0) and (compression_level <= 9))
    
    # Make sure the file exists
    assert(os.path.exists(file_path))

    # Create the zip file name
    zip_file_name = "{}.zip".format(file_path)
    
    # Zip it
    with zipfile.ZipFile(zip_file_name, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=compression_level) as archive:
        archive.write(file_path)

    # Delete the origional file 
    if(delete_file_after_compression):
        os.remove(file_path)

