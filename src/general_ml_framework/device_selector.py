
# Python Imports
import os 

# Package Imports
import pynvml
import torch
from prettytable import PrettyTable

# Project Imports
from .utils.config import *


class DeviceSelector:
    def __init__(self):
        pass

    def get_device(self, device_configs):

        # Unpack the device configs
        device_selection_string = get_mandatory_config("device", device_configs, "device_configs")
        min_free_memory_gb = get_mandatory_config("min_free_memory_gb", device_configs, "device_configs")
        max_number_of_tasks = get_mandatory_config("max_number_of_tasks", device_configs, "device_configs")

        #  The device to select 
        device = None

        # Get all the devices 
        if("cuda_use_all" == device_selection_string):

            # Get all the free GPUs:
            free_gpus = self._get_free_gpus(min_free_memory_gb, max_number_of_tasks)

            # Make sure we have some GPUs
            if(len(free_gpus) == 0):
                print("No GPUs available.... Exiting")
                assert(False)

            # Use all of them
            device = free_gpus

            # If we have 1 GPU then rename it into a pytorch format
            if(len(device) == 1):
                device = "cuda:{}".format(device[0])

        # Get the device we should use based on what the user specified
        elif("cuda_auto_multi" in device_selection_string):

            # Get all the free GPUs:
            free_gpus = self._get_free_gpus(min_free_memory_gb, max_number_of_tasks)

            # Get how many gpus we need
            num_gpus_needed = int(device_selection_string.split(":")[-1])

            # Make sure we have enough
            if(num_gpus_needed > len(free_gpus)):
                print("Not enough available GPUs for request.... Exiting")
                print("\t Requested {:d} GPUs but only had {:d}".format(num_gpus_needed, free_gpus))
                assert(False)

            # Select only the number we need
            device = free_gpus[:num_gpus_needed]

            # If we have 1 GPU then rename it into a pytorch format
            if(len(device) == 1):
                device = "cuda:{}".format(device[0])

        elif(device_selection_string == "cuda_auto"):

            # Get all the free GPUs:
            free_gpus = self._get_free_gpus(min_free_memory_gb, max_number_of_tasks)

            # Get how many gpus we need
            num_gpus_needed = int(device_selection_string.split(":")[-1])

            # Make sure we have enough
            if(len(free_gpus) == 0):
                print("Not enough available GPUs for request.... Exiting")
                assert(False)

            # Select only the number we need
            device_idx = free_gpus[0]

            # put it in pytorch format
            device = "cuda:{}".format(device_idx)

        else:
            device = device_selection_string

        return device


    def get_gpu_info_str(self, indent=""):

        # Get the labels for each of the devices
        all_devices = self._get_all_device_ids()

        # Get all the data
        device_infos = self._get_all_device_infos()

        # Pack into a pretty table
        table = PrettyTable()
        table.field_names = ["Device", "Num. Running Processes", "Used VRAM", "Free VRAM", "Total Memory"]

        for device_info in device_infos:
            row_data = []
            row_data.append("GPU: {:02d}".format(device_info["id"]))
            row_data.append("{:d}".format(device_info["num_compute_processes_running"]))
            row_data.append("{:.2f} MiB".format(device_info["used_memory"]/(1024*1024)))
            row_data.append("{:.2f} MiB".format(device_info["free_memory"]/(1024*1024)))
            row_data.append("{:.2f} MiB".format(device_info["total_memory"]/(1024*1024)))
            table.add_row(row_data)

        # Add indent
        table_str = str(table)
        table_str = table_str.split("\n")
        table_str = ["{}{}".format(indent, ts) for ts in table_str]
        table_str = "\n".join(table_str)

        return table_str


    def _get_free_gpus(self, min_free_memory_gb, max_number_of_tasks):

        # Get the labels for each of the devices
        all_devices = self._get_all_device_ids()

        # Get all the data
        device_infos = self._get_all_device_infos()

        free_gpus = []
        gpus_free_memory = []


        # Go through all the devices and
        for i, device_info in enumerate(device_infos):

            # Extract the info we care about
            free_memory = device_info["free_memory"]
            num_compute_processes_running = device_info["num_compute_processes_running"]

            # If it has free memory then we can use it
            if((free_memory >= (min_free_memory_gb * 1024 * 1024)) and (num_compute_processes_running < max_number_of_tasks)):
                free_gpus.append((i, free_memory))


        # Sort the GPUs by the ones with the most free memory.  
        # This will make the GPUs earlier in the list have more memory available
        free_gpus = sorted(free_gpus, key=lambda x:x[1], reverse=True)
        free_gpus = [x[0] for x in free_gpus]

        return free_gpus

    def _get_gpu_to_use(self):

        # Get the labels for each of the devices
        all_devices = self._get_all_device_ids()

        # Get all the data
        device_infos = self._get_all_device_infos()

        # The stats we care about when selecting a GPU
        free_memory_value = 0
        num_processes_value = 100000

        # Select the GPU
        gpu_select_index = -1

        # Go through all the devices and
        for i, device_info in enumerate(device_infos):

            # Extract the info we care about
            free_memory = device_info["free_memory"]
            num_compute_processes_running = device_info["num_compute_processes_running"]

            if(abs(free_memory_value - free_memory) < (500 * 1024 * 1024)):
                
                # If the GPU doesnt have vram free (less than 500 MiB) then select the GPU with the least number of processes
                if(num_compute_processes_running < num_processes_value):
                    gpu_select_index = i;
                    free_memory_value = free_memory
                    num_processes_value = num_compute_processes_running
            else:

                # Choose the GPU with the most free VRAM
                if(free_memory_value < free_memory):
                    gpu_select_index = i;
                    free_memory_value = free_memory
                    num_processes_value = num_compute_processes_running

        return gpu_select_index





    def _get_all_device_ids(self):


        # Get the labels for each of the devices
        if("CUDA_VISIBLE_DEVICES" in os.environ):
            all_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(",")
            all_devices = [int(s) for s in all_devices]
        else:

            # Get the number of devices
            device_count = torch.cuda.device_count()
            
            # Make into a list
            all_devices = [i for i in range(device_count)] 

        return all_devices

    def _get_all_device_infos(self):

        # Init NVML
        pynvml.nvmlInit()   

        # Get the names for each of the devices
        all_device_ids = self._get_all_device_ids()

        # Loop through all the devices and get the info
        device_infos = []
        for device_id in all_device_ids:

            # Get the info and save it
            device_info = self._get_gpu_info(device_id)
            device_infos.append(device_info)

        # Shutdown NVML
        pynvml.nvmlShutdown()

        return device_infos

    def _get_gpu_info(self, device_id):

        # Create the NVML device handle
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(device_id))
        
        # Get the info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        num_compute_processes_running = len(pynvml.nvmlDeviceGetComputeRunningProcesses(handle))

        # Pack into a dictionary
        device_info = dict()
        device_info["id"] = device_id
        device_info["free_memory"] = mem_info.free
        device_info["used_memory"] = mem_info.used
        device_info["total_memory"] = mem_info.total
        device_info["num_compute_processes_running"] = num_compute_processes_running

        return device_info