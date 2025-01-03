
# Python Imports

# Module Imports
import torch

def move_data_to_device(data, device, exclude_keys=None, include_keys=None):

    # If we are using multiple devices then we dont need to move the data, that will be handled for us
    if(isinstance(device, list)):
        return data

    if(isinstance(data, list)):
        return [move_data_to_device(d, device, exclude_keys=exclude_keys, include_keys=include_keys) for d in data]

    elif(isinstance(data, dict)):
        moved_data = dict()
        for k in data.keys():
            if(data[k] is None):
                continue

            if(include_keys is not None):
                if(k in include_keys):
                    moved_data[k] = data[k].to(device)

            elif(exclude_keys is not None):
                if(k not in exclude_keys):
                    moved_data[k] = data[k].to(device)

            else:
                moved_data[k] = move_data_to_device(data[k], device, exclude_keys=exclude_keys, include_keys=include_keys)

        return moved_data  

    elif(torch.is_tensor(data) == False):
        return data

    elif(data is not None):
        return data.to(device)

    else:
        return None


def get_device_from_dict(in_dict):

    for k in in_dict.keys():
        if(torch.is_tensor(in_dict[k])):
            return in_dict[k].device

    return "cpu"