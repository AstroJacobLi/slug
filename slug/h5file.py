'''HDF5 files related functions.'''

import numpy as np

__all__ = ['h5_print_attrs', 'h5_rewrite_dataset', 'str2dic']

# Print attributes of a HDF5 file
def h5_print_attrs(f):
    '''
    Print all attributes of a HDF5 file.

    Parameters:
    ----------
    f: HDF5 file.

    Returns:
    --------
    All attributes of 'f'
    '''
    def print_attrs(name, obj):
        print(name)
        for key, val in obj.attrs.items():
            print("    %s: %s" % (key, val))

    f.visititems(print_attrs)

# Rewrite dataset
def h5_rewrite_dataset(mother_group, key, new_data):
    '''
    Rewrite the given dataset of a HDF5 group.

    Parameters:
    ----------
    mother_group: HDF5 group class.
    key: string, the name of the dataset to be writen into.
    new_data: The data to be written into.
    '''
    if any([keys == key for keys in mother_group.keys()]):
        mother_group.__delitem__(key)
        dt = mother_group.create_dataset(key, data=new_data)
    else:
        dt = mother_group.create_dataset(key, data=new_data)
    return dt

# Create New group
def h5_new_group(mother_group, key):
    '''
    Create a new data_group

    Parameters:
    ----------
    mother_group: HDF5 group class.
    key: string, the name of the dataset to be writen into.
    new_data: The data to be written into.
    '''
    if not any([keys == key for keys in mother_group.keys()]):
        new_grp = mother_group.create_group(key)
    else:
        new_grp = mother_group[key]
    return new_grp

# String to dictionary
def str2dic(string):
    '''
    This function is used to load string dictionary and convert it into python dictionary.
    '''
    import yaml
    return yaml.load(string)
