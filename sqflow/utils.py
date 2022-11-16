import os

def get_fname(fname_w_path):
    """
    gets a file/folder name and returns the same if does not exist. 
    Otherwise, appends a 3-digit identifier at the end of the name and returns the result.
    """
    filename, extension = os.path.splitext(fname_w_path)
    counter = 0
    res = fname_w_path
    while os.path.exists(res):
        res = filename + "_" + '{:03d}'.format(counter,) + extension
        counter += 1
    return res