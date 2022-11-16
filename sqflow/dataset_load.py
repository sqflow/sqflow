import os
import numpy as np
import csv

def load_2d_array(fname='test.csv', val_to_retrieve=None):
    if val_to_retrieve is None:
        return np.array(list(csv.reader(open(fname)))).astype("float32")

    else:
        reader = csv.reader(open(fname, "r"), delimiter=",")
        locs = list(reader)
        m = int(locs[0][1])
        n = int(locs[0][2])
        a_whole = np.zeros((m,n)).astype("float32")
        
        ind_row = -1
        for row in locs[1:]:
            ind_row +=1
            for ind_elem in row:
                a_whole[ind_row][int(ind_elem)] = 1
        
        return a_whole
    
def load_dataset_from_dir(dataset_dir):
    this_h = load_2d_array(fname=os.path.join(dataset_dir, 'h.csv'), val_to_retrieve=None)
    this_t = load_2d_array(fname=os.path.join(dataset_dir, 't.csv'), val_to_retrieve=None)
    this_dp = load_2d_array(fname=os.path.join(dataset_dir, 'dp.csv'), val_to_retrieve=1)
    this_vof = load_2d_array(fname=os.path.join(dataset_dir, 'vof.csv'), val_to_retrieve=1)
    
    return this_h, this_t, this_vof, this_dp

def walk_through_dir_and_load_datasets(
    dir_base,
    fname_base = 'vof',
    fname_suffix = '.csv',
    verbose=False,
):

    x_extra_param = np.array([])
    x_field = np.array([])
    y_extra_param = np.array([])
    y_field = np.array([])

    list_files = []
    this_iter = 0
    for subdir, dirs, files in os.walk(dir_base):
        this_iter += 1
        if verbose:
            print('\n')
            print(f'processing directory {this_iter}\t{subdir}\t (dirs, files): ({len(dirs)}, {len(files)})')
        for file in files:
            if os.path.isfile(os.path.join(subdir, file)) and file.startswith(fname_base) and file.endswith(fname_suffix):
                list_files.append(os.path.join(subdir, file))
    
    arr_h = np.array([])
    arr_t = np.array([])
    arr_vof = np.array([])
    arr_dp = np.array([])
    
    if len(list_files)>0:
        if verbose:
            print('-'*100)
            print('walking through dataset directories...')
            print('-'*100)
        for this_file in list_files:
            this_dir = os.path.dirname(os.path.abspath(this_file))
            if verbose:
                print(f'processing directory {this_dir}...')
            this_h, this_t, this_vof, this_dp = load_dataset_from_dir(dataset_dir=this_dir)

            if arr_h.size==0:
                arr_h = np.copy(this_h)
                arr_t = np.copy(this_t)
                arr_vof = np.copy(this_vof)
                arr_dp = np.copy(this_dp)

            else:
                arr_h = np.append(arr_h, this_h, axis=0)
                arr_t = np.append(arr_t, this_t, axis=0)
                arr_vof = np.append(arr_vof, this_vof, axis=0)
                arr_dp = np.append(arr_dp, this_dp, axis=0)
        
        return arr_h, arr_t, arr_vof, arr_dp
    else:
        print('no dataset found.')
        return None, None, None, None