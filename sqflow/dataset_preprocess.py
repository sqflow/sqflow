import numpy as np

def get_records_with_h(arr_h, arr_t, arr_vof, arr_dp, h_mean=100e-9, h_var=10e-9):
    list_to_return = []
    for sn, elem in enumerate(arr_h):
        if elem[0]>h_mean-h_var and elem[0]<h_mean+h_var:
            list_to_return.append(sn)
    
    arr_t = arr_t[list_to_return]
    arr_h = arr_h[list_to_return]
    arr_vof = arr_vof[list_to_return]
    arr_dp = arr_dp[list_to_return]
    
    return arr_h, arr_t, arr_vof, arr_dp

def get_records_with_t(arr_h, arr_t, arr_vof, arr_dp, t_mean=10e-3, t_var=2e-3):
    list_to_return = []
    for sn, elem in enumerate(arr_t):
        if elem[0]>t_mean-t_var and elem[0]<t_mean+t_var:
            list_to_return.append(sn)
    
    arr_t = arr_t[list_to_return]
    arr_h = arr_h[list_to_return]
    arr_vof = arr_vof[list_to_return]
    arr_dp = arr_dp[list_to_return]
    
    return arr_h, arr_t, arr_vof, arr_dp

def get_records_with_fff(arr_h, arr_t, arr_vof, arr_dp, fff_max=0.5, fff_min=0):
    list_to_return = []
    for sn, elem in enumerate(arr_vof):
        this_fff = sum(elem)/arr_vof.shape[1]
        if this_fff>fff_min and this_fff<fff_max:
            list_to_return.append(sn)
    
    arr_t = arr_t[list_to_return]
    arr_h = arr_h[list_to_return]
    arr_vof = arr_vof[list_to_return]
    arr_dp = arr_dp[list_to_return]
    
    return arr_h, arr_t, arr_vof, arr_dp

def get_records_with_local_fff(
        arr_h, arr_t, arr_vof, arr_dp, 
        local_win_sz = 72, #size of interrogation window
        local_fff_max=0.9, #max of local field coverage for the interrogation window
        ):
    
    local_win_area = float(local_win_sz*local_win_sz)
    list_to_return = []
    for sn, elem in enumerate(arr_vof):
        flag = True
        sz_img = int(np.sqrt(np.prod(elem.shape)))
        this_img= elem.reshape(sz_img, sz_img)
        #moving the interrogation window across the image
        step_sz = max(4, int(sz_img/3))
        for ind_x in range(0, sz_img-local_win_sz, step_sz):
            for ind_y in range(0, sz_img-local_win_sz, step_sz):
                #calculating the local field coverage inside the interrogation window
                this_fff_count = sum(sum(this_img[ind_x:ind_x+local_win_sz,
                                        ind_y:ind_y+local_win_sz])).astype('float32')
                this_fff = this_fff_count/local_win_area
                if this_fff>local_fff_max:
                    flag = False
                    break
            if not flag:
                break
        
        if flag:
            list_to_return.append(sn)
            
    arr_t = arr_t[list_to_return]
    arr_h = arr_h[list_to_return]
    arr_vof = arr_vof[list_to_return]
    arr_dp = arr_dp[list_to_return]
    
    return arr_h, arr_t, arr_vof, arr_dp

def del_small_t_records(arr_h, arr_t, arr_vof, arr_dp, small_time_threshold=1e-9):
    list_to_del = []
    for sn, elem in enumerate(arr_t):
        if elem[0]<small_time_threshold:
            list_to_del.append(sn)
    arr_t = np.delete(arr_t, list_to_del, axis=0)
    arr_h = np.delete(arr_h, list_to_del, axis=0)
    arr_vof = np.delete(arr_vof, list_to_del, axis=0)
    arr_dp = np.delete(arr_dp, list_to_del, axis=0)
    return arr_h, arr_t, arr_vof, arr_dp

def normalize(arr, mean=0., std=1.):
    this_mean = np.mean(arr, axis=0)[0]
    this_std = np.std(arr, axis=0)[0]
    return this_mean, this_std, mean+std*(arr-this_mean)/this_std

def transform(arr, mean, std):
    return (arr-mean)/std
