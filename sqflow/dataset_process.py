import numpy as np

from .dataset_load import walk_through_dir_and_load_datasets
from .dataset_preprocess import get_records_with_h, get_records_with_t, \
    get_records_with_fff, get_records_with_local_fff, del_small_t_records, \
        normalize, transform


class dataset_process():
    def __init__(
            self,
            dataset_dir_train=None, 
            dataset_dir_val=None,
            dataset_dir_test=None, 
            opt_confine_film_thickness=False,
            h_mean_limit = 100e-9, 
            h_var_limit=10e-9,
            opt_confine_time=False,
            t_mean_limit = 5e-3, 
            t_var_limit=1e-3,
            opt_confine_fff=False,
            fff_max_limit = 0.5, 
            fff_min_limit=0,
            opt_omit_small_time=True,
            small_time_threshold = 1e-9,
            opt_confine_local_fff=False, #to limit max of local coverage 
            local_win_sz=72, #size of interrogation window
            local_fff_max=0.9, #max of local field coverage for the interrogation window
            opt_write_norm_stat=False,
            opt_shuf=False, 
            verbose=True,
            **kwargs,
            ):
        self.dataset_dir_train = dataset_dir_train
        self.dataset_dir_val = dataset_dir_val
        self.dataset_dir_test = dataset_dir_test
        
        self.opt_confine_film_thickness=opt_confine_film_thickness
        self.h_mean_limit=h_mean_limit
        self.h_var_limit=h_var_limit
        
        self.opt_confine_time=opt_confine_time
        self.t_mean_limit=t_mean_limit
        self.t_var_limit=t_var_limit
        
        self.opt_confine_fff=opt_confine_fff
        self.fff_max_limit=fff_max_limit
        self.fff_min_limit=fff_min_limit
        
        self.opt_omit_small_time=opt_omit_small_time
        self.small_time_threshold=small_time_threshold
        
        self.opt_confine_local_fff=opt_confine_local_fff
        self.local_win_sz=local_win_sz
        self.local_fff_max=local_fff_max
        
        self.opt_write_norm_stat=opt_write_norm_stat
        self.verbose=verbose
        self.opt_shuf=opt_shuf
                
        self.dataset=None
        self.dataset_with_concat_train_val=None
   
    
    def load(self,**kwargs,):
        
        dataset_dir_train=kwargs.pop("dataset_dir_train", self.dataset_dir_train)
        dataset_dir_val=kwargs.pop("dataset_dir_val", self.dataset_dir_val)
        dataset_dir_test=kwargs.pop("dataset_dir_test", self.dataset_dir_test)

        
        opt_confine_film_thickness=kwargs.pop("opt_confine_film_thickness", self.opt_confine_film_thickness)
        h_mean_limit=kwargs.pop("h_mean_limit", self.h_mean_limit)
        h_var_limit=kwargs.pop("h_var_limit", self.h_var_limit)
        
        opt_confine_time=kwargs.pop("opt_confine_time", self.opt_confine_time)
        t_mean_limit=kwargs.pop("t_mean_limit", self.t_mean_limit)
        t_var_limit=kwargs.pop("t_var_limit", self.t_var_limit)
        
        opt_confine_fff=kwargs.pop("opt_confine_fff", self.opt_confine_fff)
        fff_max_limit=kwargs.pop("fff_max_limit", self.fff_max_limit)
        fff_min_limit=kwargs.pop("fff_min_limit", self.fff_min_limit)
        
        opt_confine_local_fff=kwargs.pop("opt_confine_local_fff", self.opt_confine_local_fff)
        local_win_sz=kwargs.pop("local_win_sz", self.local_win_sz)
        local_fff_max=kwargs.pop("local_fff_max", self.local_fff_max)
        
        opt_omit_small_time=kwargs.pop("opt_omit_small_time", self.opt_omit_small_time)
        small_time_threshold=kwargs.pop("small_time_threshold", self.small_time_threshold)
        
        opt_write_norm_stat=kwargs.pop("opt_write_norm_stat", self.opt_write_norm_stat)
        
        verbose=kwargs.pop("verbose", self.verbose)
          
        if verbose:
            print('beginning of loading train dataset from the root directory:', dataset_dir_train)
        train_h_full, train_t_full, train_vof_full, train_dp_full = walk_through_dir_and_load_datasets(
            dir_base=dataset_dir_train, verbose=verbose)

        if verbose:
            print('beginning of loading val dataset from the root directory:', dataset_dir_val)
        val_h_full, val_t_full, val_vof_full, val_dp_full = walk_through_dir_and_load_datasets(
            dir_base=dataset_dir_val, verbose=verbose)

        if verbose:
            print('beginning of loading test dataset from the root directory:', dataset_dir_test)
        test_h_full, test_t_full, test_vof_full, test_dp_full = walk_through_dir_and_load_datasets(
            dir_base=dataset_dir_test, verbose=verbose)
        
        if verbose:
            print('loading ended.')
            
        if verbose:
            print('beginning of preprocessing #1 ...')
            
        #removing unwanted records
        train_h=train_h_full.copy()
        train_t=train_t_full.copy()
        train_vof=train_vof_full.copy()
        train_dp=train_dp_full.copy()
        
        val_h=val_h_full.copy()
        val_t=val_t_full.copy()
        val_vof=val_vof_full.copy()
        val_dp=val_dp_full.copy()
        
        test_h=test_h_full.copy()
        test_t=test_t_full.copy()
        test_vof=test_vof_full.copy()
        test_dp=test_dp_full.copy()
        
        if opt_confine_film_thickness:
            train_h, train_t, train_vof, train_dp = get_records_with_h(train_h_full, train_t_full, train_vof_full, train_dp_full, h_mean=h_mean_limit, h_var=h_var_limit)
            val_h, val_t, val_vof, val_dp = get_records_with_h(val_h_full, val_t_full, val_vof_full, val_dp_full, h_mean=h_mean_limit, h_var=h_var_limit)
            test_h, test_t, test_vof, test_dp = get_records_with_h(test_h_full, test_t_full, test_vof_full, test_dp_full, h_mean=h_mean_limit, h_var=h_var_limit)
            
        if opt_confine_time:
            train_h, train_t, train_vof, train_dp = get_records_with_t(train_h_full, train_t_full, train_vof_full, train_dp_full, t_mean=t_mean_limit, t_var=t_var_limit)
            val_h, val_t, val_vof, val_dp = get_records_with_t(val_h_full, val_t_full, val_vof_full, val_dp_full, t_mean=t_mean_limit, t_var=t_var_limit)
            test_h, test_t, test_vof, test_dp = get_records_with_t(test_h_full, test_t_full, test_vof_full, test_dp_full, t_mean=t_mean_limit, t_var=t_var_limit)
            
        if opt_confine_fff:
            train_h, train_t, train_vof, train_dp = get_records_with_fff(train_h_full, train_t_full, train_vof_full, train_dp_full, fff_max=fff_max_limit, fff_min=fff_min_limit)
            val_h, val_t, val_vof, val_dp = get_records_with_fff(val_h_full, val_t_full, val_vof_full, val_dp_full, fff_max=fff_max_limit, fff_min=fff_min_limit)
            test_h, test_t, test_vof, test_dp = get_records_with_fff(test_h_full, test_t_full, test_vof_full, test_dp_full, fff_max=fff_max_limit, fff_min=fff_min_limit)    
        
        if opt_confine_local_fff:
            train_h, train_t, train_vof, train_dp = get_records_with_local_fff(
                arr_h = train_h, 
                arr_t = train_t, 
                arr_vof = train_vof, 
                arr_dp = train_dp, 
                local_win_sz = local_win_sz, 
                local_fff_max=local_fff_max, 
                )
            
            val_h, val_t, val_vof, val_dp = get_records_with_local_fff(
                arr_h = val_h, 
                arr_t = val_t, 
                arr_vof = val_vof, 
                arr_dp = val_dp, 
                local_win_sz = local_win_sz, 
                local_fff_max=local_fff_max, 
                )
            
            test_h, test_t, test_vof, test_dp = get_records_with_local_fff(
                arr_h = test_h, 
                arr_t = test_t, 
                arr_vof = test_vof, 
                arr_dp = test_dp, 
                local_win_sz = local_win_sz, 
                local_fff_max=local_fff_max, 
                )            
            
        if opt_omit_small_time:
            train_h, train_t, train_vof, train_dp = del_small_t_records(
                arr_h = train_h, 
                arr_t = train_t, 
                arr_vof = train_vof, 
                arr_dp = train_dp, 
                small_time_threshold = small_time_threshold)

            val_h, val_t, val_vof, val_dp = del_small_t_records(
                arr_h = val_h, 
                arr_t = val_t, 
                arr_vof = val_vof, 
                arr_dp = val_dp, 
                small_time_threshold = small_time_threshold)
            
            test_h, test_t, test_vof, test_dp = del_small_t_records(
                arr_h = test_h, 
                arr_t = test_t, 
                arr_vof = test_vof, 
                arr_dp = test_dp, 
                small_time_threshold = small_time_threshold)
            
        #normalization
        mean_t, std_t, train_t_norm = normalize(np.log(train_t), mean=0., std=1.)
        mean_h, std_h, train_h_norm = normalize(np.log(train_h), mean=0., std=1.)
        
        #saving the norm. stat
        self.train_mean_log_t = mean_t
        self.train_std_log_t = std_t
        
        self.train_mean_log_h = mean_h
        self.train_std_log_h = std_h
        
        if verbose:
            print(f'train mean log_t: \t{mean_t}\n')
            print(f'train std log_t: \t{std_t}\n')
            
            print(f'train mean log_h: \t{mean_h}\n')
            print(f'train std log_h: \t{std_h}\n')
            
        if opt_write_norm_stat:
		    #writing the norm. stat to a file
            f = open('train_norm_stat.txt', "w")
            f.write(f'train mean log_t: \t{mean_t}\n')
            f.write(f'train std log_t: \t{std_t}\n')
			
            f.write(f'train mean log_h: \t{mean_h}\n')
            f.write(f'train std log_h: \t{std_h}\n')
            f.close()
        
        #trans. of val and test datasets
        val_t_norm = transform(np.log(val_t), mean=mean_t, std=std_t)
        val_h_norm = transform(np.log(val_h), mean=mean_h, std=std_h)
        
        test_t_norm = transform(np.log(test_t), mean=mean_t, std=std_t)
        test_h_norm = transform(np.log(test_h), mean=mean_h, std=std_h)
        
        #preparing the dataset
        self.dataset={'train':None, 'val':None, 'test':None}
        self.dataset['train'] = {
            'h_norm': train_h_norm, 
            't_norm':train_t_norm, 
            'vof':train_vof, 
            'dp':train_dp}
 
        self.dataset['val'] = {
            'h_norm': val_h_norm, 
            't_norm':val_t_norm, 
            'vof':val_vof, 
            'dp':val_dp}
 
        self.dataset['test'] = {
            'h_norm': test_h_norm, 
            't_norm':test_t_norm, 
            'vof':test_vof, 
            'dp':test_dp}
        
        #size of dataset
        self.len_train = self.dataset['train']['vof'].shape[0]
        self.len_val = self.dataset['val']['vof'].shape[0]
        self.len_test = self.dataset['test']['vof'].shape[0]
        self.validation_split = self.len_val/(self.len_val+self.len_train)
        
        
        if verbose:
            print('preprocessing #1 ended')
        
        return self.dataset
        
    def concat_shuffle(self, **kwargs,):
        dataset=kwargs.pop("dataset", self.dataset)
        verbose=kwargs.pop("verbose", self.verbose)
        opt_shuf=kwargs.pop("opt_shuf", self.opt_shuf)
                
        if not dataset:
            if verbose:
                print('Dataset needs to be loaded first.')
            dataset=self.load()
        
        if verbose:
            print('preprocessing #2 starting...')
            
        train_h_norm = dataset['train']['h_norm']
        train_t_norm = dataset['train']['t_norm']
        train_vof = dataset['train']['vof']
        train_dp = dataset['train']['dp']
        
        val_h_norm = dataset['val']['h_norm']
        val_t_norm = dataset['val']['t_norm']
        val_vof = dataset['val']['vof']
        val_dp = dataset['val']['dp']
        
        test_h_norm = dataset['test']['h_norm']
        test_t_norm = dataset['test']['t_norm']
        test_vof = dataset['test']['vof']
        test_dp = dataset['test']['dp']
        
        #concat train and val
        train_val_h_norm = np.concatenate((train_h_norm, val_h_norm), axis=0)
        train_val_t_norm = np.concatenate((train_t_norm, val_t_norm), axis=0)
        train_val_vof = np.concatenate((train_vof, val_vof), axis=0)
        train_val_dp = np.concatenate((train_dp, val_dp), axis=0)
        
        
        # concatenating x and y for shuffling
        concat_xy_train=np.concatenate((train_h_norm, train_vof, train_t_norm, train_dp), axis=1)
        concat_xy_val=np.concatenate((val_h_norm, val_vof, val_t_norm, val_dp), axis=1)
        concat_xy_test=np.concatenate((test_h_norm, test_vof, test_t_norm, test_dp), axis=1)    
        
        if opt_shuf:
            # shuffling xy datasets
            np.random.shuffle(concat_xy_train)
            np.random.shuffle(concat_xy_val)
            np.random.shuffle(concat_xy_test)
            
            # retrieving training, validation, and test shuffled datasets 
            train_img_pxls = train_vof.shape[1]
            
            shuf_train_h_norm = concat_xy_train[:,0]
            shuf_train_vof = concat_xy_train[:,1:train_img_pxls+1]
            shuf_train_t_norm = concat_xy_train[:,train_img_pxls+1]
            shuf_train_dp = concat_xy_train[:,train_img_pxls+2:]
            
            shuf_val_h_norm = concat_xy_val[:,0]
            shuf_val_vof = concat_xy_val[:,1:train_img_pxls+1]
            shuf_val_t_norm = concat_xy_val[:,train_img_pxls+1]
            shuf_val_dp = concat_xy_val[:,train_img_pxls+2:]
            
            shuf_test_h_norm = concat_xy_test[:,0]
            shuf_test_vof = concat_xy_test[:,1:train_img_pxls+1]
            shuf_test_t_norm = concat_xy_test[:,train_img_pxls+1]
            shuf_test_dp = concat_xy_test[:,train_img_pxls+2:]
            
            # Concatenation of training and validation datasets
            shuf_train_val_h_norm = np.concatenate((shuf_train_h_norm, shuf_val_h_norm), axis=0)
            shuf_train_val_t_norm = np.concatenate((shuf_train_t_norm, shuf_val_t_norm), axis=0)
            shuf_train_val_vof = np.concatenate((shuf_train_vof, shuf_val_vof), axis=0)
            shuf_train_val_dp = np.concatenate((shuf_train_dp, shuf_val_dp), axis=0)
        
        #preparing the shuffled dataset
        self.dataset_with_concat_train_val={'train_val':None, 'test':None} 
        if opt_shuf:    
            self.dataset_with_concat_train_val['train_val'] = {
                'h_norm': shuf_train_val_h_norm, 
                't_norm': shuf_train_val_t_norm, 
                'vof': shuf_train_val_vof, 
                'dp': shuf_train_val_dp}
            
            self.dataset_with_concat_train_val['test'] = {
                'h_norm': shuf_test_h_norm, 
                't_norm':shuf_test_t_norm, 
                'vof':shuf_test_vof, 
                'dp':shuf_test_dp}
        else:
            self.dataset_with_concat_train_val['train_val'] = {
                'h_norm': train_val_h_norm, 
                't_norm': train_val_t_norm, 
                'vof': train_val_vof, 
                'dp': train_val_dp}
            
            self.dataset_with_concat_train_val['test'] = {
                'h_norm': test_h_norm, 
                't_norm':test_t_norm, 
                'vof':test_vof, 
                'dp':test_dp}
     
        
        if verbose:
            print('preprocessing #2 ended')
            
        return self.dataset_with_concat_train_val
        