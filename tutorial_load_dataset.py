from sqflow.dataset_process import dataset_process as dsp

# Loading a small toy dataset consisting of 50 examples from Category #1 (each example includes one droplet in its droplet pattern image) for each of training, validation, and test datasets:
dataset_dir_train = r'./sqflow/dataset/toy_cat_01/train'
dataset_dir_val = r'./sqflow/dataset/toy_cat_01/val'
dataset_dir_test = r'./sqflow/dataset/toy_cat_01/test'

#-----------------------------------------
# loading dataset files
#-----------------------------------------
dataset = dsp(
    dataset_dir_train=dataset_dir_train,
    dataset_dir_val=dataset_dir_val,
    dataset_dir_test=dataset_dir_test,
    opt_confine_local_fff=True, #to limit max of local coverage 
    local_win_sz=72, #size of interrogation window
    local_fff_max=0.9,
    verbose=True,
)

dataset_with_concat_train_val = dataset.concat_shuffle()
validation_split=dataset.validation_split

"""
Concatenated train_val dataset & test dataset
"""
train_val_h_norm = dataset_with_concat_train_val['train_val']['h_norm']
train_val_t_norm = dataset_with_concat_train_val['train_val']['t_norm']
train_val_vof = dataset_with_concat_train_val['train_val']['vof']
train_val_dp = dataset_with_concat_train_val['train_val']['dp']

test_h_norm = dataset_with_concat_train_val['test']['h_norm']
test_t_norm = dataset_with_concat_train_val['test']['t_norm']
test_vof = dataset_with_concat_train_val['test']['vof']
test_dp = dataset_with_concat_train_val['test']['dp']

"""
Unconcatenated train and val datasets
"""
train_h_norm = dataset.dataset['train']['h_norm']
train_t_norm = dataset.dataset['train']['t_norm']
train_vof = dataset.dataset['train']['vof']
train_dp = dataset.dataset['train']['dp']

val_h_norm = dataset.dataset['val']['h_norm']
val_t_norm = dataset.dataset['val']['t_norm']
val_vof = dataset.dataset['val']['vof']
val_dp = dataset.dataset['val']['dp']

print('size of dataset -- train:', dataset.len_train)
print('size of dataset -- val:', dataset.len_val)
print('size of dataset -- test:', dataset.len_test)
print('\n')
print('train_val:')
print('h_norm:', train_val_h_norm.shape)
print('t_norm', train_val_t_norm.shape)
print('vof:', train_val_vof.shape)
print('dp:', train_val_dp.shape)
print('\n')
print('\ntest:')
print('h_norm:', test_h_norm.shape)
print('t_norm', test_t_norm.shape)
print('vof:', test_vof.shape)
print('dp:', test_dp.shape)

