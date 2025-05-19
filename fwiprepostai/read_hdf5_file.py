import numpy as np
import h5py
from skimage.measure import block_reduce

def get_data_from_hdf5(to_file, index):
    with h5py.File(to_file, 'r') as f:
        waveform_data = f['data_after_utils']['waveforms_r30'][()]
        waveform_data = waveform_data[index]
        images_vs_data = f['data_after_utils']['vs'][()][index]
        images_vs_data = truncate_read_files_vs(images_vs_data[None, ...], 60,0.25,5,55,25,0.25,0,20)
        images_vs_data = pooling_multi(images_vs_data, (4,4), 'mean')[0]
    return waveform_data, images_vs_data

def pooling_multi(array, kernel_size_2d=(2,2), method='max', dtype='float32'):
    if method==None:
        return array.astype(dtype)
    assert len(kernel_size_2d) == 2, f"kernel_size_2d should be 2-dimensional tuple. Eg.(2,2), provided dim={len(kernel_size_2d)}."
    assert method=='max' or method=='mean', f"method allowed is 'max' for maxpooling, and 'mean' for meanpooling."
    block_size = (1, kernel_size_2d[0], kernel_size_2d[1])
    if method=='mean':
        array_reduced=block_reduce(array, block_size=block_size, func=np.nanmean, cval=np.nan, func_kwargs={'dtype': np.float64})
    elif method == 'max':
        array_reduced=block_reduce(array, block_size=block_size, func=np.nanmax, cval=np.nan)
    return array_reduced.astype(dtype)
        
def truncate_read_files_vs(vs, y_lim, del_y, trunc_start_y, trunc_end_y, z_lim, del_z, trunc_start_z, trunc_end_z):
    trunc_y_index = get_trunc_y_or_z_index(y_lim, del_y, trunc_start_y, trunc_end_y)
    trunc_z_index = get_trunc_y_or_z_index(z_lim, del_z, trunc_start_z, trunc_end_z)
    return vs[:,:,trunc_y_index][:,trunc_z_index,:]

def get_trunc_y_or_z_index(y_or_z_lim, del_yz, trunc_start_yz, trunc_end_yz):
    curr_yz_index = np.arange(0, y_or_z_lim+del_yz/2, del_yz) 
    condition = (curr_yz_index >= trunc_start_yz) & (curr_yz_index < trunc_end_yz)  
    indices = np.argwhere(condition).flatten()
    return indices
