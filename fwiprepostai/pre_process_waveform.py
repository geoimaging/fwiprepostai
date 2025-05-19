# COPY FUNCTIONS DIRECTLY FROM read_batch_v8, but only required for post-training

import copy
import numpy as np
import scipy

def zero_padding_and_trim_each(waveform_np, freq=1000, pad_time=0, upto_time=None):
    #upto_time is in ms
    batch_size = np.shape(waveform_np)[0]
    org_waveform_len = np.shape(waveform_np)[2]
    
    if upto_time is not None:
        org_waveform_len = int(upto_time * 1000/freq)
        
    if isinstance(pad_time, list):
        if len(pad_time)!=batch_size:
            raise Exception(f"ERROR: len of pad_time {len(pad_time)} is not equal to batch_size{batch_size}")
        paddings = [int(i*freq) for i in pad_time]
    else: 
        if  pad_time<0:
            raise exception("pad time should be greater or equal to 0")
        paddings = [int(pad_time*freq) for _ in range(batch_size)]
    
    zero_prefixed_waveforms = []
 
    for i in range(batch_size):
        random_padding = paddings[i]
        padded_waveform = np.pad(waveform_np[i], ((0, 0), (random_padding, 0)), mode='constant')  #Removed last , (0, 0)
        zero_prefixed_waveforms.append(padded_waveform[:, :org_waveform_len])
    zero_prefixed_waveforms = np.stack(zero_prefixed_waveforms)
    
    return zero_prefixed_waveforms
    
def process_waveforms_images(waveforms_time_np, images_np, norm_x, norm_x_fft, norm_vs, fft_option, f_type, dc_correction, cosine_taper_ms, order_filter, order_norm, order_fft, filter_wn, norm_x_excl_receivers=[], fft_size_eq_time=True, norm_each_time_domain=False, detailed_print=False, summarize_head='no_summary'):
    #v8: Changes "bp_filter", "norm_before_fft" changed to order_filter, order_fft, order_norm
    
    assert fft_option in ['abs', 'real', 'imag', 'angle', 'none', 'real_imag', 'abs_angle', 'time_abs'], f"not {fft_option}"
    
    waveform_dim = waveforms_time_np.shape
    
    string = 'Augment (if any) ->'
    
    #Step 1: DC Correction: (Mean Correction)
    if dc_correction:
        waveforms_time_np = get_dc_corrected(waveforms_time_np)
        string = string + 'DC correction -> '
    
    #Step 2: Taper correction
    assert cosine_taper_ms >= 0
    if cosine_taper_ms>0:
        window = cosine_taper(waveform_dim[2], cosine_taper_ms*2/waveform_dim[2])
        waveforms_time_np*=window[np.newaxis, np.newaxis, :]
        string = string + 'Taper -> '
    
    #Step3: Filter, norm, fft
    order_fnf = check_get_orders(order_filter, order_norm, order_fft)
    
    for calc in [i for i in order_fnf.values()]:
        if calc == 'filter': #Always before fft. so always waveform_time_np will be of ndim of 3
            my_kwargs = {'filter_wn': filter_wn}
        
        elif calc == 'fft':
            my_kwargs = {'fft_size_eq_time': fft_size_eq_time, 'detailed_print':detailed_print,
                         'f_type':f_type, 'fft_option':fft_option}
            
        elif calc == 'norm':
            my_kwargs = {'norm_vs': norm_vs, 'norm_x':norm_x, 'f_type':f_type,
                         'norm_x_fft':norm_x_fft, 'fft_index':fft_index, 'norm_x_excl_receivers':norm_x_excl_receivers}
        
        waveforms_time_np, images_np, fft_index, string = do_filter_norm_fft(waveforms_time_np, images_np, calc, string, **my_kwargs)

        
    #Step4: Additional
    if norm_each_time_domain:
        waveforms_t_only = waveforms_time_np[:,:,:,0]
        min_arr = np.min(np.abs(waveforms_t_only), axis = (-1), keepdims=True)
        max_arr = np.max(np.abs(waveforms_t_only), axis = (-1), keepdims=True)
        waveforms_t_only=(waveforms_t_only - min_arr)/(max_arr-min_arr)
        waveforms_time_np[...,0] = waveforms_t_only
        string = string + 'Norm_Each_Time_Domain -> '
        
    if np.any(np.isnan(waveforms_time_np)):
        print("NAN found in the input")
    if np.any(np.isinf(waveforms_time_np)):
        print("INF found in the input")

    if summarize_head != 'no_summary':
        summarize_load_Dataset(waveforms_time_np, images_np, summarize_head)
    return waveforms_time_np, images_np, string

def exclude_reciever_in_array(array, exclude_indices):
    if len(exclude_indices)==0:
        sliced_array = array
    else:
        all_indices = list(range(array.shape[-2]))
        assert all(index in all_indices for index in exclude_indices), f"Not all indices in exclude_indices are valid: NOT ALL of {exclude_indices} in {all_indices}."
        include_indices = [i for i in all_indices if i not in exclude_indices]
        # Select the slices from the original array using the list of included indices
        sliced_array = copy.deepcopy(array[:, include_indices, :])
    return sliced_array

def data_norm_max_min(array, norm_dict, f_type, norm_x_excl_receivers=[]):
    if norm_dict['norm'] == None or norm_dict['norm'] == False:
        max_val = 1
        min_val = 0
        return array
    else:
        if norm_dict['max'] == 'max_abs':
            sliced_array = exclude_reciever_in_array(array, exclude_indices=norm_x_excl_receivers) #[1, 2, 3]
            max_val = np.max(np.abs(sliced_array), axis = (1,2), keepdims=True)

        elif norm_dict['max'] == 'max_val':
            sliced_array = exclude_reciever_in_array(array, exclude_indices=norm_x_excl_receivers) #[1, 2, 3]
            max_val = np.max(sliced_array, axis = (1,2), keepdims=True)
        else:
            max_val = np.array(norm_dict['max'])

        if norm_dict['min'] == 'min_abs':
            sliced_array = exclude_reciever_in_array(array, exclude_indices=norm_x_excl_receivers) #[1, 2, 3]
            min_val = np.min(np.abs(sliced_array), axis = (1,2), keepdims=True)
        elif norm_dict['min'] == 'min_val':
            sliced_array = exclude_reciever_in_array(array, exclude_indices=norm_x_excl_receivers) #[1, 2, 3]
            min_val = np.min(sliced_array, axis = (1,2), keepdims=True)
        else:
            min_val = norm_dict['min']
    norm_array = (array-min_val)/(max_val-min_val)
    return norm_array.astype(f_type)
    
    
def data_norm_min_max_reverse(array, max_val, min_val):
    return (array * (max_val-min_val) + min_val)

def get_wn(filter_wn):
    #filter_wn = {'min':[20], 'max':[20,100]}
    a = filter_wn[0]
    if len(a) == 1:
        a = a[0]
    elif len(a) == 2:
        a = np.random.uniform(a[0], a[1])
    else:
        raise Exception ("error format for filter_wn")
    filt_min = a
    
    a = filter_wn[1]
    if len(a) == 1:
        a = a[0]
    elif len(a) == 2:
        a = np.random.uniform(a[0], a[1])
    else:
        raise Exception ("error format for filter_wn")
    filt_max = a
    
    return [filt_min, filt_max]

def get_dc_corrected(data_np):
    assert data_np.ndim == 3
    mean = np.mean(data_np, axis=(2), keepdims=True)
    mean_corrected_data = data_np - mean
    return mean_corrected_data

def cosine_taper(n, p):
    """
    Create a cosine taper window.
    
    Parameters:
    - n: Length of the time series.
    - p: Proportion of the series to taper at each end (0 < p < 0.5).
    
    Returns:
    - window: The cosine taper window.
    """
    if p <= 0 or p >= 0.5:
        raise ValueError("Proportion p must be between 0 and 0.5.")
    
    taper_length = int(p * n)
    taper = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_length)))
    window = np.ones(n)
    window[:taper_length] = taper
    window[-taper_length:] = taper[::-1]
    return window

def check_get_orders(order_filter, order_norm, order_fft):
    assert order_filter in [0, 1, 2], f"order_filter {order_filter} is not in [0, 1, 2]. 3 is not possible as always order_fft>order_filter"
    assert order_norm in [0, 1, 2, 3], f"order_norm {order_norm} is not in [0, 1, 2, 3]"
    assert order_fft in [0, 1, 2, 3], f"order_fft {order_fft} is not in [0, 1, 2, 3]"
    if order_fft!=0:
        assert order_fft > order_filter, f"fft (order: {order_fft}) cannot be done earlier than filter (order:{order_filter}). Doesn't make sense." 
    
    order_fnf = {}
    order_rev = {'filter':order_filter, 
                 'norm': order_norm,
                 'fft': order_fft}
    keys = [i for i in order_rev.keys()]
    values = np.array([i for i in order_rev.values()])
    sorted_indices = np.argsort(values)
    
    i=1
    for key_id in sorted_indices:
        key = keys[key_id]
        #print(key)
        value = order_rev[key]
        if value!=0:
            if value in order_fnf.keys():
                raise Exception(f'Repetitive order number ({value}): Check order_filter, order_norm, and order_fft')
            if value!=i:
                raise Exception(f'Missing {i} in order numbers. Incorrect order numbers.  Check order_filter, order_norm, and order_fft')
            order_fnf[i] = key
            i+=1
    return order_fnf

def do_filter_norm_fft(waveforms_time_np, images_np, calc, string = '', **kwargs):
    #print(f"do_filter_norm_fft: shape: {waveforms_time_np.shape}")
    #waveform shape = (1,24,1000)
    
    n_recievers = np.shape(waveforms_time_np)[1]
    org_waveform_len = np.shape(waveforms_time_np)[2]
    if calc == 'filter': #Always before fft. so always waveform_time_np will be of ndim of 3
        assert waveforms_time_np.ndim == 3
        
        filter_wn = kwargs['filter_wn']
        assert np.array(filter_wn).shape == (2,) or np.array(filter_wn).shape == (2,2), f'filter_wn should a list with shape of either (2,) (eg [wn_low, wn_high] - constant filter) or (2,2) (eg [[wn_low1, wn_low2],[wn_high1, wnhigh2]], such that uniform random selection is done in given range for wn_low and wn_high. But provided {np.array(filter_wn).shape}'
        
        order=5
        
        if np.array(filter_wn).shape == (2,):
            wn = filter_wn
        else:
            wn = get_wn(filter_wn)
        sos = scipy.signal.butter(N=order, Wn=wn, btype = 'bandpass', fs=1000, output='sos')
        waveforms_fft_processed= scipy.signal.sosfiltfilt(sos,waveforms_time_np, axis=-1) #Adjusted for 4dims (No need as ndim is always 3)
        
        string = string + 'filter -> '
        fft_index = None #Only for fft
        
    elif calc == 'fft':
        assert waveforms_time_np.ndim == 3
        
        fft_size_eq_time = kwargs['fft_size_eq_time']
        detailed_print = kwargs['detailed_print']
        f_type = kwargs['f_type']
        fft_option = kwargs['fft_option']
        
        if fft_size_eq_time:
            n = waveforms_time_np.shape[-1]*2-1
        else:
            n = waveforms_time_np.shape[-1]
        waveforms_fft_processed, fft_index = fourier_transform(waveforms_time_np, n, detailed_print, f_type, fft_option)
        string = string + f'fft : {fft_option} -> '
        
    elif calc == 'norm':
        norm_vs = kwargs['norm_vs']
        norm_x = kwargs['norm_x']
        norm_x_fft = kwargs['norm_x_fft']
        f_type = kwargs['f_type']
        fft_index = kwargs['fft_index']
        norm_x_excl_receivers = kwargs['norm_x_excl_receivers']
        
        images_np = data_norm_max_min(images_np, norm_vs, f_type)
        
        if waveforms_time_np.ndim == 3: #Means fft is not done
            waveforms_fft_processed = data_norm_max_min(waveforms_time_np, norm_x, f_type, norm_x_excl_receivers)
            string = string + 'norm (freq domain not possible) -> '
        else:
            assert waveforms_time_np.ndim == 4 #fft is done
            for i,ind in enumerate(fft_index):
                assert ind in [0,1]
                if ind == 0: #Normal
                    waveforms_fft_processed[..., i] = data_norm_max_min(waveforms_fft_processed[..., i], norm_x, f_type, norm_x_excl_receivers)
                    #print(i)
                    #print(norm_x)
                else:
                    waveforms_fft_processed[..., i] = data_norm_max_min(waveforms_fft_processed[..., i], norm_x_fft, f_type)
                    #print(i)
                    #print(norm_x_fft)
                    
            string = string + 'norm -> '
            
        fft_index = None #Only for fft
    
    return waveforms_fft_processed, images_np, fft_index, string
    
def rfft(amplitude, **kwargs):
    n = kwargs.get("n", amplitude.shape[-1])
    rfft = np.fft.rfft(amplitude, **kwargs)
    rfft *= 2/min(amplitude.shape[-1], n)
    return rfft

def fourier_transform(array, n,  detailed_print, f_type='float64', option=None):
    
    assert option in ['abs', 'real', 'imag', 'angle', 'none', 'real_imag', 'abs_angle', 'time_abs'], f"not {option}"
    # Note array is of any; fourier transform will be done on -1
 
    if option == 'none':
        if detailed_print:
            print("fft = No transformation")
        return_array = np.zeros(array.shape +(1,)).astype(f_type)
        return_array[...,0] = array.astype(f_type)
        fft_index = [0]  #0 means no fft, and 1 means ffted.
    else:
        fft = rfft(array, n=n)
        if option=='abs':
            if detailed_print:
                print("fft = abs")
            return_array = np.zeros(array.shape +(1,)).astype(f_type)
            f1 = np.absolute(fft)
            return_array[...,:f1.shape[-1],0]=f1.astype(f_type)
            fft_index = [1]
        elif option=='real':
            if detailed_print:
                print("fft = real")
            return_array = np.zeros(array.shape +(1,)).astype(f_type)
            f1 = np.real(fft)
            return_array[...,:f1.shape[-1],0]=f1.astype(f_type)
            fft_index = [1]
        elif option=='imag':
            if detailed_print:
                print("fft = imag")
            return_array = np.zeros(array.shape +(1,)).astype(f_type)
            f1 = np.imag(fft)
            return_array[...,:f1.shape[-1],0]=f1.astype(f_type)
            fft_index = [1]
        elif option =='angle':
            if detailed_print:
                print("fft = angle")
            f1 = np.zeros(array.shape +(1,))   ### WTH
            return_array[...,:f1.shape[-1],0]=f1.astype(f_type)
            fft_index = [1]
        elif option == 'real_imag':
            if detailed_print:
                print("fft = no + real+ imag")
                print('corrected')
            return_array = np.zeros(array.shape +(3,)).astype(f_type)
            return_array[...,0] = array.astype(f_type)
            f1 = np.real(fft)
            return_array[...,:f1.shape[-1],1]=f1.astype(f_type)
            f2 = np.imag(fft)
            return_array[...,:f2.shape[-1],2]=f2.astype(f_type)
            fft_index = [0,1,1]
        elif option == 'abs_angle':
            if detailed_print:
                print("fft = no + abs + angle")
                print('corrected')
            return_array = np.zeros(array.shape +(3,)).astype(f_type)
            return_array[...,0] = array.astype(f_type)
            f1 = np.absolute(fft)
            return_array[...,:f1.shape[-1],1]=f1.astype(f_type)
            f2 = np.angle(fft)
            #print('corrected')
            
            return_array[...,:f2.shape[-1],2]=f2.astype(f_type)
            fft_index = [0,1,1]
        elif option == 'time_abs':
            if detailed_print:
                print("fft = no + abs")
            return_array = np.zeros(array.shape +(2,)).astype(f_type)
            return_array[...,0] = array.astype(f_type)
            f1 = np.absolute(fft)
            return_array[...,:f1.shape[-1],1]=f1.astype(f_type)
            fft_index = [0,1]
        
    return return_array.astype(f_type), fft_index

def summarize_load_Dataset(x_data, y_data, print_head):
    print(print_head)
    print(f'X: Shape={x_data.shape}')
    if x_data.ndim==3:
        print(f'X_{1}_max={np.max(x_data[:,:,:])}, min={np.min(x_data[:,:,:])}')
    else:
        for i in range(x_data.shape[-1]):
            print(f'X_{i+1}_max={np.max(x_data[:,:,:,i])}, min={np.min(x_data[:,:,:,i])}')
    print(f'Vs: Shape={y_data.shape}, max={np.max(y_data)}, min={np.min(y_data)}\n')

