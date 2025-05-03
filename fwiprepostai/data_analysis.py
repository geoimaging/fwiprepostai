import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seiskit.pre_process_waveform as pre_process

def get_prediction(model_class, waveform_time_np, pad_time_ms=0):
    assert waveform_time_np.shape == (24,1000)
    waveform_array_processed = pre_process.zero_padding_and_trim_each(waveform_time_np[None,...], freq=1000, pad_time=pad_time_ms, upto_time=model_class.time_upto_ms)   
    waveform_array_processed = pre_process.process_waveforms_images(waveform_array_processed, None, f_type='float64', **model_class.pre_processing_params)[0]
    prediction = model_class.tf_model.predict(waveform_array_processed)[0]
    return waveform_array_processed, prediction

def get_accuracy(act_vs, pred_vs, accuracy_type = 'mape'):
    assert accuracy_type in ['mape', 'mae', 'r2'], "Invalid accuracy type."
        
    if accuracy_type == 'mape':
        acc = mean_absolute_percentage_error(act_vs, pred_vs)
    elif accuracy_type == 'mae':
        acc = mean_absolute_error(act_vs, pred_vs)
    else:
        acc = r2(act_vs, pred_vs)
    return acc

def plot2d(data, ax=None, extent = None,
            vmin=None, vmax=None, cmap='gist_earth_r', 
            legend = True, legend_label = None):
    
    if ax is None:
        fig,ax = plt.subplots()

    # Create a colormap from the color mapping
    if vmin is None:
        vmin = np.min(data)

    if vmax is None:
        vmax = np.max(data)
    
    # cax = ax.imshow(plt_data.T, extent=[0, self.layer_matrix_class.x_lim, self.layer_matrix_class.z_lim, 0], interpolation='nearest')
    
    cax = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, interpolation='none') 
    
    # Colorbar
    if legend:
        cbar = plt.colorbar(cax, ax=ax, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label(legend_label)
    
    return ax

def plot_prediction(act_vs, pred_vs, accuracy_type = 'mape', extent = None,
                    ax=None, vmin=None, vmax=None, cmap='gist_earth_r', 
                    legend = True, legend_label = None):
    
    data = pred_vs
    act = act_vs
    acc = get_accuracy(act, data, accuracy_type)
    if ax is None:
        fig_len = 10
        fig, axs = plt.subplots(1,2,figsize=[fig_len,5.2/(3*5.5)*fig_len], sharex=True, sharey = True, gridspec_kw={'hspace': 0.05, 'wspace':0.08})#, height_ratios=[0.7,1,1,1])   

    # Create a colormap from the color mapping
    if vmin is None:
        vmin = np.min(data)

    if vmax is None:
        vmax = np.max(data)
        
    ax = axs[0]
    plot_data = act
    ax.imshow(plot_data, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax, extent=extent, interpolation='none')
    ax.set_aspect('equal')
    ax.set_ylabel('Depth (m)')

    ax = axs[1]
    plot_data = data
    cax = ax.imshow(plot_data, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax, extent=extent, interpolation='none')
    
    ax.set_title(f"{accuracy_type.upper()} = {acc:.0f}%", fontdict={'fontsize':11})
    fig.text(0.455, 0.183, 'Distance (m)', ha='center', va='center')
    ax.set_aspect('equal')
        
    # Colorbar
    if legend:
        cbar = plt.colorbar(cax, ax=axs, aspect=15, shrink = 0.5, pad=0.02)
        cbar.set_label("Shear Wave Velocity\n(m/s)")
    
    return ax

def r2(y, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(1.0, tf.divide(residual, total))
    return r2

def mean_absolute_error(y, y_pred):
    mae = tf.reduce_mean(tf.abs(tf.subtract(y, y_pred)))
    return mae

def mean_absolute_percentage_error(y, y_pred):
    mape = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(y, y_pred), y))) * 100.0
    return mape