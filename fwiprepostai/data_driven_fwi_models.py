import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class VantasselAndBhochhibhoya2025():
    def __init__(self, to_weight_file):
        self.name='VantasselAndBhochhibhoya2025'
        
        self.pre_processing_params = {
            'norm_x':{'norm':True, 'max':'max_abs', 'min':0},
            'norm_x_fft':{'norm':False},
            'norm_vs': {'norm':False},
            'fft_option':'time_abs',
            'dc_correction':True,
            'cosine_taper_ms':20,
            'order_filter':1,
            'order_norm':2,
            'order_fft':3,
            'filter_wn':[20,50],
            'norm_x_excl_receivers':[11,12],
            'fft_size_eq_time':False,
            'norm_each_time_domain':False,
        }
        self.time_upto_ms = 500
        self.span_x, self.span_z, self.del_x, self.del_z = 50,20,1,1
        
        self.coeff = 2
        self.coeff_2 = 2
        self.rfft_n = [1,251]
        self.fft_scale = 1000
        
        # Load weights
        self.input_shape = (24,500,2)
        self.output_shape = (20,50)
        self.tf_model = self.model()
        self.tf_model.load_weights(to_weight_file)
        
    def model(self):
        input_single = tf.keras.Input(shape=self.input_shape)
    
        # Split the input into two separate branches
        # input_1 = tf.keras.layers.Lambda(lambda x: tf.concat([x[:, :11, ::2, :1], x[:, 13:, ::2, :1]], axis=1), output_shape = (self.input_shape[0], int(self.input_shape[1]/2), 1))(input_single)
        input_1 = tf.keras.layers.Lambda(lambda x: tf.concat([x[:, :11, ::2, :1], x[:, 13:, ::2, :1]], axis=1))(input_single)
        # input_2 = tf.keras.layers.Lambda(lambda x: tf.concat([x[:, :11, self.rfft_n[0]:self.rfft_n[1], 1:2], x[:, 13:, self.rfft_n[0]:self.rfft_n[1], 1:2]], axis=1), output_shape = (self.input_shape[0], int(self.input_shape[1]/2), 1))(input_single)
        input_2 = tf.keras.layers.Lambda(lambda x: tf.concat([x[:, :11, self.rfft_n[0]:self.rfft_n[1], 1:2], x[:, 13:, self.rfft_n[0]:self.rfft_n[1], 1:2]], axis=1))(input_single)
        input_2 = scaling_tf(self.fft_scale)(input_2)      
        
        x_1 = tf.keras.layers.Conv2D(int(32*self.coeff), kernel_size=[1,3], activation="relu", kernel_initializer='he_uniform')(input_1)
        x_1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 3))(x_1)
        x_1 = tf.keras.layers.Conv2D(int(64*self.coeff), kernel_size=[1,3], activation="relu", kernel_initializer='he_uniform')(x_1)
        x_1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 3))(x_1)
        x_1 = tf.keras.layers.Conv2D(int(128*self.coeff), kernel_size=[1,3], activation="relu", kernel_initializer='he_uniform')(x_1)
        x_1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(x_1)  #Either 3,3 or 1,3:: similar to x_2

        x_2 = tf.keras.layers.Conv2D(int(32*self.coeff), kernel_size=[1,3], activation="relu", kernel_initializer='he_uniform')(input_2)
        x_2 = tf.keras.layers.MaxPooling2D(pool_size=(1, 3))(x_2)
        x_2 = tf.keras.layers.Conv2D(int(64*self.coeff), kernel_size=[1,3], activation="relu", kernel_initializer='he_uniform')(x_2)
        x_2 = tf.keras.layers.MaxPooling2D(pool_size=(1, 3))(x_2)
        x_2 = tf.keras.layers.Conv2D(int(128*self.coeff), kernel_size=[1,3], activation="relu", kernel_initializer='he_uniform')(x_2)
        x_2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(x_2)

        merged = tf.keras.layers.Concatenate(axis=-1)([x_1, x_2])

        # Additional layers to process the merged feature map
        #x = tf.keras.layers.Flatten()(merged)
        x = tf.keras.layers.Conv2D(512*self.coeff_2, kernel_size=[1,3], activation="relu", kernel_initializer='he_uniform')(merged)  #Use square from here
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(1024*self.coeff_2, kernel_size=[3,3], activation="relu", kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(500*self.coeff_2, activation='relu', kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.Dense(int(self.output_shape[0]*self.output_shape[1]), activation='linear', kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.Reshape(self.output_shape)(x)

        model = tf.keras.Model(inputs=input_single, outputs=x)
        return model
    
class scaling_tf(Layer):
    def __init__(self, fft_scale, **kwargs):
        super(scaling_tf, self).__init__(**kwargs)
        self.fft_scale = fft_scale
        
    def call(self, x):
        return tf.math.scalar_mul(self.fft_scale, x)


