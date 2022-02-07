import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
import numpy as np


# Switchable normalization
class SN(layers.Layer):
    def __init__(self, activation):
        super(SN, self).__init__()
        self.activation = activation

    def build(self, input_shape):
        self.ch = input_shape[-1]
        self.eps = 1e-7
        self.gamma = tf.Variable(tf.ones([1, 1, 1, self.ch], dtype=tf.float32), trainable=True)
        self.beta = tf.Variable(tf.zeros([1, 1, 1, self.ch], dtype=tf.float32), trainable=True)
        self.mean_weight = tf.Variable([1, 1, 1], dtype=tf.float32, trainable=True)
        self.var_weight = tf.Variable([1, 1, 1], dtype=tf.float32, trainable=True)
        self.act = layers.Activation(activation=self.activation)

    def call(self, input, act=True, training=True):
        i_mean, i_var = tf.nn.moments(input, [1, 2], keepdims=True)  # N, 1, 1, C
        l_mean, l_var = tf.nn.moments(input, [1, 2, 3], keepdims=True)  # N, 1, 1, 1
        b_mean, b_var = tf.nn.moments(input, [0, 1, 2], keepdims=True)  # 1, 1, 1, C
        mean_weight = tf.nn.softmax(self.mean_weight)
        var_weight = tf.nn.softmax(self.var_weight)
        mean = mean_weight[0] * b_mean + mean_weight[1] * i_mean + mean_weight[2] * l_mean
        var = var_weight[0] * b_var + var_weight[1] * i_var + var_weight[2] * l_var
        x = (input - mean) / (tf.sqrt(var + self.eps))
        return self.act(x * self.gamma + self.beta) if act else x * self.gamma + self.beta


# Instance Normalization
'''
class IN(layers.Layer): 
    def __init__(self, activation):
        super(IN, self).__init__()
        self.eps = 1e-7
        self.ln = tfa.layers.InstanceNormalization()
        self.activation = layers.Activation(activation=activation)
    def call(self, input, act=True, training=True):
        x = self.ln(input)
        return self.activation(x) if act else x
'''


class IN(layers.Layer):
    def __init__(self, activation):
        super(IN, self).__init__()
        self.epsilon = 1e-7
        self.activation = layers.Activation(activation=activation)

    def build(self, input_shape):
        self.ch = input_shape[-1]
        self.gamma = tf.Variable(tf.ones([1, 1, 1, self.ch], dtype=tf.float32) / self.ch, trainable=True)
        self.beta = tf.Variable(tf.zeros([1, 1, 1, self.ch], dtype=tf.float32), trainable=True)

    def call(self, input, act=True, training=True):
        i_mean, i_var = tf.nn.moments(input, [1, 2], keepdims=True)
        i_x = (input - i_mean) * (tf.math.rsqrt(i_var + self.epsilon))
        x = i_x * self.gamma + self.beta
        return self.activation(x) if act else x


# Batch Normalization
class BN(layers.Layer):
    def __init__(self, activation):
        super(BN, self).__init__()
        self.eps = 1e-7
        self.activation = layers.Activation(activation=activation)
        self.bn = layers.BatchNormalization()

    def call(self, input, act=True, training=True):
        x = self.bn(input, training=training)
        return self.activation(x) if act else x


# Layer Normalization
class LN(layers.Layer):
    def __init__(self, activation):
        super(LN, self).__init__()
        self.eps = 1e-7
        self.activation = layers.Activation(activation=activation)
        self.ln = layers.LayerNormalization()

    def call(self, input, act=True, training=True):
        x = self.ln(input)
        return self.activation(x) if act else x


class GN(layers.Layer):
    def __init__(self, activation):
        super(GN, self).__init__()
        self.eps = 1e-7
        self.activation = layers.Activation(activation=activation)
        self.gn = tfa.layers.GroupNormalization()

    def call(self, input, act=True, training=True):
        x = self.gn(input)
        return self.activation(x) if act else x


class FRN(layers.Layer):
    def __init__(self, activation):
        super(FRN, self).__init__()
        self.eps = 1e-7
        self.activation = layers.Activation(activation=activation)
        self.frn = tfa.layers.FilterResponseNormalization()

    def call(self, input, act=True, training=True):
        x = self.frn(input)
        return self.activation(x) if act else x


# Batch Instance Normalization
class BIN(layers.Layer):
    def __init__(self, activation):
        super(BIN, self).__init__()
        self.activation = layers.Activation(activation=activation)
        self.epsilon = 1e-7

    def build(self, input_shape):
        self.ch = input_shape[-1]
        self.gamma = tf.Variable(tf.ones([1, 1, 1, self.ch], dtype=tf.float32), trainable=True)
        self.beta = tf.Variable(tf.zeros([1, 1, 1, self.ch], dtype=tf.float32), trainable=True)
        self.rho = self.add_weight(
            name='rho',
            shape=input_shape[-1:],
            initializer='ones',
            constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0),
            trainable=True)

    def call(self, input, act=True, training=True):
        b_mean, b_var = tf.nn.moments(input, [0, 1, 2], keepdims=True)
        i_mean, i_var = tf.nn.moments(input, [1, 2], keepdims=True)
        b_x = (input - b_mean) * (tf.math.rsqrt(b_var + self.epsilon))
        i_x = (input - i_mean) * (tf.math.rsqrt(i_var + self.epsilon))
        x = (self.rho * b_x + (1 - self.rho) * i_x) * self.gamma + self.beta
        return self.activation(x) if act else x


class LIN(layers.Layer):
    def __init__(self, activation):
        super(LIN, self).__init__()
        self.activation = layers.Activation(activation=activation)
        self.epsilon = 1e-7

    def build(self, input_shape):
        self.ch = input_shape[-1]
        self.gamma = tf.Variable(tf.ones([1, 1, 1, self.ch], dtype=tf.float32), trainable=True)
        # self.gamma = tf.Variable(tf.random.normal([1,1,1,self.ch], mean=0.0, stddev=0.01, dtype=tf.float32), trainable=True)
        self.beta = tf.Variable(tf.zeros([1, 1, 1, self.ch], dtype=tf.float32), trainable=True)
        self.rho = self.add_weight(
            name='rho',
            shape=input_shape[-1:],
            initializer='ones',
            constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0),
            trainable=True)

    def call(self, input, act=True, training=True):
        l_mean, l_var = tf.nn.moments(input, [1, 2, 3], keepdims=True)
        i_mean, i_var = tf.nn.moments(input, [1, 2], keepdims=True)
        l_x = (input - l_mean) * (tf.math.rsqrt(l_var + self.epsilon))
        i_x = (input - i_mean) * (tf.math.rsqrt(i_var + self.epsilon))
        x = ((1 - self.rho) * l_x + self.rho * i_x) * self.gamma + self.beta
        return self.activation(x) if act else x


class GIN(layers.Layer):
    def __init__(self, activation):
        super(GIN, self).__init__()
        self.activation = layers.Activation(activation=activation)
        self.epsilon = 1e-7

    def build(self, input_shape):
        self.ch = input_shape[-1]
        self.G = min(self.ch, 16)
        self.gamma = tf.Variable(tf.ones([1, 1, 1, self.ch], dtype=tf.float32), trainable=True)
        self.beta = tf.Variable(tf.zeros([1, 1, 1, self.ch], dtype=tf.float32), trainable=True)
        self.rho = self.add_weight(
            name='rho',
            shape=input_shape[-1:],
            initializer='ones',
            constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0),
            trainable=True)

    def call(self, input, act=True, training=True):
        group = tf.concat(tf.split(tf.expand_dims(input, -1), num_or_size_splits=self.ch // self.G, axis=3),
                          axis=-1)  # b, h, w, g, ch//g
        g_mean, g_var = tf.nn.moments(group, [1, 2, 4], keepdims=True)
        i_mean, i_var = tf.nn.moments(input, [1, 2], keepdims=True)
        g_x = (group - g_mean) * (tf.math.rsqrt(g_var + self.epsilon))  # b, h, w, g, ch//g
        i_x = (input - i_mean) * (tf.math.rsqrt(i_var + self.epsilon))
        g_x = tf.squeeze(tf.concat(tf.split(g_x, num_or_size_splits=self.ch // self.G, axis=4), axis=3),
                         -1)  # 2, b, h, w, 32
        x = (self.rho * g_x + (1 - self.rho) * i_x) * self.gamma + self.beta
        return self.activation(x) if act else x


class BGN(layers.Layer):
    def __init__(self, activation):
        super(BGN, self).__init__()
        self.activation = layers.Activation(activation=activation)
        self.epsilon = 1e-7

    def build(self, input_shape):
        self.ch = input_shape[-1]
        self.G = min(self.ch, 16)
        self.gamma = tf.Variable(tf.ones([1, 1, 1, self.ch], dtype=tf.float32), trainable=True)
        self.beta = tf.Variable(tf.zeros([1, 1, 1, self.ch], dtype=tf.float32), trainable=True)
        self.rho = self.add_weight(
            name='rho',
            shape=input_shape[-1:],
            initializer='ones',
            constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0),
            trainable=True)

    def call(self, input, act=True, training=True):
        group = tf.concat(tf.split(tf.expand_dims(input, -1), num_or_size_splits=self.ch // self.G, axis=3),
                          axis=-1)  # b, h, w, g, ch//g
        b_mean, b_var = tf.nn.moments(input, [0, 1, 2], keepdims=True)
        g_mean, g_var = tf.nn.moments(group, [1, 2, 4], keepdims=True)
        g_x = (group - g_mean) * (tf.math.rsqrt(g_var + self.epsilon))  # b, h, w, g, ch//g
        b_x = (input - b_mean) * (tf.math.rsqrt(b_var + self.epsilon))
        g_x = tf.squeeze(tf.concat(tf.split(g_x, num_or_size_splits=self.ch // self.G, axis=4), axis=3),
                         -1)  # 2, b, h, w, 32
        x = (self.rho * b_x + (1 - self.rho) * g_x) * self.gamma + self.beta
        return self.activation(x) if act else x


class BLN(layers.Layer):
    def __init__(self, activation):
        super(BLN, self).__init__()
        self.activation = layers.Activation(activation=activation)
        self.epsilon = 1e-7

    def build(self, input_shape):
        self.ch = input_shape[-1]
        self.gamma = tf.Variable(tf.ones([1, 1, 1, self.ch], dtype=tf.float32), trainable=True)
        self.beta = tf.Variable(tf.zeros([1, 1, 1, self.ch], dtype=tf.float32), trainable=True)
        self.rho = self.add_weight(
            name='rho',
            shape=input_shape[-1:],
            initializer='ones',
            constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0),
            trainable=True)

    def call(self, input, act=True, training=True):
        b_mean, b_var = tf.nn.moments(input, [0, 1, 2], keepdims=True)
        l_mean, l_var = tf.nn.moments(input, [1, 2, 3], keepdims=True)
        b_x = (input - b_mean) * (tf.math.rsqrt(b_var + self.epsilon))
        l_x = (input - l_mean) * (tf.math.rsqrt(l_var + self.epsilon))
        x = (self.rho * b_x + (1 - self.rho) * l_x) * self.gamma + self.beta
        return self.activation(x) if act else x


class RBGN(layers.Layer):
    def __init__(self, activation):
        super(RBGN, self).__init__()
        self.activation = layers.Activation(activation=activation)
        self.epsilon = 1e-7

    def build(self, input_shape):
        self.ch = input_shape[-1]
        temp = int(np.log2(self.ch))
        self.G = np.power(2, np.random.randint(0, temp))
        self.gamma = tf.Variable(tf.ones([1, 1, 1, self.ch], dtype=tf.float32), trainable=True)
        self.beta = tf.Variable(tf.zeros([1, 1, 1, self.ch], dtype=tf.float32), trainable=True)
        self.rho = self.add_weight(
            name='rho',
            shape=input_shape[-1:],
            initializer='ones',
            constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0),
            trainable=True)

    def call(self, input, act=True, training=True):
        group = tf.concat(tf.split(tf.expand_dims(input, -1), num_or_size_splits=self.ch // self.G, axis=3),
                          axis=-1)  # b, h, w, g, ch//g
        b_mean, b_var = tf.nn.moments(input, [0, 1, 2], keepdims=True)
        g_mean, g_var = tf.nn.moments(group, [1, 2, 4], keepdims=True)
        g_x = (group - g_mean) * (tf.math.rsqrt(g_var + self.epsilon))  # b, h, w, g, ch//g
        b_x = (input - b_mean) * (tf.math.rsqrt(b_var + self.epsilon))  #
        g_x = tf.squeeze(tf.concat(tf.split(g_x, num_or_size_splits=self.ch // self.G, axis=4), axis=3),
                         -1)  # 2, b, h, w, 32
        x = (self.rho * b_x + (1 - self.rho) * g_x) * self.gamma + self.beta
        return self.activation(x) if act else x


"""
import tensorflow as tf
from tensorflow.keras import layers

# Switchable normalization
class SN(layers.Layer): 
    def __init__(self, activation):
        super(SN, self).__init__()
        self.activation = activation
    def build(self, input_shape):
        self.ch = input_shape[-1]
        self.eps = 1e-5
        self.momentum = 0.99
        self.gamma = tf.Variable(tf.ones([1,1,1,self.ch], dtype=tf.float32), trainable=True)
        self.beta = tf.Variable(tf.zeros([1,1,1,self.ch], dtype=tf.float32), trainable=True)
        self.mean_weight = tf.nn.softmax(tf.Variable([1,1,1], dtype=tf.float32, trainable=True))
        self.var_weight = tf.nn.softmax(tf.Variable([1,1,1], dtype=tf.float32, trainable=True))
        self.moving_average_mean = tf.Variable(tf.zeros([1,1,1,input_shape[3]]), dtype=tf.float32, trainable=False)
        self.moving_average_var = tf.Variable(tf.zeros([1,1,1,input_shape[3]]), dtype=tf.float32, trainable=False)

        self.act = layers.Activation(activation=self.activation)

    def call(self, input, act=True, training=True):
        i_mean, i_var = tf.nn.moments(input, [1,2], keepdims=True) # N, 1, 1, C
        l_mean, l_var = tf.nn.moments(input, [1,2,3], keepdims=True) # N, 1, 1, 1
        b_mean, b_var = tf.nn.moments(input, [0,1,2], keepdims=True) # 1, 1, 1, C
        if training:
            b_mean, b_var = tf.nn.moments(input, [0,1,2], keepdims=True)
            keras.backend.moving_average_update(self.moving_average_mean, b_mean, self.momentum)
            keras.backend.moving_average_update(self.moving_average_var, b_var, self.momentum)
        else:
            b_mean = self.moving_average_mean
            b_var = self.moving_average_var

        mean = self.mean_weight[0]*b_mean + self.mean_weight[1]*i_mean + self.mean_weight[2]*l_mean
        var = self.var_weight[0]*b_var + self.var_weight[1]*i_var + self.var_weight[2]*l_var
        x = (input - mean)/(tf.sqrt(var+self.eps))
        return self.act(x*self.gamma+self.beta) if act else x*self.gamma + self.beta

#Instance Normalization
class IN(layers.Layer): 
    def __init__(self, activation):
        super(IN, self).__init__()
        self.eps = 1e-7
        self.activation = layers.Activation(activation=activation)
    def call(self, input, act=True):
        i_mean, i_var = tf.nn.moments(input, [1,2], keepdims=True) # N, 1, 1, C
        x = (input-i_mean)/(tf.sqrt(i_var+self.eps))
        return self.activation(x) if act else x

#Batch Normalization
class BN(layers.Layer):
    def __init__(self, activation):
        super(BN, self).__init__()
        self.eps = 1e-7
        self.activation = layers.Activation(activation=activation)
        self.bn = layers.BatchNormalization()
    def call(self, input, act=True, training=True):
        x = self.bn(input, training=training)
        return self.activation(x) if act else x

#Batch Instance Normalization
class BIN(layers.Layer):
    def __init__(self, activation):
        super(BIN, self).__init__()
        self.activation = layers.Activation(activation=activation)
        self.epsilon = 1e-7
    def build(self, input_shape):
        self.ch = input_shape[-1]
        self.gamma = tf.Variable(tf.ones([1,1,1,self.ch], dtype=tf.float32), trainable=True)
        self.beta = tf.Variable(tf.zeros([1,1,1,self.ch], dtype=tf.float32), trainable=True)
        self.rho = self.add_weight(
            name='rho',
            shape=input_shape[-1:],
            initializer='ones',
            constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0),
            trainable=True)
    def call(self, input, act=True, training=True):
        b_mean, b_var = tf.nn.moments(input, [0,1,2], keepdims=True)
        i_mean, i_var = tf.nn.moments(input, [1,2], keepdims=True)
        b_x = (input-b_mean)*(tf.math.rsqrt(b_var+self.epsilon))
        i_x = (input-i_mean)*(tf.math.rsqrt(i_var+self.epsilon))
        x = (self.rho*b_x+(1-self.rho)*i_x)*self.gamma+self.beta
        return self.activation(x) if act else x

"""