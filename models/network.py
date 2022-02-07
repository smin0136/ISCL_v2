import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import Model



class Res_Block(Model):
    def __init__(self, dim=64):
        super(Res_Block, self).__init__()
        self.n1 = tfa.layers.InstanceNormalization()
        self.n2 = tfa.layers.InstanceNormalization()
        self.dim = dim

        self.h1 = layers.Conv2D(dim, 3, padding='valid', use_bias=False)
        self.h2 = layers.Conv2D(dim, 3, padding='valid', use_bias=False)

    def __call__(self, inputs, training=True):
        x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h1(x)
        x = self.n1(x, training=training)
        x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h2(x)
        x = self.n2(x, training=training)
        out = layers.add([inputs, x])

        return out



class Gen_with_adain(Model):
    def __init__(self, output_channels=1, dim=64, n_downsamplings=2, n_blocks=9, norm='instance_norm'):
        super(Gen_with_adain, self).__init__()
        self.initializer = 'truncated_normal'
        self.dim = dim
        self.n_downsamplings = n_downsamplings
        self.n_blocks = n_blocks
        self.AIN = AdaIN(tf.nn.leaky_relu)

        self.cv1 = layers.Conv2D(dim, 7, padding='valid', use_bias=False)
        self.cv2 = layers.Conv2D(dim * 2, 3, strides=2, padding='same', use_bias=False)
        self.cv3 = layers.Conv2D(dim * 4, 3, strides=2, padding='same', use_bias=False)

        self.res = []
        for i in range(0, n_blocks):
            self.res.append(Res_Block(dim * 4))

        self.cv4 = layers.Conv2DTranspose(dim * 2, 3, strides=2, padding='same', use_bias=False)
        self.cv5 = layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)
        self.out = layers.Conv2D(output_channels, 7, padding='valid')

        self.d1 = layers.Dense(self.dim, kernel_initializer=self.initializer)
        self.d2 = layers.Dense(self.dim, kernel_initializer=self.initializer)
        self.d3 = layers.Dense(self.dim, kernel_initializer=self.initializer)
        self.d4 = layers.Dense(self.dim, kernel_initializer=self.initializer)

        self.d5_m = layers.Dense(self.dim, kernel_initializer=self.initializer)
        self.d5_v = layers.Dense(self.dim, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d6_m = layers.Dense(self.dim * 2, kernel_initializer=self.initializer)
        self.d6_v = layers.Dense(self.dim * 2, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d7_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d7_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)

        self.d8_m = layers.Dense(self.dim * 2, kernel_initializer=self.initializer)
        self.d8_v = layers.Dense(self.dim * 2, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d9_m = layers.Dense(self.dim, kernel_initializer=self.initializer)
        self.d9_v = layers.Dense(self.dim, activation=tf.nn.relu, kernel_initializer=self.initializer)


    def __call__(self, inputs, z=None, training=True):
        if z is not None:
            l = self.d1(z)
            l = self.d2(l)
            l = self.d3(l)
            latent = self.d4(l)

        x = inputs
        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        x = self.cv1(x)

        if z is not None:
            x = self.AIN(x, self.d5_m(latent), self.d5_v(latent))
        else:
            x = self.AIN(x)

        # x = self.n1(x)
        # x = tf.nn.relu(x)
        ##### d layers dimension 확인 필요 ///  residaul block 은 어떻게 처리??

        x = self.cv2(x)
        if z is not None:
            x = self.AIN(x, self.d6_m(latent), self.d6_v(latent))
        else:
            x = self.AIN(x)
        # x = self.n2(x)
        # x = tf.nn.relu(x)

        x = self.cv3(x)
        if z is not None:
            x = self.AIN(x, self.d7_m(latent), self.d7_v(latent))
        else:
            x = self.AIN(x)
        # x = self.n3(x)
        # x = tf.nn.relu(x)

        """
        for h1 in self.res:
            if z is not None:
                x = h1(x, latent=latent, training=training)
            else:
                x = h1(x, training=training)
        """

        for h1 in self.res:
            x = h1(x, training=training)


        x = self.cv4(x)
        if z is not None:
            x = self.AIN(x, self.d8_m(latent), self.d8_v(latent))
        else:
            x = self.AIN(x)
        # x = self.n4(x)
        # x = tf.nn.relu(x)

        x = self.cv5(x)
        if z is not None:
            x = self.AIN(x, self.d9_m(latent), self.d9_v(latent))
        else:
            x = self.AIN(x)
        # x = self.n5(x)
        # x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        x = self.out(x)
        x = tf.tanh(x)

        return x


class AdaIN(layers.Layer):
    def __init__(self, activation):
        super(AdaIN, self).__init__()
        self.eps = 1e-7
        self.activation = layers.Activation(activation=activation)

    def __call__(self, input, y_mean=None, y_var=None, act=True):
        x_mean, x_var = tf.nn.moments(input, [1,2], keepdims=True) # N, 1, 1, C
        if y_mean is not None:
            y_mean = y_mean[:, tf.newaxis, tf.newaxis, :]
            y_var = y_var[:, tf.newaxis, tf.newaxis, :]

            x = (input-x_mean)/(tf.sqrt(x_var+self.eps))
            x = x*tf.sqrt(y_var+self.eps)+y_mean
        else:
            x = (input-x_mean)/(tf.sqrt(x_var+self.eps))
        return self.activation(x) if act else x


class ConvDiscriminator_cont(Model):
    def __init__(self, output_channels=1, dim=64, n_downsamplings=3, norm='instance_norm'):
        super(ConvDiscriminator_cont, self).__init__()
        self.output_channel = output_channels
        self.dim = dim
        self.n_downsampling = n_downsamplings
        self.norm = norm

        self.n1 = tfa.layers.InstanceNormalization()
        self.n2 = tfa.layers.InstanceNormalization()
        self.n3 = tfa.layers.InstanceNormalization()

        self.cv1 = keras.layers.Conv2D(dim, 4, strides=2, padding='same') # 256 -> 128
        dim = min(dim * 2, self.dim * 8)
        self.cv2 = keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)  # 128 -> 64
        dim = min(dim * 2, self.dim * 8)
        self.cv3 = keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False) # 64 -> 32
        dim = min(dim * 2, self.dim * 8)
        self.cv4 = keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)
        self.out = keras.layers.Conv2D(1, 4, strides=1, padding='same')

        #self.h1 = keras.layers.Conv2D(dim, 4, strides=4, padding='same', use_bias=False) # 16 16
        #self.h2 = keras.layers.Conv2D(dim, 4, strides=4, padding='same', use_bias=False) # 4 4
        #self.h3 = keras.layers.Conv2D(dim, 4, strides=1, padding='valid', use_bias=False) # 1 1 dim
        # h : shared features
        # Head_contrastive = keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h) 32 -> 16
        # Head_contrastive = keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(Head_contrastive) 16 -> 8,8,ch


    def __call__(self, inputs, training=True):
        x = self.cv1(inputs)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.cv2(x)
        x = self.n1(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        h = x
        #h = self.h1(x)
        #h = self.h2(h)
        #h = self.h3(h)  # batch_size, 1, 1, dim
        #h = tf.squeeze(h)  # batch_size, dim

        x = self.cv3(x)
        x = self.n2(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        x = self.cv4(x)
        x = self.n3(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.out(x)

        return x, h


class Extractor(Model):
    def __init__(self, output_channels=1, dim=64, n_blocks=3, norm='instance_norm'):
        super(Extractor, self).__init__()
        self.output_channels = output_channels
        self.dim = dim
        self.n_blocks = n_blocks
        self.norm = norm

        self.cv1 = layers.Conv2D(dim, (3,3), (1,1), activation=tf.nn.leaky_relu, padding="same", use_bias=True)

        self.res = []
        self.nor =[]
        for _ in range(n_blocks-2):
            self.res.append(keras.layers.Conv2D(dim, (3, 3), (1, 1), activation=None, padding="same", use_bias=True))
            self.nor.append(tfa.layers.InstanceNormalization())

        self.out = layers.Conv2D(output_channels, (3,3), (1,1), activation=None, padding="same", use_bias=True)

    def __call__(self, inputs, training=True):
        x = self.cv1(inputs)
        for conv, nor in zip(self.res, self.nor):
            x = conv(x)
            x = nor(x, training=training)
            x = tf.nn.leaky_relu(x)
        x = self.out(x)

        return x


class LinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate


