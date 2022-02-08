import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import optimizers

import tensorflow_addons as tfa
import numpy as np
import pylib as py
from utils.image_tool import *
from utils.manage import *
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from models.network import Gen_with_adain, ConvDiscriminator_cont, Extractor, LinearDecay
from models.loss import *

class pre_Trainer(Model):
    def __init__(self, args):
        super(pre_Trainer, self).__init__()
        self.gamma = 0.5
        self.sample_dir = py.join(args.datasets_dir, 'pre_output', args.output_date, args.dir_num, 'samples_training')
        self.output_dir = py.join(args.datasets_dir, 'pre_output', args.output_date, args.dir_num)

        self.len_dataset = 100
        self.cycle_loss_weight = args.cycle_loss_weight

        self.G = Gen_with_adain()
        self.H = Extractor()

        self.supervised_loss_fn = tf.losses.MeanSquaredError()

        self.G_lr_scheduler = LinearDecay(args.lr, args.epochs * self.len_dataset, args.epoch_decay * self.len_dataset)
        self.H_lr_scheduler = LinearDecay(0.002, args.epochs * self.len_dataset, args.epoch_decay * self.len_dataset)

        self.G_optimizer = optimizers.Adam(learning_rate=self.G_lr_scheduler, beta_1=args.beta_1)
        self.H_optimizer = optimizers.Adam(learning_rate=self.H_lr_scheduler, beta_1=args.beta_1)

        self.gloss_tracker = [tf.keras.metrics.Mean(name="N-gloss"), tf.keras.metrics.Mean(name="C-gloss")]
        self.hloss_tracker = [tf.keras.metrics.Mean(name="H-loss")]


        self.val_tracker = [tf.keras.metrics.Mean(name="psnr_g"),
                            tf.keras.metrics.Mean(name="ssim_g"),
                            tf.keras.metrics.Mean(name="psnr_h"),
                            tf.keras.metrics.Mean(name="ssim_h"),
                            tf.keras.metrics.Mean(name="psnr_e"),
                            tf.keras.metrics.Mean(name="ssim_e"),
                            ]



        for tracker in self.gloss_tracker:
            tracker.reset_state()
        for tracker in self.hloss_tracker:
            tracker.reset_state()
        for tracker in self.val_tracker:
            tracker.reset_state()




    def compile(self, **kwargs):
        self._configure_steps_per_execution(1)
        self._reset_compile_cache()
        self._is_compiled = True
        self.loss = {}

    def call(self, noisy, training=True):
        y_hat = self.F(noisy, training=training)
        y_bar = tf.clip_by_value(noisy - self.H(noisy, training=training), -1.0,
                                 1.0)  # We suppose that all images are in range [-1, 1].
        return self.gamma * (y_hat) + (1 - self.gamma) * y_bar

    def train_G(self, clean, noisy, z):
        with tf.GradientTape() as t:
            z1 = z + tf.random.normal(tf.shape(z), mean=0.0, stddev=1.0, dtype=tf.float32) * 1e-1
            # z1=z
            # z1 = tf.random.normal([1, 64], 0, 1, dtype=tf.float32)
            A2B = self.G(clean, z=z1, training=True)
            B2A = self.G(noisy, training=True)
            #############################  B-B2A <-> H(B)

            A2B_g_loss = self.supervised_loss_fn(noisy, A2B)
            B2A_g_loss = self.supervised_loss_fn(clean, B2A)


            G_loss = (A2B_g_loss + B2A_g_loss)

        G_grad = t.gradient(G_loss, self.G.trainable_variables)
        self.G_optimizer.apply_gradients(zip(G_grad, self.G.trainable_variables))
        self.gloss_tracker[0].update_state(A2B_g_loss)
        self.gloss_tracker[1].update_state(B2A_g_loss)

        return {'A2B_g_loss': A2B_g_loss,
                'B2A_g_loss': B2A_g_loss}


    def train_H(self, A, B):
        with tf.GradientTape() as t:

            sup_x = self.H(B, training=True)  # A가 noisy B가 clean noise

            loss = self.supervised_loss_fn(sup_x, B-A)

        H_grad = t.gradient(loss, self.H.trainable_variables)
        self.H_optimizer.apply_gradients(zip(H_grad, self.H.trainable_variables))
        self.hloss_tracker[0].update_state(loss)

        return {'h_loss': loss}


    def train_step(self, data):
        A, B = data
        tf.random.set_seed(5)
        z = tf.random.normal([1, 64], 0, 1, dtype=tf.float32)
        G_loss = self.train_G(A, B, z)
        H_loss = self.train_H(A, B)

        return{**G_loss, **H_loss}

    def predict(self, A, B):
        z = tf.random.normal([1, 64], 0, 1, dtype=tf.float32)

        A2B = tf.clip_by_value(self.G(A, z=z, training=False), -1.0, 1.0)
        B2A = tf.clip_by_value(self.G(B, training=False), -1.0, 1.0)
        H2A = B - self.H(B, training=False)
        H2A = tf.clip_by_value(H2A, -1.0, 1.0)
        GnH = tf.clip_by_value(0.5 * B2A + 0.5 * H2A, -1.0, 1.0)

        return A2B, B2A, H2A, GnH

    def test_step(self, data):
        A, B = data

        A2B, B2A, H2A, GnH = self.predict(A[:, :, :, np.newaxis], B[:, :, :, np.newaxis])

        # image 저장
        #img = immerge(np.concatenate([A, A2B, H2A, B, B2A, GnH], axis=0), n_rows=2)
        #imwrite(img, py.join(self.sample_dir, 'iter-%09d.jpg' % self.G_optimizer.iterations))


        # psnr 계산

        B2A = tf.squeeze(B2A, -1)
        psnr_b2a = tf.image.psnr((A + 1) * 0.5 * 255, (B2A + 1) * 0.5 * 255, max_val=255.0)
        ssim_b2a = tf.image.ssim((A[tf.newaxis,..., tf.newaxis] + 1) * 0.5 * 255, (B2A[tf.newaxis, ..., tf.newaxis] + 1) * 0.5 * 255, max_val=255.0)

        H2A = tf.squeeze(H2A, -1)
        psnr_h2a = tf.image.psnr((A + 1) * 0.5 * 255, (H2A + 1) * 0.5 * 255, max_val=255.0)
        ssim_h2a = tf.image.ssim((A[tf.newaxis,..., tf.newaxis] + 1) * 0.5 * 255, (H2A[tf.newaxis, ..., tf.newaxis] + 1) * 0.5 * 255, max_val=255.0)

        GnH = tf.squeeze(GnH, -1)
        psnr_gnh = tf.image.psnr((A + 1) * 0.5 * 255, (GnH + 1) * 0.5 * 255, max_val=255.0)
        ssim_gnh = tf.image.ssim((A[tf.newaxis,..., tf.newaxis] + 1) * 0.5 * 255, (GnH[tf.newaxis, ..., tf.newaxis] + 1) * 0.5 * 255, max_val=255.0)


        self.val_tracker[0].update_state(psnr_b2a)
        self.val_tracker[1].update_state(ssim_b2a)
        self.val_tracker[2].update_state(psnr_h2a)
        self.val_tracker[3].update_state(ssim_h2a)
        self.val_tracker[4].update_state(psnr_gnh)
        self.val_tracker[5].update_state(ssim_gnh)


        return {"PNSR_g": self.val_tracker[0].result(),
                "SSIM_g": self.val_tracker[1].result(),
                "PNSR_h": self.val_tracker[2].result(),
                "SSIM_h": self.val_tracker[3].result(),
                "PNSR_e": self.val_tracker[4].result(),
                "SSIM_e": self.val_tracker[5].result(),
                }