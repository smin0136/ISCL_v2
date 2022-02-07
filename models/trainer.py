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

class Trainer(Model):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.gamma = 0.5
        self.sample_dir = py.join(args.datasets_dir, 'output', args.output_date, args.dir_num, 'samples_training')
        self.output_dir = py.join(args.datasets_dir, 'output', args.output_date, args.dir_num)

        self.A2B_pool = ItemPool(args.pool_size)
        self.B2A_pool = ItemPool(args.pool_size)
        self.len_dataset = 100
        self.cycle_loss_weight = args.cycle_loss_weight

        self.G = Gen_with_adain()
        self.D_A = ConvDiscriminator_cont()
        self.D_B = ConvDiscriminator_cont()
        self.H = Extractor()

        self.d_loss_fn, self.g_loss_fn = get_adversarial_losses_fn(args.adversarial_loss_mode)
        self.cycle_loss_fn = tf.losses.MeanAbsoluteError()
        self.identity_loss_fn = tf.losses.MeanAbsoluteError()
        self.mae_loss_fn = tf.losses.MeanAbsoluteError()

        self.G_lr_scheduler = LinearDecay(args.lr, args.epochs * self.len_dataset, args.epoch_decay * self.len_dataset)
        self.D_lr_scheduler = LinearDecay(args.lr, args.epochs * self.len_dataset, args.epoch_decay * self.len_dataset)
        self.H_lr_scheduler = LinearDecay(0.002, args.epochs * self.len_dataset, args.epoch_decay * self.len_dataset)

        self.G_optimizer = optimizers.Adam(learning_rate=self.G_lr_scheduler, beta_1=args.beta_1)
        self.D_optimizer = optimizers.Adam(learning_rate=self.D_lr_scheduler, beta_1=args.beta_1)
        self.H_optimizer = optimizers.Adam(learning_rate=self.H_lr_scheduler, beta_1=args.beta_1)

        self.gloss_tracker = [tf.keras.metrics.Mean(name="N-gloss"), tf.keras.metrics.Mean(name="C-gloss"),
                              tf.keras.metrics.Mean(name="Cycle"), tf.keras.metrics.Mean(name="Bypass")]
        self.hloss_tracker = [tf.keras.metrics.Mean(name="Pseudo"), tf.keras.metrics.Mean(name="Noise-consistency")]
        self.dloss_tracker = [tf.keras.metrics.Mean(name="N-dloss"), tf.keras.metrics.Mean(name="C-dloss"),
                              tf.keras.metrics.Mean(name="Boosting")]

        self.val_tracker = [tf.keras.metrics.Mean(name="psnr_g"),
                            tf.keras.metrics.Mean(name="ssim_g"),
                            tf.keras.metrics.Mean(name="psnr_h"),
                            tf.keras.metrics.Mean(name="ssim_h"),
                            tf.keras.metrics.Mean(name="psnr_e"),
                            tf.keras.metrics.Mean(name="ssim_e"),
                            ]

        self.cnt = 0.0

        """
        self.checkpoint = Checkpoint(dict(G=self.G,
                                D_A=self.D_A,
                                D_B=self.D_B,
                                H = self.H,
                                G_optimizer=self.G_optimizer,
                                D_optimizer=self.D_optimizer,
                                H_optimizer=self.H_optimizer),
                           py.join(self.output_dir, 'checkpoints'),
                           max_to_keep=5)"""


        for tracker in self.gloss_tracker:
            tracker.reset_state()
        for tracker in self.hloss_tracker:
            tracker.reset_state()
        for tracker in self.val_tracker:
            tracker.reset_state()
        for tracker in self.dloss_tracker:
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
            A2B2A = self.G(A2B, training=True)
            B2A2B = self.G(B2A, z=z1, training=True)

            A2B_d_logits, _ = self.D_B(A2B, training=True)
            B2A_d_logits, _ = self.D_A(B2A, training=True)

            A2B_g_loss = self.g_loss_fn(A2B_d_logits)
            B2A_g_loss = self.g_loss_fn(B2A_d_logits)
            A2B2A_cycle_loss = self.cycle_loss_fn(clean, A2B2A)
            B2A2B_cycle_loss = self.cycle_loss_fn(noisy, B2A2B)


            ### bypass loss
            clean_H = noisy - self.H(noisy, training=False)
            noisy_H = clean + self.H(noisy, training=False)
            y_hat_j = self.G(noisy_H, training=True)

            bypass_loss = tf.reduce_mean(tf.abs(B2A - clean_H)) + tf.reduce_mean(tf.abs(clean - y_hat_j))

            G_loss = (A2B_g_loss + B2A_g_loss) + (
                        A2B2A_cycle_loss + B2A2B_cycle_loss + bypass_loss) * self.cycle_loss_weight

        G_grad = t.gradient(G_loss, self.G.trainable_variables)
        self.G_optimizer.apply_gradients(zip(G_grad, self.G.trainable_variables))
        self.gloss_tracker[0].update_state(A2B_g_loss)
        self.gloss_tracker[1].update_state(B2A_g_loss)
        self.gloss_tracker[2].update_state(A2B2A_cycle_loss + B2A2B_cycle_loss)
        self.gloss_tracker[3].update_state(bypass_loss)


        return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                          'B2A_g_loss': B2A_g_loss,
                          'A2B2A_cycle_loss': A2B2A_cycle_loss,
                          'B2A2B_cycle_loss': B2A2B_cycle_loss,
                          'bypass_loss': bypass_loss}


    def train_D(self, A, B, A2B, B2A):
        with tf.GradientTape() as t:
            A_d_logits, real_clean = self.D_A(A, training=True)
            B2A_d_logits, fake_clean = self.D_A(B2A, training=True)
            B_d_logits, real_noisy = self.D_B(B, training=True)
            A2B_d_logits, fake_noisy = self.D_B(A2B, training=True)

            A_d_loss, B2A_d_loss = self.d_loss_fn(A_d_logits, B2A_d_logits)
            B_d_loss, A2B_d_loss = self.d_loss_fn(B_d_logits, A2B_d_logits)
            """
            D_A_gp = gan.gradient_penalty(functools.partial(D_A, training=True), A, B2A,
                                          mode=args.gradient_penalty_mode)
            D_B_gp = gan.gradient_penalty(functools.partial(D_B, training=True), B, A2B,
                                          mode=args.gradient_penalty_mode)
            """

            ######## bst loss
            clean_H = B - self.H(B, training=False)
            noisy_H = A + self.H(B, training=False)
            fake_clean, _ = self.D_A(clean_H, training=True)
            fake_noisy, _ = self.D_B(noisy_H, training=True)

            bst_loss = tf.reduce_mean(tf.math.square(fake_noisy)) + tf.reduce_mean(tf.math.square(fake_clean))

            D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) +  bst_loss

        D_grad = t.gradient(D_loss, self.D_A.trainable_variables + self.D_B.trainable_variables)
        self.D_optimizer.apply_gradients(zip(D_grad, self.D_A.trainable_variables + self.D_B.trainable_variables))
        self.dloss_tracker[0].update_state(B_d_loss + A2B_d_loss)
        self.dloss_tracker[1].update_state(A_d_loss + B2A_d_loss)
        self.dloss_tracker[2].update_state(bst_loss)

        return {'A_d_loss': A_d_loss + B2A_d_loss,
                'B_d_loss': B_d_loss + A2B_d_loss,
                'bst_loss': bst_loss
                }

    def train_H(self, A, B, z):
        with tf.GradientTape() as t:
            z1 = z + tf.random.normal(tf.shape(z), mean=0.0, stddev=1.0, dtype=tf.float32) * 1e-1
            # z1 = z
            # z1 = tf.random.normal([1, 64], 0, 1, dtype=tf.float32)

            n_hat_i = self.H(B, training=True)  # A가 noisy B가 clean noise
            n_bar_i = B - self.G(B, training=True)  # nosiy - clean noise
            x_hat_j = self.G(A, z=z1, training=True)  # fake noisy
            n_tilda_j = self.H(x_hat_j, training=True)  # fake noisy noise

            pseudo_loss = tf.reduce_mean(tf.abs(n_hat_i - n_bar_i))
            noise_consistency = tf.reduce_mean(tf.abs(x_hat_j - A - n_tilda_j))
            loss = pseudo_loss  + noise_consistency

        H_grad = t.gradient(loss, self.H.trainable_variables)
        self.H_optimizer.apply_gradients(zip(H_grad, self.H.trainable_variables))
        self.hloss_tracker[0].update_state(pseudo_loss)
        self.hloss_tracker[1].update_state(noise_consistency)

        return {'pseudo_loss': pseudo_loss,
                'noise_consistency': noise_consistency}


    def train_step(self, data):
        A, B = data
        tf.random.set_seed(5)
        z = tf.random.normal([1, 64], 0, 1, dtype=tf.float32)
        A2B, B2A, G_loss = self.train_G(A, B, z)

        # cannot autograph `A2B_pool`
        #A2B = self.A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
        #B2A = self.B2A_pool(B2A)  # because of the communication between CPU and GPU

        D_loss = self.train_D(A, B, A2B, B2A)
        H_loss = self.train_H(A, B, z)



        return{**G_loss, **D_loss, **H_loss}


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