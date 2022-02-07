import tensorflow as tf
import tensorflow_addons as tfa
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np

import sys
sys.path.append('/home/Alexandrite/smin/ISCL_v2/')

from utils.image_tool import *
import pylib as py
from utils.parser import parse_args
from models.trainer import Trainer
import numpy as nps
from PIL import Image

def ISCL(args, clean, noisy, test_clean, test_noisy):

    clean = image_division(clean, patch_size=(256, 256))
    noisy = image_division(noisy, patch_size=(256, 256))

    print(clean.shape)

    dataset = tf.data.Dataset.from_tensor_slices(
        (clean, noisy))  # If you don't have enough memory, you can use tf.data.Dataset.from_generator
    dataset = dataset.cache().repeat().shuffle(len(clean), reshuffle_each_iteration=True).batch(1).prefetch(
        tf.data.experimental.AUTOTUNE)

    val_set = tf.data.Dataset.from_tensor_slices(
        (test_clean, test_noisy))
    val_set = val_set.cache().batch(1).prefetch(tf.data.experimental.AUTOTUNE)

    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    BATCH_SIZE = args.batch_size * strategy.num_replicas_in_sync


    model = Trainer(args)
    model.compile()

    model.fit(dataset, epochs=args.epochs, validation_data=val_set, steps_per_epoch=len(clean))

    # Testing

    pred_noisy = np.zeros(np.shape(test_noisy), dtype=np.float32)
    pred_clean = np.zeros(np.shape(test_noisy), dtype=np.float32)
    pred_ex = np.zeros(np.shape(test_noisy), dtype=np.float32)
    pred_en = np.zeros(np.shape(test_noisy), dtype=np.float32)

    for i in range(0, len(test_noisy)):
        tempg, tempf, temph, tempe = model.predict(test_clean[i:i + 1, :, :, np.newaxis], test_noisy[i:i + 1, :, :, np.newaxis])
        pred_noisy[i] = np.squeeze(tempg)  # noisy: [1, H, W, 1] or [1, H, W, C]
        pred_clean[i] = np.squeeze(tempf)
        pred_ex[i] = np.squeeze(temph)
        pred_en[i] = np.squeeze(tempe)

    return pred_noisy, pred_clean, pred_ex, pred_en

def main():
    args = parse_args()
    if args is None:
        print("args is none")
        exit()

    clean_data = np.array(image_read(py.join(args.datasets_dir, args.dataset, 'db_train')), dtype=np.float32)
    noisy_data = np.array(image_read(py.join(args.datasets_dir, args.dataset, 'train_noisy')), dtype=np.float32)

    clean_val = np.array(image_read(py.join(args.datasets_dir, args.dataset, 'db_valid')), dtype=np.float32)
    noisy_val = np.array(image_read(py.join(args.datasets_dir, args.dataset, 'noisy')) , dtype=np.float32)

    np.random.shuffle(clean_data)
    np.random.shuffle(noisy_data)


    output_dir = py.join(args.datasets_dir, 'output', args.output_date, args.dir_num)
    py.mkdir(output_dir)
    """
    temp = np.array(clean_val[35], dtype=np.uint8)
    im = Image.fromarray(temp)

    im.save(py.join(output_dir, "temp.png"))

    print(np.min(clean_val), np.max(clean_val))"""

    clean_data /= 255
    clean_data = clean_data*2 -1
    noisy_data /= 255
    noisy_data = noisy_data*2 -1

    clean_val /= 255.0
    clean_val = clean_val*2 -1
    noisy_val /= 255.0
    noisy_val = noisy_val*2 -1



    pred_noisy, pred_clean, pred_ex, pred_en = ISCL(args, clean_data, noisy_data, clean_val, noisy_val)

    clean_val = (clean_val + 1) * 0.5 * 255
    noisy_val = (noisy_val + 1) * 0.5 * 255



    save_dir = py.join(output_dir, 'samples_testing', 'B2A')
    py.mkdir(save_dir)

    psnr_pred_g = []
    ssim_pred_g = []
    psnr_pred_h = []
    ssim_pred_h = []
    psnr_pred_e = []
    ssim_pred_e = []

    psnr_g = 0.0
    ssim_g = 0.0
    psnr_h = 0.0
    ssim_h = 0.0
    psnr_e = 0.0
    ssim_e = 0.0

    for i in range(0, len(clean_val)):
        A = clean_val[i][np.newaxis, ...]
        B = noisy_val[i][np.newaxis, ...]
        A2B = pred_noisy[i][np.newaxis, ...]
        B2A = pred_clean[i][np.newaxis, ...]
        H2A = pred_ex[i][np.newaxis, ...]
        GnH = pred_en[i][np.newaxis, ...]

        A /= 255
        A = A * 2 -1
        B /= 255
        B = B * 2 - 1

        A = np.clip(A, -1.0, 1.0)
        B = np.clip(B, -1.0, 1.0)


        img = immerge(np.concatenate([A, A2B, H2A, B, B2A, GnH], axis=0), n_rows=2)
        imwrite(img, py.join(save_dir, 'noise2clean_%d.jpg' % i))

        clean_val[i] = clean_val[i] * 255
        pred_clean[i] = (pred_clean[i] + 1)*0.5*255
        pred_ex[i] = (pred_ex[i] + 1)*0.5*255
        pred_en[i] = (pred_en[i] + 1)*0.5*255


        psnr_g += psnr(clean_val[i], pred_clean[i], data_range=255.0)
        ssim_g += ssim(clean_val[i], pred_clean[i], data_range=255.0)

        psnr_h += psnr(clean_val[i], pred_ex[i], data_range=255.0)
        ssim_h += ssim(clean_val[i], pred_ex[i], data_range=255.0)

        psnr_e += psnr(clean_val[i], pred_en[i], data_range=255.0)
        ssim_e += ssim(clean_val[i], pred_en[i], data_range=255.0)


    psnr_pred_g.append(psnr_g / len(clean_val))
    ssim_pred_g.append(ssim_g / len(clean_val))

    psnr_pred_h.append(psnr_h / len(clean_val))
    ssim_pred_h.append(ssim_h / len(clean_val))

    psnr_pred_e.append(psnr_e / len(clean_val))
    ssim_pred_e.append(ssim_e / len(clean_val))

    print("g result PSNR: %f, SSIM %f" % (sum(psnr_pred_g), sum(ssim_pred_g)))
    print("h result PSNR: %f, SSIM %f" % (sum(psnr_pred_h), sum(ssim_pred_h)))
    print("e result PSNR: %f, SSIM %f" % (sum(psnr_pred_e), sum(ssim_pred_e)))


if __name__ == '__main__':
    main()
