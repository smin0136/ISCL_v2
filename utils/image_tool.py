import numpy as np
import os
from PIL import Image
import skimage.io as iio
import skimage.color as color
import skimage.transform as transform


def normalize(data, mean, std):
    return (data - mean) / std


def denormalize(data, mean, std):
    return (data * std) + mean


def gaussian_noise(img, mean, std):
    # The range of image must be [0, 255]
    assert img.ndim == 3 or img.ndim == 2, ("Check the dimension of input")
    gauss = np.random.normal(mean, std, img.shape)
    return np.array(img + gauss, dtype=np.float32)


def salt_and_pepper_noise(img, density):
    assert img.ndim == 3 or img.ndim == 2, ("Check the dimension of input")
    sp_ratio = 0.5  # ratio between salt and pepper
    n_salt = int(img.size * density * sp_ratio)
    n_pepper = int(img.size * density * (1 - sp_ratio))
    noisy = np.copy(img)
    idx = [np.random.randint(0, length - 1, n_salt) for length in img.shape]
    noisy[idx] = 1
    idx = [np.random.randint(0, length - 1, n_pepper) for length in img.shape]
    noisy[idx] = 0
    return noisy


def speckle_noise(img, std):
    assert img.ndim == 3 or img.ndim == 2, ("Check the dimension of input")
    assert std <= 1 and std >= 0, ("Standard deviation should be in the range [0,1]")
    noise = np.random.uniform(((12 * std) ** 2) * -.5, ((12 * std) ** 2) * .5, img.shape)
    return np.array(img + noise * img, dtype=np.float32)


def image_read(path):
    output = []
    if ('.tif' not in path) and ('.png' not in path):
        tif_list = sorted(os.listdir(path))
        for name in tif_list:
            im = Image.open(path + "/" + name)
            output.append(np.array(im, dtype=np.float32))
    else:
        im = Image.open(path)
        output = np.array(im, dtype=np.float32)
    return output


def image_division(image, patch_size):
    assert (image.ndim <= 4 and image.ndim >= 3), ("Check the dimension of inputs")  # n, h, w or n, h, w, 1
    if image.ndim == 3:
        image = np.expand_dims(image, axis=-1)

    n = len(image)
    patch_x, patch_y = patch_size
    output = []
    for i in range(0, n):
        temp = image[i]
        x, y, z = np.shape(temp)
        p = int(np.ceil(x / patch_x))
        q = int(np.ceil(y / patch_y))
        for j in range(0, p):
            for k in range(0, q):
                if j == p - 1:
                    if k == q - 1:
                        output.append(temp[-patch_x:, -patch_y:, 0:z])
                    else:
                        output.append(temp[-patch_x:, k * patch_y:(k + 1) * patch_y, 0:z])
                else:
                    if k == q - 1:
                        output.append(temp[j * patch_x:(j + 1) * patch_x, -patch_y:, 0:z])
                    else:
                        output.append(temp[j * patch_x:(j + 1) * patch_x, k * patch_y:(k + 1) * patch_y, 0:z])

    return np.array(output, dtype=np.float32)


def image_augmentation(x):
    assert (x.ndim == 4 or x.ndim == 3), ("Check the dimension of inputs")
    if x.ndim == 3:
        n, w, h = np.shape(x)[0:3]
        out = np.zeros([n * 8, w, h], dtype=np.float32)
        for f in range(0, 2):
            for r in range(0, 4):
                if f == 0 and r == 0:
                    out[:n] = x
                    continue
                for i in range(0, n):
                    out[i + (n * r) + f * 4 * n] = np.flip(np.rot90(x[i], r), f)
    elif x.ndim == 4:
        n, w, h, z = np.shape(x)[0:4]
        out = np.zeros([n * 8, w, h, z], dtype=np.float32)
        for f in range(0, 2):
            for r in range(0, 4):
                if f == 0 and r == 0:
                    out[:n] = x
                    continue
                for i in range(0, n):
                    for j in range(0, z):
                        out[i + (n * r) + f * 4 * n, :, :, j] = np.flip(np.rot90(x[i, :, :, j], r), f)
    return out


def image_division_overlap(image, patch_size):
    assert (image.ndim <= 4 and image.ndim >= 3), ("Check the dimension of inputs")  # n, h, w or n, h, w, 1
    if image.ndim == 3:
        image = np.expand_dims(image, axis=-1)

    n = len(image)
    patch_x, patch_y = patch_size
    output = []
    for i in range(0, n):
        temp = image[i]
        x, y, z = np.shape(temp)
        slide_x = int(patch_x / 2)
        slide_y = int(patch_y / 2)
        p = int(np.ceil(x / slide_x)) - 1
        q = int(np.ceil(y / slide_y)) - 1

        for j in range(0, p):
            for k in range(0, q):
                if j == p - 1:
                    if k == q - 1:
                        output.append(temp[-patch_x:, -patch_y:, 0:z])
                    else:
                        output.append(temp[-patch_x:, k * slide_y: k * slide_y + patch_y, 0:z])
                else:
                    if k == q - 1:
                        output.append(temp[j * slide_x: j * slide_x + patch_x, -patch_y:, 0:z])
                    else:
                        output.append(temp[j * slide_x: j * slide_x + patch_x, k * slide_y: k * slide_y + patch_y, 0:z])

    return np.array(output, dtype=np.float32)


def image_division_zero(image, patch_size):
    assert (image.ndim <= 4 and image.ndim >= 3), ("Check the dimension of inputs")  # n, h, w or n, h, w, 1
    # if image.ndim == 3:
    #    image = np.expand_dims(image, axis=-1)

    n = len(image)
    patch_x, patch_y = patch_size
    output = []
    for i in range(0, n):
        temp = image[i]
        x, y = np.shape(temp)
        p = int(np.ceil(x / patch_x))
        q = int(np.ceil(y / patch_y))
        for j in range(0, p):
            for k in range(0, q):
                if j == p - 1:
                    if k == q - 1:
                        tmp = np.pad(temp[-patch_x:, -patch_y:], ((20, 20), (20, 20)), 'constant', constant_values=0)
                        output.append(tmp)
                    else:
                        tmp = np.pad(temp[-patch_x:, k * patch_y:(k + 1) * patch_y], ((20, 20), (20, 20)), 'constant',
                                     constant_values=0)
                        output.append(tmp)
                else:
                    if k == q - 1:
                        tmp = np.pad(temp[j * patch_x:(j + 1) * patch_x, -patch_y:], ((20, 20), (20, 20)), 'constant',
                                     constant_values=0)
                        output.append(tmp)
                    else:
                        tmp = np.pad(temp[j * patch_x:(j + 1) * patch_x, k * patch_y:(k + 1) * patch_y],
                                     ((20, 20), (20, 20)), 'constant', constant_values=0)
                        output.append(tmp)
    output = np.array(output, dtype=np.float32)
    if output.ndim == 3:
        output = np.expand_dims(output, axis=-1)

    return output

def imread(path, as_gray=False, **kwargs):
    """Return a float64 image in [-1.0, 1.0]."""
    image = iio.imread(path, as_gray, **kwargs)
    if image.dtype == np.uint8:
        image = image / 127.5 - 1
    elif image.dtype == np.uint16:
        image = image / 32767.5 - 1
    elif image.dtype in [np.float32, np.float64]:
        image = image * 2 - 1.0
    else:
        raise Exception("Inavailable image dtype: %s!" % image.dtype)
    return image


def imwrite(image, path, quality=95, **plugin_args):
    """Save a [-1.0, 1.0] image."""
    iio.imsave(path, im2uint(image), quality=quality, **plugin_args)


def imshow(image):
    """Show a [-1.0, 1.0] image."""
    iio.imshow(im2uint(image))


show = iio.show

def _check(images, dtypes, min_value=-np.inf, max_value=np.inf):
    # check type
    assert isinstance(images, np.ndarray), '`images` should be np.ndarray!'

    # check dtype
    dtypes = dtypes if isinstance(dtypes, (list, tuple)) else [dtypes]
    assert images.dtype in dtypes, 'dtype of `images` shoud be one of %s!' % dtypes

    # check nan and inf
    assert np.all(np.isfinite(images)), '`images` contains NaN or Inf!'

    # check value
    if min_value not in [None, -np.inf]:
        l = '[' + str(min_value)
    else:
        l = '(-inf'
        min_value = -np.inf
    if max_value not in [None, np.inf]:
        r = str(max_value) + ']'
    else:
        r = 'inf)'
        max_value = np.inf
    assert np.min(images) >= min_value and np.max(images) <= max_value, \
        '`images` should be in the range of %s!' % (l + ',' + r)


def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    """Transform images from [-1.0, 1.0] to [min_value, max_value] of dtype."""
    _check(images, [np.float32, np.float64], -1.0, 1.0)
    dtype = dtype if dtype else images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)


def float2im(images):
    """Transform images from [0, 1.0] to [-1.0, 1.0]."""
    _check(images, [np.float32, np.float64], 0.0, 1.0)
    return images * 2 - 1.0


def float2uint(images):
    """Transform images from [0, 1.0] to uint8."""
    _check(images, [np.float32, np.float64], -0.0, 1.0)
    return (images * 255).astype(np.uint8)


def im2uint(images):
    """Transform images from [-1.0, 1.0] to uint8."""
    return to_range(images, 0, 255, np.uint8)


def im2float(images):
    """Transform images from [-1.0, 1.0] to [0.0, 1.0]."""
    return to_range(images, 0.0, 1.0)


def uint2im(images):
    """Transform images from uint8 to [-1.0, 1.0] of float64."""
    _check(images, np.uint8)
    return images / 127.5 - 1.0


def uint2float(images):
    """Transform images from uint8 to [0.0, 1.0] of float64."""
    _check(images, np.uint8)
    return images / 255.0


def cv2im(images):
    """Transform opencv images to [-1.0, 1.0]."""
    images = uint2im(images)
    return images[..., ::-1]


def im2cv(images):
    """Transform images from [-1.0, 1.0] to opencv images."""
    images = im2uint(images)
    return images[..., ::-1]



rgb2gray = color.rgb2gray
gray2rgb = color.gray2rgb

imresize = transform.resize
imrescale = transform.rescale


def immerge(images, n_rows=None, n_cols=None, padding=0, pad_value=0):
    """Merge images to an image with (n_rows * h) * (n_cols * w).

    Parameters
    ----------
    images : numpy.array or object which can be converted to numpy.array
        Images in shape of N * H * W(* C=1 or 3).

    """
    images = np.array(images)
    n = images.shape[0]
    if n_rows:
        n_rows = max(min(n_rows, n), 1)
        n_cols = int(n - 0.5) // n_rows + 1
    elif n_cols:
        n_cols = max(min(n_cols, n), 1)
        n_rows = int(n - 0.5) // n_cols + 1
    else:
        n_rows = int(n ** 0.5)
        n_cols = int(n - 0.5) // n_rows + 1

    h, w = images.shape[1], images.shape[2]
    shape = (h * n_rows + padding * (n_rows - 1),
             w * n_cols + padding * (n_cols - 1))
    if images.ndim == 4:
        shape += (images.shape[3],)
    img = np.full(shape, pad_value, dtype=images.dtype)

    for idx, image in enumerate(images):
        i = idx % n_cols
        j = idx // n_cols
        img[j * (h + padding):j * (h + padding) + h,
            i * (w + padding):i * (w + padding) + w, ...] = image

    return img


class ItemPool:

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.items = []

    def __call__(self, in_items):
        # `in_items` should be a batch tensor

        if self.pool_size == 0:
            return in_items

        out_items = []
        for in_item in in_items:
            if len(self.items) < self.pool_size:
                self.items.append(in_item)
                out_items.append(in_item)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, len(self.items))
                    out_item, self.items[idx] = self.items[idx], in_item
                    out_items.append(out_item)
                else:
                    out_items.append(in_item)
        return tf.stack(out_items, axis=0)
