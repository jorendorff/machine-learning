import PIL
from PIL.Image import Resampling

import numpy as np

INK_THRESHOLD = 16

def mutate_images(rng, images):
    orig_shape = images.shape
    assert len(images.shape) == 3
    assert images.dtype is np.dtype('uint8')
    mutants = np.array([mutate_image(rng, image) for image in images])
    assert mutants.shape == orig_shape
    return mutants


def mutate_image(rng, image):
    h, w = image.shape

    orig = PIL.Image.fromarray(image, mode='L')
    image = np.asarray(orig.rotate(5 * rng.normal()))

    ink = image >= INK_THRESHOLD
    ink_cols = ink.any(0)
    ink_rows = ink.any(1)
    margin_left = ink_cols.argmax()
    margin_right = ink_cols[::-1].argmax()
    margin_top = ink_rows.argmax()
    margin_bottom = ink_rows[::-1].argmax()

    orig = PIL.Image.fromarray(image, mode='L')
    dx = int(rng.uniform(-margin_left, margin_right + 1))
    dy = int(rng.uniform(-margin_top, margin_bottom + 1))
    mutant = orig.rotate(0, resample=Resampling.BICUBIC, translate=(dx, dy))

    return np.asarray(mutant)

