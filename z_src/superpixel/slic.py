# import the necessary packages
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
from skimage.util import img_as_float

import z_src.utils.config as cfg


def get_superpixels(image):
    # load the image and convert it to a floating point data type
    image = img_as_float(image)

    segments = slic(image, n_segments=cfg.SLIC['NUMBER_OF_SEGMENTS'], sigma=cfg.SLIC['SIGMA'])

    return segments


def show_segments(image, segments):
    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (cfg.SLIC['NUMBER_OF_SEGMENTS']))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments))
    plt.axis("off")

    plt.show()
