import torch.nn.functional as F
from torch import nn
from PIL import Image
import numpy as np
import config as cfg


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=0), targets)


def colorize_mask(mask):

    # 0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow,
    # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train,
    # 20=tv/monitor

    # mask: numpy array of the mask
    palette = get_palette(cfg.NUMBER_OF_CLASSES)
    # new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    output_im = Image.fromarray(mask)
    output_im.putpalette(palette)
    return output_im


def colorize_image(output_im):
    palette = get_palette(cfg.NUMBER_OF_CLASSES)
    output_im.putpalette(palette)
    return output_im


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette
