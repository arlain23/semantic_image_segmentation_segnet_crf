# Hyper parameters
NUM_EPOCHS = 82
BATCH_SIZE = 1
LEARNING_RATE = 0.01

IMAGE_SIZE = {
    'H': 256,
    'W': 315
}

MODEL_URLS = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_amazon': 'https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

MODEL_CKPT_FILE_NAME = 'model_1.pth'

MODEL_CKPT_PATH = 'ckpt'

NUMBER_OF_CLASSES = 21

CRF = {
    'ITER_MAX': 10,
    'POS_W': 3,
    'POS_XY_STD': 1,
    'BI_W': 4,
    'BI_XY_STD': 67,
    'BI_RGB_STD': 3,
}

SLIC = {
    'NUMBER_OF_SEGMENTS': 300,
    'SIGMA': 5
}
