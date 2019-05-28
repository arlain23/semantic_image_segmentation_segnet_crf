import numpy as np
import torch
from z_src.utils import config as cfg
from z_src.model.segnet_delta import SegNet as SegNet2

from z_src.net.net_trainer import train_net
from z_src.net.net_tester import test_net
from z_src.data.data_loader import get_data_loaders

if __name__ == "__main__":

    # initial setup
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    segnet = SegNet2(in_channels=3, out_channels=cfg.NUMBER_OF_CLASSES)
    segnet.load_pretrained_weights()
    segnet.to(device)

    train_loader, test_loader = get_data_loaders()

    segnet = train_net(net=segnet, n_epochs=cfg.NUM_EPOCHS, learning_rate=cfg.LEARNING_RATE, loader=train_loader,
                       n_device=device)

    test_net(net=segnet, loader=test_loader, n_device=device)
