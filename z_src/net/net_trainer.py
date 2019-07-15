from z_src.utils.voc_utils import create_loss_and_optimizer
import os
import pathlib
from z_src.utils import config as cfg
import torch
import time
from torch.autograd import Variable
import numpy as np
from z_src.utils.voc_utils import colorize_mask
import torchvision.transforms as transforms


def train_net(net, n_epochs, learning_rate, loader, n_device):
    # Print all of the hyperparameters of the training iteration:
    print("TRAINING PHASE")
    save_path = os.path.join(cfg.MODEL_CKPT_PATH, 'train_val')

    saved_epoch = 0
    epoch = 0
    # Create our loss and optimizer functions
    loss, optimizer = create_loss_and_optimizer(net, learning_rate)

    # Load checkpoint
    model_save_path = os.path.join(cfg.MODEL_CKPT_PATH, 'model')
    if not os.path.exists(model_save_path):
        pathlib.Path(model_save_path).mkdir(parents=True, exist_ok=True)

    model_save_path = os.path.join(model_save_path, cfg.MODEL_CKPT_FILE_NAME)
    if os.path.isfile(model_save_path) and os.path.getsize(model_save_path) > 0:
        print("file path ", model_save_path)
        checkpoint = torch.load(model_save_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        saved_epoch = checkpoint['epoch']
        # train_loss = checkpoint['loss']
        net.train()

    # Time for printing
    training_start_time = time.time()
    if saved_epoch >= n_epochs:
        print("model loaded from checkpoint")
    else:
        n_epochs -= saved_epoch
        print_set_length = int(len(loader) / 10)
        for epoch in range(n_epochs):
            print("epoch:", (epoch + 1 + saved_epoch))
            current_loss = 0.0
            start_time = time.time()
            total_train_loss = 0.0

            for i, (inputs, target) in enumerate(loader):
                # Get inputs
                inputs = inputs.to(n_device)
                target = target.to(n_device)

                # Wrap them in a Variable object
                inputs, target = Variable(inputs), Variable(target)

                # Set the parameter gradients to zero
                optimizer.zero_grad()

                # Forward pass, backward pass, optimize
                outputs = net(inputs)

                train_loss = loss(outputs, target)
                train_loss.backward()
                optimizer.step()

                # Print statistics
                current_loss += train_loss.item()
                total_train_loss += train_loss.item()

                # Print every 10th batch of an epoch
                if (i % print_set_length) == (print_set_length - 1):
                    print("Epoch {}, {:d}% \t train_loss: {:.8f} took: {:.2f}s".format(
                        epoch + 1, int((i / print_set_length) * 10) + 1, current_loss / print_set_length,
                        time.time() - start_time))

                    # Reset running loss and time
                    current_loss = 0.0
                    start_time = time.time()

        print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

        torch.save({
            'epoch': (epoch + 1 + saved_epoch),
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, model_save_path)
    return net
