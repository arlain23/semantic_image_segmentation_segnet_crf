import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torch.nn.functional as F
import time
from torch.autograd import Variable
import PIL
import config as cfg
from voc_utils import CrossEntropyLoss2d
from segnet import SegNet
import voc_utils
import pathlib
from segnet_delta import SegNet as SegNet2


def create_loss_and_optimizer(net, learning_rate=0.001):
    # Loss function
    # loss = CrossEntropyLoss2d()
    loss = torch.nn.CrossEntropyLoss()
    # loss = torch.nn.BCEWithLogitsLoss()

    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    return loss, optimizer


def train_net(net, n_epochs, learning_rate, loader, val_loader, n_device):
    # Print all of the hyperparameters of the training iteration:
    print("===== HYPER PARAMETERS =====")
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    saved_epoch = 0
    epoch = 0
    # Get training data
    n_batches = len(loader)

    # Create our loss and optimizer functions
    loss, optimizer = create_loss_and_optimizer(net, learning_rate)

    # Load checkpoint
    model_save_path = os.path.join(cfg.MODEL_CKPT_PATH, 'train')
    if not os.path.exists(model_save_path):
        pathlib.Path(model_save_path).mkdir(parents=True, exist_ok=True)

    model_save_path = os.path.join(model_save_path, cfg.MODEL_CKPT_FILE_NAME)
    print("file path ", model_save_path)
    if os.path.isfile(model_save_path) and os.path.getsize(model_save_path) > 0:
        checkpoint = torch.load(model_save_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        saved_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        net.train()

    # Time for printing
    training_start_time = time.time()
    if saved_epoch >= n_epochs:
        epoch = saved_epoch
        print("model loaded from checkpoint")
    else:
        # Loop for n_epochs
        for epoch in range(n_epochs):

            running_loss = 0.0
            print_every = n_batches // 10
            start_time = time.time()
            total_train_loss = 0

            for i, (inputs, labels) in enumerate(train_loader):
                # Get inputs
                inputs = inputs.to(n_device)
                labels = labels.to(n_device)

                # Wrap them in a Variable object
                inputs, labels = Variable(inputs), Variable(labels)

                # Set the parameter gradients to zero
                optimizer.zero_grad()

                # Forward pass, backward pass, optimize
                outputs = net(inputs)
                loss_size = loss(outputs, labels)
                loss_size.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss_size.item()
                total_train_loss += loss_size.item()

                # Print every 10th batch of an epoch
                if (i + 1) % (print_every + 1) == 0:
                    print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every,
                        time.time() - start_time))

                    # Reset running loss and time
                    running_loss = 0.0
                    start_time = time.time()

            # # At the end of the epoch, do a pass on the validation set
            # total_val_loss = 0
            # for inputs, labels in val_loader:
            #     # Get inputs
            #     inputs = inputs.to(n_device)
            #     labels = labels.to(n_device)
            #
            #     # Wrap tensors in Variables
            #     inputs, labels = Variable(inputs), Variable(labels)
            #
            #     # Forward pass
            #     val_outputs = net(inputs)
            #     val_loss_size = loss(val_outputs, labels)
            #     total_val_loss += val_loss_size.item()
            # print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))

        print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, model_save_path)


def test_net(net, loader, n_device):
    save_path = os.path.join(cfg.MODEL_CKPT_PATH, 'test')
    if not os.path.exists(save_path):
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    loss, optimizer = create_loss_and_optimizer(net, 0.01)
    # Test the model
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        for i, (inputs, target) in enumerate(loader):
            inputs = inputs.to(n_device)
            target = target.to(n_device)

            # Wrap them in a Variable object
            inputs, target = Variable(inputs), Variable(target)
            output = net(inputs)

            for j in range(output.size()[0]):
                # prediction = output.data[j].max(0)[0].cpu().numpy()

                softmax = torch.nn.Softmax(dim=0)
                sample_output = softmax(output.data[j])
                for k in range(21):
                    print("O: ", output.data[j][k][125][125])
                    print("S: ", sample_output.data[k][125][125])
                    print("")

                print("output ", output.size())
                print("target ", target.size())
                loss_size = loss(output, target)

                print("loss", loss_size)

                prediction = np.asarray(np.argmax(a=sample_output.cpu().numpy(), axis=0), dtype=np.uint8)
                prediction = voc_utils.colorize_mask(prediction)
                prediction.save(os.path.join(save_path, 'image' + str(i) + '.png'))

                init_img = transforms.ToPILImage()(inputs.cpu().data[j])
                init_img.save(os.path.join(save_path, 'image' + str(i) + '_i.png'))

                ground_truth = voc_utils.colorize_mask(np.asarray(target.cpu().data[j].numpy(), dtype=np.uint8))
                ground_truth.save(os.path.join(save_path, 'image' + str(i) + '_gt.png'))

            print('%d / %d' % (i + 1, len(test_loader)))


def get_rgb_image(x):
    if list(x.size())[0] == 1:
        x = x.repeat(3, 1, 1)
    return x


def rotate_image(x):
    width, height = x.size
    if width < height:
        transforms.RandomRotation(90)
    return x


def custom_tensor_transforms(x):
    # x = get_rgb_image(x)
    return x.mul_(255)


def custom_image_transforms(img):
    img = np.array(img)[:, :, ::-1]
    return PIL.Image.fromarray(img.astype(np.uint8))


def mask_to_tensor(x):
    return torch.from_numpy(np.array(x, dtype=np.int32)).long()


def img_to_rgb2(img):
    return img.convert('RGB')


if __name__ == "__main__":

    # set a standard random seed for reproducible results.
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # The compose function allows for multiple transforms
    # transforms.ToTensor() converts our PILImage to a tensor of shape (C x H x W) in the range [0,1]
    # transforms.Normalize(mean,std) normalizes a tensor to a (mean, std) for (R, G, B)
    transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=PIL.Image.BILINEAR),
            transforms.Lambda(custom_image_transforms),
            transforms.ToTensor(),
            transforms.Lambda(custom_tensor_transforms),
            # transforms.Normalize([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])
         ])

    target_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=PIL.Image.NEAREST),
        transforms.Lambda(mask_to_tensor),
        # transforms.Normalize([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=PIL.Image.BILINEAR),
        transforms.Lambda(img_to_rgb2),
        transforms.ToTensor(),
        # transforms.Normalize([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])
    ])

    train_set = torchvision.datasets.VOCSegmentation(
        root='./voc/voc2007',
        year='2007',
        image_set='train',
        download=True,
        transform=transform,
        target_transform=target_transform)
    test_set = torchvision.datasets.VOCSegmentation(
        root='./voc/voc2007',
        year='2007',
        image_set='train',
        download=True,
        transform=transform,
        target_transform=target_transform)

    print('Voc2007 train set has {} images'.format(len(train_set)))

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=cfg.BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=cfg.BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=2)

    validation_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                    batch_size=cfg.BATCH_SIZE,
                                                    shuffle=True,
                                                    num_workers=2)

    # myCNN = CNN(len(classes)).to(device)

    # https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch
    # Load the pretrained model from pytorch
    # alexnet = torchvision.models.alexnet()
    # alexnet.load_state_dict(utils.model_zoo.load_url(cfg.MODEL_URLS['alexnet']))
    #
    # # Freeze training for all layers
    # for param in alexnet.features.parameters():
    #     param.require_grad = False
    #
    # # Newly created modules have require_grad=True by default
    # num_features = alexnet.classifier[-1].in_features
    # features = list(alexnet.classifier.children())[:-1]  # Remove last layer
    # features.extend([nn.Linear(num_features, 21)])  # Add our layer with 4 outputs
    # alexnet.classifier = nn.Sequential(*features)  # Replace the model classifier
    # alexnet = alexnet.to(device)

    segnet = SegNet2(in_channels=3, out_channels=cfg.NUMBER_OF_CLASSES)
    segnet.load_pretrained_weights()
    segnet.to(device)

    train_net(net=segnet, n_epochs=cfg.NUM_EPOCHS, learning_rate=cfg.LEARNING_RATE, loader=train_loader,
              val_loader=validation_loader, n_device=device)

    test_net(net=segnet, loader=test_loader, n_device=device)
