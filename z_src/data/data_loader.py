import numpy as np
import torch
import PIL
import torchvision
import torchvision.transforms as transforms
import z_src.utils.config as cfg


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


def get_data_loaders():
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=PIL.Image.BILINEAR),
        # transforms.Lambda(custom_image_transforms),
        transforms.ToTensor(),
        transforms.Lambda(custom_tensor_transforms),
        # transforms.Normalize([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])
    ])

    target_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=PIL.Image.BILINEAR),
        transforms.Lambda(mask_to_tensor),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
        # transforms.ToPILImage()
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

    # test_set = train_set
    test_set = torchvision.datasets.VOCSegmentation(
        root='./voc/voc2007',
        year='2007',
        image_set='trainval',
        download=True,
        transform=transform,
        target_transform=target_transform)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=cfg.BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=cfg.BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=2)

    return train_loader, test_loader
