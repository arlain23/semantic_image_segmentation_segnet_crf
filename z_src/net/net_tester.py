from z_src.utils.voc_utils import create_loss_and_optimizer
from z_src.utils.voc_utils import colorize_mask
import os
import pathlib
from z_src.utils import config as cfg
import torch
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
import z_src.net.crf as CRF
from z_src.superpixel.superpixel_image import SuperpixelImage


def test_net(net, loader, n_device):
    crf_with_superpixels = True

    print("TESTING PHASE")
    save_path = os.path.join(cfg.MODEL_CKPT_PATH, 'test')
    if not os.path.exists(save_path):
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    loss, optimizer = create_loss_and_optimizer(net, 0.01)
    # Test the model
    softmax = torch.nn.Softmax(dim=1)
    # CRF
    if crf_with_superpixels:
        postprocessor = CRF.setup_postprocessor_superpixels()
    else:
        postprocessor = CRF.setup_postprocessor()

    net.train()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        for i, (inputs, target) in enumerate(loader):
            j = 0
            # Get inputs
            inputs = inputs.to(n_device)
            target = target.to(n_device)

            # Wrap them in a Variable object
            inputs, target = Variable(inputs), Variable(target)

            # initial image object
            init_img = (np.asarray(inputs.cpu().numpy()[j], dtype=np.uint8))
            init_img = np.transpose(init_img, (1, 2, 0))

            # Forward pass, backward pass, optimize
            outputs = net(inputs)

            sample_output = softmax(outputs)
            sample_output_crf = sample_output.cpu().numpy()

            # Superpixel segmentation
            image_dto = SuperpixelImage(image=init_img, probability_map=sample_output_crf)
            superpixel_features = image_dto.get_superpixel_features()
            probs = image_dto.get_probs_for_superpixels()

            # DENSE CRF
            if crf_with_superpixels:
                sample_output_crf = postprocessor(superpixel_features, probs, True)
                sample_output_crf = image_dto.get_probabilities_for_image(crf_output=sample_output_crf)

                sample_output_crf_no_pairwise = postprocessor(superpixel_features, probs, False)
                sample_output_crf_no_pairwise = image_dto.get_probabilities_for_image(
                    crf_output=sample_output_crf_no_pairwise)
            else:
                sample_output_crf = postprocessor(inputs.cpu().numpy()[j], sample_output_crf[0], True)
                sample_output_crf_no_pairwise = postprocessor(inputs.cpu().numpy()[j], sample_output_crf[0], False)

            train_loss = loss(outputs, target)

            # Result with CRF
            prediction = np.asarray(np.argmax(a=sample_output_crf, axis=0), dtype=np.uint8)
            prediction = colorize_mask(prediction)
            prediction.save(os.path.join(save_path, 'image' + str(i) + '_res_crf.png'))

            # Result with CRF no pairwise
            prediction = np.asarray(np.argmax(a=sample_output_crf_no_pairwise, axis=0), dtype=np.uint8)
            prediction = colorize_mask(prediction)
            prediction.save(os.path.join(save_path, 'image' + str(i) + '_res_crf_U.png'))

            # Result without CRF
            prediction = np.asarray(np.argmax(a=sample_output[j].cpu().detach().numpy(), axis=0), dtype=np.uint8)
            prediction = colorize_mask(prediction)
            prediction.save(os.path.join(save_path, 'image' + str(i) + '_res.png'))

            # Save image and ground truth
            init_img = transforms.ToPILImage()(init_img)
            init_img.save(os.path.join(save_path, 'image' + str(i) + '.png'))

            ground_truth = colorize_mask(np.asarray(target.cpu().data[j].numpy(), dtype=np.uint8))
            ground_truth.save(os.path.join(save_path, 'image' + str(i) + '_gt.png'))

            print("{:d} / {:d} : {:.8f}".format(
                i + 1,  len(loader), train_loss.item()))



