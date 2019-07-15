import numpy as np
import pydensecrf_src.densecrf as dcrf
import pydensecrf_src.utils as utils
import z_src.utils.config as cfg


class DenseCRFSuperpixels(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, superpixel_features, probs, include_pairwise):
        C = cfg.NUMBER_OF_CLASSES
        N = len(probs[0])

        unary = utils.unary_from_softmax(probs)
        unary = np.ascontiguousarray(unary)

        d = dcrf.DenseCRF2DGraph(N, C)
        d.setUnaryEnergy(unary)

        if include_pairwise:
            d.addPairwiseGaussian(sxy=self.pos_xy_std, superpixel_features=superpixel_features, compat=self.pos_w)
            d.addPairwiseBilateral(
                sxy=self.bi_xy_std, srgb=self.bi_rgb_std, superpixel_features=superpixel_features, compat=self.bi_w
            )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape(C, N)
        return Q


def setup_postprocessor_superpixels():
    postprocessor = DenseCRFSuperpixels(
        iter_max=cfg.CRF['ITER_MAX'],
        pos_xy_std=cfg.CRF['POS_XY_STD'],
        pos_w=cfg.CRF['POS_W'],
        bi_xy_std=cfg.CRF['BI_XY_STD'],
        bi_rgb_std=cfg.CRF['BI_RGB_STD'],
        bi_w=cfg.CRF['BI_W'],
    )
    return postprocessor


class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap, include_pairwise):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        if include_pairwise:
            d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
            d.addPairwiseBilateral(
                sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
            )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q


def setup_postprocessor():
    postprocessor = DenseCRF(
        iter_max=cfg.CRF['ITER_MAX'],
        pos_xy_std=cfg.CRF['POS_XY_STD'],
        pos_w=cfg.CRF['POS_W'],
        bi_xy_std=cfg.CRF['BI_XY_STD'],
        bi_rgb_std=cfg.CRF['BI_RGB_STD'],
        bi_w=cfg.CRF['BI_W'],
    )
    return postprocessor
