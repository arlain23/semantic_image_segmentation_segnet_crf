import numpy as np
import z_src.superpixel.slic as SLIC
from z_src.superpixel.superpixel_dto import SuperpixelDTO
import pydensecrf_src.utils as utils
import z_src.utils.config as CFG


class SuperpixelImage:
    def __init__(self, image, probability_map):
        self.image = image
        self.superpixels_segments = SLIC.get_superpixels(image)
        self.superpixel_dict = {}
        self.number_of_superpixels = 0
        B, C, self.H, self.W = probability_map.shape
        for i in range(self.H):
            for j in range(self.W):
                superpixel_index = self.superpixels_segments[i][j]
                r = image[i, j, 0]
                g = image[i, j, 1]
                b = image[i, j, 2]
                pixel_probs = probability_map[0, :, i, j]
                if superpixel_index in self.superpixel_dict:
                    dto = self.superpixel_dict[superpixel_index]
                    dto.append_pixel(x=j, y=i, r=r, g=g, b=b, probs=pixel_probs)
                else:
                    dto = SuperpixelDTO(superpixel_index)
                    dto.append_pixel(x=j, y=i, r=r, g=g, b=b, probs=pixel_probs)
                    self.superpixel_dict[superpixel_index] = dto
                    self.number_of_superpixels += 1

    def get_superpixel_features(self):
        number_of_features = self.number_of_superpixels * 6
        superpixel_array = np.zeros(number_of_features, dtype=int)
        i = 0
        for superpixel_index in self.superpixel_dict:
            superpixel = self.superpixel_dict[superpixel_index]
            r, g, b = superpixel.get_mean_rgb()
            x, y = superpixel.get_centroid_coordinates()

            superpixel_array[i+0] = superpixel_index
            superpixel_array[i+1] = x
            superpixel_array[i+2] = y
            superpixel_array[i+3] = r
            superpixel_array[i+4] = g
            superpixel_array[i+5] = b

            i += 6
        return superpixel_array

    def get_probs_for_superpixels(self):
        image_probs = np.zeros((CFG.NUMBER_OF_CLASSES, self.number_of_superpixels), dtype=float)

        i = 0
        for superpixel_index in self.superpixel_dict:
            superpixel = self.superpixel_dict[superpixel_index]
            probs = superpixel.get_mean_probs()

            image_probs[:, i] = probs

            i += 1

        return image_probs

    def get_probabilities_for_image(self, crf_output):
        C = CFG.NUMBER_OF_CLASSES
        result_image = np.zeros(shape=(C, self.H, self.W), dtype=float)

        for superpixel_index in self.superpixel_dict:
            superpixel = self.superpixel_dict[superpixel_index]
            pixel_data = superpixel.get_pixel_data()
            probabilities = crf_output[:, superpixel_index]

            for p in pixel_data:
                result_image[:, p.y, p.x] = probabilities

        return result_image
