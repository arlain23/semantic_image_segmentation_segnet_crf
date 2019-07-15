from z_src.superpixel.pixel_dto import PixelDTO
import numpy as np
import z_src.utils.config as CFG


class SuperpixelDTO:
    def __init__(self, index):
        self.index = index
        self.pixels_data = []

    def append_pixel(self, x, y, r, g, b, probs):
        self.pixels_data.append(PixelDTO(x=x, y=y, r=r, g=g, b=b, probs=probs))

    def get_mean_rgb(self):
        r = [p.r for p in self.pixels_data]
        g = [p.g for p in self.pixels_data]
        b = [p.b for p in self.pixels_data]

        mean_r = sum(r) / len(self.pixels_data)
        mean_g = sum(g) / len(self.pixels_data)
        mean_b = sum(b) / len(self.pixels_data)

        return mean_r, mean_g, mean_b

    def get_centroid_coordinates(self):
        x = [p.x for p in self.pixels_data]
        y = [p.y for p in self.pixels_data]

        centroid_x = sum(x) / len(self.pixels_data)
        centroid_y = sum(y) / len(self.pixels_data)

        return centroid_x, centroid_y

    def get_mean_probs(self):
        probs = np.array([p.probs for p in self.pixels_data], dtype=float)
        return probs.mean(axis=0)

    def get_pixel_data(self):
        return self.pixels_data
