from skimage.color import rgb2gray
import numpy as np

from ims.descriptors.base import Descriptor


class HistDescriptor(Descriptor):
    def __init__(self, num_points=32, **kwargs):
        super(HistDescriptor, self).__init__(**kwargs)
        self.num_points = num_points
        self.eps = 1e-7

    def _calculate_descriptor(self, img, *args, **kwargs):
        if len(img.shape) == 3:
            img = rgb2gray(img)
        # if 'mean' in kwargs:
        #    mean = kwargs['mean']
        #    img2 = img / mean * 128
        # else:
        img2 = img
        bins = [x * 1.0 / (self.num_points + 2) for x in np.arange(0, self.num_points + 2)]
        # (hist, _) = np.histogram(img.ravel(), bins=bins,
        #                         range=(0, self.num_points + 1))
        (hist2, _) = np.histogram(img2.ravel(), bins=bins,
                                  range=(0, self.num_points + 1 - 5))
        # normalize the histogram
        hist = hist2.astype(np.float)
        return hist

    def size(self):
        return self.num_points + 1

    def to_json(self):
        return {
            'type': '.hist',
            'num_points': self.num_points,
        }
