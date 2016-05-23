# import the necessary packages
from skimage import feature
from skimage.color import rgb2gray
import numpy as np
from ims.descriptors.base import Descriptor


class LocalBinaryPatternsDescriptor(Descriptor):
    def __init__(self, num_points=32, radius=4, **kwargs):
        super(LocalBinaryPatternsDescriptor, self).__init__(**kwargs)
        self.algo_num_points = num_points
        self.num_points = 32 - 1

        self.radius = radius
        self.eps = 1e-7

    def _calculate_descriptor(self, image, *args, **kwargs):
        if len(image.shape) == 3:
            image = rgb2gray(image)
        # if 'mean' in kwargs:
        #    mean = kwargs['mean']
        #    image = image / mean
        lbp = feature.local_binary_pattern(image, self.algo_num_points,
                                           self.radius, method="uniform")
        # return lbp.ravel()
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.num_points + 2),
                                 range=(0, self.num_points + 1))

        # normalize the histogram
        hist = hist.astype("float")
        # hist /= (hist.sum() + self.eps)

        # return the histogram of Local Binary Patterns
        return hist

    def size(self):
        return self.num_points + 1
