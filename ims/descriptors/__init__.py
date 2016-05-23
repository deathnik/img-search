import struct

from itertools import imap
from copy import deepcopy

import cv2
import numpy as np

from ims.util import crop_image

from ims.descriptors.hist import HistDescriptor
from ims.descriptors.lbp import LocalBinaryPatternsDescriptor

ALLOWED_DESCRIPTOR_TYPES = {
    '.hist': HistDescriptor,
    '.lbp': LocalBinaryPatternsDescriptor
}


class DescriptorConfig(object):
    def __init__(self):
        self.descriptor_suffix = '.hist'
        self.sizes = [[64, 64]]

    def _get_descriptor_template(self):
        return ALLOWED_DESCRIPTOR_TYPES[self.descriptor_suffix]()

    def get_descriptor_calculator(self):
        def calculate_descriptors(binary_img):
            descriptor_template = self._get_descriptor_template()

            def inner():
                # load byte array to ndarray
                arr = np.fromstring(binary_img, np.uint8)
                img = cv2.imdecode(arr, cv2.CV_LOAD_IMAGE_COLOR)

                for w, h in self.sizes:
                    descriptor = deepcopy(descriptor_template)

                    descriptor.w = w
                    descriptor.h = h
                    for i in range(0, img.shape[0] / w):
                        for j in range(0, img.shape[1] / h):
                            img_part = crop_image(w * i, h * j, w * (i + 1), h * (j + 1), img)
                            res = descriptor.calculate_descriptor(img_part)
                            yield res

            pack_template = '{}f'.format(descriptor_template.size())
            return ''.join(imap(lambda x: struct.pack(pack_template, *x), inner()))

        return calculate_descriptors

    def to_json(self):
        return {
            'sizes': self.sizes,
            'desc': self._get_descriptor_template().to_json(),
        }
