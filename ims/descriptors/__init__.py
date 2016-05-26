import struct

from itertools import imap
from copy import deepcopy

import cv2
import numpy as np

from ims.util import crop_image, debug_log

from ims.descriptors.hist import HistDescriptor
from ims.descriptors.lbp import LocalBinaryPatternsDescriptor

ALLOWED_DESCRIPTOR_TYPES = {
    '.hist': HistDescriptor,
    '.lbp': LocalBinaryPatternsDescriptor
}

ELEMENT_SIZE = 4


class DescriptorConfig(object):
    def __init__(self, descriptor_type='hist'):
        self.descriptor_suffix = '.' + descriptor_type
        self.sizes = [[64, 64]]
        self.descriptor_params = {}
        self.img_size = [286, 384]

    def _get_descriptor_template(self):
        return ALLOWED_DESCRIPTOR_TYPES[self.descriptor_suffix](**self.descriptor_params)

    def get_descriptor_calculator(self):
        pack_template = self.get_pack_template()

        def calculate_descriptors(image_name, binary_img):
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
                            debug_log('image {} with pos {}:{} got descriptor: {}'.format(image_name, i, j, res))
                            yield res

            return ''.join(imap(lambda x: struct.pack(pack_template, *x), inner()))

        return calculate_descriptors

    def calculate_one_descriptor(self, image_part):
        return self._get_descriptor_template().calculate_descriptor(image_part)

    def get_pack_template(self):
        return '{}f'.format(self._get_descriptor_template().size())

    def get_descriptor_length(self):
        return self._get_descriptor_template().size() * ELEMENT_SIZE

    def get_offset(self, search_area_size, search_position, image_index):
        offset = 0
        size = list(search_area_size)
        for sz in self.sizes:
            if list(sz) != size:
                offset += int(self.img_size[0] / sz[0]) * int(self.img_size[1] / sz[1])
            else:
                offset += search_position[0] * int(self.img_size[1] / sz[1]) + search_position[1]
                break
        offset *= self.get_descriptor_length()
        return self._full_image_descriptors_size() * image_index + offset

    def _full_image_descriptors_size(self):
        h, w = self.img_size
        descriptors_count = 0
        for sz in self.sizes:
            x, y = sz
            descriptors_count += int(h / x) * (w / y)
        return descriptors_count * self.get_descriptor_length()

    def is_inside_image(self, coordinates, search_area_size):
        h, w = self.img_size
        x, y = coordinates
        if 0 <= x <= h / search_area_size[0] and 0 <= y <= w / search_area_size[1]:
            return True
        return False

    def to_json(self):
        return {
            'sizes': self.sizes,
            'desc': self._get_descriptor_template().to_json(),
        }

    @classmethod
    def from_json(cls, js):
        instance = cls()
        instance.sizes = js['sizes']
        instance.descriptor_suffix = js['desc']['type']
        instance.descriptor_params = js['desc']
        return instance
