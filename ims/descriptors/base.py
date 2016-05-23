from abc import ABCMeta, abstractmethod
import cv2


class Descriptor(object):
    __metaclass__ = ABCMeta

    def __init__(self, h=64, w=64, **kwargs):
        self.h = h
        self.w = w

    def _auto_crop(self, img):
        res = cv2.resize(img, (self.h, self.w))
        return res

    @abstractmethod
    def _calculate_descriptor(self, img, *args, **kwargs):
        pass

    def calculate_descriptor(self, img, *args, **kwargs):
        cropped_img = self._auto_crop(img)
        return self._calculate_descriptor(cropped_img, *args, **kwargs)

    @abstractmethod
    def to_json(self):
        pass
