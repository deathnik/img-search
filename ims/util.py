import heapq


def crop_image(x1, y1, x2, y2, img):
    return img[x1: x2, y1:y2]


class Heap(object):
    def __init__(self, capacity=10):
        self.h = []
        self.capacity = capacity
        heapq.heapify(self.h)
        self.size = 0

    def push(self, elem):
        if self.size < self.capacity:
            heapq.heappush(self.h, elem)
            self.size += 1
        else:
            return heapq.heappushpop(self.h, elem)

    def data(self):
        return self.h
