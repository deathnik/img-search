import json
import struct

from ims.metrics import hist_distance
from ims.util import Heap, debug_log

ELEMENT_SIZE = 4


class _Selector(object):
    def __init__(self, heap_size, name=''):
        self.heap_size = heap_size
        self.name = name

    def __call__(self, iterator):
        heap = Heap(capacity=self.heap_size)
        for elem in iterator:
            item = self.get_item(elem)
            if item is None:
                continue
            popped = heap.push(item)
            if popped is not None:
                debug_log('Filtered out in {} : {}'.format(self.name, popped))

        for item in sorted(heap.data()):
            yield item

    # reload this method for doing something
    @staticmethod
    def get_item(elem):
        return elem


class SearchConfig(object):
    def __init__(self, selection_size=10, halo_size=0, search_area_size=[64, 64]):
        self.selection_size = selection_size
        self.halo_size = halo_size
        self.distance = hist_distance
        self.search_area_size = search_area_size

    @classmethod
    def from_json(cls, config):
        return cls(**config)


class Seeker(object):
    def __init__(self, config=None):
        if isinstance(config, basestring):
            config = json.loads(config)

        if isinstance(config, dict):
            config = SearchConfig.from_json(config)

        if config is None:
            config = SearchConfig()

        self.config = config

    def _make_selector(self, get_item):
        selector = _Selector(self.config.selection_size)
        selector.get_item = get_item
        return selector

    def _one_line_descriptors(self, i, search_position, descriptor_cfg, shard_data):
        # fetches data about one line in hallo
        offset = descriptor_cfg.get_offset(self.config.search_area_size, search_position, i)
        elements_to_read = 1 + self.config.halo_size * 2
        descriptor_len = descriptor_cfg.get_descriptor_length()
        pack_template = descriptor_cfg.get_pack_template()

        for x in xrange(0, elements_to_read):
            data = shard_data[offset + x * descriptor_len: offset + (x + 1) * descriptor_len]
            if len(data) < descriptor_len:
                continue
            debug_log('descriptor for image: {}, offset: {} , position : {}:{} value: {}'.format(
                i, offset + x * descriptor_len,
                search_position[0], search_position[1] + x,
                struct.unpack(pack_template, data)))

            yield (search_position[0], search_position[1] + x), struct.unpack(pack_template, data)

    def _get_descriptors(self, i, search_position, scale, descriptor_cfg, shard_data):
        # fetches all halo descriptors
        for levels in xrange(-self.config.halo_size, self.config.halo_size + 1):
            new_search_pos = search_position[0] - levels, search_position[1] - self.config.halo_size
            for coordinates, value in self._one_line_descriptors(i, new_search_pos, descriptor_cfg, shard_data):
                # check if we still inside db image
                if descriptor_cfg.is_inside_image(coordinates, scale):
                    yield coordinates, value

    def get_selector(self, search_descriptor, scale, search_position, descriptor_cfg):
        def get_item(item):
            images_list, packed_descriptors = item
            images_list = json.loads(images_list)
            for i, image_name in enumerate(images_list):
                debug_log('{} {}'.format(i, image_name))
                for coordinates, descriptor_value in \
                        self._get_descriptors(i, search_position, scale, descriptor_cfg, packed_descriptors):
                    debug_log('Fetched {} {} {}'.format(image_name, coordinates, descriptor_value))
                    dist = self.config.distance(search_descriptor, descriptor_value)
                    return dist, image_name, coordinates, descriptor_value

        return self._make_selector(get_item)

    def get_selector_for_prepared_descriptors(self, search_descriptor):
        def get_item(item):
            image_name, (coordinates, descriptor_value, _type) = item
            dist = self.config.distance(search_descriptor, descriptor_value)
            return dist, image_name, coordinates, descriptor_value, _type

        return self._make_selector(get_item)

    def get_results_combiner(self):
        return _Selector(self.config.selection_size)
