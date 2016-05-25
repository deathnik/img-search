import hashlib
import json

import cv2

from ims.descriptors import DescriptorConfig
from ims.util import crop_image


class DBConfig(object):
    def __init__(self, output_path, db_stats_aggregator=None, db_normalization=None):
        self.db_stats_aggregator = db_stats_aggregator
        self.db_normalization = db_normalization
        self.db_info = None
        self.descriptors_cfg = None
        self.output_path = output_path

    def to_json(self):
        return {
            'descriptors_cfg': self.descriptors_cfg.to_json(),
            'output_path': self.output_path
        }

    @classmethod
    def from_json(cls, js):
        if isinstance(js, basestring):
            js = json.loads(js)

        instance = DBConfig(output_path=js['output_path'])
        instance.descriptors_cfg = DescriptorConfig.from_json(js['descriptors_cfg'])
        return instance


class DB(object):
    def __init__(self, runtime, config=None):
        self.runtime = runtime
        self.config = config
        self.shards = None
        self.path_to_data = None

    def create_db(self, path_to_data, batch_size=100):
        batches = self.runtime.make_batches(path_to_data, batch_size=batch_size)
        self.path_to_data = path_to_data
        db_id = self.get_id()

        if self.config.db_stats_aggregator is not None:
            self.config.db_info = self.runtime.collect_db_info(self.config.db_stats_aggregator, batches=batches,
                                                               path_to_data=path_to_data)

        if self.config.db_normalization is not None:
            batches = self.runtime.normalize_db(batches, self.config.db_info, self.config.db_normalization)

        descriptors = self.runtime.calculate_descriptors(batches, self.config.descriptors_cfg)

        self.runtime.save_descriptors(descriptors, output_path=self.config.output_path)

        self.runtime.save_db_config(self.to_json(), db_id)
        return db_id

    @classmethod
    def load_db(cls, runtime, db_id):
        db = cls(runtime)
        serialized_config = db.runtime.get_serialized_config_for_id(db_id)
        db.config = DBConfig.from_json(serialized_config)
        return db

    def _crop_region(self, image, upper_corner, lower_corner, scale):
        if isinstance(image, basestring):
            image = self.runtime.load_image(image)
        # TODO: add normalize image, via normalize_one
        image_part = crop_image(upper_corner[0], upper_corner[1], lower_corner[0], lower_corner[1], image)
        return cv2.resize(image_part, scale)

    def search(self, image, region_coordinates, search_config):
        if self.config.db_normalization is not None:
            image = self.runtime.normalize_one()

        upper_corner, lower_corner = region_coordinates
        area = (upper_corner[0] - lower_corner[0]) * (upper_corner[1] - lower_corner[1])
        scale = tuple(min(self.config.descriptors_cfg.sizes, key=lambda x: abs(area - x[0] * x[1])))
        image_region = self._crop_region(image, upper_corner, lower_corner, scale)

        search_descriptor = self.config.descriptors_cfg.calculate_one_descriptor(image_region)
        search_position = upper_corner[0] / scale[0], upper_corner[1] / scale[1]

        return self.runtime.search(self.config.output_path, search_descriptor, scale, search_position,
                                   self.config.descriptors_cfg, search_config)

    def get_id(self):
        js = json.dumps(self.to_json())
        return hashlib.sha224(js).hexdigest()

    def to_json(self):
        js = self.config.to_json()
        js['path_to_data'] = self.path_to_data
        return js
