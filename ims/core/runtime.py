import hashlib
import json
import math
import os

from abc import ABCMeta, abstractmethod, abstractproperty


class Runtime(object):
    __metaclass__ = ABCMeta
    config_location = '/etc/img-search/{}'
    INTERESTING_FORMATS = ['png', 'jpg', 'pgm']

    def _list_interesting_files(self, path):
        for base_path, _, child_files in os.walk(path):
            for f_name in child_files:
                if '.' not in f_name:
                    continue
                name, ext = f_name.lower().rsplit('.', 1)

                if ext in self.INTERESTING_FORMATS:
                    yield os.path.join(base_path, f_name)

    @abstractmethod
    def make_batches(self, path_to_data, batch_size):
        pass

    @abstractmethod
    def collect_db_info(self, stats_aggregator, **kwargs):
        pass

    def save_db_config(self, config, path_to_data):
        js = config.to_json()
        js['path_to_data'] = path_to_data
        js = json.dumps(js)
        config_id = hashlib.sha224(js).hexdigest()
        with open(self.config_location.format(config_id), 'wb') as f:
            f.write(js)
        return config_id


class LocalRuntime(Runtime):
    pass


class SparkRuntime(Runtime):
    output_format = "org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat"
    key_format = "org.apache.hadoop.io.Text"
    value_format = "org.apache.hadoop.io.BytesWritable"

    def __init__(self, spark_context):
        self.sc = spark_context

    def _save(self, rdd, path):
        rdd.saveAsNewAPIHadoopFile(path=path,
                                   outputFormatClass=self.output_format,
                                   keyClass=self.key_format,
                                   valueClass=self.value_format)

    def make_batches(self, path_to_data, batch_size):
        files = list(self._list_interesting_files(path_to_data))
        partitions_count = int(math.ceil(len(files) * 1.0 / batch_size))
        batches_rdd = self.sc.binaryFiles(','.join(files), minPartitions=partitions_count).repartition(partitions_count)
        return batches_rdd

    def collect_db_info(self, stats_aggregator, **kwargs):
        batches = kwargs.get('batches')
        stats = batches.map(lambda x: stats_aggregator.get_stats(x[1]))
        collected_stats = stats.reduce(stats_aggregator.reducer).collect()
        return collected_stats

    def normalize_db(self, batches, db_info, db_normalization):
        norm = db_normalization.get_normalization_mapper(db_info)
        return batches.map(lambda x: (x[0], norm(x[1])))

    def calculate_descriptors(self, batches, descriptors_cfg):
        descriptor_calculator = descriptors_cfg.get_descriptor_calculator()
        return batches.map(lambda x: (x[0], descriptor_calculator(x[1])))

    def save_descriptors(self, descriptors, output_path):
        def merge_images(iterator):
            images = []
            data = ''
            for path, descriptors_value in iterator:
                base_path, file_name = path.rsplit('/', 1)
                images.append(file_name)
                data += descriptors_value
            yield (json.dumps(images), bytearray(data))

        self._save(descriptors.mapPartitions(merge_images), path=output_path)
