class DBConfig(object):
    def __init__(self, db_stats_aggregator=None, db_normalization=None):
        self.db_stats_aggregator = db_stats_aggregator
        self.db_normalization = db_normalization
        self.db_info = None
        self.descriptors_cfg = None


class DB(object):
    def __init__(self, runtime, config=None):
        self.runtime = runtime
        self.config = config
        self.shards = None
        self.db_id = None

    def create_db(self, path_to_data, output_path=None, batch_size=100):
        batches = self.runtime.make_batches(path_to_data, batch_size=batch_size)

        if self.config.db_stats_aggregator is not None:
            self.config.db_info = self.runtime.collect_db_info(self.config.db_stats_aggregator, batches=batches,
                                                               path_to_data=path_to_data)

        if self.config.db_normalization is not None:
            batches = self.runtime.normalize_db(batches, self.config.db_info, self.config.db_normalization)

        descriptors = self.runtime.calculate_descriptors(batches, self.config.descriptors_cfg)

        self.runtime.save_descriptors(descriptors, output_path=output_path)
        db_id = self.runtime.save_db_config(self.config)
        return db_id

    def load_db_from_id(self, db_id):
        self.db_id = db_id
        self.config = self.runtime.get_config_from_id(db_id)
        self.shards = self.runtime.get_shards_for(db_id)

    def search(self, image, region_coordinates):
        if self.config.db_normalization is not None:
            image = self.runtime.normalize_one()

        region = self.runtime.crop_region(image, region_coordinates)

        return self.runtime.search(self.config.descriptors_cfg, region, self.config.descriptors_cfg)
