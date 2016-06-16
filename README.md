# img-search

configuring:
```python
from ims.core.runtime import SparkRuntime
from ims.core.db import DBConfig,DB
from ims.descriptors import DescriptorConfig
runtime = SparkRuntime(sc)
cfg = DescriptorConfig(descriptor_type='lbp')
db_config = DBConfig(output_path=some_path)
db_config.descriptors_cfg = cfg
db = DB(runtime,db_config)
```


build db:
```python
db_id = db.create_db(path_to_data=data_path)
```
load db:
```python
db_restore = DB.load_db(runtime,db_id)
```
search:
```python
search_cfg = SearchConfig(halo_size=1)
result = db_restore.search(img_path,bounding_box,search_cfg)
bounding_box=(upper,lower)
```
