from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
from elasticsearch import Elasticsearch
from os import listdir
from os.path import isfile, join


 
import base64 # convert image to b64 for indexing
 
elastic_client = Elasticsearch("http://localhost:9200",
    basic_auth=("elastic", "changeme"))

# create the "your_index_name" index for Elasticsearch if necessary
 


settings = {
  "settings": {
    "elastiknn": True,
    "number_of_shards": 1,
    "number_of_replicas": 0
  }
}

mapping ={
  "mappings": {
    "properties": {
      "raw_data": {
        "type": "dense_vector",
        "dims": 512,
        "index": False
      },
      "image_path": {
        "type": "text"
      }
    }
  }
}

resp = elastic_client.indices.create(
    index = "vgg18-elk",
    mappings=mapping , 
    ignore = 400 
    )
print ("\nElasticsearch create() index response -->", resp)
 
if __name__ == '__main__':
    fe = FeatureExtractor()
    img_nb=1
    _index= "vgg18-elk"
    sourceList=[] 
    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(img_path)  
        feature = fe.extract(img=Image.open(img_path))
        
        feature_path = Path("./static/feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)
        _source = {}
        _source["image_path"] = str(img_path)
        _id = img_nb

        _source["raw_data"] =feature
        sourceList.append(_source)
 
        resp = elastic_client.index(
               index = _index,
               id = _id,
               body = _source,
               request_timeout=60)
        print ("\nElasticsearch index() response -->", resp)
        img_nb += 1
 