# # # # # # # instructions # # # # # # # 

# 1) copy the pictures and the json file into the same folder 

# 2) create a folder within that folder called 'filtered'

# 3) execute python3 1_Filterpersons.py -p path/to/dir

# # # # # # # # # # # # # # # # # # # # 


import os
import json
import shutil
import argparse
from pathlib import Path
import itertools
import math

pixel_limit = 120
imgs = []
files = []
j = 0

parser = argparse.ArgumentParser()
parser.add_argument("-j", dest="json_path")
parser.add_argument("-i", dest="imgs_path")
args = parser.parse_args()
json_path = Path(args.json_path)
imgs_path = Path(args.imgs_path)

with open(json_path) as json_file:
    data = json.load(json_file)
    for i in data['images']:
        files.append(i['file_name'])

print(str(len(files)) + " images found")
for f in files:
    shutil.copy(os.path.join(imgs_path, f), os.path.join(os.path.join(imgs_path, 'filtered'), f))
