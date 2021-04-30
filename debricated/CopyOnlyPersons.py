# sub-folder filtered must exist before executing this file

import os
import json
import shutil
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("-j", dest="json_path")

parser.add_argument("-i", dest="imgs_path")

args = parser.parse_args()

json_path = Path(args.json_path)
imgs_path = Path(args.imgs_path)

imgs = []

with open(json_path) as json_file:
    data = json.load(json_file)
    for i in data['images']:
        imgs.append(i['file_name'])

for i in imgs:
    shutil.copy(i, 'filtered')
