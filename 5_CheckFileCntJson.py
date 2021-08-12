import argparse
import json
import os
import cv2
from pathlib import Path
import sqlite3

conn = sqlite3.connect(':memory:')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", dest="json_path")
    args = parser.parse_args()
    json_path = Path(args.json_path)

    with open(json_path) as json_file:
        data = json.load(json_file)

        i = 0
        j = 0
        for img in data['images']:
            i += 1
        for annot in data['annotations']:
            ids = list()
            id = int(annot['image_id'])
            if id not in ids:
                j += 1
            ids.append(id)

    if i <= j:
        print(str(i) + " images found in json file")
    else:
        print(str(j) + " images found in json file")
