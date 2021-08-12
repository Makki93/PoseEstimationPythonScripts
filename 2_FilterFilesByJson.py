# # # # # # # instructions # # # # # # # 

# 1) create a folder within the image folder called 'filtered'

# 2) execute python --input_image_path=/home/uie25806/coco/images/val2017/ --input_json=/home/uie25806/coco/annotations/person_keypoints_val2017_reduced.json

# # # # # # # # # # # # # # # # # # # # 


import os
import argparse
import json
import shutil
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_json", dest="input_json", help="path to a json file in coco format")
    parser.add_argument("-p", "--input_image_path", dest="input_image_path", help="path to image folder")
    args = parser.parse_args()
    json_path = Path(args.input_json)
    imgs_path = Path(args.input_image_path)

    imgs = []
    files = []
    cnt = 0

    with open(json_path) as json_file:
        data = json.load(json_file)
        for i in data['images']:
            files.append(i['file_name'])

    print(str(len(files)) + " images found")
    folder = os.path.join(imgs_path.parent, imgs_path.name + "_reduced")
    if not os.path.exists(folder):
        os.mkdir(folder)
    for f in files:
        shutil.copy(os.path.join(imgs_path, f),
                    os.path.join(folder, f))
        cnt += 1
        if cnt % 500 == 0:
            print(str(cnt) + " images copied")

    print(str(cnt) + " images copied")
