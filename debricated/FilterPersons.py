# sub-folder filtered must exist before executing this file

import os
import json
import shutil
import argparse
from pathlib import Path
import itertools

def slice_per(source, step):
    return [source[i::step] for i in range(step)]

parser = argparse.ArgumentParser()

parser.add_argument("-j", dest="json_path")

parser.add_argument("-i", dest="imgs_path")

args = parser.parse_args()

json_path = Path(args.json_path)
imgs_path = Path(args.imgs_path)

imgs = []
files = []
j = 0
with open(json_path) as json_file:
    data = json.load(json_file)
    for a in data['annotations']:
        if (int(a['num_keypoints'])>=6):
            keypoints_x = a['keypoints'][::3]
            keypoints_y = a['keypoints'][1::3]
            keypoints_type = a['keypoints'][2::3]
            print(keypoints_x)
            sys.exit()

            found = false
            for pair in itertools.combinations(keypoints_x, repeat=2):
                if(abs(pair[0]-pair[1])>40):
                    imgs.append(a['image_id'])
                    found = true
                    break

            if (not found):
                for pair in itertools.combinations(keypoints_y, repeat=2):
                    if(abs(pair[0]-pair[1])>40):
                        imgs.append(a['image_id'])
                        found = true
                        break

    for i in data['images']:
        
        if (i['id'] in imgs):
            files.append(i['file_name'])

            # echo amount of found pictures
            j+=1
            if (j%100==0):
                print(str(j), " images found")
        
print(str(j), " images found")
for f in files:
    shutil.copy(f, 'filtered')