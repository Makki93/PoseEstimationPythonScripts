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

    # for i in data['images']:
    #     if i['file_name'] == '000000310360.jpg':
    #         print(i)
    #         id = i['id']
    #         for a in data['annotations']:
    #             if a['image_id'] == id:
    #                 print(a)

    # cnt = 0
    # for a in data['annotations']:
    #     if int(a['num_keypoints']) > 17:
    #         print(a)
    #         cnt += 1
    #         if cnt == 100:
    #             exit()
    # exit()
    for a in data['annotations']:
        if int(a['num_keypoints']) >= 6:
            # get x,y,type from keypoints
            keypoints_x = a['keypoints'][::3]
            keypoints_y = a['keypoints'][1::3]
            keypoints_type = a['keypoints'][2::3]
            # get (x,y) pair of keypoints
            keypoints_x_y = list(zip(keypoints_x, keypoints_y))

            # filter pictures with random non-zero keypoints being at least pixel_limit pixels apart
            for pair in itertools.combinations(keypoints_x_y, r=2):
                if pair[0][0] != 0 and pair[0][1] != 0 and pair[1][0] != 0 and pair[1][1] != 0:
                    if math.hypot(pair[1][0] - pair[0][0], pair[1][1] - pair[0][1]) > pixel_limit:
                        # print(math.hypot(pair[1][0] - pair[0][0], pair[1][1] - pair[0][1]))
                        imgs.append(a['image_id'])
                        break

    for i in data['images']:
        if i['id'] in imgs:
            files.append(i['file_name'])

            # print amount of found pictures
            j += 1
            if j % 100 == 0:
                print(str(j), " images found")

print(str(j), " images found")
for f in files:
    shutil.copy(os.path.join(imgs_path, f), os.path.join(os.path.join(imgs_path, 'filtered'), f))
