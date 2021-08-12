import argparse
import json
import os
import cv2
from pathlib import Path
import sqlite3

conn = sqlite3.connect(':memory:')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_json", dest="input_json", help="path to a json file in coco format")
    parser.add_argument("-p", "--input_image_path", dest="input_image_path", help="path to image folder")
    args = parser.parse_args()
    json_path = Path(args.input_json)
    imgs_path = Path(args.input_image_path)

    id_to_file = dict()
    id_to_annot = dict()

    with open(json_path) as json_file:
        data = json.load(json_file)
        for img in data['images']:
            id_to_file[int(img['id'])] = img['file_name']

        for annot in data['annotations']:
            key = int(annot['image_id'])
            if key not in id_to_annot:
                id_to_annot[key] = []
            id_to_annot[key].append(annot['keypoints'])

    file_to_annot = dict()
    for key in id_to_file.keys():
        if key in id_to_annot.keys():
            file_to_annot[id_to_file[key]] = id_to_annot[key]

    del id_to_file
    del id_to_annot

    cnt = 0
    for file in file_to_annot.keys():
        path = str(os.path.join(imgs_path, file))
        if os.path.exists(path):
            img = cv2.imread(path)

            for annot in file_to_annot[file]:
                keypoints_x = annot[::3]
                keypoints_y = annot[1::3]
                keypoints_type = annot[2::3]
                for i in range(len(keypoints_x)):
                    if int(keypoints_type[i]) == 2:
                        cv2.circle(img, (int(keypoints_x[i]), int(keypoints_y[i])), 5, (0, 255, 0), thickness=-1,
                                   lineType=cv2.FILLED)
                    elif int(keypoints_type[i]) == 1:
                        cv2.circle(img, (int(keypoints_x[i]), int(keypoints_y[i])), 5, (0, 0, 255), thickness=-1,
                                   lineType=cv2.FILLED)
                        # cv2.putText(frame, "{}".format(i), (int(x), int(y)),
                        # cv2.FONT_HERSHEY_SIMPLEX, 1.4,(0, 0, 255), 3, lineType=cv2.LINE_AA)

            path2 = str(os.path.join(os.path.join(imgs_path, 'labeled'), file))
            # cv2.imshow('test',img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            folder = os.path.join(imgs_path, 'labeled')
            if not os.path.exists(folder):
                os.mkdir(folder)

            cv2.imwrite(path2, img)
            cnt += 1
            if cnt % 500 == 0:
                print(str(cnt) + " images annotated")

    print(str(cnt) + " images annotated")
