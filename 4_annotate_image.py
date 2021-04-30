import argparse
import json
import os
import cv2
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-j", dest="json_path")
    parser.add_argument("-i", dest="imgs_path")
    args = parser.parse_args()
    json_path = Path(args.json_path)
    imgs_path = Path(args.imgs_path)

    files = []
    id_to_file = dict()
    id_to_annot = dict()
    with open(json_path) as json_file:
        data = json.load(json_file)
        for imgs in data['images']:
            id_to_file[imgs['id']] = imgs['file_name']
        for annot in data['annotations']:
            id_to_annot[annot['image_id']] = annot['keypoints']

    for key in id_to_file.keys():
        img = cv2.imread(str(os.path.join(imgs_path, id_to_file[key])))
        annotation = list(id_to_annot[key])
        # print(annotation)
        keypoints_x = annotation[::3]
        keypoints_y = annotation[1::3]
        keypoints_type = annotation[2::3]
        for i in range(len(keypoints_x)):
            if(int(keypoints_type[i])==2):
                cv2.circle(img, (int(keypoints_x[i]), int(keypoints_y[i])), 5, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
                # cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
            elif(int(keypoints_type[i])==1):
                cv2.circle(img, (int(keypoints_x[i]), int(keypoints_y[i])), 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                # cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
        #cv2.imshow("Output-Keypoints", img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv2.imwrite(str(os.path.join(os.path.join(imgs_path, 'labeled'), id_to_file[key])), img)
