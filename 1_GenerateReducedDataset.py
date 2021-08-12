import argparse
import json
import math
import os
import shutil
import cv2
from enum import IntEnum
from pathlib import Path
from datetime import date


class Keypoint(IntEnum):
    nose = 0
    left_eye = 1
    right_eye = 2
    left_ear = 3
    right_ear = 4
    left_shoulder = 5
    right_shoulder = 6
    left_elbow = 7
    right_elbow = 8
    left_wrist = 9
    right_wrist = 10
    left_hip = 11
    right_hip = 12
    left_knee = 13
    right_knee = 14
    left_ankle = 15
    right_ankle = 16


class CocoFilter:
    """ Filters COCO dataset (info, licenses, images, annotations, categories) and generates a new, filtered json file
    """

    def __init__(self, console_args):
        self.filter_for_categories = ['person']

        # amount of output
        self.max_files = int(console_args.max_count_images)
        self.min_keypoint_cnt = int(console_args.min_keypoint_cnt_per_person)

        # Verify input path exists
        if not Path(console_args.input_json).exists():
            print('Input json path not found.')
            print('Quitting early.')
            quit()
        self.input_json_path = Path(console_args.input_json)

        # Verify output path does not already exist
        if Path(console_args.output_json).exists():
            should_continue = input('Output path already exists. Overwrite? (y/n) ').lower()
            if should_continue != 'y' and should_continue != 'yes':
                print('Quitting early.')
                quit()
        self.output_json_path = Path(console_args.output_json)

        print('Loading json file...')
        with open(self.input_json_path) as json_file:
            self.jsonFile = json.load(json_file)

        self.input_image_path = Path(console_args.input_image_path)
        self.blur_threshold = console_args.threshold

    def main(self):

        # Process the json
        print('Processing input json...')
        self._generate_info()
        self._process_images()
        self._process_annotations()

        # Filter the json
        print('Filtering...')
        self._find_annotations_with_crowd()
        self._find_annotations_with_too_few_keypoints()
        self._find_blurry_images()
        self._filter_images()

        # Build new JSON
        new_master_json = {
            'info': self.info,
            'images': self.new_images,
            'annotations': self.new_annotations,
            'categories': self.jsonFile['categories']
        }

        # Write the JSON to a file
        print('Saving new json file...')
        with open(self.output_json_path, 'w+') as output_file:
            json.dump(new_master_json, output_file)

        print('Filtered json saved.')

        self._copy_images()

    def _generate_info(self):
        today = date.today()
        self.info = self.jsonFile['info']
        self.jsonFile['info']['description'] = 'Reduced COCO 2017 Dataset'
        self.jsonFile['info']['url'] = ''
        self.jsonFile['info']['version'] = '0.1'
        self.jsonFile['info']['year'] = today.strftime("%Y")
        self.jsonFile['info']['contributor'] = 'Markus Dietl'
        self.jsonFile['info']['date_created'] = today.strftime("%Y/%m/%d")

    def _process_images(self):
        self.images = dict()
        for image in self.jsonFile['images']:
            image_id = image['id']
            if image_id not in self.images:
                self.images[image_id] = image

    def _process_annotations(self):
        self.id_to_annot = dict()
        for annot in self.jsonFile['annotations']:
            image_id = annot['image_id']
            if image_id not in self.id_to_annot:
                self.id_to_annot[image_id] = []
            self.id_to_annot[image_id].append(annot)

    def _find_annotations_with_crowd(self):
        self.image_ids_with_crowd = set()
        for image_id, annotations in self.id_to_annot.items():
            for annot in annotations:
                if annot['iscrowd'] == 1:
                    self.image_ids_with_crowd.add(image_id)
                    break

    def _find_annotations_with_too_few_keypoints(self):
        self.image_ids_with_too_few_keypoints = set()
        for image_id, annotations in self.id_to_annot.items():
            for annot in annotations:
                # get x,y,type from keypoints
                keypoint_cnt = 0
                keypoints_x = annot['keypoints'][::3]
                keypoints_y = annot['keypoints'][1::3]
                # keypoints_type = annotation['keypoints'][2::3]
                for i in range(Keypoint.left_shoulder, Keypoint.right_ankle + 1):
                    if keypoints_x[i] != 0 and keypoints_y[i] != 0:
                        keypoint_cnt += 1
                if keypoint_cnt < self.min_keypoint_cnt:
                    self.image_ids_with_too_few_keypoints.add(image_id)
                    break

    def _find_blurry_images(self):
        # load the image, crop the bounding boxes, convert it to grayscale, and compute the
        # focus measure of the image using the Variance of Laplacian
        # method
        self.image_ids_blurry = set()
        for id in self.images:
            if (not id in self.image_ids_with_crowd) and (not id in self.image_ids_with_too_few_keypoints) and (
                    id in self.id_to_annot.keys()):

                complete_path = os.path.join(self.input_image_path, self.images[id]['file_name'])
                cv_image = cv2.imread(complete_path)
                for annot in self.id_to_annot[id]:
                    x, y, width, height = annot['bbox']
                    crop_img = cv_image[int(math.ceil(y)):int(math.ceil(y)+math.floor(height)), int(math.ceil(x)):math.ceil(x)+int(math.floor(width))]
                    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
                    # cv2.imshow('image', crop_img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # if the focus measure is less than the supplied threshold, then the image should be considered "blurry"
                    if fm < self.blur_threshold:
                        self.image_ids_blurry.add(id)
                        break

        for id in self.image_ids_blurry:
            print(self.images[id]['file_name'])


    def _filter_images(self):
        """ Create new json of images which were found with console argument criteria
        """
        cnt = 0
        self.new_annotations = []
        self.new_image_ids = set()
        self.new_image_filenames = set()

        for image_id, annotations in self.id_to_annot.items():
            if (not image_id in self.image_ids_with_crowd) and (not image_id in self.image_ids_with_too_few_keypoints) and (not image_id in self.image_ids_blurry):
                for annot in annotations:
                    new_annotation = dict(annot)
                    new_annotation['category_id'] = 1  # human
                    self.new_annotations.append(new_annotation)
                    self.new_image_ids.add(image_id)

                cnt += 1
                if cnt % 500 == 0:
                    print(str(cnt) + " matching images found")
                if cnt >= self.max_files:
                    break
        print(str(cnt) + " matching images found")

        self.new_images = []
        for image_id in self.new_image_ids:
            self.new_images.append(self.images[image_id])

    def _copy_images(self):
        files = []
        cnt = 0

        for i in self.new_images:
            files.append(i['file_name'])

        folder = os.path.join(self.input_image_path.parent, self.input_image_path.name + "_reduced")
        if not os.path.exists(folder):
            os.mkdir(folder)
        for f in files:
            shutil.copy(os.path.join(self.input_image_path, f),
                        os.path.join(folder, f))
            cnt += 1
            if cnt % 500 == 0:
                print(str(cnt) + " images copied")

        print(str(cnt) + " images copied")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_json", dest="input_json", help="path to a json file in coco format")
    parser.add_argument("-o", "--output_json", dest="output_json", help="path to save the output json")
    parser.add_argument("-p", "--input_image_path", dest="input_image_path", help="path to image folder")
    parser.add_argument("-c", "--max_count_images", dest="max_count_images",
                        help="maximum number of images (annotations)")
    parser.add_argument("-k", "--min_keypoint_cnt_per_person", dest="min_keypoint_cnt_per_person",
                        help="minimum count of keypoints for each person in image")
    parser.add_argument("-t", "--threshold", type=float, default=120.0,
                        help="focus measures that fall below this value will be considered 'blurry'")
    args = parser.parse_args()

    cf = CocoFilter(args)
    cf.main()
