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
        self.image_ids_being_filtered = set()
        self.filter_for_categories = ['person']

        # amount of output
        self.max_files = int(console_args.max_count_images)
        self.min_keypoint_cnt = int(console_args.min_keypoint_cnt_per_person)

        # Verify input paths exists
        if not Path(console_args.input_json).exists():
            print('Input json path not found.')
            print('Quitting early.')
            quit()

        if not Path(console_args.input_image_path).exists():
            print('Input image path not found.')
            print('Quitting early.')
            quit()

        if not Path(console_args.detections_json).exists():
            print('Person detection json path not found.')
            print('Quitting early.')
            quit()

        self.input_json_path = Path(console_args.input_json)
        self.input_image_path = Path(console_args.input_image_path)
        self.person_det_path = Path(console_args.detections_json)

        # Verify output json file does not already exist
        size = len(str(self.input_json_path))
        self.output_json_val2017_path = Path(str(self.input_json_path)[:size - 5] + '_reduced.json')
        size = len(str(self.person_det_path))
        self.output_json_pers_det_path = Path(str(self.person_det_path)[:size - 5] + '_reduced.json')

        if self.output_json_val2017_path.exists() or self.output_json_pers_det_path.exists():
            should_continue = input('At least one output file already exists. Overwrite? (y/n) ').lower()
            if should_continue != 'y' and should_continue != 'yes':
                print('Quitting early.')
                quit()

            if self.output_json_val2017_path.exists():
                os.remove(self.output_json_val2017_path)
            if self.output_json_pers_det_path.exists():
                os.remove(self.output_json_pers_det_path)

        # Clear target image folder
        self.output_image_folder = os.path.join(self.input_image_path.parent, self.input_image_path.name + "_reduced")
        if os.path.exists(self.output_image_folder):
            for file in os.listdir(self.output_image_folder):
                os.remove(os.path.join(self.output_image_folder, file))

        print('Loading json file...')
        with open(self.input_json_path) as json_file:
            self.jsonFile = json.load(json_file)

        with open(self.person_det_path) as json_file:
            self.jsonDetFile = json.load(json_file)
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
        self._find_annotations_with_too_small_persons()
        self._find_annotations_with_too_big_persons()
        self._find_blurry_images()
        self._filter_images()

        # Build new JSON
        new_master_json = {
            'info': self.info,
            'images': self.new_images,
            'annotations': self.new_annotations,
            'categories': self.jsonFile['categories']
        }

        flat_list = []
        # Iterate through the outer list
        for element in self.new_dects:
            if type(element) is list:
                # If the element is of type list, iterate through the sublist
                for item in element:
                    flat_list.append(item)
            else:
                flat_list.append(element)

        new_master_json_det = flat_list

        # Write the JSON to a file
        print('Saving new json file...')
        with open(self.output_json_val2017_path, 'w+') as output_file:
            json.dump(new_master_json, output_file)

        with open(self.output_json_pers_det_path, 'w+') as output_file:
            json.dump(new_master_json_det, output_file)

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

        self.id_to_dets = dict()
        for annot in self.jsonDetFile:
            image_id = annot['image_id']
            if image_id not in self.id_to_dets:
                self.id_to_dets[image_id] = []
            self.id_to_dets[image_id].append(annot)

    def _find_annotations_with_crowd(self):
        for id, annotations in self.id_to_annot.items():
            if id in self.id_to_annot.keys():
                for annot in annotations:
                    if annot['iscrowd'] == 1:
                        self.image_ids_being_filtered.add(id)
                        break

    def _find_annotations_with_too_few_keypoints(self):
        for id, annotations in self.id_to_annot.items():
            if id in self.id_to_annot.keys() and not id in self.image_ids_being_filtered:
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
                        self.image_ids_being_filtered.add(id)
                        break

    def _find_annotations_with_too_small_persons(self):
        for id in self.images:
            if id in self.id_to_annot.keys() and not id in self.image_ids_being_filtered:
                complete_path = os.path.join(self.input_image_path, self.images[id]['file_name'])
                assert (Path(complete_path).exists())
                img = cv2.imread(complete_path)
                image_height, image_width, channel = img.shape

                for annot in self.id_to_annot[id]:
                    bbox_width, bbox_height = annot['bbox'][2:]
                    height_ratio = bbox_height / image_height
                    if height_ratio < 0.5:
                        self.image_ids_being_filtered.add(id)
                        break
                    width_ratio = bbox_width / image_width
                    if width_ratio < 0.15:
                        self.image_ids_being_filtered.add(id)
                        break

    def _find_annotations_with_too_big_persons(self):
        for id in self.images:
            if id in self.id_to_annot.keys() and not id in self.image_ids_being_filtered:
                complete_path = os.path.join(self.input_image_path, self.images[id]['file_name'])
                assert (Path(complete_path).exists())
                img = cv2.imread(complete_path)
                image_height, image_width, channel = img.shape

                for annot in self.id_to_annot[id]:
                    bbox_width, bbox_height = annot['bbox'][2:]
                    height_ratio = bbox_height / image_height
                    if height_ratio > 0.95:
                        self.image_ids_being_filtered.add(id)
                        break
                    width_ratio = bbox_width / image_width
                    if width_ratio > 0.95:
                        self.image_ids_being_filtered.add(id)
                        break

    def _find_blurry_images(self):
        # load the image, crop the bounding boxes, convert it to grayscale, and compute the
        # focus measure of the image using the Variance of Laplacian method
        for id in self.images:
            if id in self.id_to_annot.keys() and not id in self.image_ids_being_filtered:
                complete_path = os.path.join(self.input_image_path, self.images[id]['file_name'])
                assert (Path(complete_path).exists())
                cv_image = cv2.imread(complete_path)
                for annot in self.id_to_annot[id]:
                    x, y, width, height = annot['bbox']
                    crop_img = cv_image[int(math.ceil(y)):int(math.ceil(y) + math.floor(height)),
                               int(math.ceil(x)):math.ceil(x) + int(math.floor(width))]
                    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
                    # cv2.imshow('image', crop_img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # if the focus measure is less than the supplied threshold, then the image is considered blurry
                    if fm < self.blur_threshold:
                        self.image_ids_being_filtered.add(id)
                        break

    def _filter_images(self):
        """ Create new json of images which were found with console argument criteria
        """
        cnt = 0
        self.new_annotations = []
        self.new_image_ids = set()
        self.new_image_filenames = set()

        for id, annotations in self.id_to_annot.items():
            if not id in self.image_ids_being_filtered:
                for annot in annotations:
                    new_annotation = dict(annot)
                    new_annotation['category_id'] = 1  # human
                    self.new_annotations.append(new_annotation)
                    self.new_image_ids.add(id)

                cnt += 1
                if cnt % 500 == 0:
                    print(str(cnt) + " matching images found")
                if cnt >= self.max_files:
                    break
        print(str(cnt) + " matching images found")

        self.new_images = []
        for id in self.new_image_ids:
            self.new_images.append(self.images[id])

        cnt = 0
        self.new_dects = []
        for id, annotations in self.id_to_annot.items():
            if not id in self.image_ids_being_filtered:
                self.new_dects.append(self.id_to_dets[id])
                cnt += 1
            if cnt >= self.max_files:
                break

    def _copy_images(self):
        files = []
        cnt = 0

        if not os.path.exists(self.output_image_folder):
            os.mkdir(self.output_image_folder)

        for i in self.new_images:
            files.append(i['file_name'])

        for f in files:
            shutil.copy(os.path.join(self.input_image_path, f),
                        os.path.join(self.output_image_folder, f))
            cnt += 1
            if cnt % 500 == 0:
                print(str(cnt) + " images copied")

        print(str(cnt) + " images copied")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_json", dest="input_json", help="path to a json file in coco format")
    parser.add_argument("-d", "--detections_json", dest="detections_json", help="path to person detection results json")
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
