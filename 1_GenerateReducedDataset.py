import json
from enum import IntEnum
from pathlib import Path
import itertools
import math
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


def check_body_keypoint_cnt_max(annotations: list):
    keypoints = Keypoint
    keypoint_cnt_max = 0
    for annot in annotations:
        keypoint_cnt = 0
        keypoints_x = annot['keypoints'][::3]
        keypoints_y = annot['keypoints'][1::3]
        for i in range(keypoints.left_shoulder, keypoints.right_ankle + 1):
            if keypoints_x[i] != 0 and keypoints_y[i] != 0:
                keypoint_cnt += 1
        if keypoint_cnt > keypoint_cnt_max:
            keypoint_cnt_max = keypoint_cnt
    return keypoint_cnt_max


class CocoFilter():
    """ Filters COCO dataset (info, licenses, images, annotations, categories) and generates a new, filtered json file
    """

    def __init__(self, paths):
        # Verify input path exists
        if not Path(paths.input_json).exists():
            print('Input json path not found.')
            print('Quitting early.')
            quit()
        self.input_json_path = Path(paths.input_json)

        # Verify output path does not already exist
        if Path(paths.output_json).exists():
            should_continue = input('Output path already exists. Overwrite? (y/n) ').lower()
            if should_continue != 'y' and should_continue != 'yes':
                print('Quitting early.')
                quit()
        self.output_json_path = Path(paths.output_json)

        with open(self.input_json_path) as json_file:
            self.jsonFile = json.load(json_file)

    def main(self):
        self.filter_for_categories = ['person']

        # filters pictures which does not contain at least one
        # person with more than 3 keypoints regardless the head keypoints
        self.main_person_body_keypoint_limit = 6

        # filters persons which are far away
        self.main_person_body_keypoint_min_distance_limit = 120

        # amount of output
        self.max_files = 100000

        # Process the json
        print('Processing input json...')
        self._generate_info()
        self._process_images()
        self._process_annotations()
        self._process_categories()

        # Filter the json
        print('Filtering...')
        self._filter_categories()
        self._filter_annotations()
        self._filter_images()

        # Build new JSON
        new_master_json = {
            'info': self.info,
            'images': self.new_images,
            'annotations': self.new_annotations,
            'categories': self.new_categories
        }

        # Write the JSON to a file
        print('Saving new json file...')
        with open(self.output_json_path, 'w+') as output_file:
            json.dump(new_master_json, output_file)

        print('Filtered json saved.')

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
            else:
                print(f'ERROR: Skipping duplicate image id: {image}')

    def _process_annotations(self):
        self.id_to_annot = dict()
        for annot in self.jsonFile['annotations']:
            image_id = annot['image_id']
            if image_id not in self.id_to_annot:
                self.id_to_annot[image_id] = []
            self.id_to_annot[image_id].append(annot)

    def _process_categories(self):
        self.categories = dict()
        self.super_categories = dict()
        self.category_set = set()

        for category in self.jsonFile['categories']:
            cat_id = category['id']  # 1
            super_category = category['supercategory']  # person

            # Add category to categories dict
            if cat_id not in self.categories:  # 1
                self.categories[cat_id] = category  # 1 = old entry
                self.category_set.add(category['name'])  # person
            else:
                print(f'ERROR: Skipping duplicate category id: {category}')

            # Add category id to the super_categories dict
            if super_category not in self.super_categories:
                self.super_categories[super_category] = {cat_id}  # "person" = {1}
            else:
                self.super_categories[super_category] |= {cat_id}  # e.g. {1, 2, 3} |= {4} => {1, 2, 3, 4}

    def _filter_categories(self):
        """ Find category ids matching args
            Create mapping from original category id to new category id
            Create new collection of categories
        """
        self.new_category_map = dict()
        new_id = 1
        for key, item in self.categories.items():
            if item['name'] in self.filter_for_categories:
                self.new_category_map[key] = new_id
                new_id += 1

        self.new_categories = []
        for original_cat_id, new_id in self.new_category_map.items():
            new_category = dict(self.categories[original_cat_id])
            new_category['id'] = new_id
            self.new_categories.append(new_category)

    def _filter_annotations(self):
        """ Create new collection of annotations matching category ids
            Keep track of image ids matching annotations
        """
        self.new_annotations = []
        self.new_image_ids = set()
        cnt = 0
        for image_id, annotations in self.id_to_annot.items():
            not_crowd = True
            main_person_large_enough = False
            categoryMatches = True

            for annot in annotations:
                if annot['category_id'] not in self.new_category_map.keys():
                    categoryMatches = False
                    break

                if annot['iscrowd'] == 1:
                    not_crowd = False
                    break

                # get x,y,type from keypoints
                keypoint_cnt = 0
                keypoints_x = annot['keypoints'][::3]
                keypoints_y = annot['keypoints'][1::3]
                # keypoints_type = annotation['keypoints'][2::3]
                for i in range(Keypoint.left_shoulder, Keypoint.right_ankle + 1):
                    if keypoints_x[i] != 0 and keypoints_y[i] != 0:
                        keypoint_cnt += 1

                if keypoint_cnt > self.main_person_body_keypoint_limit:
                    # get (x,y) pair of keypoints
                    keypoints_x_y = list(zip(keypoints_x, keypoints_y))

                    for pair in itertools.combinations(keypoints_x_y, r=2):
                        if pair[0][0] != 0 and pair[0][1] != 0 and pair[1][0] != 0 and pair[1][1] != 0:
                            if math.hypot(pair[1][0] - pair[0][0], pair[1][1] - pair[0][1]) > self.main_person_body_keypoint_min_distance_limit:
                                main_person_large_enough = True

            if not_crowd and main_person_large_enough and categoryMatches:
                for annot in annotations:
                    original_seg_cat = annot['category_id']
                    new_annotation = dict(annot)
                    new_annotation['category_id'] = self.new_category_map[original_seg_cat]
                    self.new_annotations.append(new_annotation)
                    self.new_image_ids.add(image_id)
                cnt += 1
            if cnt % 500 == 0:
                print(str(cnt) + " matching images found")
            if cnt >= self.max_files:
                break
        print(str(cnt) + " matching images found")

    def _filter_images(self):
        """ Create new collection of images
        """
        self.new_images = []
        for image_id in self.new_image_ids:
            self.new_images.append(self.images[image_id])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter COCO JSON: "
                                                 "Filters a COCO Instances JSON file to only include specified categories. "
                                                 "This includes images, and annotations. Does not modify 'info' or 'licenses'.")

    parser.add_argument("-i", "--input_json", dest="input_json",
                        help="path to a json file in coco format")
    parser.add_argument("-o", "--output_json", dest="output_json",
                        help="path to save the output json")

    args = parser.parse_args()

    cf = CocoFilter(args)
    cf.main()
