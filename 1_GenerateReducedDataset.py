import argparse
import json
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

    def _find_annotations_with_crowd(self):
        """ Create new collection of annotations matching filter criteria from init
            Keep track of image ids matching annotations
        """
        self.new_annotations = []
        self.new_image_ids = set()
        self.image_ids_with_crowd = set()
        cnt = 0
    
        for image_id, annotations in self.id_to_annot.items():
            for annot in annotations:
                if annot['iscrowd'] == 1:
                    self.image_ids_with_crowd.add(image_id)
                    break

    def _find_annotations_with_too_few_keypoints(self):
        """ Create new collection of annotations matching filter criteria from init
            Keep track of image ids matching annotations
        """
        self.new_annotations = []
        self.new_image_ids = set()
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

    def _filter_images(self):
        """ Create new collection of images
        """
        cnt = 0
        for image_id, annotations in self.id_to_annot.items():
            if (not image_id in self.image_ids_with_crowd) and (not image_id in self.image_ids_with_too_few_keypoints):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_json", dest="input_json", help="path to a json file in coco format")
    parser.add_argument("-o", "--output_json", dest="output_json", help="path to save the output json")
    parser.add_argument("-c", "--max_count_images", dest="max_count_images", help="maximum number of images (annotations)")
    parser.add_argument("-k", "--min_keypoint_cnt_per_person", dest="min_keypoint_cnt_per_person", help="minimum count of keypoints for each person in image")
    args = parser.parse_args()

    cf = CocoFilter(args)
    cf.main()
