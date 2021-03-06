import os
import argparse
import json
from pathlib import Path
from datetime import date


class CocoFilter:
    """ Filters the COCO dataset (info, licenses, images, annotations, categories) and generates a new, filtered json file
    """

    def __init__(self, paths):
        self.filter_for_categories = ['person']

        # Verify image path exists
        if not Path(paths.image_path).exists():
            print('Image path not found.')
            print('Quitting early.')
            quit()
        self.image_path = Path(args.image_path)

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

        print('Loading json file...')
        with open(self.input_json_path) as json_file:
            self.jsonFile = json.load(json_file)

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
        print("Original image count: " + str(len(self.images.keys())))

    def _process_annotations(self):
        self.annotations = dict()
        for annotation in self.jsonFile['annotations']:
            image_id = annotation['image_id']
            if image_id not in self.annotations:
                self.annotations[image_id] = []
            self.annotations[image_id].append(annotation)
        print("Original annotations count: " + str(len(self.jsonFile['annotations'])))

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
        missing_categories = set(self.filter_for_categories) - self.category_set
        if len(missing_categories) > 0:
            print(f'Did not find categories: {missing_categories}')
            should_continue = input('Continue? (y/n) ').lower()
            if should_continue != 'y' and should_continue != 'yes':
                print('Quitting early.')
                quit()

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
        for image_id, annotation_list in self.annotations.items():
            if os.path.isfile(os.path.join(self.image_path, self.images[image_id]['file_name'])):
                for annotation in annotation_list:
                    original_seg_cat = annotation['category_id']
                    new_annotation = dict(annotation)
                    new_annotation['category_id'] = self.new_category_map[original_seg_cat]
                    self.new_annotations.append(new_annotation)
                    self.new_image_ids.add(image_id)
        print("New image count: " + str(len(self.new_image_ids)))
        print("New annotation count: " + str(len(self.new_annotations)))


    def _filter_images(self):
        """ Create new collection of images
        """
        self.new_images = []
        for image_id in self.new_image_ids:
            self.new_images.append(self.images[image_id])

    def main(self):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_json", dest="input_json", help="path to a json file in coco format")
    parser.add_argument("-o", "--output_json", dest="output_json", help="path to save the output json")
    parser.add_argument("-p", "--image_path", dest="image_path", help="path to the images")

    args = parser.parse_args()

    cf = CocoFilter(args)
    cf.main()
