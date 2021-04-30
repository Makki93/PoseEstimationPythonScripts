from pycocotools.coco import COCO
...

train_annot_path = 'dataset_coco/annotations  /person_keypoints_train2017.json'
val_annot_path = 'dataset_coco/annotations/person_keypoints_val2017.json'
train_coco = COCO(train_annot_path) # load annotations for training set
val_coco = COCO(val_annot_path) # load annotations for validation set
...
# function iterates ofver all ocurrences of a  person and returns relevant data row by row
def get_meta(coco):
    ids = list(coco.imgs.keys())
    for i, img_id in enumerate(ids):
        img_meta = coco.imgs[img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # basic parameters of an image
        img_file_name = img_meta['file_name']
        w = img_meta['width']
        h = img_meta['height']
        # retrieve metadata for all persons in the current image
        anns = coco.loadAnns(ann_ids)

        yield [img_id, img_file_name, w, h, anns]

...

# iterate over images
for img_id, img_fname, w, h, meta in get_meta(train_coco):
    ...
    # iterate over all annotations of an image
    for m in meta:
        # m is a dictionary
        keypoints = m['keypoints']
        ...
...