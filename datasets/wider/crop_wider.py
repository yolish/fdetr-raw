
from os.path import join
import numpy as np
import json
def read_dataset_file(dataset_path, file_path, img_dir, split,  class_type, model_dir):
    """Load the dataset from the text file."""

    out_img_dir = ""

    lines = open(file_path).readlines()
    for line in lines:
        in_json_file = line.strip()

        labels = []
        for line in lines:
            json_file = line.strip()
            with open(join(dataset_path, json_file)) as f:
                d = json.load(f)
                img_path = join(img_dir, d.get("image_path"))
                bboxes = d.get("bboxes")
                # read the image

                for box in bboxes:
                    # crop

                    # save

                    # call model to label
                # format is x0,y0,x1,y1
                bboxes = np.array([np.array(box[:4]) for box in bboxes])

                # Remove invalid bboxes where w or h are 0
                invalid = np.where(np.logical_or(bboxes[:, 2] < bboxes[:, 0],
                                                 bboxes[:, 3] < bboxes[:, 1]))
                bboxes = np.delete(bboxes, invalid, 0)
                datum = {
                    "img_path": img_path,
                    "bboxes": bboxes
                }
                data.append(datum)