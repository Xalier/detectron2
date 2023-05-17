# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
import shutil
import pycocotools.mask as mask_util
from pathlib import Path
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager

from .. import DatasetCatalog, MetadataCatalog

logger = logging.getLogger(__name__)

__all__ = ["load_coco_dir", "register_coco_instances"]

def load_coco_dir(root_dir):
    dataset_name='coco_person'    
    meta = MetadataCatalog.get(dataset_name)
    meta.thing_classes = ['person']
    meta.thing_dataset_id_to_contiguous_id = {0:0}

    dataset_dicts = []
    timer = Timer()
    instance_dirs = sorted([x for x in Path(root_dir).iterdir() if x.is_dir()])
    for instance_dir in instance_dirs:
        label_file_ids = sorted(map(lambda x: x.stem, instance_dir.glob('./output/*.txt')))

        # All images for each instance share the same size because
        # they come from the same video so just look at the first
        # image.
        w,h = (0,0)
        first_label_file_id = label_file_ids[0]
        with Image.open(instance_dir / 'images' / f'{first_label_file_id}.jpg') as first_image:
            w, h = first_image.size
        
        for label_file_id in label_file_ids:
            label_data = None
            with open(instance_dir / 'output' / f'{label_file_id}.txt', 'r') as label_file:
                # Only care about the first instance.
                label_data = np.array(label_file.readline().split(' '), dtype=np.float32)

            if label_data is None:
                logger.warn('Skipping file with no invalid label data.')
                continue

            bbox = label_data[1:5]
            # Convert XYWH_REL to XYXY_ABS
            bbox[0] *= w 
            bbox[1] *= h
            bbox[2] *= w
            bbox[3] *= h
            bbox[0] -= bbox[2] / 2.0
            bbox[1] -= bbox[3] / 2.0
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            
            keypoints = label_data[5:]
            # Resize x, y to absolute pixel width and height
            # Reset visibility to coco visible (https://cocodataset.org/#keypoints-eval)            
            keypoints[0::3] *= w
            keypoints[1::3] *= h
            keypoints[2::3] = 2.0

            record = {
                'file_name': instance_dir / 'images' / f'{label_file_id}.jpg',
                'width': w,
                'height': h,
                'image_id': f'{instance_dir.name}-{label_file_id}',
                'annotations': [
                    {
                        'category_id': 0,
                        'bbox_mode': BoxMode.XYXY_ABS,
                        'bbox': bbox,
                        'keypoints': keypoints,
                    }
                ]
            }
            dataset_dicts.append(record)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(root_dir, timer.seconds()))

    return dataset_dicts


def register_coco_instances(name, metadata, root_dir):
    assert isinstance(name, str), name
    assert isinstance(root_dir, (str, os.PathLike)), root_dir
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_dir(root_dir))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        root_dir=root_dir, evaluator_type="coco", **metadata
    )


if __name__ == "__main__":
    """
    Test the COCO json dataset loader.

    Usage:
        python -m detectron2.data.datasets.coco_instance \
            path/to/root_dir

        "dataset_name" can be "coco_2014_minival_100", or other
        pre-registered ones
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys

    logger = setup_logger(name=__name__)
    meta = MetadataCatalog.get('coco_person')

    dicts = load_coco_dir(sys.argv[1])
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "/tmp/coco-data-vis"
    os.makedirs(dirname, exist_ok=True)
    # Sample first 10
    for d in dicts[:10]:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
