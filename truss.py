"""
Mask R-CNN
Train on the toy truss dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 truss.py train --dataset=/path/to/truss/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 truss.py train --dataset=/path/to/truss/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 truss.py train --dataset=/path/to/truss/dataset --weights=imagenet

    # Apply color splash to an image
    python3 truss.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 truss.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
# Root directory of the project
ROOT_DIR = os.path.abspath(".")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from skimage import img_as_ubyte
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class TrussConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "truss"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + truss

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class TrussDataset(utils.Dataset):


    
    def load_truss(self, dataset_dir, subset):
        """Load a subset of the Truss dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("truss", 1, "truss")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        print("DATASETDIE: "+str(dataset_dir))
        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        #annotations = list(annotations.values())  # don't need the dict keys
        #print(annotations[0])
        metaDat = dict(annotations["_via_img_metadata"])
        #annotations = dict([])
        #for k, v in metaDat.items():
        #    print(v["regions"])
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in metaDat.values() if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "truss",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a truss dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "truss":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "truss":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, dataset):
    """Train the model."""
    # Training dataset.
    dataset_train = TrussDataset()
    dataset_train.load_truss(dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TrussDataset()
    dataset_val.load_truss(dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
        print("<--Image dimensions--->")
        print(mask.shape[0],mask.shape[1])
        print(image.shape[0],image.shape[1])
        imageCopy = image
        rows= image.shape[0]
        cols= image.shape[1]
        for i in range(rows):
            for j in range(cols):
                if mask[i,j] == True:
                    imageCopy[i,j] = [255, 255, 255]
                else:
                    imageCopy[i,j] = [0, 0, 0]
        skimage.io.imsave("Masked.png", imageCopy)
        print(splash.shape[0],splash.shape[1])
    else:
        splash = gray.astype(np.uint8)
    return splash

"""Runs the detection pipeline.

    images: List of images, potentially of different sizes.

    Returns a list of dicts, one dict per image. The dict contains:
    rois: [N, (y1, x1, y2, x2)] detection bounding boxes
    class_ids: [N] int class IDs
    scores: [N] float probability scores for the class IDs
    masks: [H, W, N] instance binary masks
"""
def detect_and_color_splash(model, image_path):
    # Run model detection and generate the color splash effect
    print("Running on {}".format(image_path))
    # Read image
    image = skimage.io.imread(image_path)
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    # Color splash
    splash = color_splash(image, r['masks'])
    # Save output
    file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    skimage.io.imsave(file_name, splash)
    print("Saved to ", file_name)
    return r

def ClassifyCornerJoint( pathToWeights, image):
    class InferenceConfig(TrussConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir="")
    model.load_weights(pathToWeights, by_name=True)
    rVal = detect_and_color_splash(model, image_path=image)
    bboxDict = dict([])
    try:
        bBoxCoords = rVal["rois"]
        print(rVal["scores"])
        #rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        im = skimage.io.imread(image)
        cv_image = img_as_ubyte(im)
        count = 0
        for coord in bBoxCoords:
            print("|-----Found truss corner-----|")
            print("| Confidence: " +str(rVal["scores"][count]))
            print("| BBox coords (y1, x1, y2, x2): "+str(coord))
            print("|----------------------------|")
            cv2.rectangle(cv_image,(coord[1],coord[0]),(coord[3],coord[2]),(0,255,0),2)
            cv2.imwrite("BBox.png",cv_image)
            #cv2.imshow("my pic",cv_image)
            #k = cv2.waitKey(0) # 0==wait forever
            bboxDict["found"] = True
            bboxDict["x1"] = coord[1]
            bboxDict["y1"] = coord[0]
            bboxDict["x2"] = coord[3]
            bboxDict["y2"] = coord[2]
            count+=1
        return bboxDict
    except:
        #rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        bboxDict["found"] = False
        #invalid pixels
        bboxDict["x1"] = -1
        bboxDict["y1"] = -1
        bboxDict["x2"] = -1
        bboxDict["y2"] = -1
        print("No truss.")
        return bboxDict
def Train(dataPath):
    config = TrussConfig()
    model = modellib.MaskRCNN(mode="training", config=config,model_dir=None) #model_dir?
    dataset= dataPath
    # Default to coco for transfer learning and for model.
    weights_path = COCO_WEIGHTS_PATH
    # Download coco weights if not present
    if not os.path.exists(weights_path):
        utils.download_trained_weights(weights_path)
    model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
    train(model)
#ClassifyCornerJoint("/home/robert/workspace/Mask_RCNN/logs","mask_rcnn_truss_0030.h5","122.jpg")
#Train("/home/robert/workspace/Mask_RCNN/datasets/truss")
