# install dependencies not included by Colab
# use pip3 to ensure compatibility w/ Google Deep Learning Images

# install dependencies not included by Colab
# use pip3 to ensure compatibility w/ Google Deep Learning Images
# !pip3 install -q pydicom
# !pip3 install -q tqdm
# !pip3 install -q imgaug

NUM_EPOCHS = 40

import os
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa

import skimage.io
import skimage.measure
from tqdm import tqdm
from PIL import Image

import requests
import shutil
import zipfile

import mdai
mdai.__version__

mdai_client = mdai.Client(domain='public.md.ai', access_token="1aed8036b2599460d9da97084b0708e5")
p = mdai_client.project('aGq4k6NW', path='./lesson2-data')

# download MD.ai's dilated unet implementation
UNET_URL = 'https://s3.amazonaws.com/md.ai-ml-lessons/unet.zip'
UNET_ZIPPED = 'unet.zip'

if not os.path.exists(UNET_ZIPPED):
    r = requests.get(UNET_URL, stream=True)
    if r.status_code == requests.codes.ok:
        with open(UNET_ZIPPED, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    else:
        r.raise_for_status()

    with zipfile.ZipFile(UNET_ZIPPED) as zf:
        zf.extractall()

p.show_label_groups()

# this maps label ids to class ids as a dict obj
labels_dict = {'L_A8Jm3d':1 # Lung
              }

print(labels_dict)
p.set_labels_dict(labels_dict)


p.show_datasets()
dataset = p.get_dataset_by_id('D_rQLwzo')
dataset.prepare()


image_ids = dataset.get_image_ids()
len(image_ids)


imgs_anns_dict = dataset.imgs_anns_dict


from unet import dataset
from unet import train
from unet import dilated_unet


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

CONFIG_FP = 'unet/configs/11.json'
name = os.path.basename(CONFIG_FP).split('.')[0]
print(name)

with open(CONFIG_FP, 'r') as f:
    config = json.load(f)

images, masks = dataset.load_images(imgs_anns_dict)

# increase the number of epochs for better prediction
history = train.train(config, name, images ,masks, num_epochs=NUM_EPOCHS)


import matplotlib.pyplot as plt

print(history.history.keys())


from keras.models import load_model
import keras.backend as K

model_name = 'unet/trained/model_'+name+'.hdf5'
print(model_name)

model = load_model(model_name, custom_objects={'dice': train.dice, 'iou': train.iou}) # loading in the model just trained
