#!/usr/bin/python
import numpy as np
import Image
from glob import glob
import os
import math
import cPickle

def crop_image(img, target):
    width = img.size[0]
    height = img.size[1]
    if width > height:
        ratio = target * 1.0 / height
        newWidth = int(width * ratio)
        newHeight = target
        img_ret = img.resize((newWidth, newHeight), Image.ANTIALIAS)
        img_ret = img_ret.crop(((newWidth - target)/2, 0,
                               (newWidth - target)/2 + target, target))
    else:
        ratio = target * 1.0 / width
        newWidth = target
        newHeight = int(height * ratio)
        img_ret = img.resize((newWidth, newHeight))
        img_ret = img_ret.crop((0, (newHeight - target)/2, 
                               target, (newHeight - target)/2 + target))
    return img_ret

def preprocess_caltech101_image():
    caltech101_image_path = "/home/rolexye/project/data/101_ObjectCategories/Faces/"
    for idx, img_file in enumerate(glob(os.path.join(caltech101_image_path, "*.jpg"))):
        img = Image.open(img_file)
        img = img.convert("L")
        img = crop_image(img, 150)
        img.save("../data/caltech101/faces/%d.png" % idx)

def divide_kyoto_image(img_num, div_num):
    img = []
    image_path = "../data/kyoto"
    for idx in xrange(img_num):
        img.append(Image.open("%s/%d.tif" % (image_path, idx)))

    width = img[0].size[0]
    height = img[0].size[1]

    for div_idx in xrange(div_num):
        rnd_img_idx = np.random.randint(0, img_num) 
        x = np.random.randint(0, width - 64)
        y = np.random.randint(0, height - 64)
        div_img = img[rnd_img_idx].crop((x, y, x+64, y+64))
        div_img.save("%s/divide/%d.png" % (image_path, div_idx))

def raw_image_2_pkl(image_path, image_suffix, image_count, data_path):
    data = []
    for idx in xrange(image_count):
        img = Image.open("%s/%d.%s" % (image_path, idx, image_suffix))
        img_data = np.asarray(img.getdata()).reshape(img.size)
        img_data = np.array(img_data/255.0, dtype="float32")
        data.append(img_data)
    cPickle.dump(data, open(data_path, "w+"))

if __name__ == "__main__":
    divide_kyoto_image(10, 500)
    #preprocess_kyoto_image()
    #preprocess_caltech101_image()
    #raw_image_2_pkl("../data/caltech101/faces", "png", 100, "../data/faces_train.pkl")
    #raw_image_2_pkl("../data/kyoto", "tif", 10, "../data/kyoto_train.pkl")
    raw_image_2_pkl("../data/kyoto/divide", "png", 500, "../data/kyoto_train.pkl")
