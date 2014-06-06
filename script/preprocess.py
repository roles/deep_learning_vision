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
    """
import numpy as np
import Image
from glob import glob
import os
import math
import cPickle
image_path = "../data/kyoto"
image_suffix = "tif"
idx = 1
img = Image.open("%s/%d.%s" % (image_path, idx, image_suffix))
img_data = np.asarray(img.getdata()).reshape(img.size)
img_data = img_data - img_data.mean()
img_data = img_data / img_data.std()
width = img_data.shape[0]
height = img_data.shape[1]
fx, fy = np.meshgrid(np.linspace(-width/2, width/2-1, width), np.linspace(-height/2, height/2-1, height))
rho = np.sqrt(fx * fx + fy * fy)
f0 = 0.4 * np.mean([width, height])
filt = rho * np.exp(- np.power(rho/f0, 4))
If = np.fft.fft2(img_data)
img_data = np.real(np.fft.ifft2(If * np.fft.fftshift(filt)))
img_data = img_data / img_data.std()
img_data = img_data - img_data.mean()
img_data = img_data / np.sqrt(np.mean(np.power(img_data, 2)))
img_data = np.sqrt(0.1) * img_data
    """
    data = []
    for idx in xrange(image_count+1):
        try:
            img = Image.open("%s/%d.%s" % (image_path, idx, image_suffix))
            img_data = np.asarray(img.getdata()).reshape(img.size)
            img_data = img_data - img_data.mean()
            img_data = img_data / img_data.std()
            width = img_data.shape[0]
            height = img_data.shape[1]
            fx, fy = np.meshgrid(np.linspace(-width/2, width/2-1, width), np.linspace(-height/2, height/2-1, height))
            rho = np.sqrt(fx * fx + fy * fy)
            f0 = 0.4 * np.mean([width, height])
            filt = rho * np.exp(- np.power(rho/f0, 4))
            If = np.fft.fft2(img_data)
            img_data = np.real(np.fft.ifft2(If * np.fft.fftshift(filt)))
            img_data = img_data / img_data.std()
            img_data = img_data - img_data.mean()
            img_data = img_data / np.sqrt(np.mean(np.power(img_data, 2)))
            img_data = np.sqrt(0.1) * img_data
            data.append(img_data)
        except:
            pass
    cPickle.dump(data, open(data_path, "w+"))

if __name__ == "__main__":
    #divide_kyoto_image(10, 500)
    #preprocess_kyoto_image()
    #preprocess_caltech101_image()
    #raw_image_2_pkl("../data/caltech101/faces", "png", 100, "../data/faces_train.pkl")
    raw_image_2_pkl("../data/kyoto", "tif", 10, "../data/kyoto_large_train.pkl")
    #raw_image_2_pkl("../data/kyoto/divide", "png", 500, "../data/kyoto_train.pkl")

