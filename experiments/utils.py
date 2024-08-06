# _*_ coding : utf-8 _*_
# @Time : 2024/8/2 下午3:54
# @Author : Kmoon_Hs
# @File : utils

# utils.py
import os

import numpy as np
from PIL import Image

def save_image(image, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    image.save(path)

# 将矩阵保存为图像
def save_image_from_array(img_array, save_path):
    """
    Save a numpy array as an image using PIL.

    Args:
        img_array (numpy.ndarray): The image data in numpy array format.
        save_path (str): The path to save the image.
    """
    img = Image.fromarray(img_array.astype('uint8'))
    img.save(save_path, 'JPEG')


# 加载图像为矩阵
def load_image_to_array(image_path):
    """
    Load an image from a file path and return it as a numpy array.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The image data in numpy array format.
    """
    with Image.open(image_path) as img:
        img_array = np.array(img, dtype=np.float32)
    return img_array

