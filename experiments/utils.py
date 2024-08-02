# _*_ coding : utf-8 _*_
# @Time : 2024/8/2 下午3:54
# @Author : Kmoon_Hs
# @File : utils

# utils.py
import os
from PIL import Image

def save_image(image, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    image.save(path)

# 其他实用函数
