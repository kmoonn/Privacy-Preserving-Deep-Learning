# Privacy-Preserving-Deep-learning


[TOC]

# Papers

## 1. DLMT: Outsourcing Deep Learning with Privacy Protection Based on Matrix Transformation

## Information

**Year**：2023

**CCF**：C

**Publisher:** IEEE

**Published in:** [2023 26th International Conference on Computer Supported Cooperative Work in Design (CSCWD)](https://ieeexplore.ieee.org/xpl/conhome/10152543/proceeding)

**Cited**：1

## Main Content

### Abstract

To solve this problem, we propose a privacy-preserving deep learning model based on matrix transformation. Specifically, we transform original data by adding or multiplying a random matrix.

### Methods

- methods based on data processing

- Matrix Transformation

transform original data by adding or multiplying a random matrix

treat each training image data as a **pixel matrix**, and add or multiply it with a random matrix **element by element**

two matrix transformations: **matrix addition and matrix multiplication**

Each image data needs to be processed with a **same random matrix**

each training and testing data must be transformed with a same random matrix R

the dimension of R is as same as the original data

The matrix R used to transform original data is not open to public.

![image-20240802141021865](https://image.kmoon.fun//images/202408021410907.png)

![image-20240802140537439](https://image.kmoon.fun//images/202408021405572.png)

![image-20240802141225769](https://image.kmoon.fun//images/202408021412811.png)

![image-20240802141252668](https://image.kmoon.fun//images/202408021412719.png)

- parameters

|  Name  | Description                                                  |
| :----: | ------------------------------------------------------------ |
|   A    | image pixel matrix                                           |
| W×H×C  | image dimension(width、height、channels)                     |
|   R    | random matrix(same dimension、Each value in R is a random integer in the interval [1,MAX_V]) |
| MAX_V  | random positive integer                                      |
| RISE_V | a constant number                                            |

### Dataset

- MNIST
- CIFAR-10

### Experiment

- Pytorch

https://github.com/kuangliu/pytorch-cifar

- ResNet18

## Related

### Citations

- Sagar Sharma, AKM Mubashwir Alam and Keke Chen, “Image Disguising for Protecting Data and Model Confidentiality in Outsourced Deep Learning”, In 2021 IEEE 14th International Conference on Cloud Computing (CLOUD), pp. 71–77, 2021.

**data processing**: a method based on block-wise permutation and RMT (Randomized Multidimensional Transformations)

- Warit Sirichotedumrong and Hitoshi Kiya, “A gan-based image transformation scheme for privacy-preserving deep neural networks”, In 2020 28th European Signal Processing Conference (EUSIPCO), pp. 745-749, 2021.

**data processing**: a transformation network to protect visual information in images.

- Warit Sirichotedumrong, Takahiro Maekawa, Yuma Kinoshita and Hitoshi Kiya, “Privacy-Preserving Deep Neural Networks with Pixel-Based Image Encryption Considering Data Augmentation in the Encrypted Domain”, 2019 IEEE International Conference on Image Processing (ICIP), pp. 674-678, 2019.

**data processing**: each image pixel is XORed with 255 with the probability 50%, and the result is then submitted to the server for training or testing. However, since there are about half of image pixels remain unchanged, some information about original data would be recovered.

- Mahawaga Arachchige Pathum Chamikara, Peter Bert´ok, Ibrahim Khalil, Dongxi Liu and Seyit Camtepe, “Privacy preserving face recognition utilizing differential privacy”, Computers & Security, Elsevier, vol. 97, pp. 101951, 2020.

**differential privacy**: studied an image processing method for protecting human face data with differential privacy.

- Liyue Fan, “Image pixelization with differential privacy”, IFIP Annual Conference on Data and Applications Security and Privacy, Springer, pp. 148-162, 2018.

**differential privacy**: studied the m-neighborhood notion based on differential privacy to protect sensitive information

### Thinking

矩阵变换后像素值会超过像素值范围255，所以需要对结果进行一个**归一化**处理，将像素值范围控制在0-255之间，主要是考虑矩阵变换后对原始像素矩阵整体的影响。

MMT、MAT都是**逐像素**相加和相乘，对于RGB图像需要对RGB3通道分别变换后再合起来。

![image-20240802143331623](https://image.kmoon.fun//images/202408021433671.png)



## 2. Privacy-Preserving Deep Neural Networks with Pixel-Based Image Encryption Considering Data Augmentation in the Encrypted Domain

## Information

**Year**：2019

**CCF**：C

**Publisher:** IEEE

**Published in:** [2019 IEEE International Conference on Image Processing (ICIP)](https://ieeexplore.ieee.org/xpl/conhome/8791230/proceeding)

**Cited**：116

## Main Content

### Abstract

### Methods

### Dataset

### Experiment

## Related

### Reference

### Thinking

## 3. Image Pixelization with Differential Privacy

## Information

**Year**：2018

**CCF**：None

**Publisher:** IEEE

**Published in:** Data and Applications Security and Privacy XXXII(DBSec 2018)

**Cited**：153

## Main Content

### Abstract

### Methods

### Dataset

### Experiment

## Related

### Reference

### Thinking

