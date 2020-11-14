import skimage as ski
from matplotlib import pyplot as plt
from skimage import data, io, filters, exposure
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
import skimage.morphology as mp
from skimage.transform import resize
from skimage.segmentation import flood_fill
import numpy as np
import photo_spliting as ps
import math
import warnings
import cv2


def load_file(path):
    photo = io.imread(path)
    if photo.shape[0] * photo.shape[1] > 1000 * 2000:
        photo = cv2.resize(photo, (int(photo.shape[1]/4), int(photo.shape[0]/4)))
    return photo


def black_white(img):
    warnings.simplefilter("ignore")
    img = rgb2gray(img)
    img **= 3
    img = (img <= 0.04) * 1
    for i in range(1):
        img = mp.dilation(img)
    img = (img == 1) * 255
    return img


def cut(img, tag, color):
    if tag == 'allp':
        while (img[0, :] == color).all():
            img = np.delete(img, 0, 0)
        while (img[:, 0] == color).all():
            img = np.delete(img, 0, 1)
        while (img[-1, :] == color).all():
            img = np.delete(img, -1, 0)
        while (img[:, -1] == color).all():
            img = np.delete(img, -1, 1)
    elif tag == 'anyp':
        while (img[0, :] == color).any():
            img = np.delete(img, 0, 0)
        while (img[:, 0] == color).any():
            img = np.delete(img, 0, 1)
        while (img[-1, :] == color).any():
            img = np.delete(img, -1, 0)
        while (img[:, -1] == color).any():
            img = np.delete(img, -1, 1)
    return img


def cut_min(img):
    img = cut(img, 'allp', 255)
    img = cut(img, 'anyp', 255)
    img = cut(img, 'allp', 0)
    return img


def fill_board(img, top_right,color=125): #linie planszy będą czarne
    img = ski.segmentation.flood_fill(img, (top_right[1], top_right[0]), color)
    return img


def print_result(result):
    for i in result:
        print(''.join(i))


def show_img(img):
    fig, ax = plt.subplots()
    plt.grid(True)
    ax.imshow(img, cmap=plt.cm.gray)
    return ax


def rotate(rot):
    corners = ski.transform.probabilistic_hough_line(rot,line_length=int(2 / 3 * min(rot.shape[0], rot.shape[1])),
                                                         line_gap=1)
    line = corners[0]
    c = np.array([line[0][0], line[1][1]])
    len1 = np.linalg.norm(c - np.array(line[0]))
    len2 = np.linalg.norm(c - np.array(line[1]))

    if line[0][0] >= line[1][0] and line[0][1] >= line[1][0]:
        if len2 != 0:
            tangens = len1 / len2
        else:
            tangens = 0
    else:
        if len1 != 0:
            tangens = len2 / len1
        else:
            tangens = 0

    angle = np.degrees(math.atan(tangens))
    if angle > 10:
        rot = ps.make_big_square(rot)
        rot = ski.transform.rotate(rot, angle)
        rot = cut_min(rot)
        rot = (rot >= 1) * 255
        rot = mp.dilation(mp.erosion(rot))

    return rot


