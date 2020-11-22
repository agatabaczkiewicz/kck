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
    #show_img(photo)
    return photo


def tresholding(img):
    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    (thresh, img) = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY_INV)
    img = cv2.dilate(img, kernel, iterations=1)
    return img


def cut(img, color):
        while (img[0, :] == color).all(): # dla kazdej kolmny pierwszy wiersz
            img = np.delete(img, 0, 0) # usuwa pierwszy wiersz
        while (img[-1, :] == color).all():
            img = np.delete(img, -1, 0) # usuwa ostatni wiersz
        while (img[:, 0] == color).all():
            img = np.delete(img, 0, 1) # usuwa pierwsza kolumne
        while (img[:, -1] == color).all():
            img = np.delete(img, -1, 1) # usuwa ostatnia kolumne
        return img


def cut_min(img):
    img = cut(img, 0)
    show_img(img)
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


def rotate(img):
    corners = ski.transform.probabilistic_hough_line(img,line_length=int(2 / 3 * (img.shape[0])),
                                                         line_gap=1)
    line = corners[0]
    for a in corners:
        if line[0][1] > a[0][1] or line[1][1] > a[1][1]:
            line = a

    angle = math.degrees(math.atan2(line[0][1]- line[1][1], line[0][0]- line[1][0]))

    if angle > 5:
        img = ps.make_big_square(img)
        img = ski.transform.rotate(img, angle)
        img = cut_min(img)
        img = (img >= 1) * 255
        img = mp.dilation(mp.erosion(img))

    return img

