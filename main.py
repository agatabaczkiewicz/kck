#!/usr/bin/python
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
import circle_cross as cc
from numpy import *
import cv2
import imutils

def perp(a):
    b = empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return
# liczy punkt przzecięcia dwóch prostych
def cross_point(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = dot(dap, db)
    num = dot(dap, dp)
    return (num / denom.astype(float)) * db + b1


def result(result, mark):
    for i in range(3):
        if result[0][1] == result[1][i] == result[2][i] == mark:  # sprawdza w pionie
            return True
        if result[i][0] == result[i][1] == result[i][2] == mark:  # w poziomie
            return True
    if result[0][0] == result[1][1] == result[2][2] == mark:  # sprawdza ukos 1
        return True
    elif result[2][0] == result[1][1] == result[0][2] == mark:  # ukos 2
        return True

    return False


def who_win(res):
    if result(res, 'O'):
        return '\nWygrały O\n'
    elif result(res, 'X'):
        return "\nWygrały X\n"
    else:
        return "\nNikt nie wygrał\n"


def search_forXO(img, up_l, up_r, down_l, down_r):
    result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    # print(img[0:min(up_l[0], up_r[0]), up_l[1]:up_r[1]])

    if len(np.unique(img[0:up_l[0], 0:up_l[1]])) == 3:
        result[0][0] = 'X'
    elif len(np.unique(img[0:up_l[0], 0:up_l[1]])) >= 4:
        result[0][0] = 'O'
    else:
        result[0][0] = ' '
    if len(np.unique(img[0:min(up_l[0], up_r[0]), up_l[1]:up_r[1]])) == 3:
        result[0][1] = 'X'
    elif len(np.unique(img[0:min(up_l[0], up_r[0]), up_l[1]:up_r[1]])) >= 4:
        result[0][1] = 'O'
    else:
        result[0][1] = ' '
    if len(np.unique(img[:up_r[0], up_r[1]:])) == 3:
        result[0][2] = 'X'
    elif len(np.unique(img[:up_r[0], up_r[1]:])) >= 4:
        result[0][2] = 'O'
    else:
        result[0][2] = ' '
    if len(np.unique(img[up_l[0]:down_l[0], 0:up_l[1]])) == 3:
        result[1][0] = 'X'
    elif len(np.unique(img[up_l[0]:down_l[0], 0:up_l[1]])) >= 4:
        result[1][0] = 'O'
    else:
        result[1][0] = ' '
    if ((len(np.unique(img[max(up_l[0], up_r[0]):min(down_l[0], down_r[0]),
                       max(up_l[1], down_l[1]):min(up_r[1], down_r[1])]))) == 3):
        result[1][1] = 'X'
    elif ((len(np.unique(img[max(up_l[0], up_r[0]):min(down_l[0], down_r[0]),
                         max(up_l[1], down_l[1]):min(up_r[1], down_r[1])]))) >= 4):
        result[1][1] = 'O'
    else:
        result[1][1] = ' '
    if len(np.unique(img[up_r[0]:down_r[0], max(down_r[1], up_r[1]):])) == 3:
        result[1][2] = 'X'
    elif len(np.unique(img[up_r[0]:down_r[0], max(down_r[1], up_r[1]):])) >= 4:
        result[1][2] = 'O'
    else:
        result[1][2] = ' '
    if len(np.unique(img[down_l[0]:, 0:down_l[1]])) == 3:
        result[2][0] = 'X'
    elif len(np.unique(img[down_l[0]:, 0:down_l[1]])) >= 4:
        result[2][0] = 'O'
    else:
        result[2][0] = ' '
    if len(np.unique(img[max(down_l[0], down_r[0]):, down_l[1]:down_r[1]])) == 3:
        result[2][1] = 'X'
    elif len(np.unique(img[max(down_l[0], down_r[0]):, down_l[1]:down_r[1]])) >= 4:
        result[2][1] = 'O'
    else:
        result[2][1] = ' '
    if len(np.unique(img[down_r[0]:, down_r[1]:])) == 3:
        result[2][2] = 'X'
    elif len(np.unique(img[down_r[0]:, down_r[1]:])) >= 4:
        result[2][2] = 'O'
    else:
        result[2][2] = ' '
    return result


def find_intersections(i):
    up_r = []
    up_l = []
    down_r = []
    down_l = []
    up = ()
    left = ()
    right = ()
    down = ()
    done = False
    while not done:
        try:
            corners = ski.transform.probabilistic_hough_line(i, line_length=int(2 / 3 * min(i.shape[0], i.shape[1])),
                                                             line_gap=1)
            for line in corners:
                if abs(line[0][0] - line[1][0]) < i.shape[1] // 3 and line[0][0] < i.shape[1] // 2 and line[1][0] < \
                        i.shape[1] // 2:
                    left = np.float32(np.array(line))

                    break
            for line in corners:
                if abs(line[0][0] - line[1][0]) < i.shape[1] // 3 and line[0][0] > i.shape[1] // 2 and line[1][0] > \
                        i.shape[1] // 2:
                    right = np.float32(np.array(line))

                    break

            for line in corners:
                if abs(line[0][1] - line[1][1]) < i.shape[0] // 3 and line[0][1] < i.shape[0] // 2 and line[1][1] < \
                        i.shape[0] // 2:
                    up = np.float32(np.array(line))

                    break
            for line in corners:
                if abs(line[0][1] - line[1][1]) < i.shape[0] // 3 and line[0][1] > i.shape[0] // 2 and line[1][1] > \
                        i.shape[0] // 2:
                    down = np.float32(np.array(line))

                    break
            done = True

            up_l = np.uint16(cross_point(left[0], left[1], up[0], up[1]))
            down_l = np.uint16(cross_point(left[0], left[1], down[0], down[1]))
            up_r = np.uint16(cross_point(right[0], right[1], up[0], up[1]))
            down_r = np.uint16(cross_point(right[0], right[1], down[0], down[1]))
        except IndexError:
            done = False

    return up_l, up_r, down_l, down_r


def put_contour(img):
    for k in range(2):
        img = mp.dilation(img)
    contours = ski.measure.find_contours(img, 0.5)
    color = 20
    for contour in contours:
        for j in contour:
            img[int(round(j[0])), int(round(j[1]))] = color
        color += 10
    return img


def put_contour2(image):
    # for k in range(2):
    #     image = mp.dilation(image)
    img = image.astype(np.uint8)
    cnts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    color = 20
    for x in cnts:
        for j in x:
            img[int(round(j[0][1])), int(round(j[0][0]))] = color
        color += 10
    return img


def changeXYaxis(up_l, up_r, down_l, down_r):
    up_l = np.flip(up_l)
    down_l = np.flip(down_l)
    up_r = np.flip(up_r)
    down_r = np.flip(down_r)
    return up_l, up_r, down_l, down_r


if __name__ == '__main__':
    img = cc.load_file('photo13.jpg')
    img = cc.tresholding(img)
    img = cc.cut_min(img)
    images = ps.photo_division(img)
    no = 1

    for i in images:

        i = cc.rotate(i)
        # n = np.vstack((np.zeros((10, i.shape[1])), i, np.zeros((10, i.shape[1])))) # dodają czarne pasy na górzze i po bokach
        # i = np.hstack((np.zeros((n.shape[0], 10)), n, np.zeros((n.shape[0], 10))))
        up_l, up_r, down_l, down_r = find_intersections(i)
        i = cc.fill_board(i, up_r, 0)  # zamienia linia siatki na czarny
        print(type(i))
        i = put_contour2(i)
        up_l, up_r, down_l, down_r = changeXYaxis(up_l, up_r, down_l, down_r)
        print("\nplansza numer:", no)
        no += 1
        res = search_forXO(i, up_l, up_r, down_l, down_r)
        cc.print_result(res)
        print(who_win(res))
        ax = cc.show_img(i)
        plt.show()
