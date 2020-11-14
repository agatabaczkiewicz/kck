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


def perp(a):
    b = empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return
#liczy punkt przzecięcia dwóch prostych
def seg_intersect(a1, a2, b1, b2):
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = dot(dap, db)
    num = dot(dap, dp)
    return (num / denom.astype(float))*db + b1

def wynik(result,mark):
    for i in range(3):
        if result[0][1] == result[1][i] == result[2][i]==mark: # sprawdza w pionie
            return True
        elif result[i][0] == result[i][1] == result[i][2]==mark: # w poziomie
            return True
    if result[0][0] == result[1][1] == result[2][2] == mark:  # sprawdza ukos 1
        return True
    elif result[2][0] == result[1][1] == result[0][2] == mark:  # ukos 2
        return True

    return False


def find_result2(img, top_left, top_right, bottom_left, bottom_right):
    result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    print(img[0:min(top_left[0], top_right[0]), top_left[1]:top_right[1]])

    if len(np.unique(img[0:top_left[0], 0:top_left[1]])) == 3:
        result[0][0] = 'X'
    elif len(np.unique(img[0:top_left[0], 0:top_left[1]])) >= 4:
        result[0][0] = 'O'
    else:
        result[0][0] = ' '
    if len(np.unique(img[0:min(top_left[0], top_right[0]), top_left[1]:top_right[1]])) == 3:
        result[0][1] = 'X'
    elif len(np.unique(img[0:min(top_left[0], top_right[0]), top_left[1]:top_right[1]])) >= 4:
        result[0][1] = 'O'
    else:
        result[0][1] = ' '
    if len(np.unique(img[:top_right[0], top_right[1]:])) == 3:
        result[0][2] = 'X'
    elif len(np.unique(img[:top_right[0], top_right[1]:])) >= 4:
        result[0][2] = 'O'
    else:
        result[0][2] = ' '
    if len(np.unique(img[top_left[0]:bottom_left[0], 0:top_left[1]])) == 3:
        result[1][0] = 'X'
    elif len(np.unique(img[top_left[0]:bottom_left[0], 0:top_left[1]])) >= 4:
        result[1][0] = 'O'
    else:
        result[1][0] = ' '
    if ((len(np.unique(img[max(top_left[0], top_right[0]):min(bottom_left[0], bottom_right[0]),
         max(top_left[1], bottom_left[1]):min(top_right[1], bottom_right[1])]))) == 3):
        result[1][1] = 'X'
    elif((len(np.unique(img[max(top_left[0], top_right[0]):min(bottom_left[0], bottom_right[0]),
         max(top_left[1], bottom_left[1]):min(top_right[1], bottom_right[1])]))) >= 4):
        result[1][1] = 'O'
    else:
        result[1][1] = ' '
    if len(np.unique(img[top_right[0]:bottom_right[0], max(bottom_right[1], top_right[1]):])) == 3:
        result[1][2] = 'X'
    elif len(np.unique(img[top_right[0]:bottom_right[0], max(bottom_right[1], top_right[1]):])) >= 4:
        result[1][2] = 'O'
    else:
        result[1][2] = ' '
    if len(np.unique(img[bottom_left[0]:, 0:bottom_left[1]])) == 3:
        result[2][0] = 'X'
    elif len(np.unique(img[bottom_left[0]:, 0:bottom_left[1]])) >= 4:
        result[2][0] = 'O'
    else:
        result[2][0] = ' '
    if len(np.unique(img[max(bottom_left[0], bottom_right[0]):, bottom_left[1]:bottom_right[1]])) == 3:
        result[2][1] = 'X'
    elif len(np.unique(img[max(bottom_left[0], bottom_right[0]):, bottom_left[1]:bottom_right[1]])) >= 4:
        result[2][1] = 'O'
    else:
        result[2][1] = ' '
    if len(np.unique(img[bottom_right[0]:, bottom_right[1]:])) == 3:
        result[2][2] = 'X'
    elif len(np.unique(img[bottom_right[0]:, bottom_right[1]:])) >= 4:
        result[2][2] = 'O'
    else:
        result[2][2] = ' '
    return result


def find_intersections(i):
    top_right = []
    top_left = []
    bottom_right = []
    bottom_left = []
    top=()
    left=()
    right=()
    bottom=()
    done=False
    while not done:
        try:
            corners = ski.transform.probabilistic_hough_line(i, line_length=int(2 / 3 * min(i.shape[0], i.shape[1])),
                                                             line_gap=1)
            for line in corners:
                if abs(line[0][0]-line[1][0])<i.shape[1]//3 and line[0][0]<i.shape[1]//2 and line[1][0]<i.shape[1]//2:
                    left=np.float32(np.array(line))

                    break
            for line in corners:
                if abs(line[0][0]-line[1][0])<i.shape[1]//3 and line[0][0]>i.shape[1]//2 and line[1][0]>i.shape[1]//2:
                    right=np.float32(np.array(line))

                    break

            for line in corners:
                if abs(line[0][1]-line[1][1])<i.shape[0]//3 and line[0][1]<i.shape[0]//2 and line[1][1]<i.shape[0]//2:
                    top=np.float32(np.array(line))

                    break
            for line in corners:
                if abs(line[0][1]-line[1][1])<i.shape[0]//3 and line[0][1]>i.shape[0]//2 and line[1][1]>i.shape[0]//2:
                    bottom=np.float32(np.array(line))

                    break
            done=True

            top_left=np.uint16(seg_intersect(left[0],left[1],top[0],top[1]))
            bottom_left=np.uint16(seg_intersect(left[0],left[1],bottom[0],bottom[1]))
            top_right=np.uint16(seg_intersect(right[0],right[1],top[0],top[1]))
            bottom_right=np.uint16(seg_intersect(right[0],right[1],bottom[0],bottom[1]))
        except IndexError:
            done = False
    return top_left, top_right, bottom_left, bottom_right



if __name__ == '__main__':
    img = cc.load_file('photo25.jpg')
    img = cc.black_white(img)
    img = cc.cut_min(img)
    images = ps.photo_division(img)
    no = 1
    for i in images:
        i = cc.rotate(i)
        n=np.vstack((np.zeros((10,i.shape[1])),i,np.zeros((10,i.shape[1]))))
        i=np.hstack((np.zeros((n.shape[0],10)),n,np.zeros((n.shape[0],10))))
        top_left, top_right, bottom_left, bottom_right = find_intersections(i)
        i = cc.fill_board(i, top_right, 0) # zamienia linia siatki na czarny
        for j in range(2):
            i = mp.dilation(i)
        contours = ski.measure.find_contours(i, 0.5)
        print("halko")
        color = 20
        for contour in contours:
            for j in contour:
                i[int(round(j[0])), int(round(j[1]))] = color
            color += 10

        top_left = np.flip(top_left)
        bottom_left = np.flip(bottom_left)
        top_right = np.flip(top_right)
        bottom_right = np.flip(bottom_right)
        print("\nplansza numer:", no)
        no += 1
        res=find_result2(i, top_left, top_right, bottom_left, bottom_right)
        cc.print_result(res)
        if wynik(res,'O'):
            print("Wygrały O")
        elif wynik(res,'X'):
            print("Wygrały X")
        else:
            print("Nikt nie wygrał")
        ax = cc.show_img(i)
        plt.show()





