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


def find_board(img, square_range): #nie używa
    spots = []
    while len(spots)<4:
        for i in range(0, img.shape[0] - square_range):
            for j in range(0, img.shape[1] - square_range):
                square = np.array(img[i:i + square_range, j:j + square_range])
                if (square[0, 0] == 0
                        and square[0, -1] == 0
                        and square[-1, 0] == 0
                        and square[-1, -1] == 0
                        and not all(square[0, :] == 0)
                        and not all(square[-1, :] == 0)
                        and not all(square[:, 0] == 0)
                        and not all(square[:, -1] == 0)
                        and square[square_range // 2, square_range // 2] == 255):
                    spots.append([i + square_range // 2, j + square_range // 2])
        square_range+=1
    # for i in spots:
    #     img[i[0], i[1]] = 125
    top_right = []
    top_left = []
    bottom_right = []
    bottom_left = []

    for i in spots:
        if i[0] < img.shape[0] // 2 and i[1] < img.shape[1] // 2:
            top_left = i
            break
    for i in spots:
        if i[0] > img.shape[0] // 2 and i[1] < img.shape[1] // 2:
            bottom_left = i
            break
    for i in spots:
        if i[0] < img.shape[0] // 2 and i[1] > img.shape[1] // 2:
            top_right = i
            break
    for i in spots:
        if i[0] > img.shape[0] // 2 and i[1] > img.shape[1] // 2:
            bottom_right = i
            break
    return top_left, top_right, bottom_left, bottom_right


def fill_board(img, top_right,color=125): #linie planszy będą czarne
    img = ski.segmentation.flood_fill(img, (top_right[1], top_right[0]), color)
    return img


def find_crosses(img, square_range): #nie używa
    spot2 = []
    while len(spot2)==0:
        for i in range(0, img.shape[0] - square_range):
            for j in range(0, img.shape[1] - square_range):
                square = np.array(img[i:i + square_range, j:j + square_range])
                if (square[0, 0] == 255
                        and square[0, -1] == 255
                        and square[-1, 0] == 255
                        and square[-1, -1] == 255
                        and square[square_range-1, square_range // 2] == 0
                        and square[square_range // 2, square_range-1] == 0
                        and square[square_range // 2, 0] == 0
                        and square[0, square_range // 2] == 0
                        and square[square_range // 2, square_range // 2] == 0):
                    spot2.append([i + square_range // 2, j + square_range // 2])
        square_range+=1

    for i in spot2:
        img[i[0], i[1]] = 50
    return img


def find_circle(img, top_left, top_right, bottom_left, bottom_right): #nie używa
    size_x = int(((top_right[1] - top_left[1]) + (bottom_right[1] - bottom_left[1])) / 2)
    size_y = int(((bottom_left[0] - top_left[0]) + (bottom_right[0] - top_right[0])) / 2)
    percent_x = int(0.1 * size_x)
    percent_y = int(0.1 * size_y)
    new_img = img[top_left[0] + percent_y:bottom_left[0] - percent_y, top_left[1] + percent_x:top_right[1] - percent_x]
    new_img = new_img / 255
    sum_white = 0
    for row in new_img:
        sum_white += sum(row)
    img_size = new_img.shape[0] * new_img.shape[1]
    return sum_white / img_size


def mark_points(img,top_left, top_right, bottom_left, bottom_right): #nie używa
    img[top_left[0], top_left[1]] = 175
    img[top_right[0], top_right[1]] = 175
    img[bottom_right[0], bottom_right[1]] = 175
    img[bottom_left[0], bottom_left[1]] = 175
    return img


def find_result(img, top_left, top_right, bottom_left, bottom_right, circle_percent): #nie używa
    img[top_left[1],top_left[0]]=75
    result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    x1 = int((top_left[1]+bottom_left[1])/2)
    x2 = int((top_right[1]+bottom_right[1])/2)
    y1 = int((top_right[0]+top_left[0])/2)
    y2 = int((bottom_right[0]+bottom_left[0])/2)

    if (img[0:top_left[0], 0:top_left[1]] == 50).any():
        result[0][0] = 'X'
    elif find_circle(img, [0,0], [0,x1], [y1,0], top_left) > circle_percent:
        result[0][0] = 'O'
    else:
        result[0][0] = ' '
    if (img[0:min(top_left[0], top_right[0]), top_left[1]:top_right[1]] == 50).any():
        result[0][1] = 'X'
    elif find_circle(img, [0,x1], [0, x2], top_left, top_right) > circle_percent:
        result[0][1] = 'O'
    else:
        result[0][1] = ' '
    if (img[:top_right[0], top_right[1]:] == 50).any():
        result[0][2] = 'X'
    elif find_circle(img, [0, x2], [0, img.shape[1]], top_right, [y1, img.shape[1]]) > circle_percent:
        result[0][2] = 'O'
    else:
        result[0][2] = ' '
    if (img[top_left[0]:bottom_left[0], 0:top_left[1]] == 50).any():
        result[1][0] = 'X'
    elif find_circle(img, [y1,0], top_left, [y2,0], bottom_left) > circle_percent:
        result[1][0] = 'O'
    else:
        result[1][0] = ' '
    if ((img[max(top_left[0], top_right[0]):min(bottom_left[0], bottom_right[0]),
         max(top_left[1], bottom_left[1]):min(top_right[1], bottom_right[1])] == 50).any()):
        result[1][1] = 'X'
    elif find_circle(img, top_left, top_right, bottom_left, bottom_right) > circle_percent:
        result[1][1] = 'O'
    else:
        result[1][1] = ' '
    if (img[top_right[0]:bottom_right[0], max(bottom_right[1], top_right[1]):] == 50).any():
        result[1][2] = 'X'
    elif find_circle(img, top_right, [y1, img.shape[1]], bottom_right, [y2, img.shape[1]]) > circle_percent:
        result[1][2] = 'O'
    else:
        result[1][2] = ' '
    if (img[bottom_left[0]:, 0:bottom_left[1]] == 50).any():
        result[2][0] = 'X'
    elif find_circle(img, [y2,0], bottom_left, [img.shape[0],0], [img.shape[0],x1]) > circle_percent:
        result[2][0] = 'O'
    else:
        result[2][0] = ' '
    if (img[max(bottom_left[0], bottom_right[0]):, bottom_left[1]:bottom_right[1]] == 50).any():
        result[2][1] = 'X'
    elif find_circle(img, bottom_left, bottom_right, [img.shape[0], x1], [img.shape[0], x2]) > circle_percent:
        result[2][1] = 'O'
    else:
        result[2][1] = ' '
    if (img[bottom_right[0]:, bottom_right[1]:] == 50).any():
        result[2][2] = 'X'
    elif find_circle(img, bottom_right, [y2, img.shape[1]], [img.shape[0], x2], [img.shape[0], img.shape[1]]) > circle_percent:
        result[2][2] = 'O'
    else:
        result[2][2] = ' '
    return result


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


# top_right = []
# top_left = []
# bottom_right = []
# bottom_left = []
# images = []
# result = []
# img = load_file('images/photo01.jpg')
# img = black_white(img)
# img = cut_min(img)
# images = ps.photo_division(img)
# no = 1
# for i in images:
#     i = rotate(i)
#     contours = ski.measure.find_contours(i, 1)
#     square_line=min(i.shape[0],i.shape[1])//20
#     top_left, top_right, bottom_left, bottom_right = find_board(i, square_line)
#     i = find_crosses(i, 10)
#     # show_img(find_circle(i, top_left, top_right, bottom_left, bottom_right))
#
#     #i = mark_points(i, top_left, top_right, bottom_left, bottom_right)
#     for n, contour in enumerate(contours):
#         plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
#     i = fill_board(i, top_right)
#     i=mp.dilation(i)
#     show_img(i)
#     # print(top_left, top_right, bottom_left, bottom_right)
#     print("\nplansza numer:", no)
#     result = find_result(i, top_left, top_right, bottom_left, bottom_right, 0.1)
#     print_result(result)
#     no += 1
