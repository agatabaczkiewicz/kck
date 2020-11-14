from __future__ import division
import skimage
from skimage import io, filters
import skimage.morphology as mp
from skimage.color import rgb2gray
from matplotlib import pylab as plt, gridspec
import numpy as np
from ipykernel.pylab.backend_inline import flush_figures
import warnings
import math


def thresh(t, photo):
    warnings.simplefilter("ignore")
    binary = (photo > t) * 255
    binary = np.uint8(binary)
    flush_figures()
    return binary


def cut(img, tag, color):
    if tag == 'allp':
        while all(img[0, :] == color):
            img = np.delete(img, 0, 0)
        while all(img[:, 0] == color):
            img = np.delete(img, 0, 1)
        while all(img[-1, :] == color):
            img = np.delete(img, -1, 0)
        while all(img[:, -1] == color):
            img = np.delete(img, -1, 1)
    elif tag == 'anyp':
        while any(img[0, :] == color):
            img = np.delete(img, 0, 0)
        while any(img[:, 0] == color):
            img = np.delete(img, 0, 1)
        while any(img[-1, :] == color):
            img = np.delete(img, -1, 0)
        while any(img[:, -1] == color):
            img = np.delete(img, -1, 1)
    return img


def big2small(img):
    return img / 255


def find_white_line(photo):
    photo = big2small(photo)
    max_value = 0
    for row in photo:
        if sum(row) > max_value:
            max_value = sum(row)
    return max_value / len(photo[0])


def photo_division(img):
    final_list = []

    def photo_div(photo):
        for p in range(len(photo)):
            if 255 not in photo[p]:  #rzędy
                new_img1 = photo[:p, :]
                new_img2 = photo[p:, :]
                new_img1 = cut(new_img1, 'allp', 0)
                new_img2 = cut(new_img2, 'allp', 0)
                photo_div(new_img1)
                photo_div(new_img2)
                break
        else:
            iloczyn = 1
            for p in range(len(photo[0])):
                for q in range(len(photo)):  #po kolumnach
                    iloczyn *= (photo[q][p] == 0)
                if iloczyn != 0:
                    photo1 = photo[:, :p]
                    photo2 = photo[:, p:]
                    photo1 = cut(photo1, 'allp', 0)
                    photo2 = cut(photo2, 'allp', 0)
                    photo_div(photo1)
                    photo_div(photo2)
                    break
                iloczyn = 1
            else:
                final_list.append(photo)

    photo_div(img)
    print(len(final_list))
    return final_list


def to_square(photo):
    max_size = max(photo.shape)
    new_photo = 1 - np.zeros((max_size, max_size))
    new_photo[:photo.shape[0], :photo.shape[1]] = photo
    return new_photo


def find_line(photo_square, angle):
    size = photo_square.shape[0]
    tangens = round(math.tan(math.radians(angle)), 2)
    max_val = 0
    for k in range(size):
        suma = 0
        for y in range(size):
            for x in range(size):
                if x != k:
                    if tangens == round((y - k) / (x - k), 2) and photo_square[y][x] == 0:
                        suma += 1
        if suma > max_val:
            max_val = suma
    return max_val


#używa z cc
def rotate(photo):
    for i in range(90):
        photo = skimage.transform.rotate(photo, 1)
        photo = (photo != 0) * 1
        if find_white_line(photo) > 0.65:
            break
    return photo


# ta funckcja musi być, bo przy obracaniu obcinają się boki i dzięki powiększeniu obrazu plansza zostaje w całości
def make_big_square(photo):
    max_size = max(photo.shape)
    new_photo = np.zeros((int(1.4 * max_size) + 1, int(1.4 * max_size) + 1))
    sp = int(0.2 * max_size)
    new_photo[sp:sp + photo.shape[0], sp:sp + photo.shape[1]] = photo
    return new_photo

# img=cc.load_file('images/photo01.jpg')
# img=cc.cut_min(img)
# img=cc.black_white(img)
# cc.show_img(img)
