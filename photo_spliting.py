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
import circle_cross as cc


def thresh(t, photo):
    warnings.simplefilter("ignore")
    binary = (photo > t) * 255
    binary = np.uint8(binary)
    flush_figures()
    return binary


def big2small(img):
    return img / 255


def photo_division(img):
    final_list = []

    def photo_div(photo):
        for p in range(len(photo)): # ilosc rzedow
            if 255 not in photo[p]:  # rzędy
                new_img1 = photo[:p, :]
                new_img2 = photo[p:, :]
                new_img1 = cc.cut(new_img1, 0)
                new_img2 = cc.cut(new_img2, 0)
                photo_div(new_img1)
                photo_div(new_img2)
                break
        else:
            iloczyn = 1
            for p in range(len(photo[0])): # ilosc kolumnach
                for q in range(len(photo)):  # po kolumnach
                    iloczyn *= (photo[q][p] == 0)
                if iloczyn != 0: # kiedy cala kolumna bedzie czarna
                    photo1 = photo[:, :p]
                    photo2 = photo[:, p:]
                    photo1 = cc.cut(photo1, 0)
                    photo2 = cc.cut(photo2, 0)
                    photo_div(photo1)
                    photo_div(photo2)
                    break
                iloczyn = 1
            else:
                final_list.append(photo)
        #cc.show_img(photo)

    photo_div(img)
    print(len(final_list))
    for photos in final_list:
        cc.show_img(photos)
    return final_list


def to_square(photo):
    max_size = max(photo.shape)
    new_photo = 1 - np.zeros((max_size, max_size))
    new_photo[:photo.shape[0], :photo.shape[1]] = photo
    return new_photo


# ta funckcja musi być, bo przy obracaniu obcinają się boki i dzięki powiększeniu obrazu plansza zostaje w całości
def make_big_square(photo):
    max_size = max(photo.shape)
    new_photo = np.zeros((int(1.4 * max_size) + 1, int(1.4 * max_size) + 1))
    sp = int(0.2 * max_size)
    new_photo[sp:sp + photo.shape[0], sp:sp + photo.shape[1]] = photo
    return new_photo
