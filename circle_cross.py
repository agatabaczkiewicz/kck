import skimage as ski
from matplotlib import pyplot as plt
from skimage import io
import skimage.morphology as mp
from skimage.transform import resize
from skimage.segmentation import flood_fill
import numpy as np
import math
import cv2


def load_file(path):
    photo = io.imread(path)
    if photo.shape[0] * photo.shape[1] > 1000 * 2000:
        photo = cv2.resize(photo, (int(photo.shape[1]/4), int(photo.shape[0]/4)))
    #show_img(photo)
    return photo


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
# apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def tresholding2(img, gamm=1):
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    imge = adjust_gamma(blurred, gamma=gamm)
    ret, img = cv2.threshold(imge, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 130, 5)
    (thresh, img) = cv2.threshold(img, 188, 255, cv2.THRESH_BINARY_INV)
    img = cv2.dilate(img, kernel, iterations=1)
    return img


def tresholding(img):
    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    (thresh, img) = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY_INV)

    img = cv2.dilate(img, kernel, iterations=1)
    return img


def cut(img, color):
        while (img[0, :] == color).all():  # dla kazdej kolmny pierwszy wiersz
            img = np.delete(img, 0, 0)  # usuwa pierwszy wiersz
        while (img[-1, :] == color).all():
            img = np.delete(img, -1, 0)  # usuwa ostatni wiersz
        while (img[:, 0] == color).all():
            img = np.delete(img, 0, 1)  # usuwa pierwsza kolumne
        while (img[:, -1] == color).all():
            img = np.delete(img, -1, 1)  # usuwa ostatnia kolumne
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
    # plt.grid(True)
    plt.axis('off')
    ax.imshow(img, cmap=plt.cm.gray)
    return ax


def make_big_square(photo):
    max_size = max(photo.shape)
    new_photo = np.zeros((int(1.4 * max_size) + 1, int(1.4 * max_size) + 1))
    sp = int(0.2 * max_size)
    new_photo[sp:sp + photo.shape[0], sp:sp + photo.shape[1]] = photo
    return new_photo


def photo_division(img):
    final_list = []

    def photo_div(photo):
        for p in range(len(photo)):
            if 255 not in photo[p]:  # rzędy
                new_img1 = photo[:p, :]
                new_img2 = photo[p:, :]
                new_img1 = cut(new_img1, 0)
                new_img2 = cut(new_img2, 0)
                photo_div(new_img1)
                photo_div(new_img2)
                break
        else:
            iloczyn = 1
            for p in range(len(photo[0])):
                for q in range(len(photo)):  # po kolumnach
                    iloczyn *= (photo[q][p] == 0)
                if iloczyn != 0:
                    photo1 = photo[:, :p]
                    photo2 = photo[:, p:]
                    photo1 = cut(photo1, 0)
                    photo2 = cut(photo2, 0)
                    photo_div(photo1)
                    photo_div(photo2)
                    break
                iloczyn = 1
            else:
                final_list.append(photo)

    photo_div(img)
    print(len(final_list))
    return final_list


def rotate(img):
    corners = ski.transform.probabilistic_hough_line(img, line_length=int(2 / 3 * (img.shape[0])), line_gap=1)
    if len(corners) == 0:
        return img, False
    line = corners[0]
    for a in corners:
        if line[0][1] > a[0][1] or line[1][1] > a[1][1]:
            line = a

    angle = math.degrees(math.atan2(line[0][1] - line[1][1], line[0][0] - line[1][0]))

    if angle > 5:
        img = make_big_square(img)
        img = ski.transform.rotate(img, angle)
        img = cut_min(img)
        img = (img >= 1) * 255
        img = mp.dilation(mp.erosion(img))

    return img, True
