import torch
import cv2
from PIL import Image
import torch.utils.data
import numpy as np

def black_to_white(img):
    h=img.shape[0]
    w=img.shape[1]
    for i in range(h):
        for j in range(w):
            if img[i][j]==255:
                img[i][j]=0
            else:
                img[i][j]=255
    return img
def select_max_region(mask):
    nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    background = 0
    for row in range(stats.shape[0]):
        if stats[row, :][0] == 0 and stats[row, :][1] == 0:
            background = row
    stats_no_bg = np.delete(stats, background, axis=0)
    max_idx = stats_no_bg[:, 4].argmax()
    max_region = np.where(labels==max_idx+1, 1, 0)

    return max_region
def FillHole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out
def contour_extraction(img):
    best_thr, thr_OTSU = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_OTSU = black_to_white(thr_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    close = cv2.morphologyEx(thr_OTSU, cv2.MORPH_CLOSE, kernel)
    close = black_to_white(close)
    img2 = select_max_region(close)
    img2 = np.array(img2, dtype=np.uint8) * 255
    img3 = FillHole(img2)
    img4 = img3 - img2

    blur = cv2.GaussianBlur(img, (3, 3), 0)  # 用高斯滤波处理原图像降噪
    canny = cv2.Canny(blur, 200, 300)  # 50是最小阈值,150是最大阈值
    kernel1 = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(canny, kernel1, iterations=1)
    pointwise = img * dilate
    best_thr1, thr_OTSU1 = cv2.threshold(pointwise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    close1 = cv2.morphologyEx(thr_OTSU1, cv2.MORPH_CLOSE, kernel2)



    close1 = close1*img4*255

    contour = np.zeros([img.shape[0],img.shape[1],3])
    contour[:, :, 0] = close1
    contour[:, :, 1] = close1
    contour[:, :, 2] = close1

    return contour

class Dataset(torch.utils.data.Dataset):
    def __init__(self, imList, labelList, transform=None):
        self.imList = imList
        self.labelList = labelList
        self.transform = transform

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):
        image_name = self.imList[idx]
        label_name = self.labelList[idx]
        image = cv2.imread(image_name)
        label = cv2.imread(label_name, 0)
        img_contour = cv2.imread(image_name,0)
        contour = contour_extraction(img_contour)


        image = image[:, :, ::-1]#BGR-RGB
        #label = cv2.imread(label_name, 0)


        if self.transform:
            [image,contour,label] = self.transform(image,contour, label)
        return (image, contour, label)
