import cv2
import shutil

import numpy as np

from tqdm import tqdm
from pathlib import Path


def rotate_and_crop_image_v1(image_path, box):
    """
    Args:
        image_path :  图像文件的路径。
        box : 格式为[(100, 100), (200, 100), (200, 200), (100, 200)]

    Returns:
        旋转后的图像，裁剪后的图像。

    """
    # 加载图像
    img = cv2.imread(image_path)

    # 定义旋转矩形的四个顶点坐标（x, y）
    # tl, tr, br, bl = rect = [(100, 100), (200, 100), (200, 200), (100, 200)]
    tl, tr, br, bl = rect = box

    # 计算旋转矩形的中心点坐标和旋转角度
    dx, dy = tr - tl
    angle = np.degrees(-np.arctan2(dy, dx))
    center = np.mean(rect, axis=0)

    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1)

    # 应用旋转矩阵
    rotated_img = cv2.warpAffine(img, M, img.shape[:2])

    # 计算旋转后的矩形的顶点坐标
    new_rect = np.c_[rect, [1, 1, 1, 1]]
    new_rect = np.dot(M, new_rect).T[:, :2]

    # 计算旋转后的矩形的边界框
    x_min, y_min = np.round(np.min(new_rect, axis=0)).astype(int)
    x_max, y_max = np.round(np.max(new_rect, axis=0)).astype(int)

    # 剪切出对应位置的图像
    cropped_img = rotated_img[y_min:y_max, x_min:x_max]

    return rotated_img, cropped_img


def rotate_and_crop_image_v2(image_path, box):
    """
    args:
        image_path: 图像文件的路径。
        box: 旋转前的矩形框，格式为 ((xmin, ymin), (xmax, ymax))。

    return:
        旋转后的图像，裁剪后的图像。

    """
    # 加载图像
    img = cv2.imread(image_path)

    # 定义旋转矩形的四个顶点坐标
    xmin, ymin, xmax, ymax = box
    # tl, tr, br, bl = rect = np.array(
    #     [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    # )

    # # 计算旋转矩形的中心点坐标和旋转角度
    # dx, dy = tr - tl
    # angle = np.degrees(-np.arctan2(dy, dx))
    # center = np.mean(rect, axis=0)

    # # 计算旋转矩阵并应用旋转
    # M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotated_img = cv2.warpAffine(img, M, img.shape[:2])

    # 计算旋转后的矩形的顶点坐标并计算边界框
    # new_rect = np.c_[rect, [1, 1, 1, 1]]
    # new_rect = np.dot(M, new_rect).T[:, :2]
    # x_min, y_min = np.round(np.min(new_rect, axis=0)).astype(int)
    # x_max, y_max = np.round(np.max(new_rect, axis=0)).astype(int)

    # 裁剪出对应位置的图像并返回结果
    
    cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]

    return cropped_img


def xyxy2xywh(box):
    xmin, ymin, xmax, ymax = box
    xcenter = (xmin + xmax) / 2
    ycenter = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    return xcenter, ycenter, width, height


def xywh2xyxy(box):
    center_x, center_y, w, h = box
    xmin = center_x - (w / 2)
    ymin = center_y - (h / 2)
    xmax = center_x + (w / 2)
    ymax = center_y + (h / 2)
    return xmin, ymin, xmax, ymax


if __name__ == "__main__":
    pass
