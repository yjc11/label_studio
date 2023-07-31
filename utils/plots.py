import copy
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def draw_boxes(
    image: np.ndarray,
    boxes,
    arrowedLine=False,
    scores=None,
    drop_score=0.5,
) -> np.ndarray:
    """
    Args:
        image: np.ndarray
        boxes: ndarray[n, 4, 2]
    """
    if scores is None:
        scores = [1] * len(boxes)
    for box, score in zip(boxes, scores):
        if score < drop_score:
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, COLORS[2], 3)
        if arrowedLine:
            image = cv2.arrowedLine(
                image,
                (int(box[0][0][0]), int(box[0][0][1])),
                (int(box[1][0][0]), int(box[1][0][1])),
                color=(0, 255, 0),
                thickness=3,
                line_type=cv2.LINE_4,
                shift=0,
                tipLength=0.1,
            )
    return image


def rotate_box(box, image_size, angle):
    assert box.shape == (4, 2)
    w, h = image_size
    box_copy = copy.deepcopy(box)
    if angle == 0:
        return box
    if angle == -90:
        box[:, 0] = w - 1 - box_copy[:, 1]
        box[:, 1] = box_copy[:, 0]
        return box
    if angle == 90:
        box[:, 0] = box_copy[:, 1]
        box[:, 1] = h - 1 - box_copy[:, 0]
        return box
    if angle == 180:
        box[:, 0] = w - 1 - box_copy[:, 0]
        box[:, 1] = h - 1 - box_copy[:, 1]
        return box


if __name__ == '__main__':
    # img_path = '../output/卡证表单23-7_道路运输证_convert_result/images/ea2d685d-231a-4c5f-ab32-598ac42a2246.jpg'
    # image_name = Path(img_path).name
    # label_path = '../output/卡证表单23-7_道路运输证_convert_result/Labels/ea2d685d-231a-4c5f-ab32-598ac42a2246.json'
    # image = cv2.imread(img_path)

    # with open(label_path, 'r') as f:
    #     label_data: list = json.load(f)

    # image1 = image
    # for label in label_data:
    #     box = label['points']
    #     image1 = draw_boxes(image1, [box])

    # label_studio_json = '../output/卡证表单23-7_道路运输证_convert_result/label_val_studio.json'
    # with open(label_studio_json, 'r') as f:
    #     annos = json.load(f)

    # for anno in annos:
    #     anno_image = anno['data']['image'].split('-')[1]
    #     if anno['data']['image'].split('prefix-')[1] == image_name:
    #         for result in anno['annotations'][0]['result']:
    #             box = result['origin_bbox']
    #             image = draw_boxes(image, [box])

    # cv2.imshow('11.png', image1)
    # cv2.waitKey(0)
    # cv2.imshow('12.png', image)
    # cv2.waitKey(0)
    img_path = '/Users/youjiachen/Desktop/projects/label_studio_mgr/UIE-X/entity/test/123 (1).jpg'
    img = cv2.imread(img_path)
    h, w = img.shape[:-1]
    print(w)

    label_path = '/Users/youjiachen/Desktop/projects/label_studio_mgr/UIE-X/entity/test/123 (1).json'
    with open(label_path, 'r') as f:
        datas = json.load(f)

    data = datas[1]
    print(h, w)
    box = np.array(data['points'])
    print(box)
    box = rotate_box(box, (w, h), 90)
    print(box)
    img = draw_boxes(img, [box])
    cv2.imshow('im_show.png', img)
    cv2.waitKey(0)
