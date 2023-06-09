import copy
import json
import math
import os
import sys
import time
import urllib
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
from pprint import pprint

import cv2
import numpy as np
import requests
from tqdm import tqdm

sys.path.append('..')
from utils.file_func import check_folder
from utils.ocr_func import get_ocr_results

class_map = {
    "表格": "table",
    "列": "table column",
    "行": "table row",
    "表头": "table column header",
    "子标题": "table projected row header",
    "合并单元格": "table spanning cell",
    "no object": "no object",
}

color_range = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def draw_boxes(image, boxes, is_table_bbox=False, scores=None, drop_score=0.5):
    if scores is None:
        scores = [1] * len(boxes)
    for box, score in zip(boxes, scores):
        if score < drop_score:
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, color_range[2], 3)
        if is_table_bbox:
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


def draw_txt_bboxes(path, pure_nums=False):
    """
    标签格式为txt，使用此脚本画bbox
    :param pure_nums: 标签是否为数字，不包含类别信息
    :param path: path/xxx.jpg
    :return:
    """
    p = Path(path)
    for img_file_path in tqdm(list(p.glob('[!.]*'))):
        img = cv2.imread(str(img_file_path))
        # label_file_path = str(img_file_path.with_suffix('.txt')).replace('train', 'txts')
        try:
            label_file_path = str(img_file_path.with_suffix('.txt')).replace(
                'Images', 'txts'
            )
        except Exception as e:
            print(e)
            print(img_file_path)
        output_path = '/'.join([str(img_file_path.parent.parent), 'check_img'])
        Path(output_path).mkdir(exist_ok=True, parents=True)

        bboxes = list()
        with open(label_file_path, 'r') as f:
            data = f.read().strip().split('\n')
            if data[0]:
                for i in data:
                    if not pure_nums:
                        *p, _ = i.split(',')
                        p = list(map(float, p))
                        bboxes.append(p)
                    else:
                        p = list(map(float, i.split(',')))
                        bboxes.append(p)

                    # print(img_file_path)

        im_show = draw_boxes(img, bboxes, is_table_bbox=True)
        cv2.imwrite(f'{output_path}/{img_file_path.name}', im_show)


def l2_norm(pt0, pt1):
    return np.sqrt((pt0[0] - pt1[0]) ** 2 + (pt0[1] - pt1[1]) ** 2)


def crop_images(img, bboxes):
    n = bboxes.shape[0]
    mats = []
    for i in range(n):
        bbox = bboxes[i]
        ori_w = np.round(l2_norm(bbox[0], bbox[1]))
        ori_h = np.round(l2_norm(bbox[1], bbox[2]))
        new_w = ori_w
        new_h = ori_h
        src_3points = np.float32([bbox[0], bbox[1], bbox[2]])
        dest_3points = np.float32([[0, 0], [new_w, 0], [new_w, new_h]])
        M = cv2.getAffineTransform(src_3points, dest_3points)
        m = cv2.warpAffine(img, M, (int(new_w), int(new_h)))
        mats.append(m)

    return mats


def convert_rect(rrect):
    x, y, w, h, theta = rrect
    norm_theta = theta * math.pi / 180
    if w > 0 and h > 0:
        w_angle, h_angle = [norm_theta, norm_theta + math.pi / 2]
    elif w < 0 and h > 0:
        w, h = [h, -w]
        w_angle, h_angle = [norm_theta + math.pi / 2, norm_theta + math.pi]
    elif w < 0 and h < 0:
        [w, h] = [-w, -h]
        [w_angle, h_angle] = [norm_theta + math.pi, norm_theta + 1.5 * math.pi]
    else:
        [w, h] = [-h, w]
        [w_angle, h_angle] = [norm_theta + 1.5 * math.pi, norm_theta + 2 * math.pi]

    horiV = np.array([math.cos(w_angle), math.sin(w_angle)]) / np.linalg.norm(
        [math.cos(w_angle), math.sin(w_angle)]
    )
    vertV = np.array([math.cos(h_angle), math.sin(h_angle)]) / np.linalg.norm(
        [math.cos(h_angle), math.sin(h_angle)]
    )

    p0 = np.array([x, y])
    p1 = (p0 + w * horiV).astype(np.float32)
    p2 = (p1 + h * vertV).astype(np.float32)
    p3 = (p2 - w * horiV).astype(np.float32)

    return [p0.tolist(), p1.tolist(), p2.tolist(), p3.tolist()]


def parse(json_file, output_path):
    annos = json.load(open(json_file))
    img_oup_path = Path(output_path) / 'Images'
    label_oup_path = Path(output_path) / 'Labels'
    img_oup_path.mkdir(exist_ok=True, parents=True)
    label_oup_path.mkdir(exist_ok=True, parents=True)

    for anno in tqdm(annos):
        image_url = anno['data']['Image']
        bbox_infos = anno['annotations'][0]['result']
        base_name = Path(image_url).name
        decode_base_name = urllib.parse.unquote(base_name)

        # Parse bboxes
        bboxes = []
        for info in bbox_infos:
            if info.get('type') == 'labels':
                ori_w = info['original_width']
                ori_h = info['original_height']
                if not info['value'].get('points'):
                    x = info['value']['x']
                    y = info['value']['y']
                    w = info['value']['width']
                    h = info['value']['height']
                    theta = info['value']['rotation']
                    label = info['value']['labels'][0]
                    w_ = w / 100.0 * ori_w
                    h_ = h / 100.0 * ori_h
                    x_ = x / 100.0 * ori_w
                    y_ = y / 100.0 * ori_h
                    rect = convert_rect((x_, y_, w_, h_, theta))
                    tmp_dict = {
                        "category": label,
                        "value": "",
                        "shape": "polygon",
                        "points": rect,
                    }
                    bboxes.append(tmp_dict)
                elif info['value'].get('points'):
                    rect = info['value']['points']
                    label = info['value']['labels'][0]
                    tmp_dict = {
                        "category": label,
                        "value": "",
                        "shape": "polygon",
                        "points": rect,
                    }
                    bboxes.append(tmp_dict)

        if not bboxes:
            print(f'{decode_base_name} label is empty')
            continue
        json_oup = label_oup_path / Path(decode_base_name).with_suffix('.json')
        with open(json_oup, 'w', encoding='utf-8') as f:
            json.dump(bboxes, f, ensure_ascii=False, indent=2)

        # Parse Image
        response = requests.get(image_url)
        if response.status_code != 200:
            break
        bytes_data = response.content
        bytes_arr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(bytes_arr, cv2.IMREAD_COLOR)
        img_oup = str(img_oup_path / decode_base_name)
        cv2.imwrite(img_oup, img)


def process_img_degree(image_file, save_data_folder=''):
    """
    rotate image to 0 degree by its ocr results
    """
    img = cv2.imread(str(image_file))
    image_name = Path(image_file).name
    try:
        ocr_results = get_ocr_results(image_file)
    except:
        print(image_name, 'rotate erro')
        cv2.imwrite(os.path.join(save_data_folder, image_name), img)
    bboxes = ocr_results['bboxes']
    bbox_angles = []
    for index, bbox in enumerate(ocr_results['bboxes']):
        bbox = np.array(bbox)
        angle = compute_angle(bbox)
        bbox_angles.append(angle)

    bbox_angles = sorted(bbox_angles)
    bbox_angles = bbox_angles[
        int(0.25 * len(bbox_angles)) : int(0.75 * len(bbox_angles))
    ]
    mean_angle = np.mean(np.array(bbox_angles))
    # print('mean_angle:', mean_angle)

    img_rotate, old_center, new_center = rotate_image_only(img, mean_angle)
    # print(os.path.join(save_data_folder, image_name))
    cv2.imwrite(os.path.join(save_data_folder, image_name), img_rotate)

    return img_rotate, old_center, new_center, mean_angle


def compute_angle(bbox):
    angle_vector = bbox[1] - bbox[0]
    cos_angle = angle_vector[0] / np.linalg.norm(angle_vector)
    sin_angle = angle_vector[1] / np.linalg.norm(angle_vector)

    cos_angle = math.acos(cos_angle) * 180 / np.pi
    sin_angle = math.asin(sin_angle) * 180 / np.pi
    if cos_angle <= 90 and sin_angle <= 0:
        angle = 360 + sin_angle
    elif cos_angle <= 90 and sin_angle > 0:
        angle = sin_angle
    elif cos_angle > 90 and sin_angle > 0:
        angle = cos_angle
    elif cos_angle > 90 and sin_angle <= 0:
        angle = 360 - cos_angle

    # 防止有的文本角度是1-2度，有的文本角度是365度，其实方向上相差很小，但是一平均后出问题
    if angle >= 350 and angle <= 360:
        angle = angle - 360
    return angle


def rotate_image_only(im, angle):
    def rotate(src, angle, scale=1.0):  # 1
        w = src.shape[1]
        h = src.shape[0]
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]

        try:
            rotated_image = cv2.warpAffine(
                src,
                rot_mat,
                (int(math.ceil(nw)), int(math.ceil(nh))),
                flags=cv2.INTER_LANCZOS4,
            )
        except:
            return src
        return rotated_image

    old_h, old_w, _ = im.shape
    old_center = (old_w / 2, old_h / 2)

    image = rotate(im, angle)
    new_h, new_w, _ = image.shape
    new_center = (new_w / 2, new_h / 2)
    # print(old_center, '-->', new_center)

    return image, old_center, new_center


def rotate_polys_only(old_center, new_center, poly, angle):
    """
    poly:(4,2)
    """
    angle = angle * np.pi * 1.0 / 180  # 弧度
    poly = copy.deepcopy(poly)
    # print(poly.shape)

    poly[:, 0] = poly[:, 0] - old_center[0]
    poly[:, 1] = old_center[1] - poly[:, 1]
    x1 = poly[0, 0] * math.cos(angle) - poly[0, 1] * math.sin(angle) + new_center[0]
    y1 = new_center[1] - (poly[0, 0] * math.sin(angle) + poly[0, 1] * math.cos(angle))
    x2 = poly[1, 0] * math.cos(angle) - poly[1, 1] * math.sin(angle) + new_center[0]
    y2 = new_center[1] - (poly[1, 0] * math.sin(angle) + poly[1, 1] * math.cos(angle))
    x3 = poly[2, 0] * math.cos(angle) - poly[2, 1] * math.sin(angle) + new_center[0]
    y3 = new_center[1] - (poly[2, 0] * math.sin(angle) + poly[2, 1] * math.cos(angle))
    x4 = poly[3, 0] * math.cos(angle) - poly[3, 1] * math.sin(angle) + new_center[0]
    y4 = new_center[1] - (poly[3, 0] * math.sin(angle) + poly[3, 1] * math.cos(angle))

    return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]


def process_example_degree(img_file, label_folder, dst):
    img_out_folder = Path(dst) / '水平' / f'Images'
    label_out_folder = Path(dst) / '水平' / f'Labels'
    img_out_folder.mkdir(exist_ok=True, parents=True)
    label_out_folder.mkdir(exist_ok=True, parents=True)

    label_basename = img_file.stem + '.json'
    label_file = Path(label_folder) / label_basename
    _, old_center, new_center, angle = process_img_degree(str(img_file), img_out_folder)

    with open(label_file, 'r', encoding='utf-8') as f:
        img_label = json.load(f)

    rotated_img_label = []
    for elem in img_label:
        points = np.array(elem['points'])
        elem['points'] = rotate_polys_only(old_center, new_center, points, angle)
        rotated_img_label.append(elem)

    with open(label_out_folder / label_basename, 'w') as f:
        json.dump(rotated_img_label, f, ensure_ascii=False, indent=2)


def multithreadpost(img_folder, label_folder, dst, max_workers=10):
    """
    process by multithread
    """
    from concurrent.futures import ThreadPoolExecutor

    image_files = list(Path(img_folder).glob('[!.]*'))
    all_start_time = time.time()
    process_degree = partial(process_example_degree, label_folder=label_folder, dst=dst)
    with ThreadPoolExecutor(max_workers) as executor:
        for _ in tqdm(
            executor.map(process_degree, image_files), total=len(image_files)
        ):
            pass
    all_end_time = time.time()
    print('finish_time: {}'.format(all_end_time - all_start_time))


if __name__ == "__main__":
    """preprocess : rotate img"""
    # process_img_degree()

    """parse to images and labels"""
    json_file = '../data/changwailiushui_table_p2.json'
    out_folder = '../data/test_rotate'
    parse(json_file, out_folder)

    """process example degree"""
    img_path = '../data/test_rotate/Images/'
    label_path = '../data/test_rotate/Labels/'
    dst = '../data/test_rotate'
    multithreadpost(img_path, label_path, dst)

    """draw boxes"""
    from utils.draw_table_bboxes import draw_all_bboxes_row_col

    src = '/Users/youjiachen/Desktop/projects/label_studio_mgr/data/test_rotate/水平/'
    # draw_all_bboxes_row_col(src)

    encode_name = urllib.parse.quote('北京银行个人')
