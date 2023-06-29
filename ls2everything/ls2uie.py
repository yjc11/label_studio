import json
import math
import re
import sys
import urllib
from pathlib import Path

import cv2
import numpy as np
import requests
from tqdm import tqdm

sys.path.append('../')
from utils.ocr_func import get_ocr_results


# convert from LS percent units to pixels
def convert_from_ls(result):
    if 'original_width' not in result or 'original_height' not in result:
        return None

    value = result['value']
    w, h = result['original_width'], result['original_height']

    if all([key in value for key in ['x', 'y', 'width', 'height']]):
        return (
            w * value['x'] / 100.0,
            h * value['y'] / 100.0,
            w * value['width'] / 100.0,
            h * value['height'] / 100.0,
        )


# convert from pixels to LS percent units
def convert_to_ls(x, y, width, height, original_width, original_height):
    return (
        x / original_width * 100.0,
        y / original_height * 100.0,
        width / original_width * 100.0,
        height / original_height * 100,
    )


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


def short_ie_label_parse(label_path, output_path):
    """将label studio 短文档标注转为uie-x预处理前的数据格式"""
    img_oup_path = Path(output_path) / 'Images'
    img_oup_path.mkdir(exist_ok=True, parents=True)

    with open(label_path, 'r') as f:
        raw_result = json.load(f)

    post_processed_result = []
    images_num = 0
    for task in tqdm(raw_result):
        task_folder = Path(task['data']['Image']).parents[1].name
        anno_dict = {'task_name': task_folder, 'annotations': [], 'relations': []}

        # Parse Image
        image_url = task['data']['Image']
        basename = Path(image_url).name
        decode_basename = Path(urllib.parse.unquote(basename))
        new_stem = decode_basename.stem + '_page_000'
        decode_basename = new_stem + decode_basename.suffix
        response = requests.get(image_url)
        if response.status_code != 200:
            break
        bytes_data = response.content
        bytes_arr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(bytes_arr, cv2.IMREAD_COLOR)
        img_oup = str(img_oup_path / f'{decode_basename}')
        cv2.imwrite(img_oup, img)
        images_num += 1

        # Parse bboxes
        for label in task['annotations'][0]['result']:
            if label['type'] == 'labels':
                # covert box
                x, y, w, h = convert_from_ls(label)
                angle = label['value']['rotation']
                box = convert_rect([x, y, w, h, angle])
                task_row = {
                    'id': label['id'],
                    'page_name': f'{task_folder}_page_000',
                    'box': box,
                    'rotation': label['value']['rotation'],
                    'label': label['value']['labels'],  # 此处报错说明漏选标签
                }
            elif label['type'] == 'textarea':
                task_row['text'] = label['value']['text']
                anno_dict['annotations'].append(task_row)

        post_processed_result.append(anno_dict)

    with open(Path(output_path) / 'processed_labels.json', 'w') as f:
        json.dump(post_processed_result, f, ensure_ascii=False, indent=2)

    print(f'images num: {images_num}')


def long_ie_label_parse(label_path, output_path, ocr=False):
    """将label studio 长文档标注转为uie-x预处理前的数据格式"""
    img_oup_path = Path(output_path) / 'Images'
    ocr_res_oup_path = Path(output_path) / 'dataelem_ocr_res'
    img_oup_path.mkdir(exist_ok=True, parents=True)
    ocr_res_oup_path.mkdir(exist_ok=True, parents=True)

    with open(label_path, 'r') as f:
        raw_result = json.load(f)

    # for i in raw_result:
    #     for j in i['annotations'][0]['result']:
    #         if j['type'] == 'labels':
    #             total_anno_num += 1
    # pbar = tqdm(total=total_anno_num)

    post_processed_result = []
    images_num = 0
    for task in tqdm(raw_result):
        # def process_task(task, post_processed_result):
        task_folder = task['data']['Name']
        page_infos = task['data']['document']
        cur_urls = [_['page'] for _ in page_infos]

        # Parse Image and bboxes
        anno_dict = {'task_name': task_folder, 'annotations': [], 'relations': []}
        for label in task['annotations'][0]['result']:
            if label['type'] == 'labels':
                # get image
                num = int(re.search(r'_\d+', label['to_name']).group(0)[1:])
                page = f"page_{num:03d}"
                image_url = cur_urls[num]
                basename = Path(image_url).name
                decode_base_name = urllib.parse.unquote(basename)
                if not decode_base_name.startswith(task_folder):
                    decode_base_name = task_folder + '_' + decode_base_name
                response = requests.get(image_url)
                if response.status_code != 200:
                    break
                bytes_data = response.content
                bytes_arr = np.frombuffer(bytes_data, np.uint8)
                img = cv2.imdecode(bytes_arr, cv2.IMREAD_COLOR)
                img_oup = str(img_oup_path / decode_base_name)
                cv2.imwrite(img_oup, img)
                images_num += 1

                # get ocr result
                if ocr:
                    ocr_result = get_ocr_results(img_oup)
                    json_name = Path(decode_base_name).with_suffix('.json')
                    with open(ocr_res_oup_path / json_name, 'w') as f:
                        json.dump(ocr_result, f, ensure_ascii=False, indent=2)

                # covert box
                assert (label['original_width'] != 1) or (label['original_height'] != 1)
                x, y, w, h = convert_from_ls(label)
                angle = label['value']['rotation']
                box = convert_rect([x, y, w, h, angle])
                task_row = {
                    'id': label['id'],
                    'page_name': f'{task_folder}_{page}',
                    'box': box,
                    'rotation': label['value']['rotation'],
                    'label': label['value']['labels'],  # 此处报错说明漏选标签
                }

            elif label['type'] == 'textarea':
                task_row['text'] = label['value']['text']
                anno_dict['annotations'].append(task_row)

        post_processed_result.append(anno_dict)

    with open(Path(output_path) / 'processed_labels.json', 'w') as f:
        json.dump(post_processed_result, f, ensure_ascii=False, indent=2)

    print(f'images num: {images_num}')


if __name__ == '__main__':
    pass
