import json
import math
import re
import sys
import urllib
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def long_ie_label_parse_v1(label_path, output_path):
    """将label studio 长文档标注转为uie-x预处理前的数据格式"""
    img_oup_path = Path(output_path) / 'Images'
    img_oup_path.mkdir(exist_ok=True, parents=True)

    with open(label_path, 'r') as f:
        raw_result = json.load(f)

    post_processed_result = []
    for task in tqdm(raw_result):
        task_folder = task['data']['Name']
        anno_dict = {'task_name': task_folder, 'annotations': [], 'relations': []}

        # Parse Image
        page_infos = task['data']['document']
        images_num = 0
        for page in page_infos:
            image_url = page['page']
            basename = Path(image_url).name
            decode_base_name = urllib.parse.unquote(basename)
            response = requests.get(image_url)
            if response.status_code != 200:
                break
            bytes_data = response.content
            bytes_arr = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(bytes_arr, cv2.IMREAD_COLOR)
            img_oup = str(img_oup_path / decode_base_name)
            cv2.imwrite(img_oup, img)
            images_num += 1

        # Parse bboxes
        for label in task['annotations'][0]['result']:
            if label['type'] != 'labels':
                continue

            num = int(re.search(r'_\d+', label['to_name']).group(0)[1:])
            page = f"page_{num:03d}"

            # covert box
            x, y, w, h = convert_from_ls(label)
            angle = label['value']['rotation']
            box = convert_rect([x, y, w, h, angle])
            task_row = {
                'id': label['id'],
                'page_name': f'{task_folder}_{page}',
                'box': box,
                'rotation': label['value']['rotation'],
                'text': label['meta']['text'] if label.get('meta') else [],  # 写入识别结果
                'label': label['value']['labels'],  # 此处报错说明漏选标签
            }

            anno_dict['annotations'].append(task_row)

        post_processed_result.append(anno_dict)

    with open(Path(output_path) / 'processed_labels.json', 'w') as f:
        json.dump(post_processed_result, f, ensure_ascii=False, indent=2)

    print(f'images num: {images_num}')


def long_ie_label_parse_v2(label_path, output_path):
    """将label studio 长文档标注转为uie-x预处理前的数据格式"""
    img_oup_path = Path(output_path) / 'Images'
    ocr_res_oup_path = Path(output_path) / 'dataelem_ocr_res'
    img_oup_path.mkdir(exist_ok=True, parents=True)
    ocr_res_oup_path.mkdir(exist_ok=True, parents=True)

    with open(label_path, 'r') as f:
        raw_result = json.load(f)

    post_processed_result = []
    images_num = 0
    for task in tqdm(raw_result):
        # def process_task(task, post_processed_result):
        task_folder = task['data']['Name']
        page_infos = task['data']['document']
        cur_urls = [_['page'] for _ in page_infos]

        # Parse Images
        pbar = tqdm(total=len(cur_urls), desc=f'downloading {task_folder} imgs')

        def download_imgs(image_url):
            # for image_url in cur_urls:
            basename = Path(image_url).name
            decode_base_name = urllib.parse.unquote(basename)
            if not decode_base_name.startswith(task_folder):
                decode_base_name = task_folder + '_' + decode_base_name
            response = requests.get(image_url)
            if response.status_code != 200:
                print('erro')
            bytes_data = response.content
            bytes_arr = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(bytes_arr, cv2.IMREAD_COLOR)
            img_oup = str(img_oup_path / decode_base_name)
            cv2.imwrite(img_oup, img)
            pbar.update(1)

        with ThreadPoolExecutor(max_workers=10) as e:
            futures = [e.submit(download_imgs, task) for task in cur_urls]
            for future in as_completed(futures):
                images_num += 1
                future.result()
        pbar.close()

        # Parse bboxes
        anno_dict = {'task_name': task_folder, 'annotations': [], 'relations': []}
        pre_id = '-'
        for label in task['annotations'][0]['result']:
            if label['type'] in ['labels', 'textarea']:
                cur_id = label['id']
                if cur_id != pre_id:
                    anno_dict['annotations'].append({})
                pre_id = cur_id

                num = int(re.search(r'_\d+', label['to_name']).group(0)[1:])
                image_url = cur_urls[num]
                basename = Path(image_url).name
                decode_base_name = urllib.parse.unquote(basename)
                if not decode_base_name.startswith(task_folder):
                    decode_base_name = task_folder + '_' + decode_base_name
                filename_stem = Path(decode_base_name).stem

                if label['type'] == 'labels':
                    if (label['original_width'] == 1) or (
                        label['original_height'] == 1
                    ):
                        print('some original_width or original_height equal 1')
                        response = requests.get(image_url)
                        if response.status_code != 200:
                            break
                        bytes_data = response.content
                        bytes_arr = np.frombuffer(bytes_data, np.uint8)
                        img = cv2.imdecode(bytes_arr, cv2.IMREAD_COLOR)
                        label['original_width'] = img.shape[1]
                        label['original_height'] = img.shape[0]

                    x, y, w, h = convert_from_ls(label)
                    angle = label['value']['rotation']
                    box = convert_rect([x, y, w, h, angle])

                    anno_dict['annotations'][-1]['box'] = box
                    anno_dict['annotations'][-1]['label'] = label['value']['labels']
                    anno_dict['annotations'][-1]['id'] = label['id']
                    anno_dict['annotations'][-1]['page_name'] = filename_stem
                    anno_dict['annotations'][-1]['rotation'] = label['value'][
                        'rotation'
                    ]
                    anno_dict['annotations'][-1]['text'] = label['value'].get(
                        'text', []
                    )
                elif label['type'] == 'textarea':
                    anno_dict['annotations'][-1]['text'] = label['value']['text']

        post_processed_result.append(anno_dict)

    with open(Path(output_path) / 'processed_labels.json', 'w') as f:
        json.dump(post_processed_result, f, ensure_ascii=False, indent=2)

    # print(f'images num: {images_num}')


if __name__ == '__main__':
    src = '/Users/youjiachen/Downloads/zlht.json'
    long_ie_label_parse_v2(src, './zlht')
    pass
