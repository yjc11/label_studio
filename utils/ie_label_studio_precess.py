import base64
import json
import math
import os
import re
import shutil
import urllib
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import cv2
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from tqdm import tqdm

host = "192.168.106.7"
port = "8502"
http_url = f'{host}:{port}'


def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)


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


def process_label_studio(label_path, img_path, ocr_res_path, output_path):
    img_output = Path(output_path) / 'Images'
    ocr_output = Path(output_path) / 'dataelem_ocr_res'
    img_output.mkdir(exist_ok=True, parents=True)
    ocr_output.mkdir(exist_ok=True, parents=True)

    with open(label_path, 'r') as f:
        raw_result = json.load(f)

    post_processed_result = []
    for task in tqdm(raw_result):
        task_folder = task['data']['Name']
        anno_dict = {'task_name': task_folder, 'annotations': [], 'relations': []}

        # copy image and ocr result
        img_paths = Path(img_path).glob(f'{task_folder}*')
        ocr_paths = Path(ocr_res_path).glob(f'{task_folder}*')
        for image_file in img_paths:
            shutil.copy(image_file, img_output)
        for ocr_res_file in ocr_paths:
            shutil.copy(ocr_res_file, ocr_output)

        for label in task['annotations'][0]['result']:
            # relation
            if 'from_id' in label and 'to_id' in label:
                anno_dict['relations'].append(
                    {'from_id': label['from_id'], 'to_id': label['to_id']}
                )
            if label['type'] != 'labels':
                continue
            num = int(re.search(r'_\d+', label['to_name']).group(0)[1:])
            page = f"page_{num:03d}"

            # refine ori_width and ori_height
            # todo：不清楚是否pdf内所有图片都是一样的尺寸，如果是，需要改进此处
            image_path = f'{img_path}/{task_folder}_{page}.png'
            img = cv2.imread(image_path)
            height, width, _ = img.shape
            label['original_width'], label['original_height'] = width, height

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

    # 保存ori weight和ori height修正后的label studio结果
    with open(Path(output_path) / 'contract_labels_studio.json', 'w') as f:
        json.dump(raw_result, f, ensure_ascii=False, indent=2)


def process_ocr_studio(label_path, img_path, ocr_res_path, output_path):
    img_output = Path(output_path) / 'Images'
    ocr_output = Path(output_path) / 'dataelem_ocr_res'
    img_output.mkdir(exist_ok=True, parents=True)
    ocr_output.mkdir(exist_ok=True, parents=True)

    label_files = list(Path(label_path).glob('[!.]*'))

    post_processed_result = []
    invalid_data_num = 0
    for task in tqdm(label_files, desc='converting data'):
        task_folder = task.name.split('_page_')[0]
        anno_dict = {'task_name': task_folder, 'annotations': [], 'relations': []}

        with task.open('r') as f:
            task_data = json.load(f)

        # check if the task contains invalid data
        tag_set = [_['category'] for _ in task_data]
        if '无效数据' in tag_set:
            invalid_data_num += 1
            continue

        # copy image and ocr result
        img_paths = Path(img_path).glob(f'{task.stem}*')
        ocr_paths = Path(ocr_res_path).glob(f'{task.stem}*')
        for image_file in img_paths:
            shutil.copy(image_file, img_output)
        for ocr_res_file in ocr_paths:
            shutil.copy(ocr_res_file, ocr_output)

        for label in task_data:
            task_row = {
                'id': '-',
                'page_name': f'{task.stem}',
                'box': label['points'],
                'rotation': '-',
                'text': [label['value']],  # 写入识别结果
                'label': [label['category']],
            }

            anno_dict['annotations'].append(task_row)
        post_processed_result.append(anno_dict)

    print(f'ori data num: {len(label_files)}')
    print(f'invalid data num: {invalid_data_num}')
    print(f'valid data num: {len(post_processed_result)}')

    with open(Path(output_path) / 'processed_labels.json', 'w') as f:
        json.dump(post_processed_result, f, ensure_ascii=False, indent=2)


def split_ocr_res_trianval(output_path, precessed_label_path, ocr_res_path, seed=144):
    """根据train val划分ocr结果"""
    with open(precessed_label_path, 'r') as f:
        raw_result = json.load(f)

    # Get pdf name from label file
    pdfname = sorted([i['task_name'] for i in raw_result])
    ocr_res_files = list(Path(ocr_res_path).glob('*.json'))

    # Split train and val data
    train_pdf, val_pdf = train_test_split(
        pdfname, train_size=0.8, test_size=0.2, random_state=seed
    )
    train_output_path = Path(output_path) / 'train' / 'ocr_res'
    val_output_path = Path(output_path) / 'val' / 'ocr_res'
    train_output_path.mkdir(exist_ok=True, parents=True)
    val_output_path.mkdir(exist_ok=True, parents=True)

    # Copy ocr file to train or val folder
    train_count = 0
    val_count = 0
    for file in ocr_res_files:
        file_name = file.stem.split('_page_')[0]
        if file_name in train_pdf:
            shutil.copy(
                file,
                train_output_path,
            )
            train_count += 1
        elif file_name in val_pdf:
            shutil.copy(
                file,
                val_output_path,
            )
            val_count += 1
        else:
            raise ValueError(f'{file_name} not in train or val')
    print(f'train: {train_count}, val: {val_count}')


def long_ie_label_parse(label_path, output_path):
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


def SendReqWithRest(client, ep, req_body):
    def post(ep, json_data=None, timeout=10000):
        url = 'http://{}/v2/idp/ocr_app/infer'.format(ep)
        r = client.post(url=url, json=json_data, timeout=timeout)
        return r

    try:
        r = post(ep, req_body)
        return r
    except Exception as e:
        print('Exception: ', e)


def exec_transformer(image):
    recog = "transformer-v2.8-gamma-faster"

    client = requests.Session()
    ep = http_url
    bytes_data = cv2.imencode('.jpg', image)[1].tobytes()
    b64enc = base64.b64encode(bytes_data).decode()
    params = {
        'sort_filter_boxes': True,
        'rotateupright': False,
        'support_long_image_segment': True,
        'refine_boxes': True,
        'recog': recog,
    }
    req_data = {'param': params, 'data': [b64enc]}
    r = SendReqWithRest(client, ep, req_data)

    if r.status_code != 200:
        print("ERROR: can't get text result after transformer-v2.8-gamma")
        return None
    res = r.json()
    return res["result"]["texts"][0]


def crop_and_recog(label_path):
    """抠小碎图 然后获取识别结果并写入label studio格式json文件中"""
    with open(label_path, 'r') as f:
        raw_result = json.load(f)

    for task in tqdm(raw_result):
        task_folder = task['data']['Name']
        page_info = task['data']['document']
        cur_url = [i['page'] for i in page_info]

        # Parse bboxes
        for label in task['annotations'][0]['result']:
            num = int(re.search(r'_\d+', label['to_name']).group(0)[1:])
            page = f"page_{num:03d}"

            x, y, w, h = convert_from_ls(label)
            angle = label['value']['rotation']
            box = convert_rect([x, y, w, h, angle])

            image_url = next(filter(lambda x: page in x, cur_url))
            response = requests.get(image_url)
            if response.status_code != 200:
                break
            bytes_data = response.content
            bytes_arr = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(bytes_arr, cv2.IMREAD_COLOR)
            croped_img = crop_images(img, np.array([box]))[0]
            res = exec_transformer(croped_img)

            label['meta'] = {'text': [res]}

    with open(Path(dst) / 'has_rec.json', 'w') as f:
        json.dump(raw_result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    """covert label studio json to  processed json"""
    # label_path = '/home/youjiachen/workspace/Labels_06_09.json'
    # img_path = '/home/youjiachen/workspace/longtext_ie/datasets/contract_v1.3/Images'
    # ocr_res_path = '/home/youjiachen/workspace/longtext_ie/datasets/contract_v1.0/dataelem_ocr_res_rotateupright_true'
    # output_path = '/home/youjiachen/workspace/longtext_ie/datasets/contract_ds_v2.0'
    # check_folder(output_path)

    # process_label_studio(label_path, img_path, ocr_res_path, output_path)

    label_path = '/Users/youjiachen/Desktop/projects/label_studio_mgr/data/二手房-合并.json'
    dst = '/Users/youjiachen/Desktop/projects/label_studio_mgr/data/'

    # print(res)