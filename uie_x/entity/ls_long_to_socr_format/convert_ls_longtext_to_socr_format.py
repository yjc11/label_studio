import argparse
import base64
import json
import math
import os
import urllib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, List

import cv2
import numpy as np
import pandas as pd
import requests
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

IP_ADDRESS = '192.168.106.12'
PORT = 40058
# SERVER = 'http://192.168.106.12:40058'


def api_call(image, det_model, reg_model, scene='chinese_print'):
    assert det_model in ['general_text_det_mrcnn_v1.0', 'mrcnn-v5.1']
    assert reg_model in [
        'transformer-v2.8-gamma-faster',
        'transformer-blank-v0.2-faster',
        'transformer-hand-v1.16-faster',
    ]

    def general(data, ip_address, port):
        r = requests.post(
            f'http://{ip_address}:{port}/lab/ocr/predict/general', json=data
        )
        return r.json()

    def convert_b64(file):
        if os.path.isfile(file):
            with open(file, 'rb') as fh:
                x = base64.b64encode(fh.read())
                return x.decode('ascii').replace('\n', '')
        elif isinstance(file, np.ndarray):
            bytes_data = cv2.imencode('.jpg', file)[1].tobytes()
            b64enc = base64.b64encode(bytes_data).decode()
            return b64enc

    data = {
        'scene': scene,
        'image': convert_b64(image),
        'parameters': {
            'det': det_model,
            'recog': reg_model,
            'rotateupright': True,
            'refine_boxes': True,
            'sort_filter_boxes': True,
            'support_long_rotate_dense': False,
            'vis_flag': False,
            'sdk': True,
            'checkbox': ['std_checkbox'],
        },
    }

    ret = general(data, IP_ADDRESS, PORT)
    return ret['data']['json']['general_ocr_res']


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


def get_det_reg_model(excel_file, sheet_name):
    df = pd.read_excel(io=excel_file, keep_default_na=False, sheet_name=sheet_name)
    category = [i for i in df.场景 if i != '']
    assert len(category) == len(set(set(category))), 'scene name is not unique.'
    category = set(category)

    scene_info = dict()
    for scene in category:
        reg_model = ''
        det_model = ''
        items = df[df.场景 == scene].values
        reg_model = items[0][1]
        det_model = items[0][2]

        scene_info[scene] = [reg_model, det_model]
    return scene_info


def long_to_socr(input_json, output_dir, model_file, sheet_name):
    scene = Path(input_json).stem
    scene_ocr_info = get_det_reg_model(model_file, sheet_name)
    reg_model, det_model = scene_ocr_info[scene]
    print(f'SCENE: {scene}', f'DET: {det_model}', f'REGCOG: {reg_model}')

    output_dir = Path(output_dir) / scene
    labels_output_path = Path(output_dir) / 'Labels'

    images_output_path = Path(output_dir) / 'Images'
    train_images_output_path = images_output_path / 'train'
    val_images_output_path = images_output_path / 'val'

    ocr_results_output_path = images_output_path / 'ocr_results'
    train_ocr_results_output_path = ocr_results_output_path / 'train'
    val_ocr_results_output_path = ocr_results_output_path / 'val'

    file_mapping_output_path = Path(output_dir) / 'file_mapping.json'

    output_paths = [
        labels_output_path,
        images_output_path,
        train_images_output_path,
        val_images_output_path,
        ocr_results_output_path,
        train_ocr_results_output_path,
        val_ocr_results_output_path,
    ]

    for path in output_paths:
        path.mkdir(parents=True, exist_ok=True)

    def _request_image_url(url):
        response = requests.get(url)
        if response.status_code != 200:
            raise 'erro'
        bytes_data = response.content
        bytes_arr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(bytes_arr, cv2.IMREAD_COLOR)

        return img, img.shape[1], img.shape[0]

    def _url_to_filename(url):
        page = Path(url).name
        page = urllib.parse.unquote(page)  # page_xxx.png
        pdfname = Path(url).parent.name
        pdfname = urllib.parse.unquote(pdfname)
        filename = pdfname + '_' + page
        return filename

    with open(input_json, 'r') as f:
        raw_examples = json.load(f)

    train_examples, val_examples = train_test_split(
        raw_examples, train_size=0.8, test_size=0.2, shuffle=True, random_state=42
    )

    cur_datasets = {'train': train_examples, 'val': val_examples}
    print(f'train pdf: {len(train_examples)}, val pdf: {len(val_examples)}')
    label_set = set()
    file_mapping_dict = defaultdict(list)
    result_dict = defaultdict(lambda: defaultdict(dict))
    for trainval, examples in cur_datasets.items():
        for task in tqdm(examples, desc=f'process {trainval} data'):
            taskname = task['data']['Name']
            page_infos = task['data']['document']
            file_mapping_dict[taskname] = [
                _url_to_filename(info['page']) for info in page_infos
            ]

            # 1. parse images and do ocr
            # for page2url in tqdm(page_infos):
            def process_page(page2url):
                image_url = page2url['page']
                filename = _url_to_filename(image_url)
                img, w, h = _request_image_url(image_url)
                img_output_path = Path(images_output_path) / trainval / filename
                cv2.imwrite(str(img_output_path), img)

                ocr_result = api_call(img_output_path, det_model, reg_model)
                json_name = Path(filename).with_suffix('.json')
                with open(ocr_results_output_path / trainval / json_name, 'w') as f:
                    json.dump(ocr_result, f, ensure_ascii=False)

            with ThreadPoolExecutor(max_workers=10) as e:
                futures = [e.submit(process_page, page) for page in page_infos]
                for future in as_completed(futures):
                    future.result()

            # 2. parse boxes
            annos = task['annotations'][0]['result']
            for anno in annos:
                if anno['type'] == 'labels':
                    label_id = anno['id']
                    cur_label = anno['value']['labels'][0]
                    page_idx = int(anno['to_name'].split('_')[1])
                    cur_page_url = page_infos[page_idx]['page']
                    filename = _url_to_filename(cur_page_url)
                    label_set.add(cur_label)

                    if anno['original_width'] == 1 or anno['original_width'] == 1:
                        _, h, w = _request_image_url(cur_page_url)
                        anno['original_width'], anno['original_height'] = w, h

                    # convert to box
                    x, y, w, h = convert_from_ls(anno)
                    angle = anno['value']['rotation']
                    box = convert_rect([x, y, w, h, angle])
                    result_dict[filename][label_id]['points'] = box
                    result_dict[filename][label_id]['category'] = cur_label

                elif anno['type'] == 'textarea':
                    label_id = anno['id']
                    cur_text = anno['value']['text'][0]
                    page_idx = int(anno['to_name'].split('_')[1])
                    cur_page_url = page_infos[page_idx]['page']
                    filename = _url_to_filename(cur_page_url)

                    result_dict[filename][label_id]['value'] = cur_text

    # save label to socr json
    for filename, labels in tqdm(result_dict.items(), desc='saving label to socr json'):
        json_name = Path(filename).with_suffix('.json')
        json_content = [label for label in labels.values()]
        with open(labels_output_path / json_name, 'w', encoding='utf-8') as f:
            json.dump(json_content, f, ensure_ascii=False)

    # generate meta.yaml file
    meta_yaml = {
        'attributes': {
            'difficulty': 'easy',
            'language': 'Chinese',
            'license': 'public',
            'exclude_files': [],
            'exclude_keys': [],
        },
        'field_def': list(label_set),
    }

    with open(Path(output_dir) / 'meta.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(meta_yaml, f, allow_unicode=True)

    # save file mapping json
    with open(file_mapping_output_path, 'w') as f:
        json.dump(file_mapping_dict, f, ensure_ascii=False, indent=2)

    print(f'{scene} done')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input_folder',
        help='input label studio long text json',
        type=str,
        default=None,
    )
    parser.add_argument('-o', '--output_dir', help='output dir', type=str, default=None)
    parser.add_argument(
        '-m', '--model_file', help='ocr det and rocog excel ', type=str, default=None
    )
    parser.add_argument(
        '-s',
        '--sheet_name',
        help='sheet_name',
        type=str,
        default=None,
    )
    return parser.parse_args()


def main():
    args = get_args()
    input_jsons = list(Path(args.input_folder).glob('[!.]*.json'))
    for input_json in input_jsons:
        long_to_socr(input_json, args.output_dir, args.model_file, args.sheet_name)


if __name__ == '__main__':
    main()
