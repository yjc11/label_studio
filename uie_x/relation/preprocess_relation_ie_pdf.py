import argparse
import base64
import copy
import json
import math
import os
import urllib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from tqdm import tqdm

IP_ADDRESS = '192.168.106.133'
PORT = 8506
GREEN = '\033[92m'
ENDC = '\033[0m'


def threaded(func: Callable):
    def wrapper(tasks: Sequence[Any], **kwargs) -> None:
        with ThreadPoolExecutor(10) as e:
            pbar = tqdm(total=len(tasks))
            futures: List = [e.submit(func, task, **kwargs) for task in tasks]
            for future in as_completed(futures):
                pbar.update(1)
                future.result()
            pbar.close()

    return wrapper


def api_call(
    image,
    det='mrcnn-v5.1',
    recog='transformer-v2.8-gamma-faster',
    scene='chinese_print',
) -> Dict:
    def general(data, ip_address, port) -> Dict:
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
            'rotateupright': True,
            'refine_boxes': True,
            'sort_filter_boxes': True,
            'support_long_rotate_dense': False,
            'vis_flag': False,
            'sdk': True,
            'det': det,
            # 'det': 'general_text_det_mrcnn_v1.0',
            'recog': recog,
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


def rotate_image_only(im, angle) -> Tuple[np.ndarray, Tuple[Any, Any], Tuple[Any, Any]]:
    """
    rotate image in range[-10,10]
    :param polys:
    :param tags:
    :return:
    """

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
        rotated_image = cv2.warpAffine(
            src,
            rot_mat,
            (int(math.ceil(nw)), int(math.ceil(nh))),
            flags=cv2.INTER_LANCZOS4,
        )
        return rotated_image

    old_h, old_w, _ = im.shape
    old_center = (old_w / 2, old_h / 2)

    image = rotate(im, angle)
    new_h, new_w, _ = image.shape
    new_center = (new_w / 2, new_h / 2)

    return image, old_center, new_center


def _request_image(url: str) -> Tuple[np.ndarray, int, int]:
    response = requests.get(url)
    response.raise_for_status()
    bytes_data: bytes = response.content
    bytes_arr = np.frombuffer(bytes_data, np.uint8)
    img: np.ndarray = cv2.imdecode(bytes_arr, cv2.IMREAD_COLOR)

    return img, img.shape[1], img.shape[0]


def _url_to_filename(url):
    page = Path(url).name
    page = urllib.parse.unquote(page)  # page_xxx.png
    pdfname = Path(url).parent.name
    pdfname = urllib.parse.unquote(pdfname)
    filename = pdfname + '_' + page
    return filename


def convert_data(input_json: str, output_dir: str, do_ocr: bool = False) -> List:
    label_studio_train_output_path = Path(f'{output_dir}/label_train_studio.json')
    label_studio_val_output_path = Path(f'{output_dir}/label_val_studio.json')
    all_imgs_output_path = Path(f'{output_dir}/images')
    val_imgs_output_path = Path(f'{output_dir}/val_images')
    ocr_results_output_path = Path(f'{output_dir}/ocr_results')

    output_paths = [
        all_imgs_output_path,
        val_imgs_output_path,
        ocr_results_output_path,
    ]
    for path in output_paths:
        path.mkdir(exist_ok=True, parents=True)

    # Load raw label studio json
    with open(input_json, 'r') as f:
        raw_examples = json.load(f)

    train_examples, val_examples = train_test_split(
        raw_examples, train_size=0.8, test_size=0.2, shuffle=True, random_state=42
    )

    cur_datasets = {'Train': train_examples, 'Val': val_examples}
    print(f'train pdf: {len(train_examples)}, val pdf: {len(val_examples)}')

    for trainval, examples in cur_datasets.items():
        new_annos_list = list()
        for task in examples:
            result_list = task['annotations'][0]['result']

            page_to_label_id = defaultdict(list)
            for label in result_list:
                if label.get('to_name'):
                    page_idx = int(label['to_name'].split('_')[1])
                    label_id = label['id']
                    if label['type'] == 'labels':
                        page_to_label_id[page_idx].append(label_id)
            # new_annos = [{}] * len(page_to_label_id) # bug
            new_annos: List[dict] = [{} for _ in range(len(page_to_label_id))]

            taskname = task['data']['Name']
            page_infos = task['data']['document']

            # 1. parse images and do ocr
            pbar_img = tqdm(
                total=len(page_to_label_id), desc=f'Downloading {taskname} Images'
            )
            pbar_ocr = tqdm(total=len(page_to_label_id), desc=f'Do {taskname} OCR')
            for i, (label_page, label_id_list) in enumerate(page_to_label_id.items()):
                image_url = page_infos[label_page]['page']
                img_name = _url_to_filename(image_url)
                img, w, h = _request_image(image_url)

                if do_ocr:
                    ocr_result = api_call(img)
                    json_name = Path(img_name).with_suffix('.json')
                    with open(ocr_results_output_path / json_name, 'w') as f:
                        json.dump(ocr_result, f, ensure_ascii=False)
                    pbar_ocr.update(1)

                json_name = Path(img_name).with_suffix('.json')
                with open(ocr_results_output_path / json_name, 'r') as f:
                    ocr_result = json.load(f)
                    rotate_angle = ocr_result['rotate_angle']  # [-90,0,90,180]
                    image_size = ocr_result['image_size']

                rotated_img = rotate_image_only(img, rotate_angle)[0]
                if trainval == 'Val':
                    cv2.imwrite(str(val_imgs_output_path / img_name), rotated_img)
                cv2.imwrite(str(all_imgs_output_path / img_name), rotated_img)
                pbar_img.update(1)

                id_to_new_labels: Dict[str, dict] = dict()
                new_result_list = list()
                for idx, label in enumerate(result_list):
                    label_id = label.get('id')

                    if label_id and label_id not in id_to_new_labels:
                        if label['to_name'] != f'page_{label_page}':
                            continue
                        else:
                            id_to_new_labels[label_id] = dict()

                    if label['type'] == 'labels':
                        if label['to_name'] != f'page_{label_page}':
                            continue
                        else:
                            ori_w = label['original_width']
                            ori_h = label['original_height']
                            real_w, real_h = image_size[0], image_size[1]
                            if ori_w != real_w or ori_h != real_h:
                                print(f'Updating {label_id} size')
                                id_to_new_labels[label_id]['original_width'] = real_w
                                id_to_new_labels[label_id]['original_height'] = real_h
                            else:
                                id_to_new_labels[label_id]['original_width'] = ori_w
                                id_to_new_labels[label_id]['original_height'] = ori_h

                            # add entity label
                            id_to_new_labels[label_id]['value'] = {
                                'rectanglelabels': label['value']['labels'],
                            }
                            # convert bbox format
                            x, y, w, h = convert_from_ls(label)
                            # Rotate bbox
                            # 先转为原始标注的bbox
                            bbox = convert_rect(
                                [x, y, w, h, label['value']['rotation']]
                            )
                            # 再根据ocr result angle 进行旋转
                            rotated_bbox = rotate_box(
                                np.array(bbox), image_size, rotate_angle
                            )
                            id_to_new_labels[label_id][
                                'origin_bbox'
                            ] = rotated_bbox.tolist()

                            # Add type
                            id_to_new_labels[label_id]['type'] = 'rectanglelabels'
                            id_to_new_labels[label_id]['id'] = label['id']
                            id_to_new_labels[label_id]['to_name'] = label['to_name']

                    # gt text
                    elif label['type'] == 'textarea':
                        if label['to_name'] != f'page_{label_page}':
                            continue
                        else:
                            text = label['value']['text'][0]  # must str, not list
                            id_to_new_labels[label_id]['origin_text'] = text

                    elif label['type'] == 'relation':
                        if (
                            label['from_id'] in label_id_list
                            and label['to_id'] in label_id_list
                        ):
                            id_to_new_labels[f'relation_{idx}'] = label

                for new_labels in id_to_new_labels.values():
                    new_result_list.append(new_labels)

                new_annos[i]['annotations'] = [{'result': new_result_list}]
                new_annos[i]['data'] = {
                    'image': f'prefix-{img_name}',
                    'scene': Path(input_json).stem,
                }

            new_annos_list.extend(new_annos)
            pbar_img.close()
            pbar_ocr.close()

        # Save updated annotations
        output_path = (
            label_studio_train_output_path
            if trainval == 'Train'
            else label_studio_val_output_path
        )
        with open(output_path, 'w') as f:
            json.dump(new_annos_list, f, ensure_ascii=False, indent=2)

    return new_annos_list


if __name__ == '__main__':
    input_json = '/Users/youjiachen/Downloads/rmgg.json'
    output_dir = '/Users/youjiachen/Downloads/rmgg.json'
    convert_data(input_json, output_dir, do_ocr=True)
    pass
