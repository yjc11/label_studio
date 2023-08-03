import argparse
import base64
import copy
import json
import math
import os
import urllib
from pathlib import Path
from pprint import pprint
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from tqdm import tqdm

IP_ADDRESS = '192.168.106.133'
PORT = 8507
GREEN = '\033[92m'
ENDC = '\033[0m'


def api_call(
    image,
    det='mrcnn-v5.1',
    recog='transformer-v2.8-gamma-faster',
    scene='chinese_print',
) -> dict:
    def general(data, ip_address, port) -> dict:
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
            # 'recog': recog,
        },
    }

    ret = general(data, IP_ADDRESS, PORT)
    # print(rethao         )
    return ret['data']['json']['general_ocr_res']


# convert from LS percent units to pixels
def convert_from_ls(result: Dict) -> Optional[Tuple]:
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


def rotate_image_only(im, angle) -> Tuple[np.ndarray, any, any]:
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


def request_image(url):
    response = requests.get(url)
    response.raise_for_status()
    bytes_data = response.content
    bytes_arr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(bytes_arr, cv2.IMREAD_COLOR)

    return img, img.shape[1], img.shape[0]


def url_to_filename(url):
    path = Path(url)
    filename = urllib.parse.unquote(path.name)
    return filename


def convert_data(input_json, output_dir, do_ocr=False):
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
        print(len(raw_examples))

    # split train val
    train_examples, val_examples = train_test_split(
        raw_examples, train_size=0.8, test_size=0.2, shuffle=True, random_state=42
    )
    dataset = {'Train': train_examples, 'Val': val_examples}

    # do ocr on all images
    if do_ocr:
        for raw_example in tqdm(raw_examples, desc='doing ocr'):
            image_url = raw_example['data']['Image']
            image, w, h = request_image(image_url)
            image_name = url_to_filename(image_url)
            json_name = Path(image_name).with_suffix('.json')
            ocr_result = api_call(image)
            with open(ocr_results_output_path / json_name, 'w') as f:
                json.dump(ocr_result, f, ensure_ascii=False, indent=2)

    # process into uie label studio format
    no_ocr_result = list()
    no_annos = list()
    for trainval, examples in dataset.items():
        for example in tqdm(examples, desc=f'Process {trainval} Data'):
            new_result = list()

            # origin result list
            result_list = example['annotations'][0]['result']

            # get image filename
            image_url = example['data']['Image']
            img_name = url_to_filename(image_url)

            # get ocr result filename
            json_name = Path(img_name).with_suffix('.json')
            ocr_result_file = ocr_results_output_path.joinpath(json_name)

            # todo: remain no label example
            # if not ocr_result_file.exists():
            #     no_ocr_result.append(img_name)
            #     continue
            # elif not len(result_list):
            #     no_annos.append(img_name)

            # load ocr result
            with open(ocr_result_file, 'r') as f:
                ocr_result = json.load(f)
                rotate_angle = ocr_result['rotate_angle']  # [-90,0,90,180]
                image_size = ocr_result['image_size']

            # Request and rotate image
            image, w, h = request_image(image_url)
            rotated_image = rotate_image_only(image, rotate_angle)[0]
            image_output_path = str(all_imgs_output_path / img_name)
            cv2.imwrite(image_output_path, rotated_image)
            if trainval == 'Val':
                image_output_path = str(val_imgs_output_path / img_name)
                cv2.imwrite(image_output_path, rotated_image)

            # id to new converted result
            id_to_new_label = dict()

            # add scene name
            scene = Path(input_json).stem
            example['data']['scene'] = scene

            # update image name
            example['data']['image'] = 'prefix-' + img_name

            # Convert all annotations to new format
            for idx, label in enumerate(result_list):
                label_id = label.get('id')
                if label_id and label_id not in id_to_new_label:
                    id_to_new_label[label_id] = dict()

                if label['type'] == 'labels':
                    ori_w = label['original_width']
                    ori_h = label['original_height']
                    if ori_w != image_size[0] or ori_h != image_size[1]:
                        print(f'Updating {label_id} size')
                        id_to_new_label[label_id]['original_width'] = image_size[0]
                        id_to_new_label[label_id]['original_height'] = image_size[1]
                    else:
                        id_to_new_label[label_id]['original_width'] = ori_w
                        id_to_new_label[label_id]['original_height'] = ori_h

                    # Add bbox coordinates
                    angle = rotate_angle + label['value']['rotation']
                    id_to_new_label[label_id]['value'] = {
                        'x': label['value']['x'],
                        'y': label['value']['y'],
                        'width': label['value']['width'],
                        'height': label['value']['height'],
                        "rotation": label['value']['rotation'],  # 原始标注的angle
                        'rectanglelabels': label['value']['labels'],
                    }
                    # convert bbox format
                    x, y, w, h = convert_from_ls(label)
                    # Rotate bbox
                    # 先转为原始标注的bbox
                    bbox = convert_rect([x, y, w, h, label['value']['rotation']])
                    # 再根据ocr result angle 进行旋转
                    rotated_bbox = rotate_box(np.array(bbox), image_size, rotate_angle)
                    id_to_new_label[label_id]['origin_bbox'] = rotated_bbox.tolist()

                    # Add type
                    id_to_new_label[label_id]['type'] = 'rectanglelabels'
                    id_to_new_label[label_id]['id'] = label['id']

                # gt text
                elif label['type'] == 'textarea':
                    text = label['value']['text'][0]  # must str, not list
                    id_to_new_label[label_id]['origin_text'] = text

                elif label['type'] == 'relation':
                    id_to_new_label[f'relation_{idx}'] = label

            # construct final result list
            for value in id_to_new_label.values():
                if len(value):
                    new_result.append(value)
            example['annotations'][0]['result'] = new_result

        # Save updated annotations
        output_path = (
            label_studio_train_output_path
            if trainval == 'Train'
            else label_studio_val_output_path
        )
        with open(output_path, 'w') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)

    print(f'{GREEN} total image: {len(raw_examples)} {ENDC}')


def get_args():
    argparse
    pass


if __name__ == '__main__':
    input_json = '/home/youjiachen/workspace/relation_exp/完税证明_ypzh.json'
    output_dir = '/home/youjiachen/workspace/relation_exp/完税证明_ypzh'
    convert_data(input_json, output_dir, do_ocr=True)
    
    

    pass
