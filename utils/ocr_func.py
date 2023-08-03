import base64
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.dummy import Pool as ThreadPool  # 多线程
from pathlib import Path
from typing import Any, Callable, List, Sequence, Union

import cv2
import numpy as np
import requests
from tqdm import tqdm

IP_ADDRESS = '192.168.106.12'
PORT = 40058
SERVER = 'http://192.168.106.12:40058'

IMAGE_EXT = 'jpg|jpeg|bmp|png|tif|tiff|JPG|PNG|TIF|TIFF'


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


def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)


def convert_b64(file):
    if os.path.isfile(file):
        with open(file, 'rb') as fh:
            x = base64.b64encode(fh.read())
            return x.decode('ascii').replace('\n', '')
    elif isinstance(file, np.ndarray):
        bytes_data = cv2.imencode('.jpg', file)[1].tobytes()
        b64enc = base64.b64encode(bytes_data).decode()
        return b64enc


def list_image(directory, ext='jpg|jpeg|bmp|png|tif|tiff|JPG|PNG|TIF|TIFF'):
    listOfFiles = list()
    for dirpath, dirnames, filenames in os.walk(directory):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    pattern = ext + r'\Z'
    res = [f for f in listOfFiles if re.findall(pattern, f)]
    return res


def ticket(data, ip_address, port):
    r = requests.post(f'http://{ip_address}:{port}/lab/ocr/predict/ticket', json=data)
    return r.json()


def table(data, ip_address, port):
    r = requests.post(f'http://{ip_address}:{port}/lab/ocr/predict/table', json=data)
    return r.json()


def general(data, ip_address, port):
    r = requests.post(f'http://{ip_address}:{port}/lab/ocr/predict/general', json=data)
    # print(r)
    return r.json()


@threaded
def threaded_api_call(
    image_file: Union[str, np.ndarray],
    det_model='mrcnn-v5.1',
    reg_model='transformer-v2.8-gamma-faster',
    scene='chinese_print',
    output_dir=None,
):
    assert det_model in ['general_text_det_mrcnn_v1.0', 'mrcnn-v5.1']
    assert reg_model in [
        'transformer-v2.8-gamma-faster',
        'transformer-blank-v0.2-faster',
        'transformer-hand-v1.16-faster',
    ]

    data = {
        'scene': scene,
        'image': convert_b64(image_file),
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
    ocr_result = ret['data']['json']['general_ocr_res']

    if output_dir and not isinstance(image_file, np.ndarray):
        json_file = Path(image_file).stem + '.json'
        output_file = Path(output_dir) / json_file
        with open(output_file, 'w') as f:
            json.dump(ocr_result, f, ensure_ascii=False, indent=2)
    return ocr_result


def api_call(
    image_file: Union[str, np.ndarray],
    det_model='mrcnn-v5.1',
    reg_model='transformer-v2.8-gamma-faster',
    scene='chinese_print',
    output_dir=None,
):
    assert det_model in ['general_text_det_mrcnn_v1.0', 'mrcnn-v5.1']
    assert reg_model in [
        'transformer-v2.8-gamma-faster',
        'transformer-blank-v0.2-faster',
        'transformer-hand-v1.16-faster',
    ]

    data = {
        'scene': scene,
        'image': convert_b64(image_file),
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
    ocr_result = ret['data']['json']['general_ocr_res']

    if output_dir and not isinstance(image_file, np.ndarray):
        json_file = Path(image_file).stem + '.json'
        output_file = Path(output_dir) / json_file
        with open(output_file, 'w') as f:
            json.dump(ocr_result, f, ensure_ascii=False, indent=2)
    return ocr_result


if __name__ == '__main__':
    input_images = '/mnt/disk0/youjiachen/workspace/拆迁安置协议'
    output_dir = '/mnt/disk0/youjiachen/workspace/拆迁安置协议dataelem_ocr_res'
    check_folder(output_dir)

    # 多线程
    image_files = list(Path(input_images).glob('[!.]*'))
    print(image_files)
    # threaded_api_call(image_files, output_dir=output_dir)

    # 处理单个文件
