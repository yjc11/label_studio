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
from typing import Any, Callable, List, Sequence

import numpy as np
import requests
from tqdm import tqdm

IP_ADDRESS = '192.168.106.133'
PORT = 8506


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


def multithreadpost(image_files, func, max_workers=1):
    all_start_time = time.time()
    pool = ThreadPool(max_workers)
    results = pool.imap(func, image_files)
    progress_bar = tqdm(results, total=len(image_files))
    for result in progress_bar:
        pass
    pool.close()
    pool.join()
    all_end_time = time.time()
    print('finish_time: {}'.format(all_end_time - all_start_time))


def convert_b64(file):
    if os.path.isfile(file):
        with open(file, 'rb') as fh:
            x = base64.b64encode(fh.read())
            return x.decode('ascii').replace('\n', '')
    else:
        return None


def save_to_json(output_path):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            image_file = args[0]
            filename = Path(image_file).stem
            filepath = Path(output_path) / f"{filename}.json"
            with open(filepath, "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            return result

        return wrapper

    return decorator


def get_ocr_results(image_file, ip_address=IP_ADDRESS, port=PORT):
    # 配置文件 /root/socr_data/socr_models/server_settings.yaml
    # scene 这里要和yaml中的models内的name一致
    data = {
        'scene': 'chinese_print',
        'image': convert_b64(image_file),
        # 'image': image_file,
        'parameters': {
            'rotateupright': True,
            'refine_boxes': True,
            'sort_filter_boxes': True,
            'support_long_rotate_dense': False,
            'vis_flag': False,
            'sdk': True,
            'det': 'mrcnn-v5.1',
            'recog': 'transformer-v2.8-gamma-faster',
        },
    }

    ret = general(data, ip_address, port)

    return ret['data']['json']['general_ocr_res']


@threaded
def get_ocr_results_and_save(image_file, output_dir):
    """
    结构化OCR全文识别结果配置
    """
    data = {
        'scene': 'chinese_print',
        'image': convert_b64(image_file),
        'parameters': {
            'rotateupright': True,
            'refine_boxes': True,
            'sort_filter_boxes': True,
            'support_long_rotate_dense': False,
            'vis_flag': False,
            'sdk': True,
            'det': 'mrcnn-v5.1',
            # 'recog': 'transformer-v2.8-gamma-faster',
        },
    }

    ret = general(data, IP_ADDRESS, PORT)
    ocr_result = ret['data']['json']['general_ocr_res']

    if output_dir:
        json_file = Path(image_file).stem + '.json'
        output_file = Path(output_dir) / json_file
        with open(output_file, 'w') as f:
            json.dump(ocr_result, f, ensure_ascii=False, indent=2)

    return ocr_result


if __name__ == '__main__':
    input_images = '/Users/youjiachen/Desktop/长文档5个场景/公文/公文-南网项目发文'
    output_dir = '/Users/youjiachen/Desktop/长文档5个场景/公文_dataelem_ocr_res'
    check_folder(output_dir)

    image_files = list(Path(input_images).glob('[!.]*'))
    get_ocr_results_and_save(image_files, output_dir=output_dir)
