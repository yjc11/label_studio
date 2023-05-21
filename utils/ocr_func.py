import base64
import os
import requests
import time
import concurrent.futures
from multiprocessing.dummy import Pool as ThreadPool  # 多线程
import json
import numpy as np
import re
import shutil
from pathlib import Path
from tqdm import tqdm


IP_ADDRESS = '192.168.106.131'
PORT = 8506


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
        'parameters': {
            'vis_flag': False,
            'det': 'mrcnn-v5.1',
            'recog': 'transformer-v2.8-gamma-faster',
            #    'recog' : 'transformer-blank-v0.2-faster',
            'sdk': True,
            'rotateupright': False,
        },
    }

    ret = general(data, ip_address, port)

    return ret['data']['json']['general_ocr_res']


if __name__ == '__main__':
    CWD = Path().cwd()
    DATA_DIR = CWD / 'workspace/long_text/long_text_contract_ds/Images'
    OUTPUT_PATH = CWD / 'contract_longtext/dataelem_ocr_res_rotateupright_true'

    # general
    # @save_to_json(OUTPUT_PATH)
    # def get_ocr_results_and_save(image_file, ip_address=IP_ADDRESS, port=PORT):
    #     """
    #     常规ocr全文识别结果配置
    #     """
    #     data = {
    #         'scene': 'chinese_print',
    #         'image': convert_b64(image_file),
    #         'parameters': {
    #             'vis_flag': False,
    #             'det': 'mrcnn-v5.1',
    #             # 'det': 'None',
    #             'recog': 'transformer-v2.8-gamma-faster',
    #             #    'recog' : 'transformer-blank-v0.2-faster',
    #             'sdk': True,
    #             'rotateupright': False,
    #         },
    #     }

    #     ret = general(data, ip_address, port)

    #     return ret['data']['json']['general_ocr_res']

    # smart structure
    @save_to_json(OUTPUT_PATH)
    def get_ocr_results_and_save(image_file, ip_address=IP_ADDRESS, port=PORT):
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
                'recog': 'transformer-v2.8-gamma-faster',
            },
        }

        ret = general(data, ip_address, port)

        return ret['data']['json']['general_ocr_res']

    image_files = list(DATA_DIR.glob('[!.]*'))
    # multithreadpost(image_files, get_ocr_results, max_workers=10)
    # multithreadpost(image_files, get_ocr_results_and_save, max_workers=10)

    # if raise erro
    # ori_img = list(DATA_DIR.glob('[!.]*'))
    # ocr_res = list(OUTPUT_PATH.glob('[!.]*'))

    # set_ori_img = {i.stem for i in ori_img}
    # set_ocr_img = {i.stem for i in ocr_res}

    # unseen_pic = set_ocr_img ^ set_ori_img
    # unseen_image_files = [DATA_DIR / (i + '.png') for i in unseen_pic]

    # multithreadpost(unseen_image_files, get_ocr_results_and_save, max_workers=10)
