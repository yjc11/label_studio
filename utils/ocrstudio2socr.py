import argparse
import base64
import json
import os
import shutil
import urllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from pprint import pprint

import cv2
import numpy as np
import requests
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

IP_ADDRESS = '192.168.106.131'
PORT = 8506


def ocr_predict(image, scene='chinese_print'):
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
            'rotateupright': True,
            'refine_boxes': True,
            'sort_filter_boxes': True,
            'support_long_rotate_dense': False,
            'vis_flag': False,
            'sdk': True,
            'det': 'mrcnn-v5.1',
            # 'det': 'general_text_det_mrcnn_v1.0',
            'recog': 'transformer-v2.8-gamma-faster',
        },
    }

    ret = general(data, IP_ADDRESS, PORT)
    return ret['data']['json']['general_ocr_res']


def t_ocr_predict(imgs: list, path_out: Path, n_threads=10):
    def _ocr_get_save(img, path_out):
        result = ocr_predict(img)
        with open(path_out / (img.stem + '.json'), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii='false', indent=2)

    # predict & save results
    pbar = tqdm(total=len(imgs), desc="obtaining OCR results")
    with ThreadPoolExecutor(n_threads) as executor:
        futures = [executor.submit(_ocr_get_save, img, path_out) for img in imgs]
        for _ in as_completed(futures):
            pbar.update(1)
        pbar.close()


def ocrstudio2socr(input_dir, output_dir, max_workers=10):
    # paths and mkdirs
    labels_output_path = Path(output_dir) / 'Labels'
    images_output_path = Path(output_dir) / 'Images'
    train_images_output_path = images_output_path / 'train'
    val_images_output_path = images_output_path / 'val'

    ocr_results_output_path = images_output_path / 'ocr_results'
    train_ocr_results_output_path = ocr_results_output_path / 'train'
    val_ocr_results_output_path = ocr_results_output_path / 'val'

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

    # image paths
    label_path = Path(input_dir) / 'Labels'
    image_paths = list((Path(input_dir) / 'Images').glob('[!.]*'))
    # split train and val
    train_images, val_images = train_test_split(
        image_paths, train_size=0.8, test_size=0.2, shuffle=True, random_state=42
    )
    dataset = {'train': train_images, 'val': val_images}
    print(f'total image nums : {len(image_paths)}'.upper())

    # maintain a set of all
    label_set = set()

    for trainval, image_files in dataset.items():
        print(f'\n「{trainval.upper()}」')
        print(f'{trainval} image nums : {len(image_files)}'.upper())
        pbar = tqdm(total=len(image_files), desc=f'process {trainval} data'.upper())

        # for image_file in tqdm(image_files, desc=f'process {trainval} data'):
        def process_task(image_file):
            json_name = image_file.stem + '.json'
            label_file = label_path / json_name

            if trainval == 'train':
                shutil.copy(image_file, train_images_output_path)
            elif trainval == 'val':
                shutil.copy(image_file, val_images_output_path)
            shutil.copy(label_file, labels_output_path)

        with ThreadPoolExecutor(max_workers=max_workers) as e:
            futures = [e.submit(process_task, task) for task in image_files]
            for future in as_completed(futures):
                pbar.update(1)
                future.result()
        pbar.close()

        # get ocr result and save
        imgs = list(images_output_path.joinpath(trainval).glob('[!.]*.*'))
        t_ocr_predict(imgs, ocr_results_output_path / trainval, 10)

    # generate meta.yaml file
    label_paths = list(labels_output_path.glob('[!.]*'))
    for label_file in label_paths:
        with label_file.open('r') as f:
            labels = json.load(f)
            for label in labels:
                label_set.add(label['category'])

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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input_dir',
        help='input ocr studio data directory',
        type=str,
        default='/Users/youjiachen/Desktop/projects/label_studio_mgr/output/询证函-去摩尔纹',
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        help='output dir',
        type=str,
        default='../output',
    )
    parser.add_argument(
        '-nt',
        '--n-threads',
        help='number of threads to be used for getting results from OCR API',
        type=str,
        default='10',
    )
    return parser.parse_args()


def main():
    args = get_args()
    ocrstudio2socr(args.input_dir, args.output_dir, int(args.n_threads))


if __name__ == '__main__':
    main()
