import argparse
import base64
import glob
import json
import os
import re
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import yaml
from tqdm import tqdm

server = "http://192.168.106.12:40058"


def api_call(image, reg_model, det_model, scene='chinese_print'):
    url = f'{server}/lab/ocr/predict/general'
    b64 = base64.b64encode(open(image, 'rb').read()).decode()
    # 图片id(可以不添加)，用于唯一标识某张图片，可以在SDK内部日志mlserver.log/socr.log查看到，以便确定图片识别状态。建议只使用数字和字符串
    data = {
        'id': "xxxxxx",
        'scene': scene,
        'image': b64,
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
    res = requests.post(url, json=data).json()
    return res


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


def get_det_reg_model(excel_file):
    df = pd.read_excel(io=excel_file, keep_default_na=False)
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


def get_trainval_ocr_results(folder, save_folder, model_file):
    np.random.seed(42)
    scene_list = [
        scene
        for scene in os.listdir(folder)
        if (not scene.startswith('.')) and (not scene.endswith('.xlsx'))
    ]
    scene_info = get_det_reg_model(model_file)

    # check scene if exist in model.xlsx
    for scene in scene_list:
        if scene not in scene_info.keys():
            raise ValueError(
                f'{scene} not in model excel file. Pleace check scene name.\n'
            )

    for scene in scene_list:
        # if scene in ['']:
        #     continue
        scene_folder = os.path.join(folder, scene)
        scene_save_folder = os.path.join(save_folder, scene)
        reg_model, det_model = scene_info[scene]

        image_folder = os.path.join(scene_folder, 'Images')
        label_folder = os.path.join(scene_folder, 'Labels')

        save_label_folder = os.path.join(scene_save_folder, 'Labels')
        save_train_image_folder = os.path.join(scene_save_folder, 'Images', 'train')
        save_val_image_folder = os.path.join(scene_save_folder, 'Images', 'val')
        save_train_ocr_folder = os.path.join(
            scene_save_folder, 'Images', 'ocr_results', 'train'
        )
        save_val_ocr_folder = os.path.join(
            scene_save_folder, 'Images', 'ocr_results', 'val'
        )
        check_folder(save_label_folder)
        check_folder(save_train_image_folder)
        check_folder(save_val_image_folder)
        check_folder(save_train_ocr_folder)
        check_folder(save_val_ocr_folder)

        L2_keys = set()
        image_files = list_image(image_folder)
        for image_file in tqdm(image_files, desc=f'{scene}'):
            image_name = os.path.basename(image_file)
            json_name = os.path.splitext(image_name)[0] + '.json'
            label_file = os.path.join(label_folder, json_name)
            if not os.path.exists(label_file):
                print(f'{label_file} is not exist.')
                continue
            with open(label_file, 'r') as f:
                labels = json.load(f)
                labels_convert = []
                for label in labels:
                    # 去掉NAN字段
                    if label['category'] == 'NAN':
                        continue
                    L2_keys.add(label['category'])
                    labels_convert.append(label)
            with open(os.path.join(save_label_folder, json_name), 'w') as f:
                json.dump(labels_convert, f, ensure_ascii=False, indent=2)

            try:
                ocr_res = api_call(image_file, reg_model, det_model)['data']['json'][
                    'general_ocr_res'
                ]
            except Exception as e:
                print(e)
                print(f'{image_file} gets ocr reuslts failed.')

            if np.random.rand() < 0.8:
                shutil.copy(
                    image_file, os.path.join(save_train_image_folder, image_name)
                )
                with open(os.path.join(save_train_ocr_folder, json_name), 'w') as f:
                    json.dump(ocr_res, f, ensure_ascii=False, indent=2)
            else:
                shutil.copy(image_file, os.path.join(save_val_image_folder, image_name))
                with open(os.path.join(save_val_ocr_folder, json_name), 'w') as f:
                    json.dump(ocr_res, f, ensure_ascii=False, indent=2)

        meta_yaml = dict()
        meta_yaml['attributes'] = dict()
        meta_yaml['attributes']['difficulty'] = 'easy'
        meta_yaml['attributes']['language'] = 'Chinese'
        meta_yaml['attributes']['license'] = 'public'
        meta_yaml['exclude_files'] = []
        meta_yaml['exclude_keys'] = []
        meta_yaml['field_def'] = list(L2_keys)

        with open(
            os.path.join(scene_save_folder, 'meta.yaml'), 'w', encoding='utf-8'
        ) as f:
            yaml.dump(meta_yaml, f, allow_unicode=True)
        print(scene, len(image_files), len(os.listdir(label_folder)), 'done')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input_folder',
        help='input label studio long text json',
        type=str,
        default=None,
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        help='output dir',
        type=str,
        default=None,
    )
    parser.add_argument(
        '-m',
        '--model_file',
        help='ocr det and rocog excel ',
        type=str,
        default=None,
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
    get_trainval_ocr_results(args.input_folder, args.output_dir, args.model_file)


if __name__ == '__main__':
    main()
