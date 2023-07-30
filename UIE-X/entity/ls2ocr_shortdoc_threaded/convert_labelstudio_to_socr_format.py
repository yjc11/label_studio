import argparse
import base64
import json
import math
import os
import random
import shutil
import sys
import urllib
from ast import arg
from email.mime import image
from pathlib import Path
from unicodedata import category
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED

import cv2
# import ipdb as ip
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from pprint import pprint
from pathlib import Path

server = "http://192.168.106.131:8506"
def api_call(image, ocrmodel, scene='chinese_print'):
    url = f'{server}/lab/ocr/predict/general'
    b64 = base64.b64encode(open(image, 'rb').read()).decode()
    # 图片id(可以不添加)，用于唯一标识某张图片，可以在SDK内部日志mlserver.log/socr.log查看到，以便确定图片识别状态。建议只使用数字和字符串
    data = {
            'scene': scene,
            'image': b64,
            'parameters': {
                'rotateupright': True,
                'refine_boxes': True,
                'sort_filter_boxes': True,
                'support_long_rotate_dense': False,
                'vis_flag': False,
                'sdk': True,
                'det': 'mrcnn-v5.1',
                # 'det': 'general_text_det_mrcnn_v1.0',
                'recog': ocrmodel,
        },
            }
    res = requests.post(url, json=data).json()
    return res


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_json',
                        '--data_dir',
                        help='input json file path',
                        required = True,
                        type=str,
                        default=None)
    parser.add_argument('-output',
                        '--output_file',
                        help='output file path',
                        type=str,
                        required = True,
                        default=None)


    return parser.parse_args()


def download_images(json_file, imagepath):
    if not os.path.exists(imagepath):
        os.mkdir(imagepath)
        jAnnos = open(json_file, 'r', encoding='utf-8')
        annos = json.load(jAnnos)
        jAnnos.close()


        # with multithreading
        def _download_img(anno , pBar : tqdm):
            image_url = anno['data']['Image']
            base_name = Path(image_url).name
            file_name = urllib.parse.unquote(base_name)
            # ip.set_trace()
            response = requests.get(image_url)
            if response.status_code != 200:
                print('Download Error! ' + file_name + 'image')
                pBar.update(1)
                return
            bytes_data = response.content
            bytes_arr = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(bytes_arr, cv2.IMREAD_COLOR)
            try:
                cv2.imwrite(imagepath+'/' + file_name , img)
            except:
                print('Download Error! ' + file_name + 'image')

            pBar.update(1)            
            pass

        pbar = tqdm(
            total=len(annos), 
            desc="Downloading Images"
        )
        with ThreadPoolExecutor(int(os.cpu_count() * 1.5)) as executor:
            tasks = []
            for i, anno in enumerate(annos):
                tasks.append(executor.submit(_download_img, anno=anno, pBar=pbar))
            wait(tasks, return_when=ALL_COMPLETED)
        pbar.close()

        # for i, anno in tqdm(enumerate(annos), total = len(annos)):
        #     image_url = anno['data']['Image']
        #     base_name = Path(image_url).name
        #     file_name = urllib.parse.unquote(base_name)
        #     # ip.set_trace()
        #     response = requests.get(image_url)
        #     if response.status_code != 200:
        #         break
        #     bytes_data = response.content
        #     bytes_arr = np.frombuffer(bytes_data, np.uint8)
        #     img = cv2.imdecode(bytes_arr, cv2.IMREAD_COLOR)
        #     try:
        #         cv2.imwrite(imagepath+'/' + file_name , img)
        #     except:
        #         print('Download Error! ' + file_name + 'image')


def convert_label(json_file, outpath, imagepath, ocrmodel):
    annos = json.load(open(json_file))
    # ip.set_trace()
    if not os.path.exists(os.path.join(outpath, 'Labels')):
        os.system("mkdir -p {}".format(os.path.join(outpath, 'Labels')))
    if not os.path.exists(os.path.join(outpath, 'Images/val')):
        os.system("mkdir -p {}".format(os.path.join(outpath, 'Images/val')))
    if not os.path.exists(os.path.join(outpath,'Images/train')):
        os.system("mkdir -p {}".format(os.path.join(outpath,'Images/train')))
    if not os.path.exists(os.path.join(outpath,'Images/ocr_results/val')):
        os.system("mkdir -p {}".format(os.path.join(outpath,'Images/ocr_results/val')))
    if not os.path.exists(os.path.join(outpath,'Images/ocr_results/train')):
        os.system("mkdir -p {}".format(os.path.join(outpath,'Images/ocr_results/train')))
    image_list = []
    label_list = []

    def _process_label(anno, pBar : tqdm):
        # generate images and json files
        image_url = anno['data']['Image']
        bbox_infos = anno['annotations'][0]['result']
        base_name = Path(image_url).name

        file_name = urllib.parse.unquote(base_name)
        fname, ext = os.path.splitext(file_name)
        # Parse bboxes
        bboxes = []
        if len(bbox_infos) < 2:
            return
        lastid = bbox_infos[0].get('id')
        tmp_dict = {
            "category": "",
            "value": "",
            "points": "",
        }
        n = len(bbox_infos)
        clean_flag = False
        for i, info in enumerate(bbox_infos):
            if info.get('type') == 'choices' and info.get('value').get('choices')[0] == 'F002': # 清洗栏为不相关，过滤数据
                # ip.set_trace()
                clean_flag = True
                break

            if info.get('type') == 'labels':
                ori_w = info['original_width']
                ori_h = info['original_height']
                if not info['value'].get('points'):
                    x = info['value']['x']
                    y = info['value']['y']
                    w = info['value']['width']
                    h = info['value']['height']
                    theta = info['value']['rotation']
                    try:
                        label = info['value'].get('labels')[0]
                    except:
                        print('Error! Label Miss in picture!', info['value'])
                        label = ""
                    if label not in label_list:
                        label_list.append(label)
                    w_ = w / 100.0 * ori_w
                    h_ = h / 100.0 * ori_h
                    x_ = x / 100.0 * ori_w
                    y_ = y / 100.0 * ori_h
                    rect = convert_rect((x_, y_, w_, h_, theta))
                    tmp_dict['category'] = label
                    tmp_dict['points'] = rect

                if info['value'].get('points'):
                    rect = info['value']['points']
                    label = info['value'].get('labels')[0]
                    tmp_dict['category'] = label
                    tmp_dict['points'] = rect

            elif info.get('type') == 'textarea' and info.get('id') == lastid:
                text = info.get('value').get('text')[0]
                if '#wrong#' in text:
                    # ip.set_trace()
                    text = text.split('#wrong#')[1]
                tmp_dict['value'] = text

            if  tmp_dict.get('category'):
                if i == n - 1:
                    bboxes.append(tmp_dict)
                elif info.get('id') != bbox_infos[i+1].get('id'):
                    bboxes.append(tmp_dict)
                    tmp_dict = {
                        "category": "",
                        "value": "",
                        "points": "",
                    }
            lastid = info.get('id')
        if clean_flag == True:
            pBar.update(1)
            return
        image_list.append(fname + ext)
        with open(os.path.join(outpath+'/Labels', fname+".json"), 'w', encoding='utf-8') as f:
            json.dump(bboxes, f, ensure_ascii=False, indent=2)
            pBar.update(1)

    # Added : multithreading
    pBar = tqdm(
        total=len(annos), 
        desc="Processing Labels"
    )
    for anno in annos:
        with ThreadPoolExecutor(int(os.cpu_count() * 1.5)) as executor:
            futures = []
            futures.append(executor.submit(_process_label, anno=anno, pBar=pBar))
        wait(futures, return_when=ALL_COMPLETED)
    pBar.close()
        
    # ip.set_trace()
    image_num = len(image_list)
    np.random.seed(42)
    np.random.shuffle(image_list) # uses numpy's random.shuffle, faster?
    val_image = image_list[0: int(image_num*0.2)]
    train_image = image_list[int(image_num*0.2):]
    for valimage in val_image:
        if os.path.exists(os.path.join(imagepath, valimage)):
            shutil.copy(os.path.join(imagepath, valimage), os.path.join(outpath, 'Images/val'))


    for trainimage in train_image:
        if os.path.exists(os.path.join(imagepath, trainimage)):
            shutil.copy(os.path.join(imagepath, trainimage), os.path.join(outpath, 'Images/train'))


    # generate meta.yaml file
    textlist = ['attributes:',
    'difficulty: easy',
    'language: Chinese',
    'license: public',
    'exclude_files: []',
    'exclude_keys: []',
    'field_def:']
    with open(os.path.join(outpath, 'meta.yaml'), 'w') as f:
        for text in textlist:
            f.write(text + '\n')
        for label in label_list:
            f.write('- ' + label + '\n')

    # generatef ocr_results
    # Added :: Multithreaded API Calls
    def _img2ocr(img, pOut):
        ocr_res = api_call(img, ocrmodel)['data']['json']['general_ocr_res']
        nImg = Path(img).stem
        with open(os.path.join(pOut, nImg+'.json'), 'w', encoding='utf-8') as f:
            json.dump(ocr_res, f, ensure_ascii=False, indent=2)
        f.close()
        return True
    
    def _t_imgs2ocr(imgs, pOut):
        # predict & save results
        pbar = tqdm(
            total=len(imgs), 
            desc="obtaining OCR results"
        )
        with ThreadPoolExecutor(int(os.cpu_count() * 1.5)) as executor:
            futures = [executor.submit(_img2ocr, img, pOut) for img in imgs]
            for _ in as_completed(futures):
                pbar.update(1)
        pbar.close()
        pass

    # train imgs
    imgs = [p for p in Path(os.path.join(outpath,'Images/train')).iterdir()]
    _t_imgs2ocr(imgs, os.path.join(outpath,'Images/ocr_results/train'))

    # val imgs
    imgs = [p for p in Path(os.path.join(outpath,'Images/val')).iterdir()]
    _t_imgs2ocr(imgs, os.path.join(outpath,'Images/ocr_results/val'))


    # for filename in tqdm(os.listdir(os.path.join(outpath,'Images/train'))):
    #     image_file = os.path.join(outpath,'Images/train/' + filename)
    #     ocr_res = api_call(image_file, ocrmodel)['data']['json']['general_ocr_res']
    #     save_train_ocr_folder = os.path.join(outpath,'Images/ocr_results/train')
    #     with open(os.path.join(save_train_ocr_folder, filename.split('.')[0]+'.json'), 'w') as f:
    #         json.dump(ocr_res, f, ensure_ascii=False, indent=2)

    # for filename in tqdm(os.listdir(os.path.join(outpath,'Images/val'))):
    #     image_file = os.path.join(outpath,'Images/val/' + filename)
    #     ocr_res = api_call(image_file, ocrmodel)['data']['json']['general_ocr_res']
    #     save_val_ocr_folder = os.path.join(outpath,'Images/ocr_results/val')
    #     with open(os.path.join(save_val_ocr_folder, filename.split('.')[0]+'.json'), 'w') as f:
    #         json.dump(ocr_res, f, ensure_ascii=False, indent=2)


def main():
    args = get_args()

    # create out dir
    Path.mkdir(Path(args.output_file), exist_ok=True)

    # 同目录下添加 model.xlsx表格（带表格列名），映射场景（和文件夹下json文件同名）到识别模型
    model_config = pd.read_excel(io = args.data_dir + '/model.xlsx')
    data_dir = args.data_dir
    scene_model = {}
    for row in model_config.iterrows():
        scene_model[row[1][0]] = row[1][1]
    for label in os.listdir(data_dir):
        if isinstance(label, str) and not label.endswith(".json"):
            continue
        if os.path.isdir(os.path.join(data_dir, label)):
            continue
        output_path = os.path.join(args.output_file, label.split('.')[0])
        if not os.path.exists(output_path):
            os.mkdir(output_path)
            # os.chdir(output_path) # wtf??????????
        pInput = os.path.join(data_dir, label)
        print('scene:', label, 'nowuse:', scene_model[label.split('.json')[0]])
        download_images(pInput, 'download_images')
        convert_label(os.path.join(data_dir, label), output_path, 'download_images', ocrmodel = scene_model[label.split('.json')[0]])
        os.system('rm -r download_images')


if __name__ == '__main__':
    main()
