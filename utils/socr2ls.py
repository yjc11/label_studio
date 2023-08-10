import copy
import json
import shutil
import time
import urllib
from ast import List
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from tqdm import tqdm

'''
配置一个唯一的ruid
props： None
return： 
'''


class RuidGet(object):
    '''
    配置一个唯一的ruid
    props： None
    return：
    '''

    @classmethod
    def get_str_ruid(cls):
        '''
        获取16进制字符串唯一id
        :return:
        '''
        base_time = round(
            time.mktime(time.strptime('1970-01-02 00:00:00', '%Y-%m-%d %H:%M:%S'))
            * 10**3
        )

        ruid = round(time.time() * 10**3) - base_time
        time.sleep(0.001)
        return str(hex(ruid)).replace('0x', '')

    @classmethod
    def get_int_ruid(cls):
        '''
        获取10进制整数唯一id
        :return:
        '''
        base_time = round(
            time.mktime(time.strptime('1970-01-02 00:00:00', '%Y-%m-%d %H:%M:%S'))
            * 10**3
        )

        ruid = round(time.time() * 10**3) - base_time
        time.sleep(0.001)
        return str(ruid)


# bbox = [[p1] [p2] [p3] [p4]]
# reference: https://labelstud.io/tags/rectanglelabels.html
def euclidean_distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.sqrt(dx**2 + dy**2)


def bbox2ls(bbox, original_width, original_height):
    bbox = np.array(bbox)
    tl, tr, br, bl = bbox

    x, y = tl

    x_ = x / original_width * 100
    y_ = y / original_height * 100

    # width and height
    w = euclidean_distance(tl, tr) / original_width * 100
    h = euclidean_distance(tl, bl) / original_height * 100

    # get top line vector
    dy = tr[1] - tl[1]
    dx = tr[0] - tl[0]
    # get randians
    angle = np.arctan2(dy, dx)
    # convert to degrees
    r = angle * 180 / np.pi

    # fix value range
    if r < 0:
        r += 360
    # if (r >= 360): r -= 360 # don't really need this since -pi <= arctan2(x, y) <= pi

    return x_, y_, w, h, r


def demo_1():
    src = '/home/youjiachen/workspace/ELLM_V3.0_sample_30'
    output_dir = '/home/youjiachen/workspace/ELLM_V3.0_sample'
    scene_list = Path(src).glob('[!.]*')
    for scene in tqdm(scene_list):
        oup = Path(f'{output_dir}/{scene.name}')
        oup.mkdir(exist_ok=True, parents=True)
        imgs = (scene / 'Images').glob('[!.]*')
        for img in imgs:
            shutil.copy(img, oup)


def socr2ls():
    url_prefix = 'http://192.168.106.8/datasets'
    anno_temps: List = [
        {
            "original_width": 1558,
            "original_height": 1102,
            "image_rotation": 0,
            "value": {
                "x": 13.025700934579438,
                "y": 39.55408753096614,
                "width": 11.79964373524903,
                "height": 2.728974826971771,
                "rotation": 359.4327335901421,
            },
            # "id": "LpuusTPtQa",
            "from_name": "bbox",
            "to_name": "image",
            "type": "rectangle",
            "origin": "manual",
        },
        {
            "original_width": 1558,
            "original_height": 1102,
            "image_rotation": 0,
            "value": {
                "x": 13.025700934579438,
                "y": 39.55408753096614,
                "width": 11.79964373524903,
                "height": 2.728974826971771,
                "rotation": 359.4327335901421,
                "labels": ["原凭证号"],
            },
            # "id": "LpuusTPtQa",
            "from_name": "label",
            "to_name": "image",
            "type": "labels",
            "origin": "manual",
        },
        {
            "original_width": 1558,
            "original_height": 1102,
            "image_rotation": 0,
            "value": {
                "x": 13.025700934579438,
                "y": 39.55408753096614,
                "width": 11.79964373524903,
                "height": 2.728974826971771,
                "rotation": 359.4327335901421,
                "text": ["33416621060001540"],
            },
            # "id": "LpuusTPtQa",
            "from_name": "transcription",
            "to_name": "image",
            "type": "textarea",
            "origin": "manual",
        },
    ]

    data_path = Path('/home/youjiachen/workspace/ELLM_V3.0_sample_30')
    scene_list = list(data_path.glob('[!.]*'))
    result_list = list()
    for scene in tqdm(scene_list):
        # def process_task(scene):
        scene_name = scene.name
        cur_imgs = list(scene.glob('Images/[!.]*'))
        for ind, img in enumerate(cur_imgs):
            templete = {"annotations": [{"result": []}], "data": {}}

            cur_label = img.parents[1] / f'Labels/{img.stem}.json'

            # img url
            par_dir = img.parents[1].name
            page_url = f'{url_prefix}/ELLM_V3.0_sample/{par_dir}/{img.name}'
            page_url = urllib.parse.quote(page_url, safe='://')
            templete['data']['Image'] = page_url
            templete['data']['Index'] = ind + 1
            templete['data']['Tag'] = par_dir

            # annos
            image = cv2.imread(str(img))
            h, w, c = image.shape
            with cur_label.open('r') as f:
                socr_label = json.load(f)
            for label in socr_label:
                cur_id = RuidGet.get_str_ruid()
                category = label['category']
                text = label['value']
                box = label['points']
                x, y, w_, h_, r = bbox2ls(box, w, h)
                _anno_temps = copy.deepcopy(anno_temps)
                for anno_tmp in _anno_temps:
                    anno_tmp['original_width'] = w
                    anno_tmp['original_height'] = h
                    anno_tmp['value']['x'] = x
                    anno_tmp['value']['y'] = y
                    anno_tmp['value']['width'] = w_
                    anno_tmp['value']['height'] = h_
                    anno_tmp['value']['rotation'] = r
                    anno_tmp['id'] = cur_id

                    if anno_tmp['type'] == 'labels':
                        anno_tmp['value']['labels'] = ['印刷中文']

                    elif anno_tmp['type'] == 'rectangle':
                        if '金额' in category and '小写' in category:
                            anno_tmp['meta'] = {'text': ['小写金额']}
                        elif '金额' in category and '大写' in category:
                            anno_tmp['meta'] = {'text': ['大写金额']}
                        elif '日期' in category:
                            anno_tmp['meta'] = {'text': ['日期']}
                        else:
                            anno_tmp['meta'] = {'text': ['其他']}

                    elif anno_tmp['type'] == 'textarea':
                        anno_tmp['value']['text'] = [text]

                    templete['annotations'][0]['result'].append(anno_tmp)
                

            # save
            result_list.append(templete)
            
    with open('./v3test_json.json', 'w') as f:
        json.dump(result_list, f, ensure_ascii=False)


if __name__ == '__main__':
    socr2ls()
