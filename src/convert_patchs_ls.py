import base64
import json
import math
import os

import cv2
import numpy as np
import requests


def patch_recog(patchs):
    url = "http://192.168.106.7:2502/predict"
    param = {
        'enable_huarong_box_adjust': True,
        'support_long_image_segment': True,
        'recog': 'transformer-blank-v0.2-faster',
    }

    data = [base64.b64encode(cv2.imencode('.png', p)[1]).decode() for p in patchs]
    json_content = {"app_name": "ocr_general_v3", "data": data, 'param': param}
    r = requests.post(url, json=json_content).json()
    return r


def l2_norm(pt0, pt1):
    return np.sqrt((pt0[0] - pt1[0]) ** 2 + (pt0[1] - pt1[1]) ** 2)


def crop_images(img, bboxes):
    n = bboxes.shape[0]
    mats = []
    for i in range(n):
        bbox = bboxes[i]
        ori_w = np.round(l2_norm(bbox[0], bbox[1]))
        ori_h = np.round(l2_norm(bbox[1], bbox[2]))
        new_w = ori_w
        new_h = ori_h
        src_3points = np.float32([bbox[0], bbox[1], bbox[2]])
        dest_3points = np.float32([[0, 0], [new_w, 0], [new_w, new_h]])
        M = cv2.getAffineTransform(src_3points, dest_3points)
        m = cv2.warpAffine(img, M, (int(new_w), int(new_h)))
        mats.append(m)

    return mats


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


def parse(json_file, output_file):
    annos = json.load(open(json_file))
    json_content = {}

    content = []
    url_prefix = "http://192.168.106.8/datasets/patchs"

    for anno in annos:
        image_url = anno['data']['ocr']
        bbox_infos = anno['annotations'][0]['result']
        ori_w = bbox_infos[0]['original_width']
        ori_h = bbox_infos[0]['original_height']
        # Parse bboxes
        bboxes = []
        base_name = os.path.basename(image_url).rsplit('.', 1)[0]
        dir_name = os.path.basename(image_url.rsplit('/', 1)[0])

        for info in bbox_infos:
            if info.get('type', '') == 'rectangle':
                x = info['value']['x']
                y = info['value']['y']
                w = info['value']['width']
                h = info['value']['height']
                theta = info['value']['rotation']
                w_ = w / 100.0 * ori_w
                h_ = h / 100.0 * ori_h
                x_ = x / 100.0 * ori_w
                y_ = y / 100.0 * ori_h
                rect = convert_rect((x_, y_, w_, h_, theta))
                bboxes.append(rect)
        bboxes = np.array(bboxes)

        # Parse Image
        response = requests.get(image_url)
        bin_image = None
        if response.status_code != 200:
            break
        bytes_data = response.content
        bytes_arr = np.fromstring(bytes_data, np.uint8)
        img = cv2.imdecode(bytes_arr, cv2.IMREAD_COLOR)
        patchs = crop_images(img, bboxes)

        resp = patch_recog(patchs)
        texts = []
        try:
            if resp['status']['code'] == 200:
                texts = resp['result']['contents']['texts']
            else:
                texts = []
        except Exception:
            pass

        output_dir = './data/patchs'
        for i, patch in enumerate(patchs):
            file_name = "{}_{:03d}.png".format(base_name, i)
            out_d = os.path.join(output_dir, dir_name)
            if not os.path.exists(out_d):
                os.makedirs(out_d)
            out_f = os.path.join(out_d, file_name)
            cv2.imwrite(out_f, patch)

            text = texts[i]
            patch_url = "{}/{}/{}".format(url_prefix, dir_name, file_name)
            content.append({'image': patch_url, 'text1': text, 'text2': text})

    with open(output_file, 'w') as fout:
        json.dump(content, fout, indent=4)


if __name__ == "__main__":
    json_file = './data/uie_1_1.json'
    out_file = './data/patchs_1.json'
    parse(json_file, out_file)
