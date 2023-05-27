import copy
import json
import math
import os
import shutil
import urllib.parse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import cv2
import numpy as np
from tqdm import tqdm


class LabelStudio:
    def __init__(self) -> None:
        pass

    def check_folder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            shutil.rmtree(folder)
            os.makedirs(folder)

    def label_studio_to_mrcnn(self, label_path, output_path):
        Path(output_path).mkdir(exist_ok=True, parents=True)
        with open(label_path, 'r') as f:
            data = json.loads(f.read())

        for raw_expamle in tqdm(data):
            img_name = Path(raw_expamle['data']['Image']).name
            decode_img_name = urllib.parse.unquote(img_name)

            bboxes = []
            result_list = raw_expamle['annotations'][0]['result']
            for label in result_list:
                if label['value'].get('labels'):
                    x, y, w, h = self.convert_from_ls(label)
                    angle = label['value']['rotation']
                    box = self.convert_rect([x, y, w, h, angle])
                    bboxes.append(box)

            _oup_path = Path(output_path) / decode_img_name
            oup_path = _oup_path.with_suffix('.txt')

            with open(oup_path, 'w', encoding='utf-8') as f:
                for bbox in bboxes:
                    bbox = np.array(bbox).reshape(-1).tolist()
                    bbox = list(map(str, bbox))
                    line = ','.join(bbox)
                    f.write(line + '\n')

    @staticmethod
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

    @staticmethod
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


if __name__ == "__main__":
    pass
