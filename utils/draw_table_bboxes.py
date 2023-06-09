import copy
import json
import pdb
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

color_range = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def draw_boxes(image, boxes, is_table_bbox=False, scores=None, drop_score=0.5):
    if scores is None:
        scores = [1] * len(boxes)
    for box, score in zip(boxes, scores):
        if score < drop_score:
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, color_range[2], 3)
        if is_table_bbox:
            image = cv2.arrowedLine(
                image,
                (int(box[0][0][0]), int(box[0][0][1])),
                (int(box[1][0][0]), int(box[1][0][1])),
                color=(0, 255, 0),
                thickness=3,
                line_type=cv2.LINE_4,
                shift=0,
                tipLength=0.1,
            )
    return image


def draw_all_bboxes_row_col(input_path: str) -> None:
    """
    the types of label: col, row, span, table.
    draw all types of bboxes and show them in different imageview.
    And the dst is input_path/check_img.
    :param input_path: input_path/images/xxx.png or xxx.jpg ...
    """
    # imgs_path = list(Path(input_path).glob('images/[!.]*'))
    imgs_path = list(Path(input_path).glob('Images/[!.]*'))
    output_path = str(imgs_path[0].parent.parent) + '/check_img'
    Path(output_path).mkdir(exist_ok=True, parents=True)

    for img_pt in tqdm(imgs_path):
        img = cv2.imread(str(img_pt))
        table_vis_img = copy.deepcopy(img)
        row_vis_img = copy.deepcopy(img)
        col_vis_img = copy.deepcopy(img)
        header_vis_img = copy.deepcopy(img)
        span_vis_img = copy.deepcopy(img)

        # label = str(deepcopy(img_pt).with_suffix('.json')).replace('images', 'labels')
        label = str(deepcopy(img_pt).with_suffix('.json')).replace('Images', 'Labels')
        # print(label)
        with open(label, 'r') as f:
            label_info = json.load(f)
        # print(label_info)
        for ids, i in enumerate(label_info, start=1):
            if i['category'] == '行':
                bbox = i['points']
                row_vis_img = draw_boxes(row_vis_img, [bbox], is_table_bbox=True)
                row_vis_img = draw_texts(row_vis_img, ids, bbox)
            elif i['category'] == '列':
                bbox = i['points']
                col_vis_img = draw_boxes(col_vis_img, [bbox], is_table_bbox=True)
                col_vis_img = draw_texts(col_vis_img, ids, bbox)
            elif i['category'] == '表格':
                bbox = i['points']
                table_vis_img = draw_boxes(table_vis_img, [bbox], is_table_bbox=True)
                table_vis_img = draw_texts(table_vis_img, ids, bbox)
            elif i['category'] == '表头':
                bbox = i['points']
                header_vis_img = draw_boxes(header_vis_img, [bbox], is_table_bbox=True)
                header_vis_img = draw_texts(header_vis_img, ids, bbox)
            if i['category'] == '合并单元格':
                bbox = i['points']
                span_vis_img = draw_boxes(span_vis_img, [bbox], is_table_bbox=True)
                span_vis_img = draw_texts(span_vis_img, ids, bbox)

        img1 = np.concatenate([img, row_vis_img, col_vis_img], axis=1)
        img2 = np.concatenate([table_vis_img, header_vis_img, span_vis_img], axis=1)
        # img1 = img
        # img2 = span_vis_img
        img_show = np.concatenate([img1, img2], axis=0)

        cv2.imwrite(
            f'{output_path}/{img_pt.with_suffix(".jpeg").name}',
            img_show,
            [int(cv2.IMWRITE_JPEG_QUALITY), 30],
        )


def draw_all_bboxes_cells(input_path: str, is_abn=False) -> None:
    """
    the types of label: table, cell
    draw all types of bboxes and show them in different imageviews.
    And the dst is input_path/check_img.
    :param input_path: input_path/images/xxx.png or xxx.jpg ...
    """
    # imgs_path = list(Path(input_path).glob('images/[!.]*'))
    imgs_path = list(Path(input_path).glob('Images/[!.]*'))
    output_path = str(imgs_path[0].parent.parent) + '/check_img'
    Path(output_path).mkdir(exist_ok=True, parents=True)

    for img_pt in tqdm(imgs_path):
        img = cv2.imread(str(img_pt))
        table_vis_img = copy.deepcopy(img)
        cell_vis_img = copy.deepcopy(img)

        # label = str(deepcopy(img_pt).with_suffix('.json')).replace('images', 'labels')
        if is_abn:
            label = str(deepcopy(img_pt).with_suffix('.json')).replace(
                'Images', 'exception_label'
            )
        else:
            label = str(deepcopy(img_pt).with_suffix('.json')).replace(
                'Images', 'Labels'
            )
        # print(label)
        try:
            with open(label, 'r') as f:
                label_info = json.load(f)
        except:
            continue
        # print(label_info)
        for ids, i in enumerate(label_info):
            if i['category'] == '单元格':
                bbox = i['points']
                cell_vis_img = draw_boxes(cell_vis_img, [bbox], is_table_bbox=True)
                cell_vis_img = draw_texts(cell_vis_img, ids, bbox)
            elif i['category'] == '表格':
                bbox = i['points']
                table_vis_img = draw_boxes(table_vis_img, [bbox], is_table_bbox=True)
                table_vis_img = draw_texts(table_vis_img, ids, bbox)

        img_show = np.concatenate([img, table_vis_img, cell_vis_img], axis=1)
        cv2.imwrite(f'{output_path}/{img_pt.name}', img_show)


def draw_texts(img: str, texts, bbox: list):
    return cv2.putText(
        img,
        str(texts),
        (int(float(bbox[0][0])), int(float(bbox[0][1]))),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2,
        thickness=3,
        color=(0, 0, 255),
    )


def draw_txt_bboxes(path: str, pure_nums=False) -> None:
    """
    标签格式为txt，使用此脚本画bbox
    :param pure_nums: 标签是否为数字，不包含类别信息
    :param path: path/xxx.jpg
    :return:
    """
    p = Path(path)
    for img_file_path in tqdm(list(p.glob('[!.]*'))):
        img = cv2.imread(str(img_file_path))
        # label_file_path = str(img_file_path.with_suffix('.txt')).replace('train', 'txts')
        try:
            label_file_path = str(img_file_path.with_suffix('.txt')).replace(
                'Images', 'txts'
            )
        except Exception as e:
            print(e)
            print(img_file_path)
        output_path = '/'.join([str(img_file_path.parent.parent), 'check_img'])
        Path(output_path).mkdir(exist_ok=True, parents=True)

        bboxes = list()
        with open(label_file_path, 'r') as f:
            data = f.read().strip().split('\n')
            if data[0]:
                for i in data:
                    if not pure_nums:
                        *p, _ = i.split(',')
                    else:
                        p = list(map(float, i.split(',')))
                        bboxes.append(p)

                    # print(img_file_path)
        try:
            im_show = draw_boxes(img, bboxes, is_table_bbox=True)
        except Exception:
            shutil.copy(
                img_file_path,
                '/Volumes/T7-500G/数据/基础检测模型/model_第四批训练数据/内部数据/erro_img',
            )
            continue

        cv2.imwrite(f'{output_path}/{img_file_path.name}', im_show)


def label_studio_longtext_demo():
    with open('post_processed_result.json', 'r') as f:
        data = json.load(f)

    for info in data:
        draw_pics_list = defaultdict(list)
        for anno in info['annotations']:
            draw_pics_list[anno['page_name']].append(anno['box'])

        for pic, boxes in draw_pics_list.items():
            img_path = f'workspace/long_text/long_text_contract_ds/Images/{pic}.png'
            output_path = (
                f'workspace/long_text/long_text_contract_ds/check_img/{pic}.png'
            )
            img = cv2.imread(img_path)
            image = draw_boxes(img, boxes)
            cv2.imwrite(output_path, image)


if __name__ == '__main__':
    src = '/Users/youjiachen/Desktop/projects/label_studio_mgr/data/test_rotate/水平/changwai_table_p2'
    draw_all_bboxes_row_col(src)

    # p = '/Users/youjiachen/Desktop/liushui_text_det_ds/liushui_text_det_p2/Images'
    # draw_txt_bboxes(p, True)
    # import urllib

    # import pandas as pd

    # img_path = '/Users/youjiachen/Desktop/projects/label_studio_mgr/liushui_table_stru_p2/test_rotate/Images'
    # img_files = list(Path(img_path).glob('[!.]*'))
    # tmp_dict = dict()

    # for img in img_files:
    #     encode_name = urllib.parse.quote(img.name)
    #     tmp_dict[img.name] = encode_name

    # df = pd.DataFrame.from_dict(tmp_dict, orient='index')
    # df.to_excel('./图片名对应url.xlsx')
