import base64
import json
import os
import shutil
import urllib
from pathlib import Path
from pprint import pprint

import cv2
import numpy as np
import requests
from tqdm import tqdm

color_range = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


class OcrStudioApi:
    def __init__(self, ip=None, port=None, folder_id=None) -> None:
        self.username = 'test'
        self.password = 'test123456'
        self.ip = ip
        self.port = port
        self.foler_id = folder_id
        self.folder_url = f'http://{self.ip}:{self.port}/smartarchive/api/v1/training-task/query-list-page-merge?pageSize=100&pageNum=1&folderId={self.foler_id}'
        self.task_url = f'http://{self.ip}:{self.port}/smartarchive/api/v1/training-task-dataset/images/?pageSize=1000&pageNum=1&taskId='
        self.label_url = f'http://{self.ip}:{self.port}/smartarchive/api/v1/training-task-iteration-dataset/label/'

        if self.username and self.password:
            b64_auth = base64.b64encode(
                f'{self.username}:{self.password}'.encode('utf-8')
            ).decode('utf-8')
            self.headers = {'Authorization': f'Basic {b64_auth}'}
            self.task_list = self.get_task_list()
            self.taskid_to_name = self.get_taskid_to_name()

    def get_task_list(self):
        # Send the request with the headers
        response = requests.get(self.folder_url, headers=self.headers)

        # get the task list
        data = response.json()
        task_list = data['data']['trainingTaskList']

        return task_list

    def get_taskid_to_name(self):
        task_list = self.get_task_list()
        taskid_to_name = {i['id']: i['taskName'] for i in task_list}

        return taskid_to_name

    def get_all_tasks_labels(self, output_path):
        for taskid in self.taskid_to_name.keys():
            self.get_task_imgs(taskid, output_path)
            self.get_task_labels(taskid, output_path)

    def get_task_labels(self, task_id, dst):
        task_url = self.task_url + str(task_id)
        r = requests.get(task_url, headers=self.headers)
        result = r.json()
        item_list = result["data"]["list"]
        for item in tqdm(item_list, desc=f'{self.taskid_to_name[task_id]} labels'):
            img_id = item["taskDatasetId"]
            img_name = item["fileName"]
            filename = Path(img_name).with_suffix(".json")
            output_path = Path(dst) / self.taskid_to_name[task_id] / "Labels"
            output_path.mkdir(exist_ok=True, parents=True)
            r1 = requests.get(
                self.label_url + str(img_id),
                headers=self.headers,
            )
            if r1.json().get("data") is not None:
                label_info = json.loads(r1.json()["data"])
                with open(Path(output_path) / filename, "w") as f:
                    json.dump(label_info, f, ensure_ascii=False, indent=2)

    def get_task_imgs(self, task_id, dst):
        images_url = f'http://{self.ip}:{self.port}/smartarchive/api/v1/training-task-iteration-dataset/images?imageName=&error=false&taskId={task_id}&pageSize=1000&pageNum=1'
        response = requests.get(images_url, self.headers)
        images_info = response.json()
        img_list = images_info['data']['pageInfo']['list']
        for img_info in tqdm(img_list, desc=f'{self.taskid_to_name[task_id]} images'):
            img_filename = img_info['fileName']
            img_url = img_info['fileUrl']
            r = requests.get(img_url, headers=self.headers)
            bytes_data = r.content
            bytes_arr = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(bytes_arr, cv2.IMREAD_COLOR)
            output_path = Path(dst) / self.taskid_to_name[task_id] / "Images"
            output_path.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(output_path / f'{img_filename}'), img)

    def convert_to_mrcnn_and_save(self, scene_path, dst):
        """
        Args:
            scene_path : 二级目录，scene_path / Labels / xx.json
        """
        json_files = list(Path(scene_path).glob('Labels/[!.]*'))
        for json_file in tqdm(json_files):
            save_label_folder = Path(dst) / 'txts'
            save_label_folder.mkdir(exist_ok=True, parents=True)
            self.convert_to_mrcnn(json_file, save_label_folder)

    def convert_to_mrcnn(self, json_file, dst):
        # print(json_file)
        with open(json_file, 'r') as f:
            img_label = json.load(f)
            table_bboxes = []
            for elem in img_label:
                if elem['value'] != '$水印$':
                    table_bboxes.append(elem['points'])

        # save txt file
        txt_name = json_file.stem + '.txt'
        with open(dst / txt_name, 'w') as f:
            for bbox in table_bboxes:
                bbox = np.array(bbox).reshape(-1).tolist()
                bbox = list(map(str, bbox))
                line = ','.join(bbox)
                f.write(line + '\n')

    def draw_bboxes_from_label(self, file_path, json_label_path):
        with open(json_label_path, 'r') as f:
            img_label = json.load(f)
            table_bboxes = []
            for elem in img_label:
                if elem['value'] != '$水印$':
                    table_bboxes.append(elem['points'])

            img = cv2.imread(file_path)
            im_show = self.draw_boxes(img, table_bboxes)

            return im_show

    @staticmethod
    def draw_boxes(image, boxes, arrowedline=True, scores=None, drop_score=0.5):
        if scores is None:
            scores = [1] * len(boxes)
        for box, score in zip(boxes, scores):
            if score < drop_score:
                continue
            box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
            image = cv2.polylines(np.array(image), [box], True, color_range[2], 3)
            if arrowedline:
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

    def convert_to_label_studio(self):
        pass

    def label_studio_converter(self):
        pass

    def rename_with_correction(self, src, dst):
        Path(dst).mkdir(exist_ok=True, parents=True)
        src_files_path = list(Path(src).glob('[!.]*'))
        for file in tqdm(src_files_path):
            for wrong, correct in self.refine_map.items():
                if wrong in file.name:
                    file.rename(Path(dst) / file.name.replace(wrong, correct))
                    print(f'rename {wrong} to {correct}')
                # else:
                #     shutil.copy(file, dst)

    @staticmethod
    def check_folder(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            shutil.rmtree(folder)
            os.makedirs(folder)

    @staticmethod
    def copy_files(src: str, dst: str, pattern: str = '[!.]*'):
        src_path = Path(src)
        dst_path = Path(dst)
        img_list = list(src_path.glob(f'{pattern}'))
        for img in tqdm(img_list):
            prefix = img.parents[0].name
            filename = f"{prefix}_{img.name}"
            shutil.copyfile(img, dst_path / filename)


if __name__ == "__main__":
    #http://192.168.106.133:8088/task/info/scenario?taskId=235
    ip_address = "192.168.106.133"
    port = 8088
    folder_id = 5  # 10
    json_oup = '../output'
    txt_oup = '../output/'

    ocr_studio = OcrStudioApi(ip=ip_address, port=port, folder_id=folder_id)
    # ocr_studio.get_all_tasks_labels(json_oup)
    ocr_studio.get_task_labels(task_id=40, dst=json_oup)
    ocr_studio.get_task_imgs(task_id=40, dst=json_oup)
    # ocr_studio.convert_to_mrcnn_and_save(json_oup, txt_oup)
    # ocr_studio.rename_with_correction(ori_txts, ori_txts)

    # text_det_path = '/Users/youjiachen/Desktop/projects/label_studio_mgr/data/test_rotate/水平/changwai_table_p2'
    # dst = '/Users/youjiachen/Desktop/projects/label_studio_mgr/data/test_rotate/水平'
    # ocr_studio.convert_to_mrcnn_and_save(text_det_path, dst)
