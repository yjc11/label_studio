import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import requests
import yaml
from tqdm import tqdm

from crop_and_recog import long_text_crop_and_recog, short_text_crop_and_recog


class LabelStudioApi:
    def __init__(self, ip: str, port: str, token: str):
        self.ip = ip
        self.port = port
        self.server_url = f'http://{ip}:{port}'
        self.token = token
        self.headers = {"Authorization": f"Token {self.token}"}

    def get_template_yaml(self, yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as file:
            template = yaml.load(file, Loader=yaml.FullLoader)
        return template

    def get_project_details(self, project_id: int) -> Dict:
        """获取项目信息"""
        url = f"{self.server_url}/api/projects/{project_id}/"
        params = {'exportType': 'JSON'}
        response = requests.get(url, headers=self.headers, params=params)
        return response.json()

    def get_project_label_config(self, project_id: int) -> str:
        """根据项目id得到项目的配置文件"""
        return self.get_project_details(project_id)['label_config']

    def get_init_label_config(self, json_index):
        """自动获取初始化的配置文件"""
        with open(json_index, 'r', encoding='utf-8') as j:
            json_index_data = json.load(j)
        if list(json_index_data[0].keys())[0] == 'document':
            with open('template_long.yaml', 'r', encoding='utf-8') as file:
                config_template_data = yaml.load(file, Loader=yaml.FullLoader)
        if list(json_index_data[0].keys())[0] == 'Images':
            with open('template_short.yaml', 'r', encoding='utf-8') as file:
                config_template_data = yaml.load(file, Loader=yaml.FullLoader)
        return config_template_data

    def export_annotations(self, project_id, output_dir) -> List:
        """根据 project id 导出标注 json"""
        project_details = self.get_project_details(project_id)
        json_name = project_details['title']
        url = f"{self.server_url}/api/projects/{project_id}/export"
        params = {'exportType': 'JSON'}
        response = requests.get(url, headers=self.headers, params=params)
        annos = response.json()

        output_path = Path(output_dir) / f'{json_name}.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annos, f, ensure_ascii=False, indent=2)

        return annos

    def import_tasks(self, project_id, import_json):
        """为已创建且已传入图片任务导入标注"""

        # import tasks data
        headers = copy.deepcopy(self.headers)
        url = f"{self.server_url}/api/projects/{project_id}/import"
        data = {'url': self.server_url}

        # set header
        headers['content-type'] = "application/json"
        with open(import_json, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # post
        data = json.dumps(json_data, ensure_ascii=True)
        response = requests.post(url, headers=headers, data=data)

    def create_project(
        self,
        project_name,
        label_config=None,
        description='',
        import_json=None,
    ):
        """创建新标注任务 (project)
        Parameters:
            name: 项目名称
            label_config_id: 默认为None,即根据导入的json初始化文件自动判断长短文档, 导入初始化标注文件; 或者可以指定项目id
            description: 项目描述
            import_json: 初始化的json索引文件

        Returns:
            project_url: 创建项目 url
        """
        url = f"{self.server_url}/api/projects"
        data = {
            "title": project_name,
            "description": description,
        }

        data['label_config'] = label_config

        # post
        response = requests.post(url, headers=self.headers, data=data)

        # get project url
        project_id = response.json()['id']
        project_url = f"{self.server_url}/projects/{project_id}"

        self.import_tasks(project_id, import_json)
        print(project_url)
        return project_url

    # usage
    def upload_recog(
        self,
        project_id,
        description='',
    ):
        # 写入识别结果
        output_path = f'./data/'  # json的输出路径
        self.export_annotations(project_id, output_path)
        json_name = str(self.get_project_details(project_id)['title'])
        json_file = Path(output_path) / f'{json_name}.json'
        oup_path = long_text_crop_and_recog(json_file, Path(output_path).parent, 30)

        # 回传
        project_name = self.get_project_details(project_id)['title']
        label_config = self.get_project_label_config(project_id)
        upload_project_name = project_name + '_recog'  # 新项目名称
        self.create_project(upload_project_name, label_config, description, oup_path)


if __name__ == '__main__':
    ip = "192.168.106.7"
    port = '8080'
    token = 'eb71b9f42d1e1377d371d25760d49fe91e227ae1'
    label_studio_api = LabelStudioApi(ip, port, token)

    # 导出标注json文件
    # label_studio_api.export_annotations(271, output_dir=f'./data')
    label_studio_api.upload_recog(295)
