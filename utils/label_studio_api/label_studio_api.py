import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

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

    def get_project(self, project_id):
        """获取项目
        Parameters:
            project_id: 项目id
        Returns:
            project (json) - for detailes see http://192.168.106.7:8080/docs/api#operation/api_projects_validate_create
        """
        url = f"{self.server_url}/api/projects/{project_id}/"
        params = {'exportType': 'JSON'}
        response = requests.get(url, headers=self.headers, params=params)
        return response.json()

    def get_config(self, json_index):
        """自动获取初始化的配置文件
        Parameters:
            json_index: 初始化的json标注
        Returns:
            配置文件的内容
        """
        with open(json_index, 'r', encoding='utf-8') as j:
            json_index_data = json.load(j)
        if list(json_index_data[0].keys())[0] == 'document':
            with open('template_long.yaml', 'r', encoding='utf-8') as file:
                config_template_data = yaml.load(file, Loader=yaml.FullLoader)
        if list(json_index_data[0].keys())[0] == 'Images':
            with open('template_short.yaml', 'r', encoding='utf-8') as file:
                config_template_data = yaml.load(file, Loader=yaml.FullLoader)
        return config_template_data

    def export_annotations(self, id, json_data_export) -> json:
        """根据 project id 导出标注 json
        Parameters:
            id: 标注任务id
            json_data_export: 输出json文件的保存路径
        Returns:
            annos (json): 标注内容
        """
        # todo::用任务名命名json
        url = f"{self.server_url}/api/projects/{id}/export"
        params = {'exportType': 'JSON'}
        response = requests.get(url, headers=self.headers, params=params)
        annos = response.json()

        # out
        with open(json_data_export, 'w', encoding='utf-8') as f:
            json.dump(annos, f, ensure_ascii=False)

        return annos

    def import_tasks(self, project_id, json_data_import):
        """导入标记任务
        Parameters:
            project_id: 项目id号
            json_data_import: 要导入的json文件
        """

        # import tasks data
        headers = copy.deepcopy(self.headers)
        url = f"{self.server_url}/api/projects/{project_id}/import"
        data = {'url': self.server_url}

        # set header
        headers['content-type'] = "application/json"
        with open(json_data_import, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

            data = json.dumps(json_data, ensure_ascii=True)

        # post
        response = requests.post(url, headers=headers, data=data)

    def create_project(
        self,
        name,
        label_config_id=None,
        description='',
        json_data_import=None,
    ):
        """创建新标注任务 (project)
        Parameters:
            name: 项目名称
            label_config_id: 默认为None,即根据导入的json初始化文件自动判断长短文档, 导入初始化标注文件; 或者可以指定项目id
            description: 项目描述
            json_data_import: 初始化的json索引文件

        Returns:
            project_url: 创建项目 url
        """
        url = f"{self.server_url}/api/projects"
        data = {
            "title": name,
            "description": description,
        }

        # set template by type :: DEPRECATED
        if label_config_id != None:
            data['label_config'] = self.get_project(label_config_id)['label_config']
        if label_config_id == None:
            data['label_config'] = self.get_config(json_data_import)

        # post
        response = requests.post(url, headers=self.headers, data=data)

        # get project url
        project_id = response.json()['id']
        project_url = f"{self.server_url}/projects/{project_id}"

        self.import_tasks(project_id, json_data_import)
        print(project_url)
        return project_url

    # usage
    def up_rec(project_id):
        pass


if __name__ == '__main__':
    ip = "192.168.106.7"
    port = '8080'
    token = 'eb71b9f42d1e1377d371d25760d49fe91e227ae1'
    label_studio_api = LabelStudioApi(ip, port, token)

    # 导出标注json文件
    # label_studio_api.export_annotations(id=358, oup=f'./data/{id}.json')

    # 根据指定的任务id导入json文件
    # label_studio_api.import_tasks(id=358, json_data_import='./data/358.json')

    # 创建任务,并导入初始化json文件
    name = '长文档测试3'
    label_id = None
    description = ''
    json_data_import = './data/changwendangceshi.json'
    # label_studio_api.create_project(name, label_id, description, json_data_import)

    # 下载与回传
    # 下载
    id = 358  # 任务id
    json_name = str(label_studio_api.get_project(project_id=id)['title'])
    oup_json = f'./data/{id}.json'  # json的输出路径
    label_studio_api.export_annotations(id, oup_json)
    # 写入识别结果
    oup_json_path = long_text_crop_and_recog(oup_json, Path(oup_json).parent, 30)
    # oup_json_path = short_text_crop_and_recog(oup_json, Path(oup_json).parent, 10)
    # 回传
    name = str(label_studio_api.get_project(project_id=id)['title']) + str(
        '_rec'
    )  # 新项目名称
    label_studio_api.create_project(name, id, description, oup_json_path)
